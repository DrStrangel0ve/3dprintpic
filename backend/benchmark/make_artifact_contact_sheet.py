import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.benchmark.metrics import align_depth, load_bool_mask, load_mask


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_FIELDS = [
    ("full_image", "Full"),
    ("masked_image", "Masked"),
    ("mask", "Mask"),
]

OUTPUT_FIELDS = [
    ("raw_completed_image", "Raw"),
    ("completed_image", "Completed"),
]

METRIC_FIELDS = [
    ("masked_mae", "masked mae"),
    ("object_masked_mae", "object mae"),
    ("seam_mae", "seam mae"),
    ("depth_mae", "depth mae"),
    ("object_depth_mae", "object depth"),
    ("object_surface_chamfer_l1", "obj surf chamfer"),
    ("object_surface_chamfer_rmse", "obj surf rmse"),
    ("object_surface_hausdorff95", "obj surf h95"),
    ("mesh_surface_chamfer_l1", "mesh chamfer"),
    ("mesh_surface_hausdorff95", "mesh h95"),
    ("silhouette_iou_masked", "silhouette iou"),
    ("stl_is_watertight", "watertight"),
    ("stl_positive_volume", "positive volume"),
    ("stl_faces", "stl faces"),
]


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def discover_metric_paths(input_dir: Path) -> list[Path]:
    root_metrics = input_dir / "per_sample_metrics.csv"
    if root_metrics.exists():
        return [root_metrics]
    return sorted(path for path in input_dir.glob("*/per_sample_metrics.csv") if path.is_file())


def resolve_path(value, anchors: list[Path] | None = None) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path

    candidates = []
    for anchor in anchors or []:
        candidates.append(anchor / path)
    candidates.extend([PROJECT_ROOT / path, Path.cwd() / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else PROJECT_ROOT / path


def resolve_existing_path(value, anchors: list[Path] | None = None) -> Path | None:
    path = resolve_path(value, anchors)
    if path and path.exists():
        return path
    return None


def find_manifest_path(input_dir: Path, metric_paths: list[Path], explicit_manifest: str | None = None) -> Path | None:
    if explicit_manifest:
        return resolve_path(explicit_manifest, [input_dir])

    audit_paths = [input_dir / "split_audit.json"]
    audit_paths.extend(path.parent / "split_audit.json" for path in metric_paths)
    for audit_path in audit_paths:
        audit = read_json(audit_path)
        manifest = audit.get("manifest")
        if manifest:
            resolved = resolve_path(manifest, [audit_path.parent, input_dir])
            if resolved and resolved.exists():
                return resolved
    return None


def load_manifest(manifest_path: Path | None) -> dict[str, dict]:
    if not manifest_path or not manifest_path.exists():
        return {}
    samples = {}
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample_id = sample.get("id")
            if sample_id:
                samples[str(sample_id)] = sample
    return samples


def load_metric_rows(input_dir: Path) -> tuple[list[dict], list[Path]]:
    metric_paths = discover_metric_paths(input_dir)
    if not metric_paths:
        raise FileNotFoundError(f"No per_sample_metrics.csv found in {input_dir}")

    rows = []
    for metrics_path in metric_paths:
        for row in read_csv_rows(metrics_path):
            row = dict(row)
            row["_metrics_path"] = str(metrics_path)
            row["_run_dir"] = str(metrics_path.parent)
            row["_experiment_name"] = metrics_path.parent.name
            row["_method"] = str(row.get("method") or metrics_path.parent.name)
            rows.append(row)
    return rows, metric_paths


def ordered_unique(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def row_method_names(row: dict) -> set[str]:
    return {
        str(value)
        for value in (
            row.get("_method"),
            row.get("method"),
            row.get("base_method"),
            row.get("_experiment_name"),
        )
        if str(value or "").strip()
    }


def default_method_order(rows: list[dict]) -> list[str]:
    methods = ordered_unique(str(row.get("_method") or row.get("method") or "") for row in rows)
    priority = ["masked", "mirror", "biharmonic"]
    ordered = [method for method in priority if method in methods]
    ordered.extend(sorted(method for method in methods if method not in priority))
    return ordered


def filter_rows(rows: list[dict], methods: list[str] | None, samples: list[str] | None, max_samples: int) -> list[dict]:
    rows = [row for row in rows if row.get("sample_id")]
    if methods:
        method_set = set(methods)
        rows = [row for row in rows if row_method_names(row) & method_set]

    if samples:
        sample_order = samples
    else:
        sample_order = ordered_unique(str(row["sample_id"]) for row in rows)[:max_samples]
    sample_set = set(sample_order)
    method_order = methods or default_method_order(rows)
    method_index = {method: index for index, method in enumerate(method_order)}
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_order)}

    filtered = [row for row in rows if str(row["sample_id"]) in sample_set]

    def method_sort_index(row):
        matching_indexes = [method_index[name] for name in row_method_names(row) if name in method_index]
        if matching_indexes:
            return min(matching_indexes)
        return method_index.get(str(row.get("_method") or row.get("method")), 10**9)

    filtered.sort(
        key=lambda row: (
            sample_index.get(str(row["sample_id"]), 10**9),
            method_sort_index(row),
            str(row.get("_experiment_name") or ""),
        )
    )
    return filtered


def infer_depth_artifact(row: dict) -> Path | None:
    anchors = [Path(row["_run_dir"]), PROJECT_ROOT]
    for field in (
        "depth_preview",
        "output_depth_preview",
        "depth_image",
        "depth_png",
        "depth_data",
        "output_depth_data",
    ):
        path = resolve_path(row.get(field), anchors)
        if path and path.exists():
            return path

    for field in ("stl_model", "completed_image", "raw_completed_image"):
        path = resolve_path(row.get(field), anchors)
        if not path:
            continue
        parent = path.parent
        for name in ("output_depth_preview.png", "output_depth_data.npy"):
            candidate = parent / name
            if candidate.exists():
                return candidate
    return None


def infer_depth_data_artifact(row: dict) -> Path | None:
    anchors = [Path(row["_run_dir"]), PROJECT_ROOT]
    for field in ("depth_data", "output_depth_data"):
        path = resolve_path(row.get(field), anchors)
        if path and path.exists() and path.suffix.lower() == ".npy":
            return path

    for field in ("stl_model", "completed_image", "raw_completed_image", "depth_preview"):
        path = resolve_path(row.get(field), anchors)
        if not path:
            continue
        candidate = path.parent / "output_depth_data.npy"
        if candidate.exists():
            return candidate
    return None


def reference_artifacts(row: dict, manifest: dict[str, dict], manifest_path: Path | None) -> dict[str, Path | None]:
    sample = manifest.get(str(row.get("sample_id")), {})
    anchors = [Path(row["_run_dir"]), PROJECT_ROOT]
    if manifest_path:
        anchors.insert(0, manifest_path.parent)
    artifacts = {}
    for field, _ in REFERENCE_FIELDS:
        artifacts[field] = resolve_existing_path(row.get(field), anchors) or resolve_path(sample.get(field), anchors)
    return artifacts


def reference_artifact_path(
    row: dict,
    manifest: dict[str, dict],
    manifest_path: Path | None,
    fields: tuple[str, ...],
) -> Path | None:
    sample = manifest.get(str(row.get("sample_id")), {})
    anchors = [Path(row["_run_dir"]), PROJECT_ROOT]
    if manifest_path:
        anchors.insert(0, manifest_path.parent)
    for field in fields:
        path = resolve_existing_path(row.get(field), anchors)
        if path:
            return path
    for field in fields:
        path = resolve_existing_path(sample.get(field), anchors)
        if path:
            return path
    return None


def output_artifacts(row: dict) -> dict[str, Path | None]:
    anchors = [Path(row["_run_dir"]), PROJECT_ROOT]
    artifacts = {field: resolve_path(row.get(field), anchors) for field, _ in OUTPUT_FIELDS}
    artifacts["depth"] = infer_depth_artifact(row)
    artifacts["depth_data"] = infer_depth_data_artifact(row)
    artifacts["stl"] = resolve_path(row.get("stl_model"), anchors)
    return artifacts


def format_number(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value or "").strip()
        return text
    if not math.isfinite(number):
        return ""
    if number == 0:
        return "0"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) < 0.001:
        return f"{number:.2e}"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def metric_lines(row: dict) -> list[str]:
    lines = []
    for field, label in METRIC_FIELDS:
        value = row.get(field)
        formatted = format_number(value)
        if formatted:
            lines.append(f"{label}: {formatted}")
    return lines or ["no metrics"]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int, max_lines: int | None = None) -> list[str]:
    words = str(text or "").split()
    lines = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if text_width(draw, candidate, font) <= width:
            line = candidate
            continue
        if line:
            lines.append(line)
        line = word
        while text_width(draw, line, font) > width and len(line) > 1:
            chunk = line[:-1]
            while chunk and text_width(draw, chunk, font) > width:
                chunk = chunk[:-1]
            if chunk:
                lines.append(chunk)
                line = line[len(chunk) :]
            else:
                break
        if max_lines and len(lines) >= max_lines:
            break
    if line and (not max_lines or len(lines) < max_lines):
        lines.append(line)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
    if max_lines and len(lines) == max_lines and text_width(draw, lines[-1], font) > width:
        lines[-1] = lines[-1][: max(1, len(lines[-1]) - 1)]
    return lines


def placeholder(size: int, label: str) -> Image.Image:
    image = Image.new("RGB", (size, size), "#eef1f5")
    draw = ImageDraw.Draw(image)
    font = load_font(12)
    lines = wrap_text(draw, label, font, size - 16, max_lines=3)
    total_height = len(lines) * 15
    y = (size - total_height) // 2
    for line in lines:
        x = (size - text_width(draw, line, font)) // 2
        draw.text((x, y), line, fill="#6b7280", font=font)
        y += 15
    draw.rectangle((0, 0, size - 1, size - 1), outline="#c8d0dc")
    return image


def depth_npy_image(path: Path) -> Image.Image:
    depth = np.load(path)
    depth = np.asarray(depth, dtype=np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        scaled = np.zeros(depth.shape, dtype=np.uint8)
    else:
        low = float(np.percentile(finite, 2))
        high = float(np.percentile(finite, 98))
        if high <= low:
            high = low + 1.0
        scaled = np.clip((depth - low) / (high - low), 0, 1)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
        scaled = (scaled * 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.float32) > 127


def resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth.astype(np.float32)
    image = Image.fromarray(depth.astype(np.float32))
    resized = image.resize((shape[1], shape[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def error_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    stops = np.array(
        [
            [247, 250, 252],
            [254, 226, 138],
            [248, 113, 113],
            [127, 29, 29],
        ],
        dtype=np.float32,
    )
    scaled = values * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(stops) - 1)
    frac = (scaled - lower)[..., None]
    rgb = stops[lower] * (1.0 - frac) + stops[upper] * frac
    return np.clip(rgb, 0, 255).astype(np.uint8)


def depth_error_image(
    row: dict,
    manifest: dict[str, dict],
    manifest_path: Path | None,
    depth_path: Path | None,
) -> Image.Image | None:
    try:
        if not depth_path or not depth_path.exists() or depth_path.suffix.lower() != ".npy":
            return None

        gt_depth_path = reference_artifact_path(row, manifest, manifest_path, ("gt_depth",))
        mask_path = reference_artifact_path(row, manifest, manifest_path, ("mask",))
        if not gt_depth_path or not mask_path:
            return None

        gt_depth = np.load(gt_depth_path).astype(np.float32)
        pred_depth = resize_depth(np.load(depth_path), gt_depth.shape)
        mask = resize_bool_mask(load_mask(mask_path), gt_depth.shape)
        visible = ~mask

        silhouette_path = reference_artifact_path(row, manifest, manifest_path, ("gt_silhouette", "silhouette"))
        if silhouette_path:
            silhouette = resize_bool_mask(load_bool_mask(silhouette_path), gt_depth.shape)
            fit_mask = visible & silhouette
            eval_mask = mask & silhouette
            if np.count_nonzero(fit_mask) < 2 or np.count_nonzero(eval_mask) == 0:
                fit_mask = visible
                eval_mask = mask
        else:
            fit_mask = visible
            eval_mask = mask

        aligned_depth, _, _ = align_depth(pred_depth, gt_depth, fit_mask)
        error = np.abs(aligned_depth - gt_depth)
        finite_eval = error[eval_mask & np.isfinite(error)]
        if finite_eval.size == 0:
            return None

        high = float(np.percentile(finite_eval, 95))
        if high <= 0:
            high = 1.0
        normalized = np.zeros(error.shape, dtype=np.float32)
        normalized[eval_mask] = np.clip(error[eval_mask] / high, 0.0, 1.0)
        rgb = error_colormap(normalized)
        rgb[~eval_mask] = np.array([248, 250, 252], dtype=np.uint8)
        return Image.fromarray(rgb, mode="RGB")
    except Exception:
        return None


def fit_thumbnail_image(image: Image.Image, size: int) -> Image.Image:
    canvas = Image.new("RGB", (size, size), "#f8fafc")
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def stl_preview_image(path: Path, size: int, max_faces: int = 4000) -> Image.Image:
    import trimesh

    loaded = trimesh.load_mesh(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError("empty STL scene")
        mesh = trimesh.util.concatenate(geometries)
    else:
        mesh = loaded
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("empty STL mesh")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    center = (np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2.0
    extent = float(np.max(np.ptp(vertices, axis=0)))
    if extent <= 0:
        raise ValueError("degenerate STL bounds")
    vertices = (vertices - center) / extent

    rx = np.deg2rad(58.0)
    rz = np.deg2rad(-38.0)
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)],
        ],
        dtype=np.float32,
    )
    rot_z = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    vertices = vertices @ (rot_x @ rot_z).T

    if len(faces) > max_faces:
        faces = faces[np.linspace(0, len(faces) - 1, num=max_faces, dtype=np.int64)]

    xy = vertices[:, :2]
    xy_min = np.min(xy, axis=0)
    xy_max = np.max(xy, axis=0)
    xy_span = np.maximum(xy_max - xy_min, 1e-6)
    scale = (size - 18) / float(np.max(xy_span))
    projected = np.empty_like(xy)
    projected[:, 0] = (xy[:, 0] - (xy_min[0] + xy_max[0]) / 2.0) * scale + size / 2.0
    projected[:, 1] = size / 2.0 - (xy[:, 1] - (xy_min[1] + xy_max[1]) / 2.0) * scale

    z = vertices[:, 2]
    z_min = float(np.min(z))
    z_span = max(1e-6, float(np.max(z) - z_min))
    face_depth = np.mean(z[faces], axis=1)
    order = np.argsort(face_depth)

    image = Image.new("RGB", (size, size), "#f8fafc")
    draw = ImageDraw.Draw(image)
    base = np.array([89, 132, 184], dtype=np.float32)
    for face_index in order:
        face = faces[face_index]
        points = [(float(projected[index, 0]), float(projected[index, 1])) for index in face]
        shade = 0.48 + 0.42 * ((float(face_depth[face_index]) - z_min) / z_span)
        color = tuple(np.clip(base * shade, 0, 255).astype(np.uint8).tolist())
        draw.polygon(points, fill=color, outline="#485568")
    return image


def thumbnail(path: Path | None, size: int, missing_label: str) -> Image.Image:
    if not path or not path.exists():
        return placeholder(size, missing_label)
    try:
        if path.suffix.lower() == ".npy":
            image = depth_npy_image(path)
        elif path.suffix.lower() == ".stl":
            image = stl_preview_image(path, size)
        else:
            image = Image.open(path).convert("RGB")
    except Exception:
        return placeholder(size, f"unreadable {path.name}")

    return fit_thumbnail_image(image, size)


def draw_wrapped_lines(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.ImageFont,
    fill: str,
    width: int,
    line_height: int,
    max_lines: int | None = None,
):
    x, y = xy
    emitted = 0
    for line in lines:
        for wrapped in wrap_text(draw, line, font, width, max_lines=None):
            if max_lines is not None and emitted >= max_lines:
                return
            draw.text((x, y), wrapped, fill=fill, font=font)
            y += line_height
            emitted += 1


def draw_image_cell(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    row_height: int,
    image: Image.Image,
):
    image_x = x + (width - image.width) // 2
    image_y = y + (row_height - image.height) // 2
    sheet.paste(image, (image_x, image_y))
    draw.rectangle((image_x, image_y, image_x + image.width - 1, image_y + image.height - 1), outline="#d1d7e0")


def render_contact_sheet(
    rows: list[dict],
    manifest: dict[str, dict],
    manifest_path: Path | None,
    output_path: Path,
    thumb_size: int = 150,
    title: str | None = None,
) -> Path:
    if not rows:
        raise ValueError("No rows selected for contact sheet")

    font = load_font(13)
    small_font = load_font(12)
    header_font = load_font(13, bold=True)
    title_font = load_font(18, bold=True)
    columns = [
        ("Sample / Method", 190),
        ("Full", thumb_size + 18),
        ("Masked", thumb_size + 18),
        ("Mask", thumb_size + 18),
        ("Raw", thumb_size + 18),
        ("Completed", thumb_size + 18),
        ("Depth", thumb_size + 18),
        ("Obj Depth Err", thumb_size + 18),
        ("STL", thumb_size + 18),
        ("Metrics", 230),
    ]
    width = sum(column_width for _, column_width in columns)
    title_height = 48
    header_height = 30
    row_height = max(thumb_size + 32, 182)
    height = title_height + header_height + row_height * len(rows) + 1
    sheet = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(sheet)

    sheet_title = title or "Benchmark Artifact Contact Sheet"
    draw.text((14, 14), sheet_title, fill="#111827", font=title_font)
    draw.line((0, title_height - 1, width, title_height - 1), fill="#c8d0dc")

    x = 0
    for label, column_width in columns:
        draw.rectangle((x, title_height, x + column_width, title_height + header_height), fill="#f3f6fa", outline="#c8d0dc")
        draw.text((x + 8, title_height + 8), label, fill="#111827", font=header_font)
        x += column_width

    y = title_height + header_height
    for row_index, row in enumerate(rows):
        fill = "#ffffff" if row_index % 2 == 0 else "#fbfcfe"
        draw.rectangle((0, y, width, y + row_height), fill=fill)
        x = 0
        for _, column_width in columns:
            draw.rectangle((x, y, x + column_width, y + row_height), outline="#e1e6ee")
            x += column_width

        refs = reference_artifacts(row, manifest, manifest_path)
        outputs = output_artifacts(row)
        error_image = depth_error_image(row, manifest, manifest_path, outputs.get("depth_data"))
        image_paths = [
            refs.get("full_image"),
            refs.get("masked_image"),
            refs.get("mask"),
            outputs.get("raw_completed_image"),
            outputs.get("completed_image"),
            outputs.get("depth"),
        ]

        x = 0
        sample_lines = [
            str(row.get("sample_id", "")),
            str(row.get("_method") or row.get("method") or ""),
        ]
        category = row.get("asset_category")
        if category:
            sample_lines.append(str(category))
        draw_wrapped_lines(draw, (x + 10, y + 14), sample_lines, font, "#111827", columns[0][1] - 20, 17, max_lines=8)
        x += columns[0][1]

        missing_labels = ["missing full", "missing masked", "missing mask", "missing raw", "missing completed", "missing depth"]
        for image_path, missing_label, column in zip(image_paths, missing_labels, columns[1:7]):
            image = thumbnail(image_path, thumb_size, missing_label)
            draw_image_cell(sheet, draw, x, y, column[1], row_height, image)
            x += column[1]

        error_thumb = fit_thumbnail_image(error_image, thumb_size) if error_image else placeholder(thumb_size, "missing error")
        draw_image_cell(sheet, draw, x, y, columns[7][1], row_height, error_thumb)
        x += columns[7][1]

        stl_image = thumbnail(outputs.get("stl"), thumb_size, "missing stl")
        draw_image_cell(sheet, draw, x, y, columns[8][1], row_height, stl_image)
        x += columns[8][1]

        draw_wrapped_lines(draw, (x + 10, y + 12), metric_lines(row), small_font, "#374151", columns[-1][1] - 20, 16, max_lines=10)
        y += row_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def make_contact_sheet(
    input_dir: Path,
    output_path: Path | None = None,
    manifest_path: str | None = None,
    methods: list[str] | None = None,
    samples: list[str] | None = None,
    max_samples: int = 6,
    thumb_size: int = 150,
) -> dict:
    input_dir = input_dir.resolve()
    rows, metric_paths = load_metric_rows(input_dir)
    manifest_file = find_manifest_path(input_dir, metric_paths, manifest_path)
    manifest = load_manifest(manifest_file)
    selected_rows = filter_rows(rows, methods, samples, max_samples)
    output_path = output_path or (input_dir / "artifact_contact_sheet.png")
    render_contact_sheet(
        selected_rows,
        manifest,
        manifest_file,
        output_path,
        thumb_size=thumb_size,
        title=f"Artifact Contact Sheet: {input_dir.name}",
    )
    return {
        "output": str(output_path),
        "row_count": len(selected_rows),
        "metric_files": [str(path) for path in metric_paths],
        "manifest": str(manifest_file) if manifest_file else "",
    }


def parse_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Render a visual contact sheet for benchmark artifact outputs.")
    parser.add_argument("input_dir", help="A benchmark run dir or optimize_completion experiment dir.")
    parser.add_argument("--output", help="PNG path. Defaults to input_dir/artifact_contact_sheet.png.")
    parser.add_argument("--manifest", help="Optional manifest.jsonl path for full/masked/mask reference images.")
    parser.add_argument("--methods", help="Comma-separated method list to include.")
    parser.add_argument("--samples", help="Comma-separated sample_id list to include.")
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--thumb-size", type=int, default=150)
    args = parser.parse_args()

    result = make_contact_sheet(
        Path(args.input_dir),
        output_path=Path(args.output) if args.output else None,
        manifest_path=args.manifest,
        methods=parse_csv_arg(args.methods),
        samples=parse_csv_arg(args.samples),
        max_samples=args.max_samples,
        thumb_size=args.thumb_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
