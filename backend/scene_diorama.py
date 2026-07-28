from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from skimage import measure


FACADE_LABEL_HINTS = (
    "building",
    "house",
    "facade",
    "wall",
    "architecture",
    "skyscraper",
    "tower",
    "structure",
    "window",
    "door",
)
SUBJECT_LABEL_HINTS = (
    "person",
    "people",
    "human",
    "man",
    "woman",
    "boy",
    "girl",
    "face",
)
GROUND_LABEL_HINTS = ("floor", "ground", "road", "sidewalk", "pavement", "terrain")
ROLE_COLORS = {
    "facade": np.array([48, 132, 196], dtype=np.float32),
    "subject": np.array([225, 68, 58], dtype=np.float32),
    "ground": np.array([67, 160, 108], dtype=np.float32),
    "object": np.array([230, 158, 49], dtype=np.float32),
    "scene": np.array([129, 88, 176], dtype=np.float32),
}
MIN_MASK_COMPONENT_PIXELS = 8
MIN_MASK_COMPONENT_AREA_RATIO = 0.01


def _finite_percentile(values: np.ndarray, percentile: float, default: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else float(default)


def normalize_scene_depth(depth: np.ndarray, *, far_is_high: bool) -> np.ndarray:
    """Return a robust 0-near, 1-far depth map for scene placement."""
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"Scene depth must be 2D, received shape {values.shape}")

    low = _finite_percentile(values, 1.0, 0.0)
    high = _finite_percentile(values, 99.0, 1.0)
    if not math.isfinite(low) or not math.isfinite(high) or high - low <= 1e-8:
        normalized = np.zeros(values.shape, dtype=np.float32)
    else:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    if not far_is_high:
        normalized = 1.0 - normalized
    normalized[~np.isfinite(normalized)] = 0.5
    return normalized.astype(np.float32, copy=False)


def _resize_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def _load_mask(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.NEAREST)
        return np.asarray(mask, dtype=np.uint8) >= 128


def _label_text(labels: Iterable[str]) -> str:
    return " ".join(str(label).strip().lower() for label in labels if str(label).strip())


def classify_scene_layer(mask: np.ndarray, labels: Iterable[str]) -> str:
    text = _label_text(labels)
    if any(hint in text for hint in SUBJECT_LABEL_HINTS):
        return "subject"
    if any(hint in text for hint in FACADE_LABEL_HINTS):
        return "facade"
    if any(hint in text for hint in GROUND_LABEL_HINTS):
        return "ground"

    rows, columns = np.nonzero(mask)
    if not len(rows):
        return "object"
    height, width = mask.shape
    bbox_width = int(columns.max() - columns.min() + 1)
    bbox_height = int(rows.max() - rows.min() + 1)
    coverage = float(mask.mean())
    touches_upper_half = float(rows.min()) < height * 0.45
    broad = bbox_width >= width * 0.28
    if touches_upper_half and broad and (coverage >= 0.08 or bbox_height >= height * 0.35):
        return "facade"
    return "object"


def _clean_mask_with_stats(mask: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    if not np.any(mask):
        return mask.astype(bool), {
            "input_components": 0,
            "kept_components": 0,
            "removed_components": 0,
        }
    closed = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    filled = ndimage.binary_fill_holes(closed).astype(bool)
    labels, component_count = ndimage.label(filled)
    if component_count <= 1:
        return filled, {
            "input_components": int(component_count),
            "kept_components": int(component_count),
            "removed_components": 0,
        }

    component_sizes = np.bincount(labels.ravel())[1:]
    largest_size = int(component_sizes.max(initial=0))
    minimum_size = max(
        MIN_MASK_COMPONENT_PIXELS,
        int(math.ceil(largest_size * MIN_MASK_COMPONENT_AREA_RATIO)),
    )
    keep_ids = np.flatnonzero(component_sizes >= minimum_size) + 1
    if keep_ids.size == 0 and largest_size:
        keep_ids = np.array([int(np.argmax(component_sizes)) + 1], dtype=np.int32)
    cleaned = np.isin(labels, keep_ids)
    kept_count = int(len(keep_ids))
    return cleaned, {
        "input_components": int(component_count),
        "kept_components": kept_count,
        "removed_components": int(component_count - kept_count),
    }


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    cleaned, _ = _clean_mask_with_stats(mask)
    return cleaned


def _facade_silhouette(mask: np.ndarray) -> np.ndarray:
    """Complete facade pixels hidden by foreground subjects down to the base."""
    source = _clean_mask(mask)
    silhouette = np.zeros_like(source)
    for column in range(source.shape[1]):
        rows = np.flatnonzero(source[:, column])
        if rows.size:
            silhouette[int(rows.min()) :, column] = True
    silhouette = ndimage.binary_closing(
        silhouette,
        structure=np.ones((3, 5), dtype=bool),
        iterations=1,
    )
    return ndimage.binary_fill_holes(silhouette).astype(bool)


def _facade_detail_field(photo_detail: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    """Keep measured facade texture out of generated occlusion fill."""
    detail = np.zeros_like(photo_detail, dtype=np.float32)
    detail[visible_mask] = photo_detail[visible_mask]
    return detail


def _layer_fraction(role: str, median_far_depth: float) -> float:
    depth = float(np.clip(median_far_depth, 0.0, 1.0))
    if role == "subject":
        return 0.20 + 0.14 * depth
    if role == "facade":
        return 0.73 + 0.12 * depth
    if role == "ground":
        return 0.55 + 0.20 * depth
    if role == "scene":
        return 0.74
    return 0.34 + 0.34 * depth


def _stamp_variable_slab(
    occupancy: np.ndarray,
    mask: np.ndarray,
    lower_y: np.ndarray,
    upper_y: np.ndarray,
    *,
    base_layers: int,
) -> None:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return
    z_indices = base_layers + (mask.shape[0] - 1 - rows)
    x_indices = columns + 1
    lows = np.asarray(lower_y[rows, columns], dtype=np.int32)
    highs = np.asarray(upper_y[rows, columns], dtype=np.int32)
    for y_index in range(1, occupancy.shape[1] - 1):
        active = (lows <= y_index) & (highs >= y_index)
        if np.any(active):
            occupancy[z_indices[active], y_index, x_indices[active]] = True


def _attach_floating_components(
    occupancy: np.ndarray,
    mask: np.ndarray,
    lower_y: np.ndarray,
    upper_y: np.ndarray,
    *,
    base_layers: int,
    pitch_mm: float,
    minimum_feature_mm: float,
) -> int:
    labels, component_count = ndimage.label(mask)
    support_count = 0
    support_radius = max(1, int(math.ceil(minimum_feature_mm / max(pitch_mm, 1e-6) / 2.0)))
    for component_id in range(1, component_count + 1):
        rows, columns = np.nonzero(labels == component_id)
        if not len(rows):
            continue
        bottom_row = int(rows.max())
        if bottom_row >= mask.shape[0] - 2:
            continue
        bottom_columns = columns[rows >= bottom_row - 1]
        center_x = int(round(float(np.median(bottom_columns if len(bottom_columns) else columns))))
        sample_row = bottom_row
        sample_column = int(np.clip(center_x, 0, mask.shape[1] - 1))
        low = int(lower_y[sample_row, sample_column])
        high = int(upper_y[sample_row, sample_column])
        z_top = base_layers + (mask.shape[0] - 1 - bottom_row)
        x_start = max(1, center_x + 1 - support_radius)
        x_stop = min(occupancy.shape[2] - 1, center_x + 2 + support_radius)
        occupancy[base_layers : z_top + 1, low : high + 1, x_start:x_stop] = True
        support_count += 1
    return support_count


def _vertex_colors(
    mesh: trimesh.Trimesh,
    image_rgb: np.ndarray,
    *,
    width_mm: float,
    image_height_mm: float,
    base_thickness_mm: float,
    pitch_mm: float,
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    image_height, image_width = image_rgb.shape[:2]
    x = np.clip((vertices[:, 0] + width_mm / 2.0) / max(width_mm, 1e-6), 0.0, 1.0)
    z = np.clip((vertices[:, 2] - base_thickness_mm) / max(image_height_mm, 1e-6), 0.0, 1.0)
    columns = np.rint(x * (image_width - 1)).astype(np.int32)
    rows = np.rint((1.0 - z) * (image_height - 1)).astype(np.int32)
    colors = image_rgb[rows, columns].astype(np.float32)

    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    front = np.clip(-normals[:, 1], 0.0, 1.0)
    back = np.clip(normals[:, 1], 0.0, 1.0)
    shade = np.clip(0.72 + 0.28 * front - 0.22 * back, 0.45, 1.0)
    colors *= shade[:, None]
    base = vertices[:, 2] <= base_thickness_mm + pitch_mm
    colors[base] = np.array([108, 112, 118], dtype=np.float32)
    alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
    return np.concatenate((np.clip(colors, 0, 255).astype(np.uint8), alpha), axis=1)


def _write_preview(
    image: Image.Image,
    layers: list[dict],
    output_path: Path,
) -> None:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    tinted = base.copy()
    for layer in layers:
        mask = layer["source_mask"]
        color = ROLE_COLORS[layer["role"]]
        tinted[mask] = tinted[mask] * 0.62 + color * 0.38
    preview = Image.fromarray(np.clip(tinted, 0, 255).astype(np.uint8), mode="RGB")
    preview.save(output_path)


def build_scene_diorama(
    image_path: str | Path,
    depth_path: str | Path,
    output_stl_path: str | Path,
    output_glb_path: str | Path,
    output_preview_path: str | Path,
    *,
    selections: list[dict] | None = None,
    far_is_high: bool = False,
    max_size_mm: float = 180.0,
    scene_depth_mm: float = 64.0,
    base_thickness_mm: float = 2.4,
    facade_detail_mm: float = 0.8,
    subject_depth_mm: float = 12.0,
    depth_compression: float = 0.65,
    minimum_feature_mm: float = 0.8,
    max_samples: int = 240,
) -> dict:
    """Build a camera-free layered diorama from one image and selected scene masks.

    Visible pixels remain measured evidence. Facade completion, layer separation, side
    walls, backs, and supports are intentionally generated geometric assumptions.
    """
    max_size_mm = float(np.clip(max_size_mm, 20.0, 500.0))
    scene_depth_mm = float(np.clip(scene_depth_mm, 12.0, 240.0))
    base_thickness_mm = float(np.clip(base_thickness_mm, 0.8, 20.0))
    facade_detail_mm = float(np.clip(facade_detail_mm, 0.0, 4.0))
    subject_depth_mm = float(np.clip(subject_depth_mm, 2.0, 60.0))
    depth_compression = float(np.clip(depth_compression, 0.0, 1.0))
    minimum_feature_mm = float(np.clip(minimum_feature_mm, 0.2, 8.0))
    max_samples = int(np.clip(max_samples, 48, 400))

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image_width, image_height = image.size
    if image_width < 2 or image_height < 2:
        raise ValueError("Scene diorama requires an image at least 2x2 pixels")

    scale = max_size_mm / float(max(image_width, image_height))
    requested_width_mm = image_width * scale
    requested_height_mm = image_height * scale
    feature_limited_cells = int(math.floor(max_size_mm / max(minimum_feature_mm / 2.0, 1e-6)))
    long_axis_cells = min(max_samples, max(24, feature_limited_cells))
    pitch_mm = max_size_mm / float(long_axis_cells)
    render_width = max(2, min(long_axis_cells, int(round(requested_width_mm / pitch_mm))))
    render_height = max(2, min(long_axis_cells, int(round(requested_height_mm / pitch_mm))))
    width_mm = render_width * pitch_mm
    image_height_mm = render_height * pitch_mm
    y_samples = max(2, int(math.floor(scene_depth_mm / pitch_mm + 1e-9)))
    scene_depth_mm = y_samples * pitch_mm
    base_layers = max(2, int(math.ceil(base_thickness_mm / pitch_mm)))
    base_thickness_mm = base_layers * pitch_mm

    render_size = (render_width, render_height)
    render_image = image.resize(render_size, Image.Resampling.LANCZOS)
    image_rgb = np.asarray(render_image, dtype=np.uint8)
    luminance = np.asarray(render_image.convert("L"), dtype=np.float32) / 255.0

    depth = np.load(depth_path)
    normalized_depth = normalize_scene_depth(depth, far_is_high=far_is_high)
    normalized_depth = _resize_float(normalized_depth, render_size)

    prepared_layers: list[dict] = []
    for index, selection in enumerate(selections or []):
        mask_path = selection.get("mask_path")
        if not mask_path:
            continue
        source_mask = _clean_mask(_load_mask(mask_path, image.size))
        resized_mask, mask_component_stats = _clean_mask_with_stats(
            _load_mask(mask_path, render_size)
        )
        if not np.any(resized_mask):
            continue
        labels = [str(value) for value in selection.get("labels", []) if str(value).strip()]
        role = classify_scene_layer(resized_mask, labels)
        prepared_layers.append(
            {
                "index": index,
                "labels": labels,
                "role": role,
                "mask": resized_mask,
                "source_mask": source_mask,
                "mask_component_stats": mask_component_stats,
            }
        )

    if not prepared_layers:
        prepared_layers.append(
            {
                "index": 0,
                "labels": ["whole scene"],
                "role": "scene",
                "mask": np.ones((render_height, render_width), dtype=bool),
                "source_mask": np.ones((image_height, image_width), dtype=bool),
                "mask_component_stats": {
                    "input_components": 1,
                    "kept_components": 1,
                    "removed_components": 0,
                },
            }
        )

    occupancy = np.zeros(
        (base_layers + render_height + 2, y_samples + 2, render_width + 2),
        dtype=bool,
    )
    occupancy[1 : base_layers + 1, 1 : y_samples + 1, 1 : render_width + 1] = True

    blurred_luminance = ndimage.gaussian_filter(luminance, sigma=1.2)
    photo_detail = luminance - blurred_luminance
    detail_scale = _finite_percentile(np.abs(photo_detail), 95.0, 1.0)
    if detail_scale > 1e-6:
        photo_detail = np.clip(photo_detail / detail_scale, -1.0, 1.0)
    else:
        photo_detail = np.zeros_like(photo_detail)

    layer_metadata: list[dict] = []
    generated_support_count = 0
    for layer in prepared_layers:
        mask = layer["mask"]
        role = layer["role"]
        sampled_depth = normalized_depth[mask]
        median_far = float(np.median(sampled_depth)) if sampled_depth.size else 0.5
        center_mm = _layer_fraction(role, median_far) * scene_depth_mm

        if role in {"facade", "scene"}:
            geometry_mask = _facade_silhouette(mask) if role == "facade" else mask
            wall_depth_mm = max(minimum_feature_mm * 2.0, min(10.0, subject_depth_mm * 0.55))
            detail = (
                _facade_detail_field(photo_detail, mask)
                if role == "facade"
                else photo_detail
            ) * facade_detail_mm
            front_mm = center_mm - wall_depth_mm / 2.0 - detail
            back_mm = np.full(mask.shape, center_mm + wall_depth_mm / 2.0, dtype=np.float32)
            lower = np.clip(np.rint(front_mm / pitch_mm).astype(np.int32) + 1, 1, y_samples)
            upper = np.clip(np.rint(back_mm / pitch_mm).astype(np.int32) + 1, 1, y_samples)
        else:
            geometry_mask = mask
            distance = ndimage.distance_transform_edt(geometry_mask) * pitch_mm
            max_distance = float(distance[geometry_mask].max(initial=0.0))
            if max_distance > 1e-6:
                roundness = np.sqrt(np.clip(distance / max_distance, 0.0, 1.0))
            else:
                roundness = np.zeros_like(distance)
            local_depth = (normalized_depth - median_far) * depth_compression * subject_depth_mm * 0.5
            half_depth = minimum_feature_mm + roundness * max(0.0, subject_depth_mm / 2.0 - minimum_feature_mm)
            front_mm = center_mm + local_depth - half_depth
            back_mm = center_mm + local_depth + half_depth
            lower = np.clip(np.rint(front_mm / pitch_mm).astype(np.int32) + 1, 1, y_samples)
            upper = np.clip(np.rint(back_mm / pitch_mm).astype(np.int32) + 1, 1, y_samples)

        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        _stamp_variable_slab(
            occupancy,
            geometry_mask,
            lower,
            upper,
            base_layers=base_layers,
        )
        supports = _attach_floating_components(
            occupancy,
            geometry_mask,
            lower,
            upper,
            base_layers=base_layers,
            pitch_mm=pitch_mm,
            minimum_feature_mm=minimum_feature_mm,
        )
        generated_support_count += supports
        layer_metadata.append(
            {
                "index": int(layer["index"]),
                "labels": layer["labels"],
                "role": role,
                "source_coverage": float(layer["source_mask"].mean()),
                "geometry_coverage": float(geometry_mask.mean()),
                "median_far_depth": median_far,
                "scene_y_mm": float(center_mm),
                "generated_occlusion_fill": bool(role == "facade"),
                "generated_supports": int(supports),
                "mask_components": layer["mask_component_stats"],
            }
        )

    vertices_zyx, faces, _, _ = measure.marching_cubes(
        occupancy.astype(np.uint8),
        level=0.5,
        spacing=(pitch_mm, pitch_mm, pitch_mm),
        allow_degenerate=False,
    )
    vertices = np.column_stack(
        (
            vertices_zyx[:, 2] - pitch_mm / 2.0 - width_mm / 2.0,
            vertices_zyx[:, 1] - pitch_mm / 2.0,
            vertices_zyx[:, 0] - pitch_mm / 2.0,
        )
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    if math.isfinite(float(mesh.volume)) and mesh.volume < 0:
        mesh.invert()
    if not len(mesh.faces) or not np.all(np.isfinite(mesh.vertices)):
        raise RuntimeError("Scene reconstruction produced an invalid mesh")

    output_stl = Path(output_stl_path)
    output_glb = Path(output_glb_path)
    output_preview = Path(output_preview_path)
    for output in (output_stl, output_glb, output_preview):
        output.parent.mkdir(parents=True, exist_ok=True)

    mesh.export(output_stl, file_type="stl")
    colored_mesh = mesh.copy()
    colored_mesh.visual.vertex_colors = _vertex_colors(
        colored_mesh,
        image_rgb,
        width_mm=width_mm,
        image_height_mm=image_height_mm,
        base_thickness_mm=base_thickness_mm,
        pitch_mm=pitch_mm,
    )
    colored_mesh.export(output_glb, file_type="glb")
    _write_preview(image, prepared_layers, output_preview)

    extents = np.asarray(mesh.extents, dtype=np.float64)
    return {
        "mode": "single-photo-layered-diorama",
        "camera_free_view": True,
        "visible_evidence": "source image pixels, selected masks, monocular depth",
        "generated_geometry": [
            "occluded facade continuation",
            "layer separation",
            "side and rear extrusion",
            "print supports",
        ],
        "novel_view_provider": "none",
        "multiview_fusion_provider": "none",
        "future_provider_boundary": {
            "novel_views": "stable-virtual-camera",
            "camera_depth_fusion": "vggt-omega",
        },
        "mesh": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "volume_mm3": float(mesh.volume),
            "extents_mm": [float(value) for value in extents],
        },
        "dimensions": {
            "image_width_mm": float(width_mm),
            "image_height_mm": float(image_height_mm),
            "scene_depth_mm": float(scene_depth_mm),
            "base_thickness_mm": float(base_thickness_mm),
            "voxel_pitch_mm": float(pitch_mm),
            "voxel_grid": [int(value) for value in occupancy.shape],
        },
        "layers": layer_metadata,
        "generated_support_count": int(generated_support_count),
    }
