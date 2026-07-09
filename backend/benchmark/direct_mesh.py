from __future__ import annotations

import math
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.mesh_rendering import load_mesh, mesh_in_render_frame
from backend.pic_to_3d import _masked_edit_image


DIRECT_MESH_METHODS = {"source-mesh-oracle", "external-image-to-mesh", "external-multiview-to-mesh"}
DIRECT_MESH_INPUT_MODES = ("masked", "full", "mirror", "biharmonic")
MESH_REPAIR_MODES = ("none", "basic", "convex-hull", "printable")


def is_direct_mesh_method(method: str) -> bool:
    return method in DIRECT_MESH_METHODS


def resolve_existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def sample_mesh_path(sample: dict) -> Path | None:
    return resolve_existing_path(sample.get("mesh") or sample.get("asset_path"))


def direct_mesh_input_path(sample: dict, mode: str = "masked", output_dir: Path | None = None) -> Path:
    if mode == "full":
        key = "full_image"
    elif mode == "masked":
        key = "masked_image"
    elif mode in ("mirror", "biharmonic"):
        if output_dir is None:
            raise ValueError(f"direct mesh input mode {mode!r} requires an output directory")
        masked_path = direct_mesh_input_path(sample, "masked")
        mask_path = resolve_existing_path(sample.get("mask"))
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for sample {sample.get('id', '')}: {sample.get('mask', '')}")
        output_dir.mkdir(parents=True, exist_ok=True)
        prefill_path = output_dir / f"direct_mesh_input_{mode}.png"
        image = Image.open(masked_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        _masked_edit_image(image, mask, mode).save(prefill_path)
        return prefill_path
    else:
        raise ValueError(f"Unsupported direct mesh input mode: {mode}")
    path = resolve_existing_path(sample.get(key))
    if path is None:
        raise FileNotFoundError(f"Missing {key} for sample {sample.get('id', '')}: {sample.get(key, '')}")
    return path


def _jsonish_list(value) -> list:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [value]
            return parsed if isinstance(parsed, list) else [parsed]
        return [item.strip() for item in text.split(",") if item.strip()]
    return [value]


def _path_text(value) -> str:
    if value in (None, ""):
        return ""
    resolved = resolve_existing_path(str(value))
    return str(resolved or value)


def write_multiview_input_bundle(sample: dict, primary_image: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = _jsonish_list(sample.get("multiview_images"))
    masks = _jsonish_list(sample.get("multiview_masks"))
    cameras = _jsonish_list(sample.get("multiview_cameras"))
    view_ids = _jsonish_list(sample.get("multiview_view_ids"))
    if not images:
        images = [sample.get("full_image") or primary_image]
    views = []
    for index, image in enumerate(images):
        views.append(
            {
                "index": index,
                "sample_id": str(view_ids[index]) if index < len(view_ids) else "",
                "image": _path_text(image),
                "mask": _path_text(masks[index]) if index < len(masks) else "",
                "camera": cameras[index] if index < len(cameras) else {},
            }
        )
    bundle = {
        "sample_id": str(sample.get("id", "")),
        "asset_key": str(sample.get("asset_key", "")),
        "primary_image": str(primary_image),
        "masked_image": _path_text(sample.get("masked_image")),
        "full_image": _path_text(sample.get("full_image")),
        "mask": _path_text(sample.get("mask")),
        "camera": sample.get("camera", {}),
        "primary_view_index": sample.get("view_index", ""),
        "multiview_primary_index": sample.get("multiview_primary_index", ""),
        "video_path": _path_text(sample.get("video_path")),
        "frames_dir": _path_text(sample.get("frames_dir")),
        "views": views,
    }
    bundle_path = output_dir / "multiview_input.json"
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return bundle_path


def convert_mesh_to_stl(mesh_path: Path, stl_path: Path) -> Path:
    if mesh_path.resolve() == stl_path.resolve():
        return stl_path
    mesh = load_mesh(mesh_path)
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Direct mesh output has no triangles: {mesh_path}")
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(stl_path)
    return stl_path


def _finite_positive_volume(mesh) -> bool:
    try:
        volume = float(mesh.volume)
    except Exception:
        return False
    return math.isfinite(volume) and volume > 0


def mesh_is_printable_volume(mesh) -> bool:
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)) or not np.all(extents > 0):
        return False
    try:
        component_count = len(mesh.split(only_watertight=False))
    except Exception:
        component_count = math.inf
    nonmanifold_edge_count, degenerate_face_count = mesh_face_health(mesh)
    return bool(
        len(mesh.vertices)
        and len(mesh.faces)
        and mesh.is_watertight
        and mesh.is_volume
        and mesh.is_winding_consistent
        and component_count == 1
        and nonmanifold_edge_count == 0
        and degenerate_face_count == 0
        and _finite_positive_volume(mesh)
    )


def _update_faces(mesh, mask) -> None:
    if mask is None:
        return
    if hasattr(mesh, "update_faces"):
        mesh.update_faces(mask)
    else:
        mesh.faces = mesh.faces[mask]


def mesh_face_health(mesh) -> tuple[int, int]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(faces):
        return 0, 0
    face_edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    face_edges = np.sort(face_edges, axis=1)
    _, edge_counts = np.unique(face_edges, axis=0, return_counts=True)
    nonmanifold_edge_count = int(np.count_nonzero(edge_counts != 2))
    try:
        face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    except Exception:
        face_areas = np.full(len(faces), math.nan, dtype=np.float64)
    degenerate_face_count = int(np.count_nonzero((~np.isfinite(face_areas)) | (face_areas <= 1e-12)))
    return nonmanifold_edge_count, degenerate_face_count


def _drop_duplicate_and_degenerate_faces(mesh) -> None:
    if hasattr(mesh, "unique_faces"):
        _update_faces(mesh, mesh.unique_faces())
    elif hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, "nondegenerate_faces"):
        _update_faces(mesh, mesh.nondegenerate_faces())
    elif hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()


def _component_score(mesh) -> tuple[float, float, int]:
    try:
        volume = abs(float(mesh.volume))
    except Exception:
        volume = 0.0
    try:
        area = float(mesh.area)
    except Exception:
        area = 0.0
    volume = volume if math.isfinite(volume) else 0.0
    area = area if math.isfinite(area) else 0.0
    return (volume, area, int(len(mesh.faces)))


def _largest_component(mesh):
    try:
        components = [
            component
            for component in mesh.split(only_watertight=False)
            if len(component.vertices) and len(component.faces)
        ]
    except Exception:
        components = []
    if len(components) <= 1:
        return mesh
    return max(components, key=_component_score)


def _clean_mesh(mesh):
    import trimesh

    cleaned = mesh.copy()
    if hasattr(cleaned, "remove_infinite_values"):
        cleaned.remove_infinite_values()
    _drop_duplicate_and_degenerate_faces(cleaned)
    cleaned.merge_vertices()
    cleaned.process(validate=True)
    cleaned = _largest_component(cleaned).copy()
    trimesh.repair.fill_holes(cleaned)
    trimesh.repair.fix_winding(cleaned)
    trimesh.repair.fix_normals(cleaned)
    trimesh.repair.fix_inversion(cleaned)
    _drop_duplicate_and_degenerate_faces(cleaned)
    cleaned.process(validate=True)
    return _largest_component(cleaned).copy()


def _convex_hull_mesh(mesh):
    import trimesh

    source = _largest_component(mesh).copy()
    try:
        hull = source.convex_hull
        if callable(hull):
            hull = hull()
    except Exception:
        hull = source.bounding_box.to_mesh()
    if not len(hull.vertices) or not len(hull.faces):
        hull = trimesh.creation.box(extents=np.maximum(np.asarray(source.extents), 1e-6))
        hull.apply_translation(source.bounds.mean(axis=0))
    _drop_duplicate_and_degenerate_faces(hull)
    hull.process(validate=True)
    trimesh.repair.fix_winding(hull)
    trimesh.repair.fix_normals(hull)
    trimesh.repair.fix_inversion(hull)
    _drop_duplicate_and_degenerate_faces(hull)
    return hull


def repair_mesh_for_printable_stl(mesh_path: Path, output_path: Path, mode: str = "printable") -> Path:
    if mode not in MESH_REPAIR_MODES or mode == "none":
        raise ValueError(f"Unsupported mesh repair mode: {mode}")
    mesh = load_mesh(mesh_path)
    if mode == "convex-hull":
        repaired = _convex_hull_mesh(mesh)
    else:
        repaired = _clean_mesh(mesh)
        if mode == "printable" and not mesh_is_printable_volume(repaired):
            repaired = _convex_hull_mesh(repaired)
    if not len(repaired.vertices) or not len(repaired.faces):
        raise ValueError(f"Mesh repair produced no triangles: {mesh_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(output_path)
    return output_path


def _valid_extents(mesh) -> np.ndarray | None:
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)) or not np.all(extents > 0):
        return None
    return extents


def _target_extents_array(target_bbox_extents) -> np.ndarray | None:
    if target_bbox_extents is None:
        return None
    if isinstance(target_bbox_extents, str) and not target_bbox_extents.strip():
        return None
    extents = np.asarray(target_bbox_extents, dtype=np.float64)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)) or not np.all(extents > 0):
        return None
    return extents


def _scale_extents_to_max_dimension(extents: np.ndarray | None, target_max_dimension: float) -> np.ndarray | None:
    if extents is None:
        return None
    target = float(target_max_dimension or 0.0)
    if target <= 0:
        return extents
    max_extent = float(np.max(extents))
    if max_extent <= 0 or not math.isfinite(max_extent):
        return None
    return extents * (target / max_extent)


def _format_bbox_extents(extents: np.ndarray | None) -> str:
    if extents is None:
        return ""
    return ",".join(f"{float(value):.10g}" for value in extents)


def _bbox_placeholder_values(prefix: str, extents: np.ndarray | None) -> dict[str, str]:
    values = {
        f"{prefix}_bbox_extents": _format_bbox_extents(extents),
        f"{prefix}_bbox_x": "",
        f"{prefix}_bbox_y": "",
        f"{prefix}_bbox_z": "",
    }
    if extents is not None:
        values[f"{prefix}_bbox_x"] = f"{float(extents[0]):.10g}"
        values[f"{prefix}_bbox_y"] = f"{float(extents[1]):.10g}"
        values[f"{prefix}_bbox_z"] = f"{float(extents[2]):.10g}"
    return values


def source_mesh_bbox_extents(sample: dict, target_max_dimension: float = 0.0) -> np.ndarray | None:
    source_path = sample_mesh_path(sample)
    if source_path is None:
        return None
    source_mesh = load_mesh(source_path)
    if sample.get("camera"):
        source_mesh = mesh_in_render_frame(source_mesh, sample.get("camera"))
    return _scale_extents_to_max_dimension(_valid_extents(source_mesh), target_max_dimension)


def stl_file_bbox_extents(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        return _valid_extents(load_mesh(path))
    except Exception:
        return None


def direct_mesh_bbox_placeholders(sample: dict, output_dir: Path, args) -> dict[str, str]:
    stl_target_dimension = float(getattr(args, "stl_target_dimension", 0.0) or 0.0)
    source_extents = source_mesh_bbox_extents(sample, stl_target_dimension)
    mirror_stl = output_dir.parent / "mirror" / "output_model.stl"
    mirror_extents = stl_file_bbox_extents(mirror_stl)
    values = {
        "stl_target_dimension": f"{stl_target_dimension:.10g}" if stl_target_dimension > 0 else "",
        "mirror_stl": str(mirror_stl) if mirror_stl.exists() else "",
    }
    values.update(_bbox_placeholder_values("source", source_extents))
    values.update(_bbox_placeholder_values("mirror", mirror_extents))
    return values


def _center_mesh_on_origin(mesh):
    centered = mesh.copy()
    bounds = np.asarray(centered.bounds, dtype=np.float64)
    if bounds.shape == (2, 3) and np.all(np.isfinite(bounds)):
        centered.apply_translation(-bounds.mean(axis=0))
    return centered


def _scale_to_max_dimension(mesh, target_max_dimension: float):
    if target_max_dimension <= 0:
        return mesh
    extents = _valid_extents(mesh)
    if extents is None:
        return mesh
    max_extent = float(np.max(extents))
    if max_extent <= 0 or not math.isfinite(max_extent):
        return mesh
    scaled = _center_mesh_on_origin(mesh)
    scaled.apply_scale(float(target_max_dimension) / max_extent)
    return scaled


def _enforce_min_bbox_dimension(mesh, min_bbox_dimension: float):
    if min_bbox_dimension <= 0:
        return mesh
    extents = _valid_extents(mesh)
    if extents is None:
        return mesh
    factors = np.ones(3, dtype=np.float64)
    small = extents < float(min_bbox_dimension)
    if not np.any(small):
        return mesh
    factors[small] = float(min_bbox_dimension) / extents[small]
    compacted = mesh.copy()
    bounds = np.asarray(compacted.bounds, dtype=np.float64)
    center = bounds.mean(axis=0) if bounds.shape == (2, 3) and np.all(np.isfinite(bounds)) else np.zeros(3)
    compacted.vertices = (np.asarray(compacted.vertices, dtype=np.float64) - center) * factors + center
    return compacted


def _clamp_bbox_aspect_ratio(mesh, max_bbox_aspect_ratio: float):
    target_ratio = float(max_bbox_aspect_ratio or 0.0)
    if target_ratio < 1.0:
        return mesh
    extents = _valid_extents(mesh)
    if extents is None:
        return mesh
    max_extent = float(np.max(extents))
    min_extent = float(np.min(extents))
    if min_extent <= 0 or max_extent / min_extent <= target_ratio:
        return mesh
    min_allowed_extent = max_extent / target_ratio
    factors = np.ones(3, dtype=np.float64)
    small = extents < min_allowed_extent
    factors[small] = min_allowed_extent / extents[small]
    clamped = mesh.copy()
    bounds = np.asarray(clamped.bounds, dtype=np.float64)
    center = bounds.mean(axis=0) if bounds.shape == (2, 3) and np.all(np.isfinite(bounds)) else np.zeros(3)
    clamped.vertices = (np.asarray(clamped.vertices, dtype=np.float64) - center) * factors + center
    return clamped


def _match_bbox_extents(mesh, target_bbox_extents):
    target = _target_extents_array(target_bbox_extents)
    if target is None:
        return mesh
    extents = _valid_extents(mesh)
    if extents is None:
        return mesh
    matched = mesh.copy()
    bounds = np.asarray(matched.bounds, dtype=np.float64)
    center = bounds.mean(axis=0) if bounds.shape == (2, 3) and np.all(np.isfinite(bounds)) else np.zeros(3)
    matched.vertices = (np.asarray(matched.vertices, dtype=np.float64) - center) * (target / extents) + center
    return matched


def _simplify_to_face_count(mesh, target_faces: int):
    target_faces = int(target_faces or 0)
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    simplify = getattr(mesh, "simplify_quadric_decimation", None)
    if simplify is None:
        return mesh
    try:
        simplified = simplify(face_count=target_faces)
    except Exception:
        return mesh
    if not len(simplified.vertices) or not len(simplified.faces):
        return mesh
    return simplified


def postprocess_mesh_for_stl(
    mesh_path: Path,
    output_path: Path,
    *,
    target_max_dimension: float = 0.0,
    min_bbox_dimension: float = 0.0,
    max_bbox_aspect_ratio: float = 0.0,
    target_bbox_extents=None,
    target_faces: int = 0,
) -> Path:
    mesh = load_mesh(mesh_path)
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Mesh postprocess input has no triangles: {mesh_path}")
    processed = _scale_to_max_dimension(mesh, float(target_max_dimension or 0.0))
    processed = _enforce_min_bbox_dimension(processed, float(min_bbox_dimension or 0.0))
    processed = _clamp_bbox_aspect_ratio(processed, float(max_bbox_aspect_ratio or 0.0))
    processed = _match_bbox_extents(processed, target_bbox_extents)
    processed = _simplify_to_face_count(processed, int(target_faces or 0))
    processed.remove_unreferenced_vertices()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.export(output_path)
    return output_path


def run_direct_mesh(sample: dict, method: str, output_dir: Path, args) -> tuple[Path, Path, Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_image = direct_mesh_input_path(sample, getattr(args, "direct_mesh_input", "masked"), output_dir)
    input_bundle = write_multiview_input_bundle(sample, input_image, output_dir) if method == "external-multiview-to-mesh" else None
    stl_path = output_dir / "output_model.stl"
    output_ext = str(getattr(args, "direct_mesh_output_ext", "glb") or "glb").lstrip(".")
    mesh_output_path = output_dir / f"output_mesh.{output_ext}"

    if method == "source-mesh-oracle":
        source_path = sample_mesh_path(sample)
        if source_path is None:
            raise FileNotFoundError(f"Sample {sample.get('id', '')} does not include an existing source mesh")
        source_mesh_repair = str(getattr(args, "source_mesh_repair", "none") or "none")
        if source_mesh_repair not in MESH_REPAIR_MODES:
            raise ValueError(f"Unsupported source mesh repair mode: {source_mesh_repair}")
        mesh_output_path = output_dir / f"source_mesh{source_path.suffix or '.mesh'}"
        raw_mesh_output_path = (
            output_dir / f"source_mesh_raw{source_path.suffix or '.mesh'}"
            if source_mesh_repair != "none"
            else mesh_output_path
        )
        if sample.get("camera"):
            source_mesh = mesh_in_render_frame(load_mesh(source_path), sample.get("camera"))
            source_mesh.export(raw_mesh_output_path)
        else:
            shutil.copy2(source_path, raw_mesh_output_path)
        if source_mesh_repair != "none":
            mesh_output_path = output_dir / "source_mesh_repaired.ply"
            repair_mesh_for_printable_stl(raw_mesh_output_path, mesh_output_path, source_mesh_repair)
        else:
            mesh_output_path = raw_mesh_output_path
        convert_mesh_to_stl(mesh_output_path, stl_path)
        return input_image, mesh_output_path, stl_path, input_bundle

    if method in ("external-image-to-mesh", "external-multiview-to-mesh"):
        command_template = getattr(args, "direct_mesh_command", None)
        if not command_template:
            raise ValueError(f"--direct-mesh-command is required for {method}")
        values = {
            "input_image": str(input_image),
            "input_bundle": str(input_bundle or ""),
            "masked_image": str(direct_mesh_input_path(sample, "masked")),
            "full_image": str(direct_mesh_input_path(sample, "full")),
            "mask": str(resolve_existing_path(sample.get("mask")) or ""),
            "output_dir": str(output_dir),
            "output_mesh": str(mesh_output_path),
            "output_stl": str(stl_path),
            "sample_id": str(sample.get("id", "")),
            "method": method,
        }
        values.update(direct_mesh_bbox_placeholders(sample, output_dir, args))
        command = command_template.format(**values)
        subprocess.run(
            command,
            shell=True,
            check=True,
            timeout=max(1, int(getattr(args, "direct_mesh_timeout", 1800))),
        )
        if not mesh_output_path.exists() and stl_path.exists():
            return input_image, stl_path, stl_path, input_bundle
        if not mesh_output_path.exists():
            raise FileNotFoundError(
                f"External image-to-mesh command produced neither {mesh_output_path} nor {stl_path}"
            )
        if not stl_path.exists():
            convert_mesh_to_stl(mesh_output_path, stl_path)
        return input_image, mesh_output_path, stl_path, input_bundle

    raise ValueError(f"Unsupported direct mesh method: {method}")
