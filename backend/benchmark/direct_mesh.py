from __future__ import annotations

import math
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.mesh_rendering import load_mesh, mesh_in_render_frame
from backend.pic_to_3d import _masked_edit_image


DIRECT_MESH_METHODS = {"source-mesh-oracle", "external-image-to-mesh", "external-multiview-to-mesh"}
DIRECT_MESH_INPUT_MODES = ("masked", "full", "mirror", "biharmonic")
MESH_REPAIR_MODES = ("none", "basic", "convex-hull", "printable")
MESH_REPAIR_PRECONDITIONERS = ("legacy", "voxel-close")
MESH_REPAIR_VOXEL_FILL_METHODS = ("base", "holes", "orthographic")
DEFAULT_MESH_REPAIR_VOXEL_RESOLUTION = 192
MAX_MESH_REPAIR_VOXEL_RESOLUTION = 384
MAX_MESH_REPAIR_COMPONENT_FILTER_FACES = 2_000_000
MIN_SAFE_TOPOLOGY_REPAIR_FACES = 131_072
MAX_FAST_COMPONENT_FILTER_FACES = 500_000
RETRIANGULATION_SURFACE_SAMPLE_POINTS = 4_096
RETRIANGULATION_MAX_FACE_COUNT_RELATIVE_CHANGE = 0.02
RETRIANGULATION_MAX_VERTEX_COUNT_RELATIVE_CHANGE = 0.02
RETRIANGULATION_MAX_VOLUME_RELATIVE_CHANGE = 0.01
RETRIANGULATION_MAX_BBOX_EXTENT_RELATIVE_CHANGE = 0.01
RETRIANGULATION_MAX_BOUNDS_CENTER_SHIFT_NORMALIZED = 0.005
RETRIANGULATION_MAX_VERTEX_DISPLACEMENT_NORMALIZED = 0.001
RETRIANGULATION_MAX_SURFACE_CHAMFER_NORMALIZED = 0.01
RETRIANGULATION_MAX_SURFACE_HAUSDORFF95_NORMALIZED = 0.03
DIRECT_MESH_BBOX_SOURCES = ("none", "source", "mirror", "inferred", "reference")
DIRECT_MESH_BBOX_PLACEHOLDERS = {
    "source": "{source_bbox_extents}",
    "inferred": "{inferred_bbox_extents}",
    "mirror": "{mirror_bbox_extents}",
    "reference": "{reference_bbox_extents}",
}


def is_direct_mesh_method(method: str) -> bool:
    return method in DIRECT_MESH_METHODS


def resolve_direct_mesh_bbox_source(command: str | None, declared_source: str | None = None) -> str:
    command_text = str(command or "")
    used_sources = [
        source
        for source in DIRECT_MESH_BBOX_PLACEHOLDERS
        if f"{{{source}_bbox_" in command_text
    ]
    declared = str(declared_source or "").strip().lower()
    if declared and declared not in DIRECT_MESH_BBOX_SOURCES:
        expected = ", ".join(DIRECT_MESH_BBOX_SOURCES)
        raise ValueError(f"Unknown direct_mesh_bbox_source `{declared}`. Expected one of: {expected}")

    if "source" in used_sources:
        if declared and declared != "source":
            raise ValueError(
                f"direct_mesh_bbox_source `{declared}` conflicts with hidden-source placeholder "
                "{source_bbox_extents}"
            )
        return "source"
    if declared:
        if used_sources and declared == "none":
            raise ValueError(
                "direct_mesh_bbox_source `none` conflicts with command bbox placeholders: "
                + ", ".join(used_sources)
            )
        if used_sources and declared not in used_sources:
            raise ValueError(
                f"direct_mesh_bbox_source `{declared}` conflicts with command bbox placeholders: "
                + ", ".join(used_sources)
            )
        return declared
    return next((source for source in ("inferred", "mirror", "reference") if source in used_sources), "none")


def direct_mesh_bbox_uses_hidden_source(bbox_source: str | None, reference_method: str | None = None) -> bool:
    source = str(bbox_source or "none").strip().lower()
    reference = str(reference_method or "").strip().lower().replace("-", "_")
    source_reference = "source_bbox" in reference or ("source" in reference and "oracle" in reference)
    return source == "source" or (source == "reference" and source_reference)


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
        "source_mesh": str(sample_mesh_path(sample) or ""),
        "asset_path": _path_text(sample.get("asset_path")),
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


def _printability_audit(mesh, prefix: str) -> dict:
    try:
        extents = np.asarray(mesh.extents, dtype=np.float64)
    except Exception:
        extents = np.asarray([], dtype=np.float64)
    bbox_has_volume = bool(
        extents.shape == (3,)
        and np.all(np.isfinite(extents))
        and np.all(extents > 0)
    )
    try:
        component_count = len(mesh.split(only_watertight=False))
    except Exception:
        component_count = -1
    nonmanifold_edge_count, degenerate_face_count = mesh_face_health(mesh)
    positive_volume = _finite_positive_volume(mesh)
    try:
        is_watertight = bool(mesh.is_watertight)
    except Exception:
        is_watertight = False
    try:
        is_volume = bool(mesh.is_volume)
    except Exception:
        is_volume = False
    try:
        winding_consistent = bool(mesh.is_winding_consistent)
    except Exception:
        winding_consistent = False
    vertices = getattr(mesh, "vertices", ())
    printable = bool(
        len(vertices)
        and len(mesh.faces)
        and bbox_has_volume
        and is_watertight
        and is_volume
        and winding_consistent
        and component_count == 1
        and nonmanifold_edge_count == 0
        and degenerate_face_count == 0
        and positive_volume
    )
    return {
        f"{prefix}_printable": printable,
        f"{prefix}_is_watertight": is_watertight,
        f"{prefix}_is_volume": is_volume,
        f"{prefix}_winding_consistent": winding_consistent,
        f"{prefix}_component_count": int(component_count),
        f"{prefix}_nonmanifold_edge_count": int(nonmanifold_edge_count),
        f"{prefix}_degenerate_face_count": int(degenerate_face_count),
        f"{prefix}_positive_volume": bool(positive_volume),
        f"{prefix}_bbox_has_volume": bbox_has_volume,
    }


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


def _largest_face_component(mesh):
    """Select the dominant face component without materializing every fragment."""
    if len(mesh.faces) <= 1:
        return mesh
    try:
        import trimesh

        labels = trimesh.graph.connected_component_labels(
            mesh.face_adjacency,
            node_count=len(mesh.faces),
        )
        labels = np.asarray(labels, dtype=np.int64)
        if labels.shape != (len(mesh.faces),) or not len(labels):
            return mesh
        counts = np.bincount(labels)
        face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
        safe_areas = np.where(np.isfinite(face_areas), np.maximum(face_areas, 0.0), 0.0)
        areas = np.bincount(labels, weights=safe_areas, minlength=len(counts))
        triangles = np.asarray(mesh.triangles, dtype=np.float64)
        crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        volume_terms = crosses[:, 0] * triangles[:, :, 0].sum(axis=1) / 6.0
        safe_volume_terms = np.where(np.isfinite(volume_terms), volume_terms, 0.0)
        volumes = np.abs(
            np.bincount(labels, weights=safe_volume_terms, minlength=len(counts))
        )
        component = int(np.lexsort((counts, areas, volumes))[-1])
        face_indices = np.flatnonzero(labels == component)
        if not len(face_indices) or len(face_indices) == len(mesh.faces):
            return mesh
        reduced = mesh.submesh([face_indices], append=True, repair=False)
        reduced.remove_unreferenced_vertices()
        return reduced
    except Exception:
        return mesh


def _filter_face_components_by_area(mesh, min_area_ratio: float):
    """Drop disconnected surface fragments below a fraction of the largest area."""
    ratio = float(min_area_ratio or 0.0)
    if ratio <= 0.0 or len(mesh.faces) <= 1:
        return mesh
    if ratio > 1.0 or not math.isfinite(ratio):
        raise ValueError("Mesh repair component area ratio must be in [0, 1]")
    if len(mesh.faces) > MAX_MESH_REPAIR_COMPONENT_FILTER_FACES:
        raise RuntimeError(
            "Mesh repair component filtering exceeds the bounded face limit: "
            f"faces={len(mesh.faces)}, limit={MAX_MESH_REPAIR_COMPONENT_FILTER_FACES}"
        )
    try:
        import trimesh

        labels = np.asarray(
            trimesh.graph.connected_component_labels(
                mesh.face_adjacency,
                node_count=len(mesh.faces),
            ),
            dtype=np.int64,
        )
        if labels.shape != (len(mesh.faces),) or not len(labels):
            return mesh
        counts = np.bincount(labels)
        face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
        safe_areas = np.where(np.isfinite(face_areas), np.maximum(face_areas, 0.0), 0.0)
        areas = np.bincount(labels, weights=safe_areas, minlength=len(counts))
        scores = areas if float(np.max(areas, initial=0.0)) > 0.0 else counts.astype(np.float64)
        keep_components = scores >= float(np.max(scores)) * ratio
        face_indices = np.flatnonzero(keep_components[labels])
        if not len(face_indices) or len(face_indices) == len(mesh.faces):
            return mesh
        reduced = mesh.submesh([face_indices], append=True, repair=False)
        reduced.remove_unreferenced_vertices()
        return reduced
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Mesh repair could not filter components at area ratio {ratio:.10g}"
        ) from exc


def _voxel_close_mesh(mesh, resolution: int, fill_method: str):
    """Create a watertight surface while retaining concavities lost by a hull."""
    import trimesh

    resolution = int(resolution or 0)
    if resolution < 16:
        raise ValueError("Mesh repair voxel resolution must be at least 16")
    if resolution > MAX_MESH_REPAIR_VOXEL_RESOLUTION:
        raise ValueError(
            "Mesh repair voxel resolution must be at most "
            f"{MAX_MESH_REPAIR_VOXEL_RESOLUTION}"
        )
    if fill_method not in MESH_REPAIR_VOXEL_FILL_METHODS:
        expected = ", ".join(MESH_REPAIR_VOXEL_FILL_METHODS)
        raise ValueError(f"Unsupported mesh repair voxel fill method {fill_method!r}; expected {expected}")
    extents = _valid_extents(mesh)
    if extents is None:
        raise ValueError("Mesh repair voxel closure requires a finite non-zero bounding box")
    pitch = float(np.max(extents)) / float(resolution)
    try:
        voxels = mesh.voxelized(pitch=pitch, method="subdivide")
        if int(voxels.filled_count) <= 0:
            raise ValueError("voxelization produced no occupied cells")
        voxels.fill(method=fill_method)
        closed = voxels.marching_cubes.copy()
        closed.apply_transform(voxels.transform)
    except Exception as exc:
        raise RuntimeError(
            f"Mesh repair voxel closure failed at resolution {resolution} with fill {fill_method!r}"
        ) from exc
    if not len(closed.vertices) or not len(closed.faces):
        raise ValueError("Mesh repair voxel closure produced no triangles")
    closed.process(validate=True)
    trimesh.repair.fix_winding(closed)
    trimesh.repair.fix_normals(closed)
    trimesh.repair.fix_inversion(closed)
    _drop_duplicate_and_degenerate_faces(closed)
    closed.process(validate=True)
    return _largest_component(closed).copy()


def _simplify_preserving_topology(mesh, target_faces: int, *, strict: bool = False):
    target_faces = int(target_faces or 0)
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        import pymeshlab
        import trimesh
    except ImportError as exc:
        if strict:
            raise RuntimeError(
                "Topology-preserving voxel-closed repair requires pymeshlab"
            ) from exc
        return mesh
    try:
        mesh_set = pymeshlab.MeshSet()
        mesh_set.add_mesh(
            pymeshlab.Mesh(
                vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
                face_matrix=np.asarray(mesh.faces, dtype=np.int32),
            )
        )
        mesh_set.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=target_faces,
            preservetopology=True,
            preserveboundary=True,
            optimalplacement=True,
            autoclean=True,
        )
        simplified_mesh = mesh_set.current_mesh()
        simplified = trimesh.Trimesh(
            vertices=np.asarray(simplified_mesh.vertex_matrix(), dtype=np.float64),
            faces=np.asarray(simplified_mesh.face_matrix(), dtype=np.int64),
            process=True,
        )
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"Topology-preserving mesh repair failed while simplifying to {target_faces} faces"
            ) from exc
        return mesh
    if not len(simplified.vertices) or not len(simplified.faces):
        if strict:
            raise RuntimeError("Topology-preserving mesh repair produced an empty mesh")
        return mesh
    if strict and len(simplified.faces) > target_faces:
        raise RuntimeError(
            "Topology-preserving mesh repair could not reach the requested face budget: "
            f"target={target_faces}, remaining={len(simplified.faces)}"
        )
    return simplified


def _retriangulate_marching_cubes_mesh(mesh):
    try:
        import pymeshlab
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Marching-cubes retriangulation requires pymeshlab") from exc
    try:
        mesh_set = pymeshlab.MeshSet()
        mesh_set.add_mesh(
            pymeshlab.Mesh(
                vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
                face_matrix=np.asarray(mesh.faces, dtype=np.int32),
            )
        )
        mesh_set.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=max(len(mesh.faces) - 2, 4),
            preservetopology=True,
            preserveboundary=True,
            optimalplacement=False,
            autoclean=True,
        )
        result = mesh_set.current_mesh()
        retriangulated = trimesh.Trimesh(
            vertices=np.asarray(result.vertex_matrix(), dtype=np.float64),
            faces=np.asarray(result.face_matrix(), dtype=np.int64),
            process=False,
        )
    except Exception as exc:
        raise RuntimeError("Single-edge marching-cubes repair failed") from exc
    if not len(retriangulated.vertices) or not len(retriangulated.faces):
        raise RuntimeError("Single-edge marching-cubes repair produced an empty mesh")
    removed_faces = len(mesh.faces) - len(retriangulated.faces)
    removed_vertices = len(mesh.vertices) - len(retriangulated.vertices)
    if not (0 <= removed_faces <= 2 and 0 <= removed_vertices <= 1):
        raise RuntimeError(
            "Single-edge marching-cubes repair exceeded its local edit budget: "
            f"removed_faces={removed_faces}, removed_vertices={removed_vertices}"
        )
    return retriangulated


def _retriangulation_geometry_audit(reference, candidate, prefix: str) -> dict:
    from scipy.spatial import cKDTree

    from backend.benchmark.metrics import _mesh_surface_points

    reference_extents = np.asarray(reference.extents, dtype=np.float64)
    candidate_extents = np.asarray(candidate.extents, dtype=np.float64)
    reference_bounds_center = np.asarray(reference.bounds, dtype=np.float64).mean(axis=0)
    candidate_bounds_center = np.asarray(candidate.bounds, dtype=np.float64).mean(axis=0)
    reference_diagonal = float(np.linalg.norm(reference_extents))
    reference_volume = abs(float(reference.volume))

    face_count_relative_change = abs(len(candidate.faces) - len(reference.faces)) / max(
        len(reference.faces),
        1,
    )
    vertex_count_relative_change = abs(len(candidate.vertices) - len(reference.vertices)) / max(
        len(reference.vertices),
        1,
    )
    volume_relative_change = abs(abs(float(candidate.volume)) - reference_volume) / max(
        reference_volume,
        1e-12,
    )
    bbox_extent_relative_change = float(
        np.max(
            np.abs(candidate_extents - reference_extents)
            / np.maximum(np.abs(reference_extents), 1e-12)
        )
    )
    bounds_center_shift_normalized = float(
        np.linalg.norm(candidate_bounds_center - reference_bounds_center)
        / max(reference_diagonal, 1e-12)
    )

    reference_vertices = np.asarray(reference.vertices, dtype=np.float64)
    candidate_vertices = np.asarray(candidate.vertices, dtype=np.float64)
    reference_vertex_tree = cKDTree(reference_vertices)
    candidate_vertex_tree = cKDTree(candidate_vertices)
    candidate_to_reference_vertices, _ = reference_vertex_tree.query(candidate_vertices, k=1)
    reference_to_candidate_vertices, _ = candidate_vertex_tree.query(reference_vertices, k=1)
    removed_vertex_count = max(len(reference_vertices) - len(candidate_vertices), 0)
    retained_reference_distances = np.sort(reference_to_candidate_vertices)
    if removed_vertex_count:
        retained_reference_distances = retained_reference_distances[:-removed_vertex_count]
    vertex_displacement_max_normalized = float(
        max(
            np.max(candidate_to_reference_vertices, initial=0.0),
            np.max(retained_reference_distances, initial=0.0),
        )
        / max(reference_diagonal, 1e-12)
    )

    reference_points = _mesh_surface_points(
        reference,
        max_points=RETRIANGULATION_SURFACE_SAMPLE_POINTS,
    )
    candidate_points = _mesh_surface_points(
        candidate,
        max_points=RETRIANGULATION_SURFACE_SAMPLE_POINTS,
    )
    if not len(reference_points) or not len(candidate_points):
        surface_chamfer_normalized = math.nan
        surface_hausdorff95_normalized = math.nan
    else:
        reference_tree = cKDTree(reference_points)
        candidate_tree = cKDTree(candidate_points)
        candidate_to_reference, _ = reference_tree.query(candidate_points, k=1)
        reference_to_candidate, _ = candidate_tree.query(reference_points, k=1)
        surface_chamfer_normalized = float(
            (np.mean(candidate_to_reference) + np.mean(reference_to_candidate))
            / 2.0
            / max(reference_diagonal, 1e-12)
        )
        surface_hausdorff95_normalized = float(
            max(
                np.quantile(candidate_to_reference, 0.95),
                np.quantile(reference_to_candidate, 0.95),
            )
            / max(reference_diagonal, 1e-12)
        )

    values = (
        face_count_relative_change,
        vertex_count_relative_change,
        volume_relative_change,
        bbox_extent_relative_change,
        bounds_center_shift_normalized,
        vertex_displacement_max_normalized,
        surface_chamfer_normalized,
        surface_hausdorff95_normalized,
    )
    geometry_preserved = bool(
        all(math.isfinite(value) for value in values)
        and face_count_relative_change <= RETRIANGULATION_MAX_FACE_COUNT_RELATIVE_CHANGE
        and vertex_count_relative_change <= RETRIANGULATION_MAX_VERTEX_COUNT_RELATIVE_CHANGE
        and volume_relative_change <= RETRIANGULATION_MAX_VOLUME_RELATIVE_CHANGE
        and bbox_extent_relative_change <= RETRIANGULATION_MAX_BBOX_EXTENT_RELATIVE_CHANGE
        and bounds_center_shift_normalized
        <= RETRIANGULATION_MAX_BOUNDS_CENTER_SHIFT_NORMALIZED
        and vertex_displacement_max_normalized
        <= RETRIANGULATION_MAX_VERTEX_DISPLACEMENT_NORMALIZED
        and surface_chamfer_normalized <= RETRIANGULATION_MAX_SURFACE_CHAMFER_NORMALIZED
        and surface_hausdorff95_normalized
        <= RETRIANGULATION_MAX_SURFACE_HAUSDORFF95_NORMALIZED
    )
    return {
        f"{prefix}_face_count_relative_change_abs": float(face_count_relative_change),
        f"{prefix}_vertex_count_relative_change_abs": float(vertex_count_relative_change),
        f"{prefix}_volume_relative_change_abs": float(volume_relative_change),
        f"{prefix}_bbox_extent_relative_change_max": bbox_extent_relative_change,
        f"{prefix}_bounds_center_shift_normalized": bounds_center_shift_normalized,
        f"{prefix}_vertex_displacement_max_normalized": vertex_displacement_max_normalized,
        f"{prefix}_surface_chamfer_l1_normalized": surface_chamfer_normalized,
        f"{prefix}_surface_hausdorff95_normalized": surface_hausdorff95_normalized,
        f"{prefix}_geometry_preserved": geometry_preserved,
    }


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


def repair_mesh_for_printable_stl(
    mesh_path: Path,
    output_path: Path,
    mode: str = "printable",
    *,
    target_faces: int = 0,
    max_normalized_face_density_log1p: float = 0.0,
    preconditioner: str = "legacy",
    component_area_ratio: float = 0.0,
    voxel_resolution: int = DEFAULT_MESH_REPAIR_VOXEL_RESOLUTION,
    voxel_fill_method: str = "orthographic",
    metrics: dict | None = None,
) -> Path:
    if mode not in MESH_REPAIR_MODES or mode == "none":
        raise ValueError(f"Unsupported mesh repair mode: {mode}")
    if preconditioner not in MESH_REPAIR_PRECONDITIONERS:
        expected = ", ".join(MESH_REPAIR_PRECONDITIONERS)
        raise ValueError(f"Unsupported mesh repair preconditioner {preconditioner!r}; expected {expected}")
    component_area_ratio = float(component_area_ratio or 0.0)
    if component_area_ratio < 0.0 or component_area_ratio > 1.0 or not math.isfinite(component_area_ratio):
        raise ValueError("Mesh repair component area ratio must be in [0, 1]")
    mesh = load_mesh(mesh_path)
    if metrics is not None:
        metrics["repair_input_faces"] = int(len(mesh.faces))
    if preconditioner == "voxel-close":
        mesh = _filter_face_components_by_area(mesh, component_area_ratio)
        if metrics is not None:
            metrics["repair_precondition_filtered_faces"] = int(len(mesh.faces))
        mesh = _voxel_close_mesh(mesh, int(voxel_resolution), voxel_fill_method)
        if metrics is not None:
            metrics["repair_precondition_voxel_faces"] = int(len(mesh.faces))
    target_faces = int(target_faces or 0)
    if target_faces <= 0:
        target_faces = max_faces_for_normalized_bbox_complexity(
            _valid_extents(mesh),
            max_normalized_face_density_log1p,
        )
    if target_faces > 0 and len(mesh.faces) > target_faces:
        original_faces = len(mesh.faces)
        if preconditioner == "voxel-close":
            mesh = _simplify_preserving_topology(mesh, target_faces, strict=True)
        else:
            mesh = _simplify_to_face_count(mesh, target_faces, strict=True)
        if metrics is not None:
            metrics["repair_simplified_faces"] = int(len(mesh.faces))
        safe_repair_faces = max(MIN_SAFE_TOPOLOGY_REPAIR_FACES, target_faces * 4)
        if safe_repair_faces < len(mesh.faces) <= MAX_FAST_COMPONENT_FILTER_FACES:
            filtered = _largest_face_component(mesh)
            if len(filtered.faces) < len(mesh.faces):
                mesh = filtered
                if len(mesh.faces) > target_faces:
                    mesh = _simplify_to_face_count(mesh, target_faces, strict=True)
        if len(mesh.faces) > safe_repair_faces:
            raise RuntimeError(
                "Mesh repair preconditioning could not reach the safe topology-repair limit: "
                f"faces={original_faces}, target={target_faces}, safe_limit={safe_repair_faces}, "
                f"remaining={len(mesh.faces)}. "
                "Install fast-simplification or use a provider with native face-count control."
            )
    preclean_printable = False
    if preconditioner == "voxel-close":
        preclean_audit = _printability_audit(mesh, "repair_preclean")
        preclean_printable = preclean_audit["repair_preclean_printable"]
        if metrics is not None:
            metrics.update(preclean_audit)
        retriangulation_attempted = bool(
            not preclean_printable
            and preclean_audit["repair_preclean_degenerate_face_count"] == 1
            and preclean_audit["repair_preclean_bbox_has_volume"]
            and preclean_audit["repair_preclean_is_watertight"]
            and preclean_audit["repair_preclean_is_volume"]
            and preclean_audit["repair_preclean_winding_consistent"]
            and preclean_audit["repair_preclean_component_count"] == 1
            and preclean_audit["repair_preclean_nonmanifold_edge_count"] == 0
            and preclean_audit["repair_preclean_positive_volume"]
        )
        retriangulation_accepted = False
        if retriangulation_attempted:
            try:
                retriangulated = _retriangulate_marching_cubes_mesh(mesh)
            except RuntimeError:
                retriangulated = None
            if retriangulated is not None:
                retriangulated_audit = _printability_audit(
                    retriangulated,
                    "repair_preclean_retriangulated",
                )
                try:
                    retriangulated_geometry_audit = _retriangulation_geometry_audit(
                        mesh,
                        retriangulated,
                        "repair_preclean_retriangulated",
                    )
                except Exception as exc:
                    retriangulated_geometry_audit = {
                        "repair_preclean_retriangulated_geometry_preserved": False,
                        "repair_preclean_retriangulated_geometry_audit_error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                if metrics is not None:
                    metrics.update(retriangulated_audit)
                    metrics.update(retriangulated_geometry_audit)
                if (
                    retriangulated_audit["repair_preclean_retriangulated_printable"]
                    and retriangulated_geometry_audit[
                        "repair_preclean_retriangulated_geometry_preserved"
                    ]
                ):
                    mesh = retriangulated
                    preclean_printable = True
                    retriangulation_accepted = True
        if metrics is not None:
            metrics["repair_preclean_retriangulation_attempted"] = retriangulation_attempted
            metrics["repair_preclean_retriangulation_accepted"] = retriangulation_accepted
    if mode == "convex-hull":
        repaired = _convex_hull_mesh(mesh)
        used_convex_hull = True
        cleaning_skipped = True
    elif preconditioner == "voxel-close" and preclean_printable:
        repaired = mesh.copy()
        used_convex_hull = False
        cleaning_skipped = True
    else:
        repaired = _clean_mesh(mesh)
        cleaning_skipped = False
        if metrics is not None:
            metrics.update(_printability_audit(repaired, "repair_cleaned"))
        used_convex_hull = mode == "printable" and not mesh_is_printable_volume(repaired)
        if used_convex_hull:
            repaired = _convex_hull_mesh(repaired)
    if not len(repaired.vertices) or not len(repaired.faces):
        raise ValueError(f"Mesh repair produced no triangles: {mesh_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(output_path)
    if metrics is not None:
        metrics["repair_cleaning_skipped"] = bool(cleaning_skipped)
        metrics["repair_convex_hull_used"] = bool(used_convex_hull)
        metrics["repair_output_faces"] = int(len(repaired.faces))
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


def bbox_extent_comparison_metrics(source_extents, inferred_extents) -> dict[str, float]:
    source = _target_extents_array(source_extents)
    inferred = _target_extents_array(inferred_extents)
    if source is None or inferred is None:
        return {}

    source_shape = source / float(np.max(source))
    inferred_shape = inferred / float(np.max(inferred))
    intersection = float(np.prod(np.minimum(source_shape, inferred_shape)))
    union = float(np.prod(source_shape) + np.prod(inferred_shape) - intersection)
    return {
        "inferred_bbox_shape_log_mae": float(np.mean(np.abs(np.log(inferred_shape / source_shape)))),
        "inferred_bbox_shape_relative_mae": float(
            np.mean(np.abs(inferred_shape - source_shape) / source_shape)
        ),
        "inferred_bbox_centered_iou": intersection / union if union > 0 else 0.0,
    }


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


def direct_mesh_reference_stl_path(sample: dict, output_dir: Path, args, reference_method: str) -> Path | None:
    sample_id = str(sample.get("id", ""))
    candidates = [output_dir.parent / reference_method / "output_model.stl"]
    reference_root = getattr(args, "direct_mesh_reference_output_dir", None)
    if reference_root and sample_id and reference_method:
        candidates.append(Path(reference_root) / reference_method / sample_id / reference_method / "output_model.stl")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def direct_mesh_bbox_placeholders(sample: dict, output_dir: Path, args) -> dict[str, str]:
    stl_target_dimension = float(getattr(args, "stl_target_dimension", 0.0) or 0.0)
    source_extents = source_mesh_bbox_extents(sample, stl_target_dimension)
    reference_method = str(getattr(args, "direct_mesh_reference_method", "mirror") or "mirror")
    reference_stl = direct_mesh_reference_stl_path(sample, output_dir, args, reference_method)
    reference_extents = stl_file_bbox_extents(reference_stl)
    mirror_stl = direct_mesh_reference_stl_path(sample, output_dir, args, "mirror")
    mirror_extents = stl_file_bbox_extents(mirror_stl)
    values = {
        "stl_target_dimension": f"{stl_target_dimension:.10g}" if stl_target_dimension > 0 else "",
        "reference_method": reference_method,
        "reference_stl": str(reference_stl or ""),
        "mirror_stl": str(mirror_stl or ""),
        "inferred_bbox_method": "mirror",
        "inferred_bbox_stl": str(mirror_stl or ""),
    }
    values.update(_bbox_placeholder_values("source", source_extents))
    values.update(_bbox_placeholder_values("mirror", mirror_extents))
    values.update(_bbox_placeholder_values("inferred", mirror_extents))
    values.update(_bbox_placeholder_values("reference", reference_extents))
    values.update(
        {
            key: f"{value:.10g}"
            for key, value in bbox_extent_comparison_metrics(source_extents, mirror_extents).items()
        }
    )
    return values


def _tail_file(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def _external_command_failure_message(
    *,
    command: str,
    returncode: int | str,
    stdout_path: Path,
    stderr_path: Path,
) -> str:
    stdout_tail = _tail_file(stdout_path)
    stderr_tail = _tail_file(stderr_path)
    parts = [
        f"External image-to-mesh command failed with exit status {returncode}.",
        f"Command: {command}",
        f"stdout_log: {stdout_path}",
        f"stderr_log: {stderr_path}",
    ]
    if stdout_tail:
        parts.append(f"stdout_tail:\n{stdout_tail}")
    if stderr_tail:
        parts.append(f"stderr_tail:\n{stderr_tail}")
    return "\n".join(parts)


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


def max_faces_for_normalized_bbox_complexity(
    bbox_extents,
    max_normalized_face_density_log1p: float,
) -> int:
    """Convert the scale-free STL complexity limit into a per-mesh face cap."""
    target = _target_extents_array(bbox_extents)
    limit = float(max_normalized_face_density_log1p or 0.0)
    if target is None or limit <= 0 or not math.isfinite(limit):
        return 0
    max_extent = float(np.max(target))
    normalized_bbox_volume = float(np.prod(target) / (max_extent**3))
    if normalized_bbox_volume <= 0 or not math.isfinite(normalized_bbox_volume):
        return 0
    try:
        max_face_density = math.expm1(limit)
    except OverflowError:
        return 0
    if not math.isfinite(max_face_density):
        return 0
    return max(4, int(math.floor(max_face_density * normalized_bbox_volume)))


def normalized_bbox_complexity_log1p(mesh) -> float:
    extents = _valid_extents(mesh)
    if extents is None or not len(mesh.faces):
        return math.inf
    max_extent = float(np.max(extents))
    normalized_bbox_volume = float(np.prod(extents) / (max_extent**3))
    if normalized_bbox_volume <= 0 or not math.isfinite(normalized_bbox_volume):
        return math.inf
    return float(math.log1p(len(mesh.faces) / normalized_bbox_volume))


def _simplify_to_face_count(mesh, target_faces: int, *, strict: bool = False):
    target_faces = int(target_faces or 0)
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    simplify = getattr(mesh, "simplify_quadric_decimation", None)
    if simplify is None:
        if strict:
            raise RuntimeError("Mesh repair preconditioning requires fast-simplification")
        return mesh
    try:
        simplified = simplify(face_count=target_faces)
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"Mesh repair preconditioning failed while simplifying to {target_faces} faces"
            ) from exc
        return mesh
    if not len(simplified.vertices) or not len(simplified.faces):
        if strict:
            raise RuntimeError("Mesh repair preconditioning produced an empty mesh")
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
    max_normalized_face_density_log1p: float = 0.0,
    preserve_printability: bool = False,
) -> Path:
    mesh = load_mesh(mesh_path)
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Mesh postprocess input has no triangles: {mesh_path}")
    processed = _scale_to_max_dimension(mesh, float(target_max_dimension or 0.0))
    processed = _enforce_min_bbox_dimension(processed, float(min_bbox_dimension or 0.0))
    processed = _clamp_bbox_aspect_ratio(processed, float(max_bbox_aspect_ratio or 0.0))
    processed = _match_bbox_extents(processed, target_bbox_extents)
    fixed_target = int(target_faces or 0)
    if fixed_target > 0:
        if preserve_printability:
            processed = _simplify_preserving_topology(
                processed,
                fixed_target,
                strict=True,
            )
        else:
            processed = _simplify_to_face_count(processed, fixed_target)

    complexity_limit = float(max_normalized_face_density_log1p or 0.0)
    if complexity_limit > 0:
        for _ in range(8):
            complexity = normalized_bbox_complexity_log1p(processed)
            if complexity <= complexity_limit:
                break
            adaptive_target = max_faces_for_normalized_bbox_complexity(
                _valid_extents(processed),
                complexity_limit,
            )
            previous_faces = len(processed.faces)
            if preserve_printability:
                processed = _simplify_preserving_topology(
                    processed,
                    adaptive_target,
                    strict=True,
                )
            else:
                processed = _simplify_to_face_count(processed, adaptive_target)
            if len(processed.faces) >= previous_faces:
                break
        final_complexity = normalized_bbox_complexity_log1p(processed)
        if final_complexity > complexity_limit:
            raise RuntimeError(
                "Mesh still exceeds the adaptive scale-free complexity limit after simplification: "
                f"value={final_complexity:.10g}, limit={complexity_limit:.10g}, faces={len(processed.faces)}. "
                "Install fast-simplification or use a provider with native face-count control."
            )
    processed.remove_unreferenced_vertices()
    if preserve_printability and not mesh_is_printable_volume(processed):
        raise RuntimeError(
            "Mesh postprocess would emit a non-printable volume after topology-preserving simplification"
        )
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
        stdout_path = output_dir / "external_command.stdout.log"
        stderr_path = output_dir / "external_command.stderr.log"
        command_metrics_path = output_dir / "direct_mesh_command_metrics.json"
        for stale_path in (
            mesh_output_path,
            stl_path,
            output_dir / "provider_metrics.json",
            command_metrics_path,
        ):
            if stale_path.is_file():
                stale_path.unlink()
        command_started = time.perf_counter()
        command_status = "failed"
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            try:
                subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    timeout=max(1, int(getattr(args, "direct_mesh_timeout", 1800))),
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                command_status = "ok"
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    _external_command_failure_message(
                        command=command,
                        returncode=exc.returncode,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                    )
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    _external_command_failure_message(
                        command=command,
                        returncode=f"timeout after {exc.timeout}s",
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                    )
                ) from exc
            finally:
                command_metrics = {
                    "direct_mesh_command_runtime_seconds": time.perf_counter() - command_started,
                    "direct_mesh_command_status": command_status,
                }
                temporary_metrics = command_metrics_path.with_name(
                    f".{command_metrics_path.name}.{os.getpid()}.tmp"
                )
                temporary_metrics.write_text(
                    json.dumps(command_metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary_metrics.replace(command_metrics_path)
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
