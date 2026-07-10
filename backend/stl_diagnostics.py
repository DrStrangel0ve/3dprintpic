from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def stl_diagnostics(stl_path):
    import trimesh

    path = Path(stl_path)
    diagnostics = {
        "stl_model": str(path),
        "stl_exists": path.exists(),
        "stl_file_size_bytes": path.stat().st_size if path.exists() else 0,
    }
    if not path.exists():
        return diagnostics

    loaded = trimesh.load_mesh(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    extents = np.asarray(mesh.extents, dtype=np.float64)
    has_3d_extents = extents.shape == (3,) and np.all(np.isfinite(extents))
    finite_extents = extents[np.isfinite(extents)]
    bbox_has_volume = bool(has_3d_extents and np.all(extents > 0))
    bbox_volume = float(np.prod(extents)) if has_3d_extents else math.nan
    z_range = float(extents[2]) if extents.shape == (3,) and np.isfinite(extents[2]) else math.nan
    min_extent = float(np.min(finite_extents)) if len(finite_extents) else math.nan
    max_extent = float(np.max(finite_extents)) if len(finite_extents) else math.nan
    if not len(finite_extents):
        aspect_ratio = math.nan
    elif min_extent <= 0:
        aspect_ratio = math.inf
    else:
        aspect_ratio = float(max_extent / min_extent)
    signed_volume = float(mesh.volume) if np.isfinite(mesh.volume) else math.nan
    try:
        component_count = len(mesh.split(only_watertight=False))
    except Exception:
        component_count = math.nan
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces):
        face_edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        face_edges = np.sort(face_edges, axis=1)
        _, edge_counts = np.unique(face_edges, axis=0, return_counts=True)
        nonmanifold_edge_count = int(np.count_nonzero(edge_counts != 2))
        face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
        degenerate_face_count = int(np.count_nonzero((~np.isfinite(face_areas)) | (face_areas <= 1e-12)))
        degenerate_face_ratio = float(degenerate_face_count / len(faces))
    else:
        nonmanifold_edge_count = 0
        degenerate_face_count = 0
        degenerate_face_ratio = math.nan
    is_manifold = bool(len(faces) > 0 and nonmanifold_edge_count == 0 and degenerate_face_count == 0)
    component_excess = abs(component_count - 1) if np.isfinite(component_count) else math.nan
    faces_per_bbox_volume = (
        float(len(mesh.faces) / bbox_volume) if np.isfinite(bbox_volume) and bbox_volume > 0 else math.nan
    )
    diagnostics.update(
        {
            "stl_vertices": int(len(mesh.vertices)),
            "stl_faces": int(len(mesh.faces)),
            "stl_is_watertight": bool(mesh.is_watertight),
            "stl_is_volume": bool(mesh.is_volume),
            "stl_is_manifold": is_manifold,
            "stl_nonmanifold_edge_count": nonmanifold_edge_count,
            "stl_nonmanifold_edge_count_log1p": float(math.log1p(nonmanifold_edge_count)),
            "stl_degenerate_face_count": degenerate_face_count,
            "stl_degenerate_face_ratio": degenerate_face_ratio,
            "stl_winding_consistent": bool(mesh.is_winding_consistent),
            "stl_component_count": component_count,
            "stl_single_component": bool(component_count == 1) if np.isfinite(component_count) else False,
            "stl_component_excess": component_excess,
            "stl_component_excess_log1p": (
                float(math.log1p(component_excess)) if np.isfinite(component_excess) else math.nan
            ),
            "stl_euler_number": int(mesh.euler_number) if mesh.euler_number is not None else math.nan,
            "stl_surface_area": float(mesh.area) if np.isfinite(mesh.area) else math.nan,
            "stl_volume": signed_volume,
            "stl_volume_abs": abs(signed_volume) if np.isfinite(signed_volume) else math.nan,
            "stl_positive_volume": bool(np.isfinite(signed_volume) and signed_volume > 0),
            "stl_z_range": z_range,
            "stl_bbox_x": float(extents[0]) if extents.shape == (3,) and np.isfinite(extents[0]) else math.nan,
            "stl_bbox_y": float(extents[1]) if extents.shape == (3,) and np.isfinite(extents[1]) else math.nan,
            "stl_bbox_z": z_range,
            "stl_bbox_min_dimension": min_extent,
            "stl_bbox_max_dimension": max_extent,
            "stl_bbox_has_volume": bbox_has_volume,
            "stl_bbox_aspect_ratio": aspect_ratio,
            "stl_bbox_volume": bbox_volume,
            "stl_faces_per_bbox_volume": faces_per_bbox_volume,
            "stl_faces_per_bbox_volume_log1p": (
                float(math.log1p(faces_per_bbox_volume)) if np.isfinite(faces_per_bbox_volume) else math.nan
            ),
        }
    )
    return diagnostics


def json_safe_stl_diagnostics(value):
    if isinstance(value, dict):
        return {key: json_safe_stl_diagnostics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_stl_diagnostics(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
