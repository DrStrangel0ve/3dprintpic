from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np


MAX_SURFACE_FILL_PROXY_FACES = 1_000_000
SURFACE_FILL_PROXY_CHUNK_FACES = 100_000
MAX_SIGNED_VOLUME_RELIABILITY_FACES = 250_000


def _face_component_labels(mesh) -> np.ndarray:
    import trimesh

    labels = np.asarray(
        trimesh.graph.connected_component_labels(
            mesh.face_adjacency,
            node_count=len(mesh.faces),
        ),
        dtype=np.int64,
    )
    if labels.shape != (len(mesh.faces),) or np.any(labels < 0):
        raise ValueError("could not label every face component")
    return labels


def _face_component_count(mesh) -> int:
    if not len(mesh.faces):
        return 0
    return int(_face_component_labels(mesh).max(initial=-1) + 1)


def _signed_volume_reliability(
    mesh,
    *,
    include_topology: bool,
    component_count,
    signed_volume: float,
    bbox_volume: float,
) -> tuple[bool, str, bool]:
    if not len(mesh.faces):
        return False, "empty-mesh", True
    if not math.isfinite(signed_volume):
        return False, "non-finite-signed-volume", True
    if not math.isfinite(bbox_volume) or bbox_volume <= 0:
        return False, "invalid-bounding-box-volume", True
    if abs(signed_volume) / bbox_volume <= 1e-12:
        return False, "non-finite-or-zero-signed-volume", True
    if not include_topology and len(mesh.faces) > MAX_SIGNED_VOLUME_RELIABILITY_FACES:
        return False, "topology-not-assessed-face-limit", False
    try:
        is_watertight = bool(mesh.is_watertight)
        winding_consistent = bool(mesh.is_winding_consistent)
        body_count = (
            int(component_count)
            if include_topology and np.isfinite(component_count)
            else _face_component_count(mesh)
        )
    except Exception as exc:
        return False, f"topology-assessment-error:{type(exc).__name__}", False
    if not is_watertight:
        return False, "not-watertight", True
    if not winding_consistent:
        return False, "inconsistent-winding", True
    if body_count != 1:
        return False, f"component-count:{body_count}", True
    orientation = "positive" if signed_volume > 0 else "reversed"
    return True, f"closed-single-component-consistent-winding:{orientation}", True


def _surface_fill_proxy(mesh) -> dict:
    started = time.perf_counter()
    result = {
        "surface_fill_ratio_supported": False,
        "surface_fill_ratio_method": "component-centered-unsigned-tetrahedra",
        "surface_fill_ratio": math.nan,
    }
    try:
        if not len(mesh.faces):
            raise ValueError("surface fill proxy requires at least one face")
        if len(mesh.faces) > MAX_SURFACE_FILL_PROXY_FACES:
            raise ValueError(
                "surface fill proxy exceeds bounded face limit: "
                f"faces={len(mesh.faces)}, limit={MAX_SURFACE_FILL_PROXY_FACES}"
            )
        extents = np.asarray(mesh.extents, dtype=np.float64)
        if extents.shape != (3,) or not np.all(np.isfinite(extents)) or np.any(extents <= 0):
            raise ValueError("surface fill proxy requires a finite non-zero bounding box")
        bbox_volume = float(np.prod(extents))
        faces = np.asarray(mesh.faces, dtype=np.int64)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        labels = _face_component_labels(mesh)
        component_count = int(labels.max(initial=-1) + 1)
        if component_count <= 0:
            raise ValueError("surface fill proxy found no face components")
        component_mins = np.full((component_count, 3), np.inf, dtype=np.float64)
        component_maxs = np.full((component_count, 3), -np.inf, dtype=np.float64)
        for start in range(0, len(faces), SURFACE_FILL_PROXY_CHUNK_FACES):
            stop = min(start + SURFACE_FILL_PROXY_CHUNK_FACES, len(faces))
            triangles = vertices[faces[start:stop]]
            chunk_labels = labels[start:stop]
            np.minimum.at(component_mins, chunk_labels, np.min(triangles, axis=1))
            np.maximum.at(component_maxs, chunk_labels, np.max(triangles, axis=1))
        component_centers = (component_mins + component_maxs) / 2.0
        proxy_volume = 0.0
        for start in range(0, len(faces), SURFACE_FILL_PROXY_CHUNK_FACES):
            stop = min(start + SURFACE_FILL_PROXY_CHUNK_FACES, len(faces))
            triangles = vertices[faces[start:stop]]
            centered = triangles - component_centers[labels[start:stop], None, :]
            signed_six_volumes = np.einsum(
                "ij,ij->i",
                centered[:, 0],
                np.cross(centered[:, 1], centered[:, 2]),
            )
            proxy_volume += float(np.sum(np.abs(signed_six_volumes)) / 6.0)
        unclipped_ratio = float(proxy_volume / bbox_volume)
        if not math.isfinite(unclipped_ratio) or unclipped_ratio <= 0:
            raise ValueError(
                f"surface fill proxy produced invalid ratio {unclipped_ratio!r}"
            )
        ratio = float(np.clip(unclipped_ratio, 0.0, 1.0))
        result.update(
            {
                "surface_fill_ratio_supported": True,
                "surface_fill_ratio": ratio,
                "surface_fill_ratio_comparison": unclipped_ratio,
                "surface_fill_proxy_component_count": component_count,
                "surface_fill_proxy_volume": proxy_volume,
                "surface_fill_proxy_unclipped_ratio": unclipped_ratio,
                "surface_fill_proxy_clipped": bool(unclipped_ratio > 1.0),
            }
        )
    except Exception as exc:
        result["surface_fill_ratio_error"] = f"{type(exc).__name__}: {exc}"
    result["surface_fill_proxy_runtime_seconds"] = float(time.perf_counter() - started)
    return result


def mesh_diagnostics(
    mesh_path,
    prefix="mesh",
    *,
    include_topology=True,
    include_surface_fill_proxy=False,
    force_surface_fill_proxy=False,
):
    import trimesh

    prefix = f"{str(prefix).rstrip('_')}_"

    def field(name):
        return f"{prefix}{name}"

    path = Path(mesh_path)
    diagnostics = {
        field("model"): str(path),
        field("exists"): path.exists(),
        field("file_size_bytes"): path.stat().st_size if path.exists() else 0,
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
    if include_topology:
        try:
            component_count = _face_component_count(mesh)
        except Exception:
            component_count = math.nan
    else:
        component_count = math.nan
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) and include_topology:
        face_edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        face_edges = np.sort(face_edges, axis=1)
        _, edge_counts = np.unique(face_edges, axis=0, return_counts=True)
        nonmanifold_edge_count = int(np.count_nonzero(edge_counts != 2))
        face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
        degenerate_face_count = int(np.count_nonzero((~np.isfinite(face_areas)) | (face_areas <= 1e-12)))
        degenerate_face_ratio = float(degenerate_face_count / len(faces))
    elif include_topology:
        nonmanifold_edge_count = 0
        degenerate_face_count = 0
        degenerate_face_ratio = math.nan
    else:
        nonmanifold_edge_count = math.nan
        degenerate_face_count = math.nan
        degenerate_face_ratio = math.nan
    is_manifold = (
        bool(len(faces) > 0 and nonmanifold_edge_count == 0 and degenerate_face_count == 0)
        if include_topology
        else math.nan
    )
    component_excess = abs(component_count - 1) if np.isfinite(component_count) else math.nan
    faces_per_bbox_volume = (
        float(len(mesh.faces) / bbox_volume) if np.isfinite(bbox_volume) and bbox_volume > 0 else math.nan
    )
    normalized_bbox_volume = (
        float(bbox_volume / (max_extent**3))
        if np.isfinite(bbox_volume) and bbox_volume > 0 and np.isfinite(max_extent) and max_extent > 0
        else math.nan
    )
    faces_per_normalized_bbox_volume = (
        float(len(mesh.faces) / normalized_bbox_volume)
        if np.isfinite(normalized_bbox_volume) and normalized_bbox_volume > 0
        else math.nan
    )
    volume_fill_ratio = (
        float(abs(signed_volume) / bbox_volume)
        if np.isfinite(signed_volume) and np.isfinite(bbox_volume) and bbox_volume > 0
        else math.nan
    )
    (
        volume_fill_ratio_reliable,
        volume_fill_ratio_reliability_reason,
        volume_fill_ratio_topology_assessed,
    ) = _signed_volume_reliability(
        mesh,
        include_topology=include_topology,
        component_count=component_count,
        signed_volume=signed_volume,
        bbox_volume=bbox_volume,
    )
    volume_fill_ratio_reliability_status = (
        "reliable"
        if volume_fill_ratio_reliable
        else "unreliable"
        if volume_fill_ratio_topology_assessed
        else "unknown"
    )
    diagnostics.update(
        {
            field("vertices"): int(len(mesh.vertices)),
            field("faces"): int(len(mesh.faces)),
            field("is_watertight"): bool(mesh.is_watertight) if include_topology else math.nan,
            field("is_volume"): bool(mesh.is_volume) if include_topology else math.nan,
            field("is_manifold"): is_manifold,
            field("nonmanifold_edge_count"): nonmanifold_edge_count,
            field("nonmanifold_edge_count_log1p"): (
                float(math.log1p(nonmanifold_edge_count))
                if np.isfinite(nonmanifold_edge_count)
                else math.nan
            ),
            field("degenerate_face_count"): degenerate_face_count,
            field("degenerate_face_ratio"): degenerate_face_ratio,
            field("winding_consistent"): (
                bool(mesh.is_winding_consistent) if include_topology else math.nan
            ),
            field("component_count"): component_count,
            field("single_component"): (
                bool(component_count == 1) if np.isfinite(component_count) else math.nan
            ),
            field("component_excess"): component_excess,
            field("component_excess_log1p"): (
                float(math.log1p(component_excess)) if np.isfinite(component_excess) else math.nan
            ),
            field("euler_number"): (
                int(mesh.euler_number)
                if include_topology and mesh.euler_number is not None
                else math.nan
            ),
            field("surface_area"): float(mesh.area) if np.isfinite(mesh.area) else math.nan,
            field("volume"): signed_volume,
            field("volume_abs"): abs(signed_volume) if np.isfinite(signed_volume) else math.nan,
            field("volume_fill_ratio"): volume_fill_ratio,
            field("volume_fill_ratio_reliable"): volume_fill_ratio_reliable,
            field("volume_fill_ratio_reliability_status"): (
                volume_fill_ratio_reliability_status
            ),
            field("volume_fill_ratio_reliability_reason"): volume_fill_ratio_reliability_reason,
            field("volume_fill_ratio_topology_assessed"): volume_fill_ratio_topology_assessed,
            field("volume_fill_ratio_self_intersection_assessed"): False,
            field("volume_fill_ratio_reliability_scope"): (
                "watertightness,winding,components;self-intersections-unassessed"
            ),
            field("positive_volume"): bool(np.isfinite(signed_volume) and signed_volume > 0),
            field("z_range"): z_range,
            field("bbox_x"): float(extents[0]) if extents.shape == (3,) and np.isfinite(extents[0]) else math.nan,
            field("bbox_y"): float(extents[1]) if extents.shape == (3,) and np.isfinite(extents[1]) else math.nan,
            field("bbox_z"): z_range,
            field("bbox_min_dimension"): min_extent,
            field("bbox_max_dimension"): max_extent,
            field("bbox_has_volume"): bbox_has_volume,
            field("bbox_aspect_ratio"): aspect_ratio,
            field("bbox_volume"): bbox_volume,
            field("faces_per_bbox_volume"): faces_per_bbox_volume,
            field("faces_per_bbox_volume_log1p"): (
                float(math.log1p(faces_per_bbox_volume)) if np.isfinite(faces_per_bbox_volume) else math.nan
            ),
            field("normalized_bbox_volume"): normalized_bbox_volume,
            field("faces_per_normalized_bbox_volume"): faces_per_normalized_bbox_volume,
            field("faces_per_normalized_bbox_volume_log1p"): (
                float(math.log1p(faces_per_normalized_bbox_volume))
                if np.isfinite(faces_per_normalized_bbox_volume)
                else math.nan
            ),
            field("self_intersection_supported"): False,
            field("self_intersection_count"): math.nan,
        }
    )
    if include_surface_fill_proxy and (
        force_surface_fill_proxy or volume_fill_ratio_reliability_status != "reliable"
    ):
        diagnostics.update(
            {
                field(name): value
                for name, value in _surface_fill_proxy(mesh).items()
            }
        )
    return diagnostics


def stl_diagnostics(
    stl_path,
    *,
    include_surface_fill_proxy=False,
    force_surface_fill_proxy=False,
):
    return mesh_diagnostics(
        stl_path,
        prefix="stl",
        include_surface_fill_proxy=include_surface_fill_proxy,
        force_surface_fill_proxy=force_surface_fill_proxy,
    )


def json_safe_stl_diagnostics(value):
    if isinstance(value, dict):
        return {key: json_safe_stl_diagnostics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_stl_diagnostics(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
