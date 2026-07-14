"""Deterministic, topology-preserving variants of the pinned canonical face."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import trimesh

from backend.face_depth_refinement import FACE_PART_INDEX_GROUPS


DEFORMATION_GATES = {
    "minimum_normal_cosine": 0.90,
    "minimum_signed_triangle_jacobian": 0.50,
    "minimum_area_ratio": 0.50,
    "maximum_area_ratio": 1.60,
    "minimum_edge_ratio": 0.65,
    "maximum_edge_ratio": 1.45,
}


@dataclass(frozen=True)
class CanonicalFaceVariant:
    name: str
    face_width_scale: float = 1.0
    lower_face_length_cm: float = 0.0
    nose_projection_cm: float = 0.0
    eye_projection_cm: float = 0.0
    brow_projection_cm: float = 0.0
    cheek_projection_cm: float = 0.0
    mouth_projection_cm: float = 0.0
    mouth_corner_lift_cm: float = 0.0
    mouth_tilt_cm: float = 0.0
    depth_asymmetry_cm: float = 0.0


CANONICAL_FACE_VARIANTS = (
    CanonicalFaceVariant(name="neutral"),
    CanonicalFaceVariant(
        name="broad_shallow_nose",
        face_width_scale=1.12,
        nose_projection_cm=-0.65,
        eye_projection_cm=-0.12,
        cheek_projection_cm=0.28,
    ),
    CanonicalFaceVariant(
        name="narrow_projected_nose",
        face_width_scale=0.90,
        lower_face_length_cm=0.35,
        nose_projection_cm=1.00,
        brow_projection_cm=0.18,
    ),
    CanonicalFaceVariant(
        name="smile_high_cheek",
        cheek_projection_cm=0.42,
        mouth_projection_cm=0.22,
        mouth_corner_lift_cm=0.55,
    ),
    CanonicalFaceVariant(
        name="asymmetric_expression",
        mouth_corner_lift_cm=0.22,
        mouth_tilt_cm=0.48,
        depth_asymmetry_cm=0.24,
    ),
    CanonicalFaceVariant(
        name="deep_set_angular",
        face_width_scale=1.04,
        lower_face_length_cm=0.55,
        nose_projection_cm=0.34,
        eye_projection_cm=-0.42,
        brow_projection_cm=0.25,
        cheek_projection_cm=0.20,
    ),
)


def _part_center(vertices: np.ndarray, name: str) -> np.ndarray:
    indices = np.asarray(FACE_PART_INDEX_GROUPS[name], dtype=np.int64)
    return np.mean(vertices[indices], axis=0)


def _elliptical_weight(
    vertices: np.ndarray,
    center: np.ndarray,
    sigma_x_cm: float,
    sigma_y_cm: float,
) -> np.ndarray:
    dx = (vertices[:, 0] - float(center[0])) / float(sigma_x_cm)
    dy = (vertices[:, 1] - float(center[1])) / float(sigma_y_cm)
    return np.exp(-0.5 * (dx * dx + dy * dy))


def deform_canonical_face(
    mesh: trimesh.Trimesh,
    variant: CanonicalFaceVariant,
) -> tuple[trimesh.Trimesh, dict]:
    """Apply a deterministic smooth deformation without changing topology."""
    source = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = source.copy()
    if source.shape != (468, 3):
        raise ValueError(f"Canonical variant expects 468x3 vertices, got {source.shape}")

    vertices[:, 0] *= float(variant.face_width_scale)
    lower_weight = np.clip((-vertices[:, 1] - 1.5) / 7.0, 0.0, 1.0)
    vertices[:, 1] -= float(variant.lower_face_length_cm) * lower_weight

    nose_center = _part_center(source, "nose")
    mouth_center = _part_center(source, "mouth")
    eye_centers = (
        _part_center(source, "left_eye"),
        _part_center(source, "right_eye"),
    )
    brow_centers = (
        _part_center(source, "left_eyebrow"),
        _part_center(source, "right_eyebrow"),
    )
    nose_weight = _elliptical_weight(source, nose_center, 1.75, 2.65)
    mouth_weight = _elliptical_weight(source, mouth_center, 2.7, 1.35)
    vertices[:, 2] += float(variant.nose_projection_cm) * nose_weight
    vertices[:, 2] += float(variant.mouth_projection_cm) * mouth_weight

    for center in eye_centers:
        vertices[:, 2] += float(variant.eye_projection_cm) * _elliptical_weight(
            source,
            center,
            1.55,
            1.10,
        )
    for center in brow_centers:
        vertices[:, 2] += float(variant.brow_projection_cm) * _elliptical_weight(
            source,
            center,
            1.85,
            1.25,
        )

    cheek_centers = (
        np.array((-3.75, -0.15, 0.0), dtype=np.float64),
        np.array((3.75, -0.15, 0.0), dtype=np.float64),
    )
    for center in cheek_centers:
        vertices[:, 2] += float(variant.cheek_projection_cm) * _elliptical_weight(
            source,
            center,
            2.0,
            2.1,
        )

    mouth_indices = np.asarray(FACE_PART_INDEX_GROUPS["mouth"], dtype=np.int64)
    mouth_x = source[mouth_indices, 0]
    mouth_half_width = max(float(np.max(np.abs(mouth_x))), 1e-6)
    normalized_mouth_x = np.clip(source[:, 0] / mouth_half_width, -1.0, 1.0)
    corner_weight = mouth_weight * np.power(np.abs(normalized_mouth_x), 1.7)
    vertices[:, 1] += float(variant.mouth_corner_lift_cm) * corner_weight
    vertices[:, 1] += (
        float(variant.mouth_tilt_cm) * normalized_mouth_x * mouth_weight
    )

    face_scale = max(float(np.max(np.abs(source[:, 0]))), 1e-6)
    vertical_face_weight = np.exp(-0.5 * np.square((source[:, 1] + 0.5) / 5.8))
    vertices[:, 2] += (
        float(variant.depth_asymmetry_cm)
        * np.clip(source[:, 0] / face_scale, -1.0, 1.0)
        * vertical_face_weight
    )

    deformed = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
        process=False,
        maintain_order=True,
    )
    return deformed, canonical_face_deformation_audit(mesh, deformed, variant)


def canonical_face_deformation_audit(
    source_mesh: trimesh.Trimesh,
    candidate_mesh: trimesh.Trimesh,
    variant: CanonicalFaceVariant,
) -> dict:
    source = np.asarray(source_mesh.vertices, dtype=np.float64)
    candidate = np.asarray(candidate_mesh.vertices, dtype=np.float64)
    faces = np.asarray(source_mesh.faces, dtype=np.int64)
    if source.shape != candidate.shape or not np.array_equal(
        faces,
        np.asarray(candidate_mesh.faces, dtype=np.int64),
    ):
        return {
            "available": False,
            "passed": False,
            "reason": "topology_mismatch",
            "variant": asdict(variant),
        }

    source_triangles = source[faces]
    candidate_triangles = candidate[faces]
    source_cross = np.cross(
        source_triangles[:, 1] - source_triangles[:, 0],
        source_triangles[:, 2] - source_triangles[:, 0],
    )
    candidate_cross = np.cross(
        candidate_triangles[:, 1] - candidate_triangles[:, 0],
        candidate_triangles[:, 2] - candidate_triangles[:, 0],
    )
    source_area2 = np.linalg.norm(source_cross, axis=1)
    candidate_area2 = np.linalg.norm(candidate_cross, axis=1)
    area_ratio = candidate_area2 / np.maximum(source_area2, 1e-12)
    normal_cosine = np.sum(source_cross * candidate_cross, axis=1) / np.maximum(
        source_area2 * candidate_area2,
        1e-12,
    )
    signed_triangle_jacobian = np.sum(
        source_cross * candidate_cross,
        axis=1,
    ) / np.maximum(np.square(source_area2), 1e-12)
    edges = np.asarray(source_mesh.edges_unique, dtype=np.int64)
    source_edge_length = np.linalg.norm(
        source[edges[:, 1]] - source[edges[:, 0]],
        axis=1,
    )
    candidate_edge_length = np.linalg.norm(
        candidate[edges[:, 1]] - candidate[edges[:, 0]],
        axis=1,
    )
    edge_ratio = candidate_edge_length / np.maximum(source_edge_length, 1e-12)
    displacement = np.linalg.norm(candidate - source, axis=1)
    geometry_sha256 = hashlib.sha256(
        np.ascontiguousarray(candidate, dtype="<f8").tobytes()
    ).hexdigest()
    self_intersection_supported = False
    self_intersection_count = None
    self_intersection_error = None
    try:
        import pymeshlab

        mesh_set = pymeshlab.MeshSet()
        mesh_set.add_mesh(
            pymeshlab.Mesh(
                vertex_matrix=candidate,
                face_matrix=np.asarray(candidate_mesh.faces, dtype=np.int32),
            )
        )
        mesh_set.apply_filter("compute_selection_by_self_intersections_per_face")
        self_intersection_count = int(
            mesh_set.current_mesh().selected_face_number()
        )
        self_intersection_supported = True
    except Exception as exc:
        self_intersection_error = f"{type(exc).__name__}: {exc}"
    checks = {
        "finite_vertices": bool(np.all(np.isfinite(candidate))),
        "positive_triangle_area": bool(np.min(candidate_area2) > 1e-8),
        "normal_bending": bool(
            np.min(normal_cosine) >= DEFORMATION_GATES["minimum_normal_cosine"]
        ),
        "signed_triangle_jacobian": bool(
            np.min(signed_triangle_jacobian)
            >= DEFORMATION_GATES["minimum_signed_triangle_jacobian"]
        ),
        "area_ratio": bool(
            np.min(area_ratio) >= DEFORMATION_GATES["minimum_area_ratio"]
            and np.max(area_ratio) <= DEFORMATION_GATES["maximum_area_ratio"]
        ),
        "edge_stretch": bool(
            np.min(edge_ratio) >= DEFORMATION_GATES["minimum_edge_ratio"]
            and np.max(edge_ratio) <= DEFORMATION_GATES["maximum_edge_ratio"]
        ),
        "single_component": len(candidate_mesh.split(only_watertight=False)) == 1,
        "self_intersection_audited": bool(self_intersection_supported),
        "self_intersection_free": bool(
            self_intersection_supported and self_intersection_count == 0
        ),
    }
    part_centers = {
        name: np.mean(candidate[np.asarray(indices, dtype=np.int64)], axis=0).tolist()
        for name, indices in FACE_PART_INDEX_GROUPS.items()
    }
    mouth_indices = np.asarray(FACE_PART_INDEX_GROUPS["mouth"], dtype=np.int64)
    mouth_vertices = candidate[mouth_indices]
    left_corner = mouth_vertices[int(np.argmin(mouth_vertices[:, 0]))]
    right_corner = mouth_vertices[int(np.argmax(mouth_vertices[:, 0]))]
    eye_centers = np.stack(
        (
            np.asarray(part_centers["left_eye"], dtype=np.float64),
            np.asarray(part_centers["right_eye"], dtype=np.float64),
        )
    )
    nose_center = np.asarray(part_centers["nose"], dtype=np.float64)
    shape_metrics = {
        "face_width_cm": float(np.ptp(candidate[:, 0])),
        "face_height_cm": float(np.ptp(candidate[:, 1])),
        "interocular_distance_cm": float(
            np.linalg.norm(eye_centers[0, :2] - eye_centers[1, :2])
        ),
        "nose_projection_from_eye_plane_cm": float(
            nose_center[2] - np.mean(eye_centers[:, 2])
        ),
        "mouth_center_y_cm": float(np.mean(mouth_vertices[:, 1])),
        "mouth_corner_tilt_cm": float(right_corner[1] - left_corner[1]),
        "mouth_corner_mean_y_cm": float(0.5 * (right_corner[1] + left_corner[1])),
    }
    return {
        "available": True,
        "passed": bool(all(checks.values())),
        "variant": asdict(variant),
        "checks": {**checks, "passed": bool(all(checks.values()))},
        "geometry_sha256": geometry_sha256,
        "vertex_count": int(len(candidate)),
        "face_count": int(len(faces)),
        "component_count": len(candidate_mesh.split(only_watertight=False)),
        "displacement_p95_cm": float(np.percentile(displacement, 95.0)),
        "displacement_max_cm": float(np.max(displacement)),
        "minimum_normal_cosine": float(np.min(normal_cosine)),
        "minimum_signed_triangle_jacobian": float(
            np.min(signed_triangle_jacobian)
        ),
        "minimum_area_ratio": float(np.min(area_ratio)),
        "maximum_area_ratio": float(np.max(area_ratio)),
        "minimum_edge_ratio": float(np.min(edge_ratio)),
        "maximum_edge_ratio": float(np.max(edge_ratio)),
        "self_intersection_supported": bool(self_intersection_supported),
        "self_intersection_count": self_intersection_count,
        "self_intersection_error": self_intersection_error,
        "deformation_gates": DEFORMATION_GATES,
        "bounds_cm": np.stack((np.min(candidate, axis=0), np.max(candidate, axis=0))).tolist(),
        "extents_cm": np.ptp(candidate, axis=0).tolist(),
        "part_centers_cm": part_centers,
        "shape_metrics": shape_metrics,
    }


def canonical_face_variant_by_name(name: str) -> CanonicalFaceVariant:
    for variant in CANONICAL_FACE_VARIANTS:
        if variant.name == str(name):
            return variant
    raise KeyError(f"Unknown canonical face variant: {name}")
