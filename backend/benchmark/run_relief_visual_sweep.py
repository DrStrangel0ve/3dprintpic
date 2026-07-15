"""Run privacy-safe relief-height and mask-topology appearance regressions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
import trimesh
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
)

from backend.benchmark.run_relief_scene_regression import (
    SceneSpec,
    _background_surface,
    _boundary_shape_metrics,
    _finite_metric,
    _git_provenance,
    _mesh_topology,
    _scene_checks,
    _synthetic_scene,
)
from backend.benchmark.face_part_metrics import (
    FACE_PART_GATES,
    face_part_cross_height_metrics,
)
from backend.pic_to_3d import (
    _surface_lighting_agreement_metrics,
    compose_selection_depth_with_context,
    depth_data_to_3d_model,
)


RELIEF_HEIGHTS_MM = (20.0, 30.0, 40.0)
PROVENANCE_PATHS = (
    "backend/pic_to_3d.py",
    "backend/face_depth_refinement.py",
    "backend/face_relief_geometry.py",
    "backend/benchmark/face_part_metrics.py",
    "backend/benchmark/run_relief_scene_regression.py",
    "backend/benchmark/run_relief_visual_sweep.py",
    "backend/benchmark/mesh_rendering.py",
)
FACE_APPEARANCE_GATES = {
    "minimum_normal_mean_cosine": 0.95,
    "minimum_normal_p05_cosine": 0.85,
    "maximum_normal_angle_p95_deg": 30.0,
    "minimum_lighting_correlation": 0.80,
    "maximum_lighting_mae": 0.08,
    "minimum_lighting_rms_retention": 0.75,
    "maximum_lighting_rms_retention": 1.40,
}
BACKGROUND_APPEARANCE_GATES = {
    "minimum_normal_mean_cosine": 0.98,
    "minimum_normal_p05_cosine": 0.90,
    "maximum_normal_angle_p95_deg": 25.0,
    "minimum_lighting_correlation": 0.95,
    "maximum_lighting_mae": 0.05,
    "minimum_lighting_rms_retention": 0.75,
    "maximum_lighting_rms_retention": 1.25,
}
SELECTION_APPEARANCE_GATES = dict(FACE_APPEARANCE_GATES)
SELECTION_SOLVER_GATES = {
    "expected_screening_weight": 2.0,
    "expected_detail_gradient_retention": 0.9,
    "maximum_output_edge_p99_ratio": 12.0,
    "maximum_output_edge_ratio": 24.0,
    "maximum_diagonal_edge_ratio": 24.0,
}
CROSS_HEIGHT_FACE_GATES = {
    "minimum_shape_correlation": 0.98,
    "maximum_normalized_shape_rmse": 0.08,
    "minimum_gradient_correlation": 0.97,
}


@dataclass(frozen=True)
class SweepSpec:
    topology_id: str
    scene: SceneSpec
    dilation_iterations: int = 0
    frame_border_px: int = 0
    topology_mode: str = "base"
    expected_components: int = 1
    expected_holes: int = 0
    expected_background_components: int = 1


def _sweep_specs() -> tuple[SweepSpec, ...]:
    return (
        SweepSpec(
            "centered_subject",
            SceneSpec("visual_centered", "planar_room"),
        ),
        SweepSpec(
            "off_axis_clipped_subject",
            SceneSpec(
                "visual_off_axis",
                "framed_wall",
                center_col=39.0,
                yaw=-0.35,
                expression=0.2,
            ),
        ),
        SweepSpec(
            "near_full_frame_subject",
            SceneSpec(
                "visual_near_full_frame",
                "low_contrast",
                center_col=47.0,
                center_row=41.0,
                yaw=-0.18,
                expression=-0.55,
            ),
            dilation_iterations=24,
            frame_border_px=8,
        ),
        SweepSpec(
            "disconnected_subject_with_hole",
            SceneSpec(
                "visual_disconnected_hole",
                "shelving",
                expression=0.1,
            ),
            topology_mode="disconnected_hole",
            expected_components=2,
            expected_holes=1,
            expected_background_components=2,
        ),
    )


def _topology_scene(spec: SweepSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, face, subject = _synthetic_scene(spec.scene)
    if spec.topology_mode == "disconnected_hole":
        rows, cols = np.indices(source.shape, dtype=np.float32)
        island = (rows - 18.0) ** 2 + (cols - 103.0) ** 2 <= 9.0**2
        hole = (rows - 95.0) ** 2 + (cols - 60.0) ** 2 <= 12.0**2
        subject = (subject | island) & ~hole
        source = source.copy()
        source[island] = (
            0.44
            + 0.08
            * np.exp(-((rows[island] - 18.0) ** 2 + (cols[island] - 103.0) ** 2) / 20.0)
        )
        background = _background_surface(spec.scene.background, rows, cols)
        source[hole] = background[hole]
        return source.astype(np.float32, copy=False), face, subject
    if spec.topology_mode != "base":
        raise ValueError(f"Unsupported sweep topology mode: {spec.topology_mode}")
    if spec.dilation_iterations <= 0:
        return source, face, subject

    expanded = binary_dilation(subject, iterations=spec.dilation_iterations)
    border = max(0, int(spec.frame_border_px))
    if border:
        frame = np.ones(expanded.shape, dtype=bool)
        frame[:border, :] = False
        frame[-border:, :] = False
        frame[:, :border] = False
        frame[:, -border:] = False
        expanded &= frame

    _, nearest = distance_transform_edt(~subject, return_indices=True)
    nearest_subject = source[tuple(nearest)]
    added = expanded & ~subject
    expanded_source = source.copy()
    expanded_source[added] = (
        0.85 * nearest_subject[added] + 0.15 * source[added]
    )
    return expanded_source.astype(np.float32, copy=False), face, expanded


def _mask_topology(mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    _, component_count = label(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    holes = binary_fill_holes(mask) & ~mask
    _, hole_count = label(
        holes,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    border_contacts = {
        "top": bool(np.any(mask[0, :])),
        "bottom": bool(np.any(mask[-1, :])),
        "left": bool(np.any(mask[:, 0])),
        "right": bool(np.any(mask[:, -1])),
    }
    return {
        "component_count": int(component_count),
        "hole_count": int(hole_count),
        "euler_characteristic": int(component_count - hole_count),
        "border_contacts": border_contacts,
        "coverage_ratio": float(np.mean(mask)),
    }


def _synthetic_face_part_masks(face_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Create deterministic semantic regions inside an analytic face mask."""
    face = np.asarray(face_mask, dtype=bool)
    rows, cols = np.indices(face.shape, dtype=np.float32)
    face_rows, face_cols = np.where(face)
    if not len(face_rows):
        return {}
    top, bottom = float(face_rows.min()), float(face_rows.max())
    left, right = float(face_cols.min()), float(face_cols.max())
    height = max(bottom - top + 1.0, 1.0)
    width = max(right - left + 1.0, 1.0)

    def ellipse(center_x, center_y, radius_x, radius_y):
        mask = (
            np.square((cols - (left + width * center_x)) / max(width * radius_x, 1.0))
            + np.square((rows - (top + height * center_y)) / max(height * radius_y, 1.0))
            <= 1.0
        )
        return mask & face

    return {
        "left_eye": ellipse(0.68, 0.39, 0.14, 0.07),
        "right_eye": ellipse(0.32, 0.39, 0.14, 0.07),
        "left_eyebrow": ellipse(0.68, 0.29, 0.16, 0.05),
        "right_eyebrow": ellipse(0.32, 0.29, 0.16, 0.05),
        "mouth": ellipse(0.50, 0.74, 0.22, 0.08),
        "nose": ellipse(0.50, 0.54, 0.12, 0.18),
    }


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _selection_solver_record(stats: dict) -> dict:
    quality_gates = stats.get("quality_gates", {})
    quality_gate_failures = quality_gates.get("failures", [])
    if not isinstance(quality_gate_failures, (list, tuple)):
        quality_gate_failures = [str(quality_gate_failures)]
    screening_weight = _finite_metric(stats, "screening_weight")
    detail_gradient_retention = _finite_metric(
        stats,
        "detail_gradient_retention",
    )
    output_edge_ratio_p99 = _finite_metric(stats, "output_edge_ratio_p99")
    output_edge_ratio_max = _finite_metric(stats, "output_edge_ratio_max")
    diagonal_edge_ratio_max = _finite_metric(stats, "diagonal_edge_ratio_max")
    telemetry = all(
        _finite(value)
        for value in (
            screening_weight,
            detail_gradient_retention,
            output_edge_ratio_p99,
            output_edge_ratio_max,
            diagonal_edge_ratio_max,
        )
    )
    checks = {
        "telemetry": telemetry,
        "enabled": bool(stats.get("enabled", False)),
        "reason_clear": stats.get("reason") is None,
        "quality_gate": bool(
            quality_gates.get("passed", False) and not quality_gate_failures
        ),
        "face_protection": bool(stats.get("face_protection_passed", False)),
        "calibration": bool(
            telemetry
            and np.isclose(
                screening_weight,
                SELECTION_SOLVER_GATES["expected_screening_weight"],
            )
            and np.isclose(
                detail_gradient_retention,
                SELECTION_SOLVER_GATES["expected_detail_gradient_retention"],
            )
        ),
        "cardinal_edge_p99": bool(
            telemetry
            and output_edge_ratio_p99
            <= SELECTION_SOLVER_GATES["maximum_output_edge_p99_ratio"]
        ),
        "cardinal_edge_max": bool(
            telemetry
            and output_edge_ratio_max
            <= SELECTION_SOLVER_GATES["maximum_output_edge_ratio"]
        ),
        "diagonal_edge_max": bool(
            telemetry
            and diagonal_edge_ratio_max
            <= SELECTION_SOLVER_GATES["maximum_diagonal_edge_ratio"]
        ),
    }
    return {
        "enabled": bool(stats.get("enabled", False)),
        "reason": stats.get("reason"),
        "screening_weight": screening_weight,
        "detail_gradient_retention": detail_gradient_retention,
        "output_edge_ratio_p99": output_edge_ratio_p99,
        "output_edge_ratio_max": output_edge_ratio_max,
        "diagonal_edge_ratio_max": diagonal_edge_ratio_max,
        "dropped_detail_components": (
            int(stats["dropped_detail_components"])
            if _finite(stats.get("dropped_detail_components"))
            else None
        ),
        "dropped_detail_pixels": (
            int(stats["dropped_detail_pixels"])
            if _finite(stats.get("dropped_detail_pixels"))
            else None
        ),
        "quality_gate_failures": list(quality_gate_failures),
        "checks": {**checks, "passed": all(checks.values())},
    }


def _appearance_checks(metrics: dict, gates: dict) -> dict[str, bool]:
    required = (
        "normal_mean_cosine",
        "normal_p05_cosine",
        "normal_angle_p95_deg",
        "minimum_lighting_correlation",
        "maximum_lighting_mae",
        "minimum_lighting_rms_retention",
        "maximum_lighting_rms_retention",
    )
    telemetry = bool(metrics.get("available", False)) and all(
        _finite(metrics.get(key)) for key in required
    )
    if not telemetry:
        return {
            "telemetry": False,
            "coverage": False,
            "normal_field": False,
            "lighting": False,
            "lighting_energy": False,
            "components": False,
            "passed": False,
        }
    coverage = float(metrics.get("candidate_coverage_ratio", 0.0)) >= 1.0
    normal_field = bool(
        float(metrics["normal_mean_cosine"])
        >= gates["minimum_normal_mean_cosine"]
        and float(metrics["normal_p05_cosine"])
        >= gates["minimum_normal_p05_cosine"]
        and float(metrics["normal_angle_p95_deg"])
        <= gates["maximum_normal_angle_p95_deg"]
    )
    lighting = bool(
        float(metrics["minimum_lighting_correlation"])
        >= gates["minimum_lighting_correlation"]
        and float(metrics["maximum_lighting_mae"])
        <= gates["maximum_lighting_mae"]
    )
    lighting_energy = bool(
        gates["minimum_lighting_rms_retention"]
        <= float(metrics["minimum_lighting_rms_retention"])
        and float(metrics["maximum_lighting_rms_retention"])
        <= gates["maximum_lighting_rms_retention"]
    )
    component_records = metrics.get("components")
    components = True
    if component_records is not None:
        components = bool(
            int(metrics.get("component_count", 0)) > 0
            and len(component_records) == int(metrics.get("component_count", 0))
            and all(
                _appearance_checks(record, gates)["passed"]
                for record in component_records
            )
        )
    return {
        "telemetry": telemetry,
        "coverage": coverage,
        "normal_field": normal_field,
        "lighting": lighting,
        "lighting_energy": lighting_energy,
        "components": components,
        "passed": bool(
            telemetry
            and coverage
            and normal_field
            and lighting
            and lighting_energy
            and components
        ),
    }


def _appearance_record(metrics: dict) -> dict:
    return {
        "component": metrics.get("component"),
        "available": bool(metrics.get("available", False)),
        "samples": int(metrics.get("samples", 0)),
        "reference_samples": int(metrics.get("reference_samples", 0)),
        "missing_candidate_samples": int(metrics.get("missing_candidate_samples", 0)),
        "candidate_coverage_ratio": _finite_metric(
            metrics,
            "candidate_coverage_ratio",
        ),
        "normal_mean_cosine": _finite_metric(metrics, "normal_mean_cosine"),
        "normal_p05_cosine": _finite_metric(metrics, "normal_p05_cosine"),
        "normal_angle_median_deg": _finite_metric(
            metrics,
            "normal_angle_median_deg",
        ),
        "normal_angle_p95_deg": _finite_metric(metrics, "normal_angle_p95_deg"),
        "minimum_lighting_correlation": _finite_metric(
            metrics,
            "minimum_lighting_correlation",
        ),
        "maximum_lighting_mae": _finite_metric(metrics, "maximum_lighting_mae"),
        "minimum_lighting_rms_retention": _finite_metric(
            metrics,
            "minimum_lighting_rms_retention",
        ),
        "maximum_lighting_rms_retention": _finite_metric(
            metrics,
            "maximum_lighting_rms_retention",
        ),
        "lights": metrics.get("lights", []),
        "reason": metrics.get("reason"),
        "component_count": int(metrics.get("component_count", 0)),
        "components": [
            _appearance_record(record)
            for record in metrics.get("components", [])
        ],
    }


def _stl_top_surface_agreement(
    stl_path: str | Path,
    surface_path: str | Path,
    *,
    maximum_error_mm=1e-5,
    maximum_rms_error_mm=1e-6,
) -> dict:
    """Verify reopened STL samples, top triangles, and oriented facet normals."""
    expected = np.load(surface_path).astype(np.float64)
    mesh = trimesh.load_mesh(stl_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    stats = {
        "available": False,
        "passed": False,
        "expected_shape": [int(value) for value in expected.shape],
        "maximum_error_mm": float(maximum_error_mm),
        "maximum_rms_error_mm": float(maximum_rms_error_mm),
    }
    if (
        expected.ndim != 2
        or not len(vertices)
        or not len(triangles)
        or not np.all(np.isfinite(vertices))
        or not np.all(np.isfinite(triangles))
    ):
        stats["reason"] = "invalid_surface_or_mesh"
        return stats
    unique_x = np.unique(vertices[:, 0])
    unique_y = np.unique(vertices[:, 1])
    reconstructed_shape = (len(unique_y), len(unique_x))
    stats["reconstructed_shape"] = [int(value) for value in reconstructed_shape]
    if reconstructed_shape != expected.shape:
        stats["reason"] = "grid_shape_mismatch"
        return stats

    x_indices = np.searchsorted(unique_x, vertices[:, 0])
    y_indices = np.searchsorted(unique_y, vertices[:, 1])
    reconstructed = np.full(reconstructed_shape, -np.inf, dtype=np.float64)
    np.maximum.at(reconstructed, (y_indices, x_indices), vertices[:, 2])
    expected_valid = np.isfinite(expected)
    reconstructed_valid = np.isfinite(reconstructed)
    expected_samples = int(np.count_nonzero(expected_valid))
    measured = expected_valid & reconstructed_valid
    samples = int(np.count_nonzero(measured))
    coverage_ratio = samples / max(expected_samples, 1)
    stats.update(
        {
            "available": bool(expected_samples and samples),
            "reference_samples": expected_samples,
            "samples": samples,
            "missing_samples": int(expected_samples - samples),
            "coverage_ratio": float(coverage_ratio),
        }
    )
    if not samples:
        stats["reason"] = "no_shared_surface_samples"
        return stats
    errors = np.abs(reconstructed[measured] - expected[measured])
    max_error = float(np.max(errors))
    rms_error = float(np.sqrt(np.mean(np.square(errors))))

    cell_valid = (
        expected_valid[:-1, :-1]
        & expected_valid[1:, :-1]
        & expected_valid[:-1, 1:]
        & expected_valid[1:, 1:]
    )
    cell_rows, cell_cols = np.where(cell_valid)
    if not len(cell_rows):
        stats["reason"] = "no_emittable_cells"
        return stats
    v0 = np.column_stack(
        (unique_x[cell_cols], unique_y[cell_rows], expected[cell_rows, cell_cols])
    )
    v1 = np.column_stack(
        (
            unique_x[cell_cols],
            unique_y[cell_rows + 1],
            expected[cell_rows + 1, cell_cols],
        )
    )
    v2 = np.column_stack(
        (
            unique_x[cell_cols + 1],
            unique_y[cell_rows],
            expected[cell_rows, cell_cols + 1],
        )
    )
    v3 = np.column_stack(
        (
            unique_x[cell_cols + 1],
            unique_y[cell_rows + 1],
            expected[cell_rows + 1, cell_cols + 1],
        )
    )
    expected_triangles = np.concatenate(
        (
            np.stack((v0, v2, v1), axis=1),
            np.stack((v1, v2, v3), axis=1),
        ),
        axis=0,
    )

    base_z = float(np.min(vertices[:, 2]))
    surface_candidates = triangles[
        np.all(triangles[:, :, 2] > base_z + float(maximum_error_mm), axis=1)
    ]
    candidate_x = np.searchsorted(unique_x, surface_candidates[:, :, 0])
    candidate_y = np.searchsorted(unique_y, surface_candidates[:, :, 1])
    candidate_indices_valid = (
        (candidate_x >= 0)
        & (candidate_x < len(unique_x))
        & (candidate_y >= 0)
        & (candidate_y < len(unique_y))
    )
    candidate_expected_z = np.full(candidate_x.shape, np.nan, dtype=np.float64)
    valid_candidate_vertices = candidate_indices_valid
    candidate_expected_z[valid_candidate_vertices] = expected[
        candidate_y[valid_candidate_vertices],
        candidate_x[valid_candidate_vertices],
    ]
    candidate_matches_surface = np.all(
        candidate_indices_valid
        & np.isfinite(candidate_expected_z)
        & (
            np.abs(surface_candidates[:, :, 2] - candidate_expected_z)
            <= float(maximum_error_mm)
        ),
        axis=1,
    )
    actual_top_triangles = surface_candidates[candidate_matches_surface]
    unexpected_top_triangle_count = int(
        len(surface_candidates) - len(actual_top_triangles)
    )

    def canonicalize(values):
        values = np.asarray(values, dtype=np.float64)
        quantized = np.rint(values / float(maximum_error_mm)).astype(np.int64)
        vertex_order = np.broadcast_to(
            np.arange(3, dtype=np.int64),
            quantized.shape[:2],
        ).copy()
        for axis in (2, 1, 0):
            current = np.take_along_axis(
                quantized[:, :, axis],
                vertex_order,
                axis=1,
            )
            local_order = np.argsort(current, axis=1, kind="stable")
            vertex_order = np.take_along_axis(vertex_order, local_order, axis=1)
        canonical_quantized = np.take_along_axis(
            quantized,
            vertex_order[:, :, None],
            axis=1,
        )
        canonical_values = np.take_along_axis(
            values,
            vertex_order[:, :, None],
            axis=1,
        )
        flattened = canonical_quantized.reshape((len(values), 9))
        row_order = np.arange(len(values), dtype=np.int64)
        for column in range(flattened.shape[1] - 1, -1, -1):
            local_order = np.argsort(
                flattened[row_order, column],
                kind="stable",
            )
            row_order = row_order[local_order]
        return (
            canonical_quantized[row_order],
            canonical_values[row_order],
            row_order,
        )

    expected_key, expected_canonical, expected_order = canonicalize(
        expected_triangles
    )
    actual_key, actual_canonical, actual_order = canonicalize(actual_top_triangles)
    triangle_count_match = len(expected_triangles) == len(actual_top_triangles)
    triangle_set_match = bool(
        triangle_count_match
        and unexpected_top_triangle_count == 0
        and np.array_equal(expected_key, actual_key)
    )
    triangle_coordinate_error = None
    minimum_normal_cosine = None
    mean_normal_cosine = None
    if triangle_set_match:
        triangle_coordinate_error = float(
            np.max(np.abs(expected_canonical - actual_canonical))
        )

        def oriented_normals(values):
            normals = np.cross(
                values[:, 1] - values[:, 0],
                values[:, 2] - values[:, 0],
            )
            lengths = np.linalg.norm(normals, axis=1)
            return normals / np.maximum(lengths[:, None], 1e-12)

        expected_normals = oriented_normals(expected_triangles)[expected_order]
        actual_normals = oriented_normals(actual_top_triangles)[actual_order]
        normal_cosines = np.einsum("ij,ij->i", expected_normals, actual_normals)
        minimum_normal_cosine = float(np.min(normal_cosines))
        mean_normal_cosine = float(np.mean(normal_cosines))

    sample_passed = bool(
        coverage_ratio >= 1.0
        and max_error <= float(maximum_error_mm)
        and rms_error <= float(maximum_rms_error_mm)
    )
    facet_passed = bool(
        triangle_set_match
        and triangle_coordinate_error is not None
        and triangle_coordinate_error <= float(maximum_error_mm)
        and minimum_normal_cosine is not None
        and minimum_normal_cosine >= 1.0 - 1e-6
    )
    stats.update(
        {
            "max_abs_error_mm": max_error,
            "rms_error_mm": rms_error,
            "sample_grid_passed": sample_passed,
            "expected_top_triangle_count": int(len(expected_triangles)),
            "actual_top_triangle_count": int(len(actual_top_triangles)),
            "unexpected_top_triangle_count": unexpected_top_triangle_count,
            "triangle_count_match": triangle_count_match,
            "triangle_set_match": triangle_set_match,
            "max_triangle_coordinate_error_mm": triangle_coordinate_error,
            "minimum_facet_normal_cosine": minimum_normal_cosine,
            "mean_facet_normal_cosine": mean_normal_cosine,
            "facet_geometry_passed": facet_passed,
            "passed": bool(sample_passed and facet_passed),
        }
    )
    return stats


def _stl_heightfield_agreement(
    stl_path: str | Path,
    surface_path: str | Path,
    *,
    maximum_error_mm=1e-5,
    maximum_rms_error_mm=1e-6,
    expected_max_xy_size_mm=48.0,
) -> dict:
    """Verify samples and the complete exporter shell, including facet winding."""
    top_stats = _stl_top_surface_agreement(
        stl_path,
        surface_path,
        maximum_error_mm=maximum_error_mm,
        maximum_rms_error_mm=maximum_rms_error_mm,
    )
    stats = dict(top_stats)
    stats["top_facet_geometry_passed"] = bool(
        top_stats.get("facet_geometry_passed", False)
    )
    stats["complete_shell_verified"] = False
    stats["facet_geometry_passed"] = False
    stats["passed"] = False
    try:
        tolerance = float(maximum_error_mm)
        max_xy_size = float(expected_max_xy_size_mm)
    except (TypeError, ValueError):
        stats["reason"] = "invalid_shell_configuration"
        return stats
    stats["expected_max_xy_size_mm"] = max_xy_size
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0
        or not np.isfinite(max_xy_size)
        or max_xy_size <= 0
    ):
        stats["reason"] = "invalid_shell_configuration"
        return stats
    if not top_stats.get("available", False):
        return stats

    expected = np.load(surface_path).astype(np.float64)
    mesh = trimesh.load_mesh(stl_path, process=False)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if (
        expected.ndim != 2
        or min(expected.shape, default=0) < 2
        or not len(triangles)
        or not np.all(np.isfinite(triangles))
    ):
        stats["reason"] = "invalid_shell_geometry"
        return stats
    row_count, column_count = expected.shape
    coordinate_max = max(row_count - 1, column_count - 1)
    sample_pitch = max_xy_size / float(coordinate_max)
    stats["sample_pitch_mm"] = float(sample_pitch)
    if sample_pitch <= 2.0 * tolerance:
        stats["reason"] = "shell_grid_spacing_within_tolerance"
        return stats

    expected_valid = np.isfinite(expected)
    cell_valid = (
        expected_valid[:-1, :-1]
        & expected_valid[1:, :-1]
        & expected_valid[:-1, 1:]
        & expected_valid[1:, 1:]
    )
    cell_rows, cell_cols = np.where(cell_valid)
    if not len(cell_rows):
        stats["reason"] = "no_emittable_cells"
        return stats
    emittable = np.zeros(expected.shape, dtype=bool)
    emittable[cell_rows, cell_cols] = True
    emittable[cell_rows + 1, cell_cols] = True
    emittable[cell_rows, cell_cols + 1] = True
    emittable[cell_rows + 1, cell_cols + 1] = True
    stats["finite_reference_samples"] = int(np.count_nonzero(expected_valid))
    stats["unemittable_reference_samples"] = int(
        np.count_nonzero(expected_valid & ~emittable)
    )
    if np.any(np.abs(expected[emittable]) <= 2.0 * tolerance):
        stats["reason"] = "ambiguous_top_bottom_surface"
        return stats

    x_axis = np.arange(column_count, dtype=np.float64) * sample_pitch
    y_axis = np.arange(row_count, dtype=np.float64) * sample_pitch
    expected_triangles = []
    expected_keys = []

    def vertex(row, col, top):
        layer = int(bool(top))
        return (
            np.asarray(
                (
                    x_axis[col],
                    y_axis[row],
                    expected[row, col] if layer else 0.0,
                ),
                dtype=np.float64,
            ),
            int(((row * column_count + col) * 2) + layer),
        )

    def add_triangle(first, second, third):
        coordinates, keys = zip(first, second, third)
        expected_triangles.append(np.stack(coordinates, axis=0))
        expected_keys.append(keys)

    for row, col in zip(cell_rows.tolist(), cell_cols.tolist()):
        v0 = vertex(row, col, True)
        v1 = vertex(row + 1, col, True)
        v2 = vertex(row, col + 1, True)
        v3 = vertex(row + 1, col + 1, True)
        b0 = vertex(row, col, False)
        b1 = vertex(row + 1, col, False)
        b2 = vertex(row, col + 1, False)
        b3 = vertex(row + 1, col + 1, False)
        add_triangle(v0, v2, v1)
        add_triangle(v1, v2, v3)
        add_triangle(b2, b0, b1)
        add_triangle(b2, b1, b3)
        if row == 0 or not cell_valid[row - 1, col]:
            add_triangle(v0, b0, v2)
            add_triangle(v2, b0, b2)
        if row == cell_valid.shape[0] - 1 or not cell_valid[row + 1, col]:
            add_triangle(v1, v3, b1)
            add_triangle(v3, b3, b1)
        if col == 0 or not cell_valid[row, col - 1]:
            add_triangle(v0, v1, b0)
            add_triangle(v1, b1, b0)
        if col == cell_valid.shape[1] - 1 or not cell_valid[row, col + 1]:
            add_triangle(v2, b2, v3)
            add_triangle(v3, b2, b3)

    expected_triangles = np.asarray(expected_triangles, dtype=np.float64)
    expected_keys = np.asarray(expected_keys, dtype=np.int64)
    signed_volume = float(
        np.einsum(
            "ij,ij->i",
            expected_triangles[:, 0],
            np.cross(expected_triangles[:, 1], expected_triangles[:, 2]),
        ).sum()
        / 6.0
    )
    if signed_volume < 0:
        expected_triangles = expected_triangles[:, [0, 2, 1], :]
        expected_keys = expected_keys[:, [0, 2, 1]]

    actual_vertices = triangles.reshape(-1, 3)
    actual_cols = np.rint(actual_vertices[:, 0] / sample_pitch).astype(np.int64)
    actual_rows = np.rint(actual_vertices[:, 1] / sample_pitch).astype(np.int64)
    grid_valid = (
        (actual_cols >= 0)
        & (actual_cols < column_count)
        & (actual_rows >= 0)
        & (actual_rows < row_count)
    )
    clipped_cols = np.clip(actual_cols, 0, column_count - 1)
    clipped_rows = np.clip(actual_rows, 0, row_count - 1)
    x_error = np.abs(actual_vertices[:, 0] - x_axis[clipped_cols])
    y_error = np.abs(actual_vertices[:, 1] - y_axis[clipped_rows])
    grid_valid &= x_error <= tolerance
    grid_valid &= y_error <= tolerance
    top_z = expected[clipped_rows, clipped_cols]
    top_match = (
        grid_valid
        & np.isfinite(top_z)
        & (np.abs(actual_vertices[:, 2] - top_z) <= tolerance)
    )
    bottom_match = grid_valid & (np.abs(actual_vertices[:, 2]) <= tolerance)
    ambiguous_layer = top_match & bottom_match
    vertex_valid = (top_match | bottom_match) & ~ambiguous_layer
    layers = top_match.astype(np.int64)
    actual_vertex_keys = np.full(len(actual_vertices), -1, dtype=np.int64)
    actual_vertex_keys[vertex_valid] = (
        (actual_rows[vertex_valid] * column_count + actual_cols[vertex_valid]) * 2
        + layers[vertex_valid]
    )
    actual_keys = actual_vertex_keys.reshape(-1, 3)
    actual_triangle_valid = np.all(actual_keys >= 0, axis=1)
    mapped_coordinates = np.column_stack(
        (
            x_axis[clipped_cols],
            y_axis[clipped_rows],
            np.where(layers > 0, top_z, 0.0),
        )
    )
    coordinate_errors = np.abs(
        actual_vertices[vertex_valid] - mapped_coordinates[vertex_valid]
    )
    maximum_coordinate_error = (
        float(np.max(coordinate_errors)) if coordinate_errors.size else None
    )

    def canonical_keys(keys):
        keys = np.sort(np.asarray(keys, dtype=np.int64), axis=1)
        if not len(keys):
            return keys, np.empty(0, dtype=np.int64)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        return keys[order], order

    canonical_expected, expected_order = canonical_keys(expected_keys)
    canonical_actual, actual_order = canonical_keys(actual_keys)
    shell_count_match = len(expected_triangles) == len(triangles)
    shell_connectivity_match = bool(
        shell_count_match
        and np.all(actual_triangle_valid)
        and np.array_equal(canonical_expected, canonical_actual)
    )
    minimum_shell_normal_cosine = None
    mean_shell_normal_cosine = None
    if shell_connectivity_match:

        def oriented_normals(values):
            normals = np.cross(
                values[:, 1] - values[:, 0],
                values[:, 2] - values[:, 0],
            )
            lengths = np.linalg.norm(normals, axis=1)
            valid_lengths = lengths > 1e-12
            normalized = np.zeros_like(normals)
            normalized[valid_lengths] = (
                normals[valid_lengths] / lengths[valid_lengths, None]
            )
            return normalized, valid_lengths

        expected_normals, expected_normals_valid = oriented_normals(
            expected_triangles
        )
        actual_normals, actual_normals_valid = oriented_normals(triangles)
        expected_normals = expected_normals[expected_order]
        actual_normals = actual_normals[actual_order]
        normals_valid = (
            expected_normals_valid[expected_order]
            & actual_normals_valid[actual_order]
        )
        normal_cosines = np.einsum(
            "ij,ij->i",
            expected_normals,
            actual_normals,
        )
        normal_cosines[~normals_valid] = -1.0
        minimum_shell_normal_cosine = float(np.min(normal_cosines))
        mean_shell_normal_cosine = float(np.mean(normal_cosines))

    complete_shell_verified = bool(
        shell_connectivity_match
        and maximum_coordinate_error is not None
        and maximum_coordinate_error <= tolerance
        and minimum_shell_normal_cosine is not None
        and minimum_shell_normal_cosine >= 1.0 - 1e-6
    )
    top_coordinate_error = top_stats.get("max_triangle_coordinate_error_mm")
    combined_coordinate_error = (
        max(float(top_coordinate_error), maximum_coordinate_error)
        if top_coordinate_error is not None and maximum_coordinate_error is not None
        else maximum_coordinate_error
    )
    top_normal_cosine = top_stats.get("minimum_facet_normal_cosine")
    combined_normal_cosine = (
        min(float(top_normal_cosine), minimum_shell_normal_cosine)
        if top_normal_cosine is not None and minimum_shell_normal_cosine is not None
        else minimum_shell_normal_cosine
    )
    stats.update(
        {
            "expected_shell_triangle_count": int(len(expected_triangles)),
            "actual_shell_triangle_count": int(len(triangles)),
            "invalid_shell_triangle_count": int(
                len(triangles) - np.count_nonzero(actual_triangle_valid)
            ),
            "shell_triangle_count_match": shell_count_match,
            "shell_connectivity_match": shell_connectivity_match,
            "max_shell_coordinate_error_mm": maximum_coordinate_error,
            "minimum_shell_normal_cosine": minimum_shell_normal_cosine,
            "mean_shell_normal_cosine": mean_shell_normal_cosine,
            "max_triangle_coordinate_error_mm": combined_coordinate_error,
            "minimum_facet_normal_cosine": combined_normal_cosine,
            "complete_shell_verified": complete_shell_verified,
            "facet_geometry_passed": bool(
                top_stats.get("facet_geometry_passed", False)
                and complete_shell_verified
            ),
        }
    )
    stats["passed"] = bool(
        stats.get("sample_grid_passed", False)
        and stats["facet_geometry_passed"]
    )
    if not stats["passed"] and stats.get("reason") is None:
        stats["reason"] = "shell_geometry_mismatch"
    return stats


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: str | Path) -> dict:
    path = Path(path)
    return {
        "name": path.name,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _centered_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_centered = reference - float(np.mean(reference))
    candidate_centered = candidate - float(np.mean(candidate))
    reference_norm = float(np.linalg.norm(reference_centered))
    candidate_norm = float(np.linalg.norm(candidate_centered))
    if reference_norm <= 1e-12:
        return 1.0 if candidate_norm <= 1e-12 else 0.0
    if candidate_norm <= 1e-12:
        return 0.0
    return float(
        np.dot(reference_centered, candidate_centered)
        / (reference_norm * candidate_norm)
    )


def _cross_height_face_shape_metrics(
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    face_mask: np.ndarray,
    *,
    boundary_exclusion_px: int = 2,
    minimum_samples: int = 64,
) -> dict:
    """Compare normalized face shape while allowing height scale and offset changes."""
    reference = np.asarray(reference_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    region = np.asarray(face_mask, dtype=bool)
    stats = {
        "available": False,
        "passed": False,
        "samples": 0,
        "reference_samples": 0,
        "candidate_coverage_ratio": 0.0,
        "shape_correlation": None,
        "normalized_shape_rmse": None,
        "minimum_gradient_correlation": None,
    }
    if reference.shape != candidate.shape or reference.shape != region.shape:
        stats["reason"] = "shape_mismatch"
        return stats

    exclusion = max(1, int(boundary_exclusion_px))
    interior = binary_erosion(
        region,
        structure=np.ones((3, 3), dtype=bool),
        iterations=exclusion,
        border_value=0,
    )
    reference_valid = binary_erosion(
        np.isfinite(reference),
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    candidate_valid = binary_erosion(
        np.isfinite(candidate),
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    expected = interior & reference_valid
    measured = expected & candidate_valid
    reference_samples = int(np.count_nonzero(expected))
    samples = int(np.count_nonzero(measured))
    coverage = samples / max(reference_samples, 1)
    stats.update(
        {
            "samples": samples,
            "reference_samples": reference_samples,
            "missing_candidate_samples": int(reference_samples - samples),
            "candidate_coverage_ratio": float(coverage),
            "boundary_exclusion_px": exclusion,
            "minimum_samples": int(minimum_samples),
        }
    )
    if reference_samples < int(minimum_samples):
        stats["reason"] = "insufficient_reference_samples"
        return stats
    if samples < int(minimum_samples):
        stats["reason"] = "insufficient_candidate_samples"
        return stats

    reference_signal = reference[measured]
    candidate_signal = candidate[measured]
    reference_span = float(
        np.percentile(reference_signal, 95.0)
        - np.percentile(reference_signal, 5.0)
    )
    candidate_span = float(
        np.percentile(candidate_signal, 95.0)
        - np.percentile(candidate_signal, 5.0)
    )
    stats.update(
        {
            "reference_p05_p95_span_mm": reference_span,
            "candidate_p05_p95_span_mm": candidate_span,
        }
    )
    if reference_span <= 1e-8 or candidate_span <= 1e-8:
        stats["reason"] = "flat_surface"
        return stats

    reference_normalized = (
        reference_signal - float(np.median(reference_signal))
    ) / reference_span
    candidate_normalized = (
        candidate_signal - float(np.median(candidate_signal))
    ) / candidate_span
    gradient_correlations = [
        _centered_correlation(reference_gradient[measured], candidate_gradient[measured])
        for reference_gradient, candidate_gradient in zip(
            np.gradient(reference),
            np.gradient(candidate),
        )
    ]
    stats.update(
        {
            "available": True,
            "shape_correlation": _centered_correlation(
                reference_signal,
                candidate_signal,
            ),
            "normalized_shape_rmse": float(
                np.sqrt(
                    np.mean(
                        np.square(reference_normalized - candidate_normalized)
                    )
                )
            ),
            "gradient_correlations": [
                float(value) for value in gradient_correlations
            ],
            "minimum_gradient_correlation": float(min(gradient_correlations)),
        }
    )
    checks = {
        "coverage": coverage >= 1.0,
        "shape_correlation": stats["shape_correlation"]
        >= CROSS_HEIGHT_FACE_GATES["minimum_shape_correlation"],
        "normalized_shape_rmse": stats["normalized_shape_rmse"]
        <= CROSS_HEIGHT_FACE_GATES["maximum_normalized_shape_rmse"],
        "gradient_correlation": stats["minimum_gradient_correlation"]
        >= CROSS_HEIGHT_FACE_GATES["minimum_gradient_correlation"],
    }
    stats["checks"] = {**checks, "passed": all(checks.values())}
    stats["passed"] = bool(stats["checks"]["passed"])
    return stats


def _cross_height_face_consistency(
    output_dir: str | Path,
    specs: tuple[SweepSpec, ...],
    completed_pairs: set[tuple[str, float]],
) -> dict:
    output_dir = Path(output_dir)
    records = []
    missing = []
    for spec in specs:
        _, face_mask, _ = _topology_scene(spec)
        emitted_face_mask = np.flip(face_mask, axis=1)
        emitted_part_masks = {
            name: np.flip(mask, axis=1)
            for name, mask in _synthetic_face_part_masks(face_mask).items()
        }
        available_heights = [
            height_mm
            for height_mm in RELIEF_HEIGHTS_MM
            if (spec.topology_id, float(height_mm)) in completed_pairs
        ]
        missing.extend(
            [spec.topology_id, float(height_mm)]
            for height_mm in RELIEF_HEIGHTS_MM
            if height_mm not in available_heights
        )
        surfaces = {
            height_mm: np.load(
                output_dir
                / f"{spec.topology_id}_{int(height_mm)}mm"
                / "emitted_surface.npy"
            )
            for height_mm in available_heights
        }
        for reference_height, candidate_height in combinations(available_heights, 2):
            metrics = _cross_height_face_shape_metrics(
                surfaces[reference_height],
                surfaces[candidate_height],
                emitted_face_mask,
            )
            named_parts = face_part_cross_height_metrics(
                surfaces[reference_height],
                surfaces[candidate_height],
                emitted_face_mask,
                emitted_part_masks,
                sample_pitch_mm=0.4,
            )
            whole_face_passed = bool(metrics["passed"])
            metrics["whole_face_passed"] = whole_face_passed
            metrics["named_parts"] = named_parts
            metrics["passed"] = bool(whole_face_passed and named_parts["passed"])
            records.append(
                {
                    "topology_id": spec.topology_id,
                    "reference_height_mm": float(reference_height),
                    "candidate_height_mm": float(candidate_height),
                    **metrics,
                }
            )
    expected_comparisons = len(specs) * len(tuple(combinations(RELIEF_HEIGHTS_MM, 2)))
    coverage_complete = len(records) == expected_comparisons and not missing
    quality_passed = all(record["passed"] for record in records)
    finite_shape_correlations = [
        record["shape_correlation"]
        for record in records
        if _finite(record.get("shape_correlation"))
    ]
    finite_shape_errors = [
        record["normalized_shape_rmse"]
        for record in records
        if _finite(record.get("normalized_shape_rmse"))
    ]
    finite_gradient_correlations = [
        record["minimum_gradient_correlation"]
        for record in records
        if _finite(record.get("minimum_gradient_correlation"))
    ]
    named_part_records = [
        part
        for record in records
        for part in record.get("named_parts", {}).get("parts", [])
        if part.get("available", False)
    ]
    return {
        "method": "robust_span_normalized_face_shape_with_named_parts_v2",
        "gates": CROSS_HEIGHT_FACE_GATES,
        "named_part_gates": FACE_PART_GATES,
        "expected_comparison_count": expected_comparisons,
        "comparison_count": len(records),
        "missing_rows": missing,
        "coverage_complete": coverage_complete,
        "quality_passed": quality_passed,
        "minimum_shape_correlation": (
            min(finite_shape_correlations) if finite_shape_correlations else None
        ),
        "maximum_normalized_shape_rmse": (
            max(finite_shape_errors) if finite_shape_errors else None
        ),
        "minimum_gradient_correlation": (
            min(finite_gradient_correlations)
            if finite_gradient_correlations
            else None
        ),
        "minimum_named_part_shape_correlation": (
            min(part["shape_correlation"] for part in named_part_records)
            if named_part_records
            else None
        ),
        "maximum_named_part_face_normalized_shape_rmse": (
            max(part["face_normalized_shape_rmse"] for part in named_part_records)
            if named_part_records
            else None
        ),
        "minimum_named_part_gradient_correlation": (
            min(part["minimum_gradient_correlation"] for part in named_part_records)
            if named_part_records
            else None
        ),
        "failed_named_parts": [
            {
                "topology_id": record["topology_id"],
                "reference_height_mm": record["reference_height_mm"],
                "candidate_height_mm": record["candidate_height_mm"],
                "parts": record.get("named_parts", {}).get("failed_parts", []),
            }
            for record in records
            if record.get("named_parts", {}).get("failed_parts")
        ],
        "passed": bool(coverage_complete and quality_passed),
        "records": records,
    }


def _appearance_negative_controls() -> dict:
    rows, cols = np.indices((81, 101), dtype=np.float32)
    region = np.ones((81, 101), dtype=bool)
    reference = (
        2.0
        + 0.025 * cols
        + 0.015 * rows
        + 4.0 * np.exp(-((rows - 39.0) ** 2 + (cols - 51.0) ** 2) / 125.0)
        + 0.7 * np.sin(cols / 7.0)
    ).astype(np.float32)
    shifted = _surface_lighting_agreement_metrics(
        reference,
        reference + 9.0,
        region,
        sample_pitch_mm=0.4,
    )
    flattened = _surface_lighting_agreement_metrics(
        reference,
        np.full(reference.shape, float(np.mean(reference)), dtype=np.float32),
        region,
        sample_pitch_mm=0.4,
    )
    smoothed = _surface_lighting_agreement_metrics(
        reference,
        gaussian_filter(reference, sigma=5.0),
        region,
        sample_pitch_mm=0.4,
    )
    missing_candidate = reference.copy()
    missing_candidate[18:36, 22:82] = np.nan
    missing = _surface_lighting_agreement_metrics(
        reference,
        missing_candidate,
        region,
        sample_pitch_mm=0.4,
    )
    first_component = (rows - 41.0) ** 2 + (cols - 28.0) ** 2 <= 15.0**2
    second_component = (rows - 41.0) ** 2 + (cols - 75.0) ** 2 <= 13.0**2
    component_region = first_component | second_component
    component_reference = (
        2.0
        + 3.0 * np.exp(-((rows - 41.0) ** 2 + (cols - 28.0) ** 2) / 75.0)
        + 2.6 * np.exp(-((rows - 41.0) ** 2 + (cols - 75.0) ** 2) / 60.0)
    ).astype(np.float32)
    component_candidate = component_reference.copy()
    component_candidate[second_component] = float(
        np.mean(component_reference[second_component])
    )
    component_damage = _surface_lighting_agreement_metrics(
        component_reference,
        component_candidate,
        component_region,
        sample_pitch_mm=0.4,
        component_metrics=True,
    )
    height_damage = reference.copy()
    height_damage[22:60, 34:70] = float(np.mean(height_damage[22:60, 34:70]))
    cross_height_damage = _cross_height_face_shape_metrics(
        reference,
        height_damage,
        region,
    )
    part_face = (rows - 41.0) ** 2 / 31.0**2 + (cols - 50.0) ** 2 / 27.0**2 <= 1.0
    part_masks = _synthetic_face_part_masks(part_face)
    mouth_flattened = reference.copy()
    mouth_flattened[part_masks["mouth"]] = float(
        np.mean(reference[part_masks["mouth"]])
    )
    nose_oversharpened = reference.copy()
    nose_reference = reference[part_masks["nose"]]
    nose_oversharpened[part_masks["nose"]] = (
        float(np.mean(nose_reference))
        + 4.0 * (nose_reference - float(np.mean(nose_reference)))
    )
    mouth_part_damage = face_part_cross_height_metrics(
        reference,
        mouth_flattened,
        part_face,
        part_masks,
        sample_pitch_mm=0.4,
    )
    nose_part_damage = face_part_cross_height_metrics(
        reference,
        nose_oversharpened,
        part_face,
        part_masks,
        sample_pitch_mm=0.4,
    )
    shifted_checks = _appearance_checks(shifted, FACE_APPEARANCE_GATES)
    flattened_checks = _appearance_checks(flattened, FACE_APPEARANCE_GATES)
    smoothed_checks = _appearance_checks(smoothed, FACE_APPEARANCE_GATES)
    missing_checks = _appearance_checks(missing, FACE_APPEARANCE_GATES)
    component_checks = _appearance_checks(component_damage, FACE_APPEARANCE_GATES)
    checks = {
        "constant_height_offset_passes": shifted_checks["passed"],
        "flattened_surface_rejected": not flattened_checks["passed"],
        "heavy_smoothing_rejected": not smoothed_checks["passed"],
        "missing_candidate_pixels_rejected": not missing_checks["passed"],
        "single_component_damage_rejected": not component_checks["passed"],
        "cross_height_face_damage_rejected": not cross_height_damage["passed"],
        "mouth_flattening_rejected": bool(
            not mouth_part_damage["passed"]
            and "mouth" in mouth_part_damage["failed_parts"]
        ),
        "nose_oversharpening_rejected": bool(
            not nose_part_damage["passed"]
            and "nose" in nose_part_damage["failed_parts"]
        ),
    }
    return {
        "checks": {**checks, "passed": all(checks.values())},
        "shifted": _appearance_record(shifted),
        "flattened": _appearance_record(flattened),
        "smoothed": _appearance_record(smoothed),
        "missing": _appearance_record(missing),
        "component_damage": _appearance_record(component_damage),
        "cross_height_damage": cross_height_damage,
        "mouth_part_damage": mouth_part_damage,
        "nose_part_damage": nose_part_damage,
    }


def _extreme(rows: list[dict], section: str, key: str, function) -> float | None:
    values = [
        row[section].get(key)
        for row in rows
        if _finite(row.get(section, {}).get(key))
    ]
    return float(function(values)) if values else None


def _matrix_coverage(rows: list[dict], expected_pairs: set[tuple[str, float]]) -> dict:
    actual_pairs = [
        (str(row["topology_id"]), float(row["relief_height_mm"]))
        for row in rows
    ]
    actual_pair_set = set(actual_pairs)
    return {
        "exact": actual_pair_set == expected_pairs and len(actual_pairs) == len(expected_pairs),
        "unique": len(actual_pairs) == len(actual_pair_set),
        "missing": [
            [topology_id, height_mm]
            for topology_id, height_mm in sorted(expected_pairs - actual_pair_set)
        ],
        "unexpected": [
            [topology_id, height_mm]
            for topology_id, height_mm in sorted(actual_pair_set - expected_pairs)
        ],
    }


def run(
    output_dir: str | Path,
    summary_path: str | Path | None = None,
    limit: int | None = None,
    allow_dirty: bool = False,
    smoke: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = [
        (spec, height_mm)
        for spec in _sweep_specs()
        for height_mm in RELIEF_HEIGHTS_MM
    ]
    selected_matrix = (
        matrix
        if limit is None
        else matrix[: max(0, int(limit))]
    )
    if not selected_matrix:
        raise ValueError("Relief visual sweep requires at least one matrix row")

    rows = []
    started = time.perf_counter()
    for spec, height_mm in selected_matrix:
        row_started = time.perf_counter()
        row_id = f"{spec.topology_id}_{int(height_mm)}mm"
        row_dir = output_dir / row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        source, face_mask, subject_mask = _topology_scene(spec)
        mask_topology = _mask_topology(subject_mask)
        composed, compose_stats = compose_selection_depth_with_context(
            source,
            subject_mask,
            relief_height_mm=height_mm,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            background_depth_ratio=0.45,
            background_feather_mm=1.5,
            background_smoothing_mm=0.6,
        )
        depth_path = row_dir / "depth.npy"
        stl_path = row_dir / "relief.stl"
        surface_path = row_dir / "emitted_surface.npy"
        np.save(depth_path, composed.astype(np.float32, copy=False))
        postprocess = depth_data_to_3d_model(
            depth_path,
            output_stl_path=str(stl_path),
            target_dimension=-1,
            z_scale=height_mm,
            max_xy_size=48.0,
            sigma=0.0,
            relief_gamma=1.0,
            detail_boost=0.0,
            low_percentile=0.0,
            high_percentile=100.0,
            base_border_px=1,
            value_transform="linear",
            minimum_feature_mm=0.8,
            max_relief_slope=2.0,
            face_region_mask=face_mask,
            selection_region_mask=subject_mask,
            selection_background_depth_ratio=0.45,
            surface_output_path=surface_path,
        )
        mesh = trimesh.load_mesh(stl_path, process=True)
        topology = _mesh_topology(mesh)
        emitted_surface = _stl_heightfield_agreement(stl_path, surface_path)
        base_checks = _scene_checks(compose_stats, postprocess, topology)
        appearance = postprocess["surface_appearance_agreement"]
        face_appearance = appearance["face"]
        selection_appearance = appearance["selection_nonface"]
        background_appearance = appearance["background"]
        boundary_shape = _boundary_shape_metrics(
            postprocess["background_depth_preservation"],
            postprocess["selection_background_physical_cap"],
        )
        selection_solver = _selection_solver_record(
            postprocess.get("selection_gradient_compression", {})
        )
        face_checks = _appearance_checks(face_appearance, FACE_APPEARANCE_GATES)
        selection_checks = _appearance_checks(
            selection_appearance,
            SELECTION_APPEARANCE_GATES,
        )
        background_checks = _appearance_checks(
            background_appearance,
            BACKGROUND_APPEARANCE_GATES,
        )
        row_checks = {
            **base_checks,
            "face_appearance": face_checks["passed"],
            "selection_appearance": selection_checks["passed"],
            "background_appearance": background_checks["passed"],
            "selection_solver": selection_solver["checks"]["passed"],
            "emitted_surface": emitted_surface["passed"],
            "mask_topology": bool(
                mask_topology["component_count"] == spec.expected_components
                and mask_topology["hole_count"] == spec.expected_holes
            ),
            "appearance_component_coverage": bool(
                selection_appearance.get("component_count")
                == spec.expected_components
                and background_appearance.get("component_count")
                == spec.expected_background_components
            ),
        }
        rows.append(
            {
                "row_id": row_id,
                "topology_id": spec.topology_id,
                "background_archetype": spec.scene.background,
                "relief_height_mm": float(height_mm),
                "subject_coverage_ratio": float(np.mean(subject_mask)),
                "mask_topology": mask_topology,
                "runtime_seconds": float(time.perf_counter() - row_started),
                "checks": {**row_checks, "passed": all(row_checks.values())},
                "face_appearance_checks": face_checks,
                "selection_appearance_checks": selection_checks,
                "background_appearance_checks": background_checks,
                "compose": {
                    "normalized_context_correlation": _finite_metric(
                        compose_stats,
                        "background_context_normalized_correlation",
                    ),
                    "normalized_context_rms_retention": _finite_metric(
                        compose_stats,
                        "background_context_normalized_rms_retention",
                    ),
                    "recoverable_context_coverage_ratio": _finite_metric(
                        compose_stats,
                        "background_context_recoverable_coverage_ratio",
                    ),
                    "measured_context_coverage_ratio": _finite_metric(
                        compose_stats,
                        "background_context_measured_coverage_ratio",
                    ),
                    "recoverable_measured_context_ratio": _finite_metric(
                        compose_stats,
                        "background_context_recoverable_measured_ratio",
                    ),
                },
                "face": {
                    "correlation": _finite_metric(
                        postprocess["face_detail_guard"].get("final", {}),
                        "correlation",
                    ),
                    "rms_retention": _finite_metric(
                        postprocess["face_detail_guard"].get("final", {}),
                        "rms_retention",
                    ),
                },
                "face_appearance": _appearance_record(face_appearance),
                "selection_appearance": _appearance_record(selection_appearance),
                "background_appearance": _appearance_record(background_appearance),
                "selection_solver": selection_solver,
                "emitted_surface": emitted_surface,
                "background": {
                    "correlation": _finite_metric(
                        postprocess["background_depth_preservation"],
                        "correlation",
                    ),
                    "gradient_correlation": _finite_metric(
                        postprocess["background_depth_preservation"],
                        "gradient_correlation",
                    ),
                },
                "boundary_shape": boundary_shape,
                "physical_cap": {
                    "far_background_max_mm": _finite_metric(
                        postprocess["selection_background_physical_cap"],
                        "far_background_max_mm",
                    ),
                    "feasible_attachment_jump_max_mm": _finite_metric(
                        postprocess["selection_background_physical_cap"],
                        "feasible_attachment_jump_max_mm",
                    ),
                    "emission_passed": bool(
                        postprocess["selection_background_physical_cap"].get(
                            "emission_passed",
                            False,
                        )
                    ),
                    "attachment_constraint_conflicts": int(
                        postprocess["selection_background_physical_cap"].get(
                            "attachment_constraint_conflicts",
                            0,
                        )
                    ),
                    "attachment_constraint_conflict_max_mm": _finite_metric(
                        postprocess["selection_background_physical_cap"],
                        "attachment_constraint_conflict_max_mm",
                    ),
                },
                "topology": topology,
                "artifacts": {
                    "depth": _artifact_record(depth_path),
                    "surface": _artifact_record(surface_path),
                    "stl": _artifact_record(stl_path),
                },
            }
        )

    provenance = _git_provenance(PROVENANCE_PATHS)
    negative_controls = _appearance_negative_controls()
    expected_rows = len(matrix)
    expected_pairs = {
        (spec.topology_id, float(height_mm))
        for spec, height_mm in matrix
    }
    matrix_coverage = _matrix_coverage(rows, expected_pairs)
    completed_pairs = {
        (str(row["topology_id"]), float(row["relief_height_mm"]))
        for row in rows
    }
    cross_height_face = _cross_height_face_consistency(
        output_dir,
        _sweep_specs(),
        completed_pairs,
    )
    topologies = {row["topology_id"] for row in rows}
    heights = {row["relief_height_mm"] for row in rows}
    checks = {
        "full_matrix_complete": matrix_coverage["exact"],
        "matrix_rows_unique": matrix_coverage["unique"],
        "expected_row_count": len(rows) == len(selected_matrix),
        "topology_coverage_complete": topologies
        == {spec.topology_id for spec in _sweep_specs()},
        "height_coverage_complete": heights == set(RELIEF_HEIGHTS_MM),
        "implementation_provenance_clean": bool(
            provenance.get("available", False) and provenance.get("clean", False)
        ),
        "negative_controls_passed": bool(
            negative_controls.get("checks", {}).get("passed", False)
        ),
        "cross_height_face_coverage_complete": bool(
            cross_height_face.get("coverage_complete", False)
        ),
        "cross_height_face_quality_passed": bool(
            cross_height_face.get("quality_passed", False)
        ),
        "all_row_gates_passed": bool(rows)
        and all(row["checks"]["passed"] for row in rows),
        "all_face_appearance_gates_passed": bool(rows)
        and all(row["checks"]["face_appearance"] for row in rows),
        "all_selection_appearance_gates_passed": bool(rows)
        and all(row["checks"]["selection_appearance"] for row in rows),
        "all_background_appearance_gates_passed": bool(rows)
        and all(row["checks"]["background_appearance"] for row in rows),
        "all_selection_solver_gates_passed": bool(rows)
        and all(row["checks"]["selection_solver"] for row in rows),
        "all_emitted_surfaces_match": bool(rows)
        and all(row["checks"]["emitted_surface"] for row in rows),
        "all_meshes_printable": bool(rows)
        and all(row["topology"]["printable"] for row in rows),
    }
    summary = {
        "schema_version": 2,
        "run_kind": "deterministic_privacy_safe_relief_visual_sweep",
        "privacy": "all inputs are analytic arrays; no private artifacts are used",
        "implementation_provenance": provenance,
        "allow_dirty": bool(allow_dirty),
        "smoke": bool(smoke),
        "matrix": {
            "relief_heights_mm": list(RELIEF_HEIGHTS_MM),
            "topology_ids": [spec.topology_id for spec in _sweep_specs()],
            "expected_rows": expected_rows,
            "completed_rows": len(rows),
            "coverage": matrix_coverage,
        },
        "appearance_method": "physical_heightfield_normals_lambertian_v1",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "trimesh": trimesh.__version__,
        },
        "face_appearance_gates": FACE_APPEARANCE_GATES,
        "selection_appearance_gates": SELECTION_APPEARANCE_GATES,
        "selection_solver_gates": SELECTION_SOLVER_GATES,
        "background_appearance_gates": BACKGROUND_APPEARANCE_GATES,
        "cross_height_face_gates": CROSS_HEIGHT_FACE_GATES,
        "named_face_part_gates": FACE_PART_GATES,
        "runtime_seconds": float(time.perf_counter() - started),
        "checks": {**checks, "passed": all(checks.values())},
        "aggregate": {
            "minimum_face_normal_mean_cosine": _extreme(
                rows,
                "face_appearance",
                "normal_mean_cosine",
                min,
            ),
            "maximum_face_normal_angle_p95_deg": _extreme(
                rows,
                "face_appearance",
                "normal_angle_p95_deg",
                max,
            ),
            "minimum_face_lighting_correlation": _extreme(
                rows,
                "face_appearance",
                "minimum_lighting_correlation",
                min,
            ),
            "maximum_face_lighting_mae": _extreme(
                rows,
                "face_appearance",
                "maximum_lighting_mae",
                max,
            ),
            "minimum_background_normal_mean_cosine": _extreme(
                rows,
                "background_appearance",
                "normal_mean_cosine",
                min,
            ),
            "minimum_selection_normal_mean_cosine": _extreme(
                rows,
                "selection_appearance",
                "normal_mean_cosine",
                min,
            ),
            "minimum_selection_lighting_correlation": _extreme(
                rows,
                "selection_appearance",
                "minimum_lighting_correlation",
                min,
            ),
            "minimum_background_lighting_correlation": _extreme(
                rows,
                "background_appearance",
                "minimum_lighting_correlation",
                min,
            ),
            "maximum_background_lighting_mae": _extreme(
                rows,
                "background_appearance",
                "maximum_lighting_mae",
                max,
            ),
            "minimum_face_detail_correlation": _extreme(
                rows,
                "face",
                "correlation",
                min,
            ),
            "minimum_background_depth_correlation": _extreme(
                rows,
                "background",
                "correlation",
                min,
            ),
            "maximum_selection_solver_edge_p99_ratio": _extreme(
                rows,
                "selection_solver",
                "output_edge_ratio_p99",
                max,
            ),
            "maximum_selection_solver_edge_ratio": _extreme(
                rows,
                "selection_solver",
                "output_edge_ratio_max",
                max,
            ),
            "maximum_selection_solver_diagonal_edge_ratio": _extreme(
                rows,
                "selection_solver",
                "diagonal_edge_ratio_max",
                max,
            ),
            "maximum_far_background_mm": _extreme(
                rows,
                "physical_cap",
                "far_background_max_mm",
                max,
            ),
        },
        "negative_controls": negative_controls,
        "cross_height_face_consistency": cross_height_face,
        "rows": rows,
    }
    summary_path = Path(summary_path) if summary_path is not None else output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary_summary_path.replace(summary_path)
    print(json.dumps(summary, indent=2, allow_nan=False))
    failed = {name for name, passed in checks.items() if not passed}
    permitted_failures = set()
    if allow_dirty:
        permitted_failures.add("implementation_provenance_clean")
    if smoke:
        permitted_failures.update(
            {
                "full_matrix_complete",
                "topology_coverage_complete",
                "height_coverage_complete",
                "cross_height_face_coverage_complete",
            }
        )
    blocking_failures = sorted(failed - permitted_failures)
    if blocking_failures:
        raise RuntimeError(
            f"Relief visual sweep failed: {', '.join(blocking_failures)}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="backend/output/relief-visual-sweep-local-n12",
    )
    parser.add_argument("--summary-path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(
        args.output_dir,
        summary_path=args.summary_path,
        limit=args.limit,
        allow_dirty=args.allow_dirty,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
