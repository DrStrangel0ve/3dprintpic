"""Part-aware physical relief metrics for fixed facial landmark regions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    laplace,
    zoom,
)
from scipy.stats import wasserstein_distance

from backend.face_depth_refinement import FACE_PART_NAMES


FACE_PART_METRIC_SCHEMA_VERSION = 3
FACE_PART_SMOOTHING_RADII_MM = (0.0, 0.8)
FACE_PART_MINIMUM_ERODED_SUPPORT_RATIO = 0.35
FACE_PART_GATES = {
    "minimum_coverage_ratio": 1.0,
    "minimum_shape_correlation": 0.90,
    "maximum_face_normalized_shape_rmse": 0.20,
    "minimum_raw_gradient_correlation": 0.25,
    "minimum_gradient_correlation": 0.75,
    "minimum_slope_q95_retention": 0.45,
    "maximum_slope_q95_retention": 2.20,
    "minimum_curvature_q95_retention": 0.35,
    "maximum_curvature_q95_retention": 2.80,
    "maximum_slope_wasserstein_ratio": 0.75,
    "maximum_curvature_wasserstein_ratio": 0.90,
}
FACE_PART_AFFINE_MM_GATES = {
    "minimum_coverage_ratio": 1.0,
    "minimum_face_affine_scale": 0.10,
    "maximum_face_affine_scale": 2.00,
    "maximum_rmse_mm": 0.80,
    "maximum_p95_absolute_error_mm": 1.50,
    "maximum_absolute_bias_mm": 0.50,
    "minimum_span_retention": 0.65,
    "maximum_span_retention": 1.35,
}


def _centered_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
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


def _retention(reference_value: float, candidate_value: float) -> float | None:
    if not np.isfinite(reference_value) or reference_value <= 1e-12:
        return None
    if not np.isfinite(candidate_value):
        return None
    return float(candidate_value / reference_value)


def _all_finite(values) -> bool:
    return bool(values) and all(
        value is not None and np.isfinite(float(value)) for value in values
    )


def _complete_extreme(values, function) -> float | None:
    if not _all_finite(values):
        return None
    return float(function(float(value) for value in values))


def _fill_nonfinite_nearest(values: np.ndarray) -> np.ndarray:
    """Supply derivative stencils without hiding missing in-mask coverage."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if np.all(finite):
        return values
    if not np.any(finite):
        return values
    nearest = distance_transform_edt(
        ~finite,
        return_distances=False,
        return_indices=True,
    )
    return values[tuple(nearest)]


def _distribution_metrics(
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
) -> dict:
    reference = np.asarray(reference_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    candidate = candidate[np.isfinite(candidate)]
    stats = {"available": False, "samples": int(min(len(reference), len(candidate)))}
    if not len(reference) or not len(candidate):
        stats["reason"] = "no_finite_samples"
        return stats
    reference_q50, reference_q90, reference_q95 = np.percentile(
        reference, (50.0, 90.0, 95.0)
    )
    candidate_q50, candidate_q90, candidate_q95 = np.percentile(
        candidate, (50.0, 90.0, 95.0)
    )
    reference_rms = float(np.sqrt(np.mean(np.square(reference))))
    candidate_rms = float(np.sqrt(np.mean(np.square(candidate))))
    scale = max(float(reference_q95), reference_rms, 1e-12)
    stats.update(
        {
            "available": True,
            "reference_q50": float(reference_q50),
            "reference_q90": float(reference_q90),
            "reference_q95": float(reference_q95),
            "candidate_q50": float(candidate_q50),
            "candidate_q90": float(candidate_q90),
            "candidate_q95": float(candidate_q95),
            "q95_retention": _retention(reference_q95, candidate_q95),
            "rms_retention": _retention(reference_rms, candidate_rms),
            "wasserstein_ratio": float(
                wasserstein_distance(reference, candidate) / scale
            ),
        }
    )
    return stats


def _eroded_or_original(
    mask: np.ndarray,
    iterations: int,
    minimum_samples: int,
    minimum_retained_fraction: float = 0.0,
) -> np.ndarray:
    return _select_eroded_support(
        mask,
        iterations,
        minimum_samples,
        minimum_retained_fraction,
    )[0]


def _select_eroded_support(
    mask: np.ndarray,
    iterations: int,
    minimum_samples: int,
    minimum_retained_fraction: float = 0.0,
) -> tuple[np.ndarray, dict]:
    mask = np.asarray(mask, dtype=bool)
    original_samples = int(np.count_nonzero(mask))
    if iterations <= 0:
        return mask, {
            "attempted": False,
            "applied": False,
            "reason": "not_requested",
            "visible_samples": original_samples,
            "eroded_samples": original_samples,
            "attempted_retained_fraction": 1.0,
            "selected_samples": original_samples,
        }
    eroded = binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        iterations=int(iterations),
        border_value=0,
    )
    eroded_samples = int(np.count_nonzero(eroded))
    retained_fraction = eroded_samples / max(original_samples, 1)
    enough_samples = eroded_samples >= int(minimum_samples)
    enough_support = retained_fraction >= float(minimum_retained_fraction)
    applied = bool(enough_samples and enough_support)
    reason = (
        "accepted"
        if applied
        else "insufficient_samples"
        if not enough_samples
        else "insufficient_retained_fraction"
    )
    selected = eroded if applied else mask
    return selected, {
        "attempted": True,
        "applied": applied,
        "reason": reason,
        "visible_samples": original_samples,
        "eroded_samples": eroded_samples,
        "attempted_retained_fraction": float(retained_fraction),
        "selected_samples": int(np.count_nonzero(selected)),
    }


def face_part_cross_height_metrics(
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    face_mask: np.ndarray,
    part_masks: dict[str, np.ndarray],
    *,
    sample_pitch_mm: float,
    boundary_exclusion_px: int = 1,
    minimum_face_samples: int = 64,
    minimum_part_samples: int = 12,
    smoothing_radii_mm: tuple[float, ...] = FACE_PART_SMOOTHING_RADII_MM,
    gates: dict | None = None,
) -> dict:
    """Compare fixed facial parts with one normalization shared by the face."""
    reference = np.asarray(reference_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    face = np.asarray(face_mask, dtype=bool)
    try:
        radii_mm = tuple(float(value) for value in smoothing_radii_mm)
    except (TypeError, ValueError):
        radii_mm = ()
    effective_gates = dict(FACE_PART_GATES)
    if gates is not None:
        effective_gates.update(gates)
    stats = {
        "schema_version": FACE_PART_METRIC_SCHEMA_VERSION,
        "available": False,
        "passed": False,
        "sample_pitch_mm": None,
        "normalization": "per-face-median-and-p05-p95-span",
        "smoothing_radii_mm": list(radii_mm),
        "gates": effective_gates,
        "parts": [],
    }
    try:
        pitch = float(sample_pitch_mm)
    except (TypeError, ValueError):
        stats["reason"] = "invalid_sample_pitch"
        return stats
    stats["sample_pitch_mm"] = pitch
    if not np.isfinite(pitch) or pitch <= 0:
        stats["reason"] = "invalid_sample_pitch"
        return stats
    if not radii_mm or any(
        not np.isfinite(radius) or radius < 0 for radius in radii_mm
    ):
        stats["reason"] = "invalid_smoothing_radii"
        return stats
    if reference.shape != candidate.shape or reference.shape != face.shape:
        stats["reason"] = "shape_mismatch"
        return stats
    if not part_masks:
        stats["reason"] = "no_part_masks"
        return stats
    if any(np.asarray(mask).shape != face.shape for mask in part_masks.values()):
        stats["reason"] = "part_shape_mismatch"
        return stats

    reference_valid = np.isfinite(reference)
    candidate_valid = np.isfinite(candidate)
    face_interior = _eroded_or_original(face, 2, minimum_face_samples)
    expected_face = face_interior & reference_valid
    measured_face = expected_face & candidate_valid
    reference_face_samples = int(np.count_nonzero(expected_face))
    measured_face_samples = int(np.count_nonzero(measured_face))
    face_coverage = measured_face_samples / max(reference_face_samples, 1)
    stats.update(
        {
            "reference_face_samples": reference_face_samples,
            "face_samples": measured_face_samples,
            "face_coverage_ratio": float(face_coverage),
        }
    )
    if reference_face_samples < int(minimum_face_samples):
        stats["reason"] = "insufficient_reference_face_samples"
        return stats
    if measured_face_samples < int(minimum_face_samples):
        stats["reason"] = "insufficient_candidate_face_samples"
        return stats

    reference_face_values = reference[measured_face]
    candidate_face_values = candidate[measured_face]
    reference_p05, reference_p95 = np.percentile(reference_face_values, (5.0, 95.0))
    candidate_p05, candidate_p95 = np.percentile(candidate_face_values, (5.0, 95.0))
    reference_span = float(reference_p95 - reference_p05)
    candidate_span = float(candidate_p95 - candidate_p05)
    if reference_span <= 1e-8 or candidate_span <= 1e-8:
        stats["reason"] = "flat_face_surface"
        return stats
    reference_median = float(np.median(reference_face_values))
    candidate_median = float(np.median(candidate_face_values))
    reference_normalized = (reference - reference_median) / reference_span
    candidate_normalized = (candidate - candidate_median) / candidate_span
    reference_derivative_field = _fill_nonfinite_nearest(reference_normalized)
    candidate_derivative_field = _fill_nonfinite_nearest(candidate_normalized)
    stats["face_normalization"] = {
        "reference_median_mm": reference_median,
        "candidate_median_mm": candidate_median,
        "reference_p05_p95_span_mm": reference_span,
        "candidate_p05_p95_span_mm": candidate_span,
    }

    scale_fields = []
    for radius_mm in radii_mm:
        radius = max(0.0, float(radius_mm))
        sigma = radius / pitch
        reference_scale = (
            gaussian_filter(reference_derivative_field, sigma=sigma, mode="nearest")
            if sigma > 0
            else reference_derivative_field
        )
        candidate_scale = (
            gaussian_filter(candidate_derivative_field, sigma=sigma, mode="nearest")
            if sigma > 0
            else candidate_derivative_field
        )
        reference_gradients = np.gradient(reference_scale, pitch)
        candidate_gradients = np.gradient(candidate_scale, pitch)
        reference_slope = np.hypot(*reference_gradients)
        candidate_slope = np.hypot(*candidate_gradients)
        reference_curvature = np.abs(laplace(reference_scale, mode="nearest")) / pitch**2
        candidate_curvature = np.abs(laplace(candidate_scale, mode="nearest")) / pitch**2
        scale_fields.append(
            {
                "radius_mm": radius,
                "reference_gradients": reference_gradients,
                "candidate_gradients": candidate_gradients,
                "reference_slope": reference_slope,
                "candidate_slope": candidate_slope,
                "reference_curvature": reference_curvature,
                "candidate_curvature": candidate_curvature,
            }
        )

    for name in sorted(part_masks):
        visible_part = np.asarray(part_masks[name], dtype=bool) & face
        visible_part_samples = int(np.count_nonzero(visible_part))
        part, boundary_support = _select_eroded_support(
            visible_part,
            boundary_exclusion_px,
            minimum_part_samples,
            FACE_PART_MINIMUM_ERODED_SUPPORT_RATIO,
        )
        selected_part_samples = int(np.count_nonzero(part))
        expected = part & reference_valid
        measured = expected & candidate_valid
        reference_samples = int(np.count_nonzero(expected))
        samples = int(np.count_nonzero(measured))
        coverage = samples / max(reference_samples, 1)
        record = {
            "name": str(name),
            "available": False,
            "passed": False,
            "reference_samples": reference_samples,
            "samples": samples,
            "candidate_coverage_ratio": float(coverage),
            "visible_samples_before_boundary_exclusion": visible_part_samples,
            "boundary_exclusion_requested_px": int(boundary_exclusion_px),
            "boundary_exclusion_applied": bool(boundary_support["applied"]),
            "boundary_exclusion_reason": boundary_support["reason"],
            "boundary_exclusion_eroded_samples": int(
                boundary_support["eroded_samples"]
            ),
            "boundary_exclusion_attempted_retained_fraction": float(
                boundary_support["attempted_retained_fraction"]
            ),
            "boundary_exclusion_retained_fraction": float(
                selected_part_samples / max(visible_part_samples, 1)
            ),
            "minimum_eroded_support_ratio": (
                FACE_PART_MINIMUM_ERODED_SUPPORT_RATIO
            ),
            "scales": [],
        }
        if reference_samples < int(minimum_part_samples):
            record["reason"] = "insufficient_reference_samples"
            stats["parts"].append(record)
            continue
        if samples < int(minimum_part_samples):
            record["reason"] = "insufficient_candidate_samples"
            stats["parts"].append(record)
            continue

        reference_signal = reference_normalized[measured]
        candidate_signal = candidate_normalized[measured]
        record.update(
            {
                "available": True,
                "shape_correlation": _centered_correlation(
                    reference_signal,
                    candidate_signal,
                ),
                "face_normalized_shape_rmse": float(
                    np.sqrt(np.mean(np.square(reference_signal - candidate_signal)))
                ),
            }
        )
        for fields in scale_fields:
            gradient_correlations = [
                _centered_correlation(
                    reference_gradient[measured],
                    candidate_gradient[measured],
                )
                for reference_gradient, candidate_gradient in zip(
                    fields["reference_gradients"],
                    fields["candidate_gradients"],
                )
            ]
            record["scales"].append(
                {
                    "radius_mm": fields["radius_mm"],
                    "gradient_correlations": gradient_correlations,
                    "minimum_gradient_correlation": float(min(gradient_correlations)),
                    "slope": _distribution_metrics(
                        fields["reference_slope"][measured],
                        fields["candidate_slope"][measured],
                    ),
                    "curvature": _distribution_metrics(
                        fields["reference_curvature"][measured],
                        fields["candidate_curvature"][measured],
                    ),
                }
            )
        minimum_all_scale_gradient_correlation = min(
            scale["minimum_gradient_correlation"] for scale in record["scales"]
        )
        raw_gradient_correlations = [
            scale["minimum_gradient_correlation"]
            for scale in record["scales"]
            if float(scale["radius_mm"]) <= 0.0
        ]
        physical_gradient_correlations = [
            scale["minimum_gradient_correlation"]
            for scale in record["scales"]
            if float(scale["radius_mm"]) > 0.0
        ]
        minimum_raw_gradient_correlation = min(
            raw_gradient_correlations
            or [minimum_all_scale_gradient_correlation]
        )
        minimum_gradient_correlation = min(
            physical_gradient_correlations
            or [minimum_all_scale_gradient_correlation]
        )
        slope_retentions = [
            scale["slope"].get("q95_retention") for scale in record["scales"]
        ]
        curvature_retentions = [
            scale["curvature"].get("q95_retention") for scale in record["scales"]
        ]
        slope_distances = [
            scale["slope"].get("wasserstein_ratio") for scale in record["scales"]
        ]
        curvature_distances = [
            scale["curvature"].get("wasserstein_ratio")
            for scale in record["scales"]
        ]
        checks = {
            "coverage": coverage >= effective_gates["minimum_coverage_ratio"],
            "shape_correlation": record["shape_correlation"]
            >= effective_gates["minimum_shape_correlation"],
            "shape_rmse": record["face_normalized_shape_rmse"]
            <= effective_gates["maximum_face_normalized_shape_rmse"],
            "gradient_correlation": minimum_gradient_correlation
            >= effective_gates["minimum_gradient_correlation"],
            "raw_gradient_correlation": minimum_raw_gradient_correlation
            >= effective_gates["minimum_raw_gradient_correlation"],
            "slope_retention": all(
                _all_finite(slope_retentions)
                and value is not None
                and effective_gates["minimum_slope_q95_retention"]
                <= value
                <= effective_gates["maximum_slope_q95_retention"]
                for value in slope_retentions
            ),
            "curvature_retention": all(
                _all_finite(curvature_retentions)
                and value is not None
                and effective_gates["minimum_curvature_q95_retention"]
                <= value
                <= effective_gates["maximum_curvature_q95_retention"]
                for value in curvature_retentions
            ),
            "slope_distribution": all(
                _all_finite(slope_distances)
                and value is not None
                and value <= effective_gates["maximum_slope_wasserstein_ratio"]
                for value in slope_distances
            ),
            "curvature_distribution": all(
                _all_finite(curvature_distances)
                and value is not None
                and value <= effective_gates["maximum_curvature_wasserstein_ratio"]
                for value in curvature_distances
            ),
        }
        record.update(
            {
                "minimum_gradient_correlation": float(minimum_gradient_correlation),
                "minimum_raw_gradient_correlation": float(
                    minimum_raw_gradient_correlation
                ),
                "minimum_all_scale_gradient_correlation": float(
                    minimum_all_scale_gradient_correlation
                ),
                "minimum_slope_q95_retention": _complete_extreme(
                    slope_retentions, min
                ),
                "maximum_slope_q95_retention": _complete_extreme(
                    slope_retentions, max
                ),
                "minimum_curvature_q95_retention": _complete_extreme(
                    curvature_retentions, min
                ),
                "maximum_curvature_q95_retention": _complete_extreme(
                    curvature_retentions, max
                ),
                "maximum_slope_wasserstein_ratio": _complete_extreme(
                    slope_distances, max
                ),
                "maximum_curvature_wasserstein_ratio": _complete_extreme(
                    curvature_distances, max
                ),
                "checks": {**checks, "passed": all(checks.values())},
                "passed": bool(all(checks.values())),
            }
        )
        stats["parts"].append(record)

    measured_parts = [record for record in stats["parts"] if record["available"]]
    stats.update(
        {
            "available": bool(measured_parts),
            "part_count": len(stats["parts"]),
            "measured_part_count": len(measured_parts),
            "unavailable_part_count": len(stats["parts"]) - len(measured_parts),
            "failed_parts": [
                record["name"] for record in stats["parts"] if not record["passed"]
            ],
        }
    )
    stats["passed"] = bool(
        face_coverage >= effective_gates["minimum_coverage_ratio"]
        and len(measured_parts) == len(stats["parts"])
        and all(record["passed"] for record in stats["parts"])
    )
    return stats


def face_part_affine_surface_error_metrics(
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    face_mask: np.ndarray,
    part_masks: dict[str, np.ndarray],
    *,
    minimum_face_samples: int = 64,
    minimum_part_samples: int = 12,
    boundary_exclusion_px: int = 1,
    gates: dict | None = None,
) -> dict:
    """Measure local millimeter error after one shared face-wide affine fit."""
    reference = np.asarray(reference_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    face = np.asarray(face_mask, dtype=bool)
    effective_gates = dict(FACE_PART_AFFINE_MM_GATES)
    if gates is not None:
        effective_gates.update(gates)
    stats = {
        "schema_version": 1,
        "method": "shared_face_affine_then_part_mm_error",
        "available": False,
        "passed": False,
        "gates": effective_gates,
        "parts": [],
        "failed_parts": sorted(str(name) for name in part_masks),
        "face_affine_fit": None,
    }
    if reference.shape != candidate.shape or reference.shape != face.shape:
        stats["reason"] = "shape_mismatch"
        return stats
    if not part_masks:
        stats["reason"] = "no_part_masks"
        return stats
    if any(np.asarray(mask).shape != face.shape for mask in part_masks.values()):
        stats["reason"] = "part_shape_mismatch"
        return stats

    reference_valid = np.isfinite(reference)
    candidate_valid = np.isfinite(candidate)
    face_interior = _eroded_or_original(face, 2, minimum_face_samples)
    expected_face = face_interior & reference_valid
    measured_face = expected_face & candidate_valid
    reference_face_samples = int(np.count_nonzero(expected_face))
    face_samples = int(np.count_nonzero(measured_face))
    face_coverage = face_samples / max(reference_face_samples, 1)
    stats.update(
        {
            "reference_face_samples": reference_face_samples,
            "face_samples": face_samples,
            "face_coverage_ratio": float(face_coverage),
        }
    )
    if reference_face_samples < int(minimum_face_samples):
        stats["reason"] = "insufficient_reference_face_samples"
        return stats
    if face_samples < int(minimum_face_samples):
        stats["reason"] = "insufficient_candidate_face_samples"
        return stats

    reference_face = reference[measured_face]
    candidate_face = candidate[measured_face]
    design = np.column_stack((reference_face, np.ones_like(reference_face)))
    scale, offset = np.linalg.lstsq(design, candidate_face, rcond=None)[0]
    if not np.isfinite(scale) or not np.isfinite(offset):
        stats["reason"] = "invalid_face_affine_fit"
        return stats
    expected_candidate = scale * reference + offset
    stats["face_affine_fit"] = {
        "scale": float(scale),
        "offset_mm": float(offset),
    }

    for name in sorted(part_masks):
        visible_part = np.asarray(part_masks[name], dtype=bool) & face
        visible_part_samples = int(np.count_nonzero(visible_part))
        part, boundary_support = _select_eroded_support(
            visible_part,
            boundary_exclusion_px,
            minimum_part_samples,
            FACE_PART_MINIMUM_ERODED_SUPPORT_RATIO,
        )
        selected_part_samples = int(np.count_nonzero(part))
        expected = part & reference_valid
        measured = expected & candidate_valid
        reference_samples = int(np.count_nonzero(expected))
        samples = int(np.count_nonzero(measured))
        coverage = samples / max(reference_samples, 1)
        record = {
            "name": str(name),
            "available": False,
            "passed": False,
            "reference_samples": reference_samples,
            "samples": samples,
            "candidate_coverage_ratio": float(coverage),
            "visible_samples_before_boundary_exclusion": visible_part_samples,
            "boundary_exclusion_requested_px": int(boundary_exclusion_px),
            "boundary_exclusion_applied": bool(boundary_support["applied"]),
            "boundary_exclusion_reason": boundary_support["reason"],
            "boundary_exclusion_eroded_samples": int(
                boundary_support["eroded_samples"]
            ),
            "boundary_exclusion_attempted_retained_fraction": float(
                boundary_support["attempted_retained_fraction"]
            ),
            "boundary_exclusion_retained_fraction": float(
                selected_part_samples / max(visible_part_samples, 1)
            ),
            "minimum_eroded_support_ratio": (
                FACE_PART_MINIMUM_ERODED_SUPPORT_RATIO
            ),
        }
        if reference_samples < int(minimum_part_samples):
            record["reason"] = "insufficient_reference_samples"
            stats["parts"].append(record)
            continue
        if samples < int(minimum_part_samples):
            record["reason"] = "insufficient_candidate_samples"
            stats["parts"].append(record)
            continue

        errors = candidate[measured] - expected_candidate[measured]
        expected_values = expected_candidate[measured]
        candidate_values_part = candidate[measured]
        expected_p05, expected_p95 = np.percentile(expected_values, (5.0, 95.0))
        candidate_p05, candidate_p95 = np.percentile(
            candidate_values_part,
            (5.0, 95.0),
        )
        expected_span = float(expected_p95 - expected_p05)
        candidate_span = float(candidate_p95 - candidate_p05)
        span_retention = _retention(expected_span, candidate_span)
        rmse_mm = float(np.sqrt(np.mean(np.square(errors))))
        bias_mm = float(np.mean(errors))
        p95_mm = float(np.percentile(np.abs(errors), 95.0))
        checks = {
            "coverage": coverage >= effective_gates["minimum_coverage_ratio"],
            "rmse": rmse_mm <= effective_gates["maximum_rmse_mm"],
            "p95_absolute_error": p95_mm
            <= effective_gates["maximum_p95_absolute_error_mm"],
            "bias": abs(bias_mm) <= effective_gates["maximum_absolute_bias_mm"],
            "span_retention": bool(
                span_retention is not None
                and effective_gates["minimum_span_retention"]
                <= span_retention
                <= effective_gates["maximum_span_retention"]
            ),
        }
        record.update(
            {
                "available": True,
                "rmse_mm": rmse_mm,
                "bias_mm": bias_mm,
                "p95_absolute_error_mm": p95_mm,
                "expected_span_p05_p95_mm": expected_span,
                "candidate_span_p05_p95_mm": candidate_span,
                "span_retention": span_retention,
                "checks": {**checks, "passed": bool(all(checks.values()))},
                "passed": bool(all(checks.values())),
            }
        )
        stats["parts"].append(record)

    measured_parts = [record for record in stats["parts"] if record["available"]]
    fit_scale_passed = bool(
        effective_gates["minimum_face_affine_scale"]
        <= float(scale)
        <= effective_gates["maximum_face_affine_scale"]
    )
    stats.update(
        {
            "available": bool(measured_parts),
            "part_count": len(stats["parts"]),
            "measured_part_count": len(measured_parts),
            "unavailable_part_count": len(stats["parts"]) - len(measured_parts),
            "failed_parts": [
                record["name"] for record in stats["parts"] if not record["passed"]
            ],
            "face_affine_scale_passed": fit_scale_passed,
        }
    )
    stats["passed"] = bool(
        face_coverage >= effective_gates["minimum_coverage_ratio"]
        and fit_scale_passed
        and len(measured_parts) == len(stats["parts"])
        and all(record["passed"] for record in stats["parts"])
    )
    return stats


def load_persisted_face_part_masks(
    metadata_path: str | Path,
    target_shape: tuple[int, int] | None = None,
    *,
    flip_horizontal: bool = False,
    surface_grid_transform: dict | None = None,
) -> list[dict]:
    """Load fixed per-face masks emitted by face depth refinement."""
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = []
    for face in metadata.get("faces", []):
        part_metadata = face.get("part_masks", {})
        if not part_metadata.get("complete", False):
            continue
        face_path = part_metadata.get("face_file")
        files = part_metadata.get("files", {})
        if not face_path or not files:
            continue
        file_names = set(files)
        required_names = set(FACE_PART_NAMES)
        if file_names != required_names:
            missing = sorted(required_names - file_names)
            unexpected = sorted(file_names - required_names)
            raise ValueError(
                "Complete face-part metadata must contain exactly the required "
                f"parts; missing={missing}, unexpected={unexpected}"
            )

        def resize_mask(mask: np.ndarray, shape) -> np.ndarray:
            shape = (int(shape[0]), int(shape[1]))
            if mask.shape == shape:
                return mask
            factors = (shape[0] / mask.shape[0], shape[1] / mask.shape[1])
            resized = zoom(mask.astype(np.uint8), factors, order=0) > 0
            resized = resized[: shape[0], : shape[1]]
            pad_height = shape[0] - resized.shape[0]
            pad_width = shape[1] - resized.shape[1]
            if pad_height > 0 or pad_width > 0:
                resized = np.pad(
                    resized,
                    ((0, max(0, pad_height)), (0, max(0, pad_width))),
                    mode="edge",
                )
            return resized

        def load_mask(relative_path: str) -> np.ndarray:
            image = Image.open(metadata_path.parent / relative_path).convert("L")
            mask = np.asarray(image) > 0
            if surface_grid_transform is not None:
                transform = surface_grid_transform
                required_transform_fields = {
                    "input_depth_shape",
                    "target_depth_shape",
                    "mesh_shape_before_crop",
                    "crop_bbox_rc",
                    "emitted_shape",
                }
                missing_fields = sorted(required_transform_fields - set(transform))
                if missing_fields:
                    raise ValueError(
                        "Surface-grid transform is missing required fields: "
                        + ", ".join(missing_fields)
                    )
                input_shape = tuple(
                    int(value) for value in transform["input_depth_shape"]
                )
                if mask.shape != input_shape:
                    raise ValueError(
                        f"Persisted face-part mask has shape {mask.shape}, "
                        f"expected recorded input depth shape {input_shape}"
                    )
                mask = resize_mask(mask, transform["target_depth_shape"])
                if bool(transform.get("flip_x", False)):
                    mask = np.flip(mask, axis=1)
                mask = resize_mask(mask, transform["mesh_shape_before_crop"])
                top, left, bottom, right = (
                    int(value) for value in transform["crop_bbox_rc"]
                )
                mask = mask[top:bottom, left:right]
                emitted_shape = tuple(int(value) for value in transform["emitted_shape"])
                if mask.shape != emitted_shape:
                    raise ValueError(
                        f"Transformed face-part mask has shape {mask.shape}, "
                        f"expected {emitted_shape}"
                    )
                return mask
            if target_shape is None:
                raise ValueError("target_shape or surface_grid_transform is required")
            mask = resize_mask(mask, target_shape)
            return np.flip(mask, axis=1) if flip_horizontal else mask

        records.append(
            {
                "face_index": int(face.get("index", len(records))),
                "detector": face.get("detector"),
                "landmark_count": int(face.get("landmark_count") or 0),
                "face_mask": load_mask(face_path),
                "part_masks": {
                    name: load_mask(relative_path)
                    for name, relative_path in sorted(files.items())
                },
            }
        )
    return records
