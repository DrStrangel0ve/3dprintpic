"""Check that protected photo detail does not carve a moat around a subject."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

from backend.benchmark.run_background_photo_detail_provider_smoke import (
    _detail_telemetry_checks,
)
from backend.benchmark.run_background_photo_detail_sweep import (
    _detail_metrics,
    _photo_detail_signal,
    _resize_signal,
)
from backend.benchmark.run_makehuman_face_depth_smoke import _correlation
from backend.benchmark.run_private_background_photo_detail_replay import (
    _transform_mask_to_surface_grid,
)
from backend.benchmark.run_relief_scene_regression import _git_provenance, _mesh_topology
from backend.benchmark.run_relief_visual_sweep import _stl_heightfield_agreement
from backend.pic_to_3d import (
    BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM,
    BACKGROUND_PHOTO_DETAIL_ZERO_GUARD_MM,
    _resize_binary_mask,
    compose_selection_depth_with_context,
    depth_data_to_3d_model,
)


DETAIL_LEVELS_MM = (0.0, 0.60)
RELIEF_HEIGHT_MM = 30.0
PHYSICAL_SIZE_MM = 128.0
BACKGROUND_DEPTH_RATIO = 0.65
RECOVERY_OUTER_MM = 12.0
HALO_GATES = {
    "minimum_finite_coverage_ratio": 1.0,
    # Gains are relative to an unattenuated, production-normalized 0.60 mm template.
    "minimum_calibration_gain": 0.70,
    "maximum_calibration_gain": 1.05,
    "minimum_relative_gain_5_8mm": 0.70,
    "maximum_relative_gain_5_8mm": 1.10,
    "minimum_relative_gain_8_12mm": 0.90,
    "maximum_relative_gain_8_12mm": 1.10,
    "minimum_source_correlation_5_8mm": 0.75,
    "minimum_local_window_relative_gain": 0.50,
    # A structural ring below half an 0.08 mm print layer is treated as invisible.
    "maximum_lowpass_sector_bin_mean_mm": 0.04,
    "maximum_lowpass_absolute_p99_mm": 0.06,
    "maximum_adjacent_radial_bin_mean_change_mm": 0.04,
    "maximum_new_radial_gradient_reversal_ratio": 0.01,
}
PROVENANCE_PATHS = (
    "backend/pic_to_3d.py",
    "backend/benchmark/run_background_photo_detail_provider_smoke.py",
    "backend/benchmark/run_background_photo_detail_sweep.py",
    "backend/benchmark/run_background_halo_continuity_smoke.py",
    "backend/benchmark/run_private_background_photo_detail_replay.py",
    "backend/benchmark/run_relief_scene_regression.py",
    "backend/benchmark/run_relief_visual_sweep.py",
)


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _halo_telemetry_checks(stats: dict) -> dict:
    checks = {
        "method": stats.get("protection_method")
        == "euclidean_inner_guard_smoothstep_to_full_gain",
        "halo_mm": abs(
            float(stats.get("protection_halo_mm", -1.0))
            - BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM
        )
        <= 1e-9,
        "zero_guard_mm": abs(
            float(stats.get("protection_zero_guard_mm", -1.0))
            - BACKGROUND_PHOTO_DETAIL_ZERO_GUARD_MM
        )
        <= 1e-9,
    }
    return {**checks, "passed": bool(all(checks.values()))}


def _structured_scene(shape: tuple[int, int] = (384, 512)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic vase-on-shelf scene with textured architecture."""
    height, width = (int(shape[0]), int(shape[1]))
    rows, cols = np.indices((height, width), dtype=np.float32)
    x = 2.0 * cols / max(width - 1, 1) - 1.0
    y = 2.0 * rows / max(height - 1, 1) - 1.0

    depth = 0.18 + 0.035 * (x + 1.0) / 2.0
    depth += 0.018 * np.sin(2.6 * np.pi * x) * np.cos(1.4 * np.pi * y)

    window = (x > 0.18) & (x < 0.82) & (y > -0.78) & (y < 0.15)
    frame = window & (
        (np.abs(x - 0.18) < 0.035)
        | (np.abs(x - 0.82) < 0.035)
        | (np.abs(y + 0.78) < 0.04)
        | (np.abs(y - 0.15) < 0.04)
        | (np.abs(x - 0.50) < 0.025)
        | (np.abs(y + 0.31) < 0.025)
    )
    depth[window] = 0.12 + 0.015 * np.sin(7.0 * x[window])
    depth[frame] = 0.36

    shelf = (y > 0.35) & (y < 0.52)
    shelf_edge = (y >= 0.35) & (y < 0.39)
    depth[shelf] = 0.39 + 0.025 * np.sin(18.0 * x[shelf])
    depth[shelf_edge] = 0.46

    vase_body = ((x + 0.18) / 0.235) ** 2 + ((y - 0.08) / 0.43) ** 2 <= 1.0
    vase_body &= y > -0.28
    vase_neck = (np.abs(x + 0.18) < 0.095) & (y >= -0.47) & (y <= -0.20)
    vase_lip = ((x + 0.18) / 0.13) ** 2 + ((y + 0.47) / 0.055) ** 2 <= 1.0
    selection = vase_body | vase_neck | vase_lip
    bulge = np.clip(
        1.0 - ((x + 0.18) / 0.24) ** 2 - ((y - 0.05) / 0.46) ** 2,
        0.0,
        1.0,
    )
    depth[selection] = 0.60 + 0.22 * np.sqrt(bulge[selection])

    rgb = np.empty((height, width, 3), dtype=np.float32)
    wall_texture = 0.055 * np.sin(cols * 0.34) + 0.035 * np.cos(rows * 0.27)
    rgb[..., 0] = 0.39 + wall_texture
    rgb[..., 1] = 0.50 + 0.65 * wall_texture
    rgb[..., 2] = 0.46 + 0.45 * wall_texture
    panel_seam = (np.mod(cols + 9, 74) < 3) | (np.mod(rows + 17, 92) < 2)
    rgb[panel_seam] *= np.asarray((0.72, 0.78, 0.76), dtype=np.float32)

    glass_texture = 0.055 * np.sin(cols * 0.51) * np.cos(rows * 0.19)
    rgb[window, 0] = 0.36 + glass_texture[window]
    rgb[window, 1] = 0.55 + glass_texture[window]
    rgb[window, 2] = 0.63 + glass_texture[window]
    rgb[frame] = np.asarray((0.84, 0.82, 0.74), dtype=np.float32)

    wood = 0.055 * np.sin(cols * 0.43) + 0.025 * np.sin(cols * 1.17)
    rgb[shelf, 0] = 0.47 + wood[shelf]
    rgb[shelf, 1] = 0.30 + 0.55 * wood[shelf]
    rgb[shelf, 2] = 0.20 + 0.35 * wood[shelf]
    rgb[shelf_edge] *= np.asarray((0.78, 0.78, 0.78), dtype=np.float32)

    vase_light = 0.18 * np.clip(1.0 - np.abs(x + 0.25) / 0.25, 0.0, 1.0)
    vase_texture = 0.025 * np.sin(rows * 0.48)
    rgb[selection, 0] = 0.15 + vase_light[selection] + vase_texture[selection]
    rgb[selection, 1] = 0.35 + vase_light[selection] + vase_texture[selection]
    rgb[selection, 2] = 0.58 + vase_light[selection]

    source = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return depth.astype(np.float32), source, selection


def _photo_detail_template(
    source_path: Path,
    transform: dict,
    *,
    selection_mask: np.ndarray,
    physical_size_mm: float,
    requested_detail_mm: float,
) -> np.ndarray:
    """Recreate the unattenuated production-normalized detail on the STL grid."""
    target_shape = tuple(int(value) for value in transform["target_depth_shape"])
    raw = _photo_detail_signal(source_path, target_shape)
    selection = _resize_binary_mask(selection_mask, target_shape)
    pitch_mm = float(physical_size_mm) / max(max(target_shape) - 1, 1)
    calibration_support = (
        distance_transform_edt(~selection) * pitch_mm > 12.0
    )
    scale_values = np.abs(raw[calibration_support & np.isfinite(raw)])
    scale = float(np.percentile(scale_values, 96.0)) if scale_values.size else 0.0
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Photo-detail template requires a finite positive scale")
    normalized = np.clip(raw / scale, -1.0, 1.0)
    normalized = np.sign(normalized) * np.power(np.abs(normalized), 0.75)
    template = float(requested_detail_mm) * normalized
    template = _resize_signal(template, tuple(transform["mesh_shape_before_crop"]))
    top, left, bottom, right = (int(value) for value in transform["crop_bbox_rc"])
    template = template[top:bottom, left:right]
    emitted_shape = tuple(int(value) for value in transform["emitted_shape"])
    if template.shape != emitted_shape:
        raise ValueError("Photo-detail template does not match the emitted surface")
    return template.astype(np.float64, copy=False)


def halo_continuity_metrics(
    baseline_surface: np.ndarray,
    candidate_surface: np.ndarray,
    protection_mask: np.ndarray,
    source_template_mm: np.ndarray,
    *,
    sample_pitch_mm: float,
    halo_mm: float = BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM,
    recovery_outer_mm: float = RECOVERY_OUTER_MM,
) -> dict:
    baseline = np.asarray(baseline_surface, dtype=np.float64)
    candidate = np.asarray(candidate_surface, dtype=np.float64)
    protected = np.asarray(protection_mask, dtype=bool)
    template = np.asarray(source_template_mm, dtype=np.float64)
    if not (
        baseline.shape == candidate.shape == protected.shape == template.shape
        and baseline.ndim == 2
    ):
        raise ValueError("Halo metrics require one shared 2D surface grid")
    pitch = float(sample_pitch_mm)
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("Halo metrics require a positive sample pitch")
    if not 0 < float(halo_mm) < float(recovery_outer_mm):
        raise ValueError("Halo metric distances are invalid")

    reference_valid = np.isfinite(baseline) & np.isfinite(template)
    valid = reference_valid & np.isfinite(candidate)
    distance_mm = distance_transform_edt(~protected) * pitch
    radial_zone = valid & (distance_mm > 0) & (distance_mm <= recovery_outer_mm)
    recovery = valid & (distance_mm > halo_mm) & (distance_mm <= recovery_outer_mm)
    if np.count_nonzero(radial_zone) < 128 or np.count_nonzero(recovery) < 128:
        raise ValueError("Halo metric bands do not contain enough samples")

    candidate_filled = np.where(valid, candidate, baseline)
    delta = candidate_filled - baseline
    lowpass_sigma_px = max(1.0, 2.0 / pitch)
    lowpass_delta = gaussian_filter(delta, sigma=lowpass_sigma_px)
    lowpass_baseline = gaussian_filter(baseline, sigma=lowpass_sigma_px)
    lowpass_candidate = gaussian_filter(candidate_filled, sigma=lowpass_sigma_px)

    active_template = np.abs(template) >= 0.024

    def band_stats(inner: float, outer: float) -> dict:
        reference_band = (
            reference_valid & (distance_mm > inner) & (distance_mm <= outer)
        )
        band = valid & (distance_mm > inner) & (distance_mm <= outer)
        reference_active = reference_band & active_template
        active = band & active_template
        denominator = float(
            np.dot(template[reference_active], template[reference_active])
        )
        gain = (
            float(np.dot(delta[active], template[active]) / denominator)
            if denominator > 1e-12
            else 0.0
        )
        correlation = (
            _correlation(template[active], delta[active])
            if np.count_nonzero(active) >= 64
            else 0.0
        )
        return {
            "inner_mm": float(inner),
            "outer_mm": float(outer),
            "reference_samples": int(np.count_nonzero(reference_band)),
            "finite_samples": int(np.count_nonzero(band)),
            "active_template_samples": int(np.count_nonzero(active)),
            "reference_active_template_samples": int(
                np.count_nonzero(reference_active)
            ),
            "finite_coverage_ratio": float(
                np.count_nonzero(band) / max(int(np.count_nonzero(reference_band)), 1)
            ),
            "source_aligned_gain": gain,
            "source_correlation": float(correlation),
        }

    gain_5_8 = band_stats(5.0, 8.0)
    gain_8_12 = band_stats(8.0, 12.0)
    calibration = band_stats(12.0, 20.0)
    calibration_gain = calibration["source_aligned_gain"]
    relative_gain_5_8 = gain_5_8["source_aligned_gain"] / max(calibration_gain, 1e-8)
    relative_gain_8_12 = gain_8_12["source_aligned_gain"] / max(calibration_gain, 1e-8)

    centroid = np.mean(np.argwhere(protected), axis=0)
    rows, cols = np.indices(protected.shape, dtype=np.float64)
    angles = np.mod(np.arctan2(rows - centroid[0], cols - centroid[1]), 2.0 * np.pi)
    radial_bin_records = []
    sector_bin_means = []
    radial_bin_means = []
    for radial_index in range(10):
        inner = float(radial_index)
        outer = float(radial_index + 1)
        radial_bin = valid & (distance_mm > inner) & (distance_mm <= outer)
        radial_values = lowpass_delta[radial_bin]
        radial_mean = float(np.mean(radial_values)) if radial_values.size else 0.0
        radial_bin_means.append(radial_mean)
        sectors = []
        for sector_index in range(12):
            angle_low = sector_index * 2.0 * np.pi / 12.0
            angle_high = (sector_index + 1) * 2.0 * np.pi / 12.0
            sector = radial_bin & (angles >= angle_low) & (angles < angle_high)
            values = lowpass_delta[sector]
            if values.size < 12:
                continue
            mean = float(np.mean(values))
            sector_bin_means.append(mean)
            sectors.append(
                {
                    "sector": int(sector_index),
                    "samples": int(values.size),
                    "mean_delta_mm": mean,
                }
            )
        radial_bin_records.append(
            {
                "inner_mm": inner,
                "outer_mm": outer,
                "samples": int(radial_values.size),
                "mean_delta_mm": radial_mean,
                "sectors": sectors,
            }
        )
    maximum_sector_bin_mean = max(
        (abs(value) for value in sector_bin_means), default=0.0
    )
    maximum_adjacent_bin_change = max(
        (
            abs(right - left)
            for left, right in zip(radial_bin_means[:-1], radial_bin_means[1:])
        ),
        default=0.0,
    )
    lowpass_absolute_p99 = float(
        np.percentile(np.abs(lowpass_delta[radial_zone]), 99.0)
    )

    distance_y, distance_x = np.gradient(distance_mm)
    distance_norm = np.hypot(distance_x, distance_y)
    normal_x = np.divide(
        distance_x,
        distance_norm,
        out=np.zeros_like(distance_x),
        where=distance_norm > 1e-8,
    )
    normal_y = np.divide(
        distance_y,
        distance_norm,
        out=np.zeros_like(distance_y),
        where=distance_norm > 1e-8,
    )

    def radial_gradient(values: np.ndarray) -> np.ndarray:
        grad_y, grad_x = np.gradient(values, pitch, pitch)
        return grad_x * normal_x + grad_y * normal_y

    baseline_radial_gradient = radial_gradient(lowpass_baseline)
    candidate_radial_gradient = radial_gradient(lowpass_candidate)
    reversal_eligible = radial_zone & (np.abs(baseline_radial_gradient) >= 0.02)
    new_reversals = reversal_eligible & (
        baseline_radial_gradient * candidate_radial_gradient < 0
    ) & (np.abs(candidate_radial_gradient) >= 0.02)
    reversal_ratio = float(
        np.count_nonzero(new_reversals)
        / max(int(np.count_nonzero(reversal_eligible)), 1)
    )

    window_px = max(3, int(round(6.0 / pitch)))
    stride_px = max(1, int(round(3.0 / pitch)))
    local_window_gains = []
    for row in range(0, baseline.shape[0] - window_px + 1, stride_px):
        for col in range(0, baseline.shape[1] - window_px + 1, stride_px):
            window = np.zeros_like(valid)
            window[row : row + window_px, col : col + window_px] = True
            active = window & recovery & active_template
            if np.count_nonzero(active) < 32:
                continue
            denominator = float(np.dot(template[active], template[active]))
            if denominator <= 1e-12:
                continue
            gain = float(np.dot(delta[active], template[active]) / denominator)
            local_window_gains.append(gain / max(calibration_gain, 1e-8))
    minimum_local_window_gain = min(local_window_gains, default=0.0)

    finite_coverages = [
        gain_5_8["finite_coverage_ratio"],
        gain_8_12["finite_coverage_ratio"],
        calibration["finite_coverage_ratio"],
    ]

    checks = {
        "finite_coverage": min(finite_coverages)
        >= HALO_GATES["minimum_finite_coverage_ratio"],
        "calibration_gain": HALO_GATES["minimum_calibration_gain"]
        <= calibration_gain
        <= HALO_GATES["maximum_calibration_gain"],
        "relative_gain_5_8mm": HALO_GATES["minimum_relative_gain_5_8mm"]
        <= relative_gain_5_8
        <= HALO_GATES["maximum_relative_gain_5_8mm"],
        "relative_gain_8_12mm": HALO_GATES["minimum_relative_gain_8_12mm"]
        <= relative_gain_8_12
        <= HALO_GATES["maximum_relative_gain_8_12mm"],
        "source_correlation_5_8mm": gain_5_8["source_correlation"]
        >= HALO_GATES["minimum_source_correlation_5_8mm"],
        "local_window_gain": minimum_local_window_gain
        >= HALO_GATES["minimum_local_window_relative_gain"],
        "sector_bin_mean": maximum_sector_bin_mean
        <= HALO_GATES["maximum_lowpass_sector_bin_mean_mm"],
        "lowpass_amplitude": lowpass_absolute_p99
        <= HALO_GATES["maximum_lowpass_absolute_p99_mm"],
        "adjacent_radial_bin_change": maximum_adjacent_bin_change
        <= HALO_GATES["maximum_adjacent_radial_bin_mean_change_mm"],
        "gradient_reversal": reversal_ratio
        <= HALO_GATES["maximum_new_radial_gradient_reversal_ratio"],
    }
    return {
        "halo_mm": float(halo_mm),
        "recovery_outer_mm": float(recovery_outer_mm),
        "lowpass_sigma_mm": float(lowpass_sigma_px * pitch),
        "gain_bands": {
            "5_8mm": gain_5_8,
            "8_12mm": gain_8_12,
            "12_20mm_calibration": calibration,
        },
        "relative_gain_5_8mm": float(relative_gain_5_8),
        "relative_gain_8_12mm": float(relative_gain_8_12),
        "radial_bins": radial_bin_records,
        "maximum_lowpass_sector_bin_mean_mm": float(maximum_sector_bin_mean),
        "lowpass_absolute_p99_mm": lowpass_absolute_p99,
        "maximum_adjacent_radial_bin_mean_change_mm": float(
            maximum_adjacent_bin_change
        ),
        "radial_gradient_reversal_eligible_samples": int(
            np.count_nonzero(reversal_eligible)
        ),
        "new_radial_gradient_reversal_ratio": reversal_ratio,
        "local_window_mm": 6.0,
        "local_window_stride_mm": 3.0,
        "measured_local_windows": int(len(local_window_gains)),
        "minimum_local_window_relative_gain": float(minimum_local_window_gain),
        "checks": {**checks, "passed": bool(all(checks.values()))},
    }


def _compact_cap(cap: dict) -> dict:
    return {
        "strict_passed": bool(cap.get("passed", False)),
        "emission_passed": bool(cap.get("emission_passed", False)),
        "far_background_cap_passed": bool(cap.get("far_background_cap_passed", False)),
        "feasible_attachment_constraints_passed": bool(
            cap.get("feasible_attachment_constraints_passed", False)
        ),
        "far_background_cap_violation_mm": _finite(
            cap.get("far_background_cap_violation_mm")
        ),
        "feasible_attachment_jump_max_mm": _finite(
            cap.get("feasible_attachment_jump_max_mm")
        ),
        "attachment_constraint_conflicts": int(
            cap.get("attachment_constraint_conflicts", 0)
        ),
    }


def _compact_shell(shell: dict) -> dict:
    return {
        key: shell.get(key)
        for key in (
            "passed",
            "complete_shell_verified",
            "facet_geometry_passed",
            "coverage_ratio",
            "max_abs_error_mm",
            "rms_error_mm",
            "expected_shell_triangle_count",
            "actual_shell_triangle_count",
            "invalid_shell_triangle_count",
        )
    }


def run(
    output_dir: str | Path,
    *,
    allow_dirty: bool = False,
    allow_failures: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _git_provenance(PROVENANCE_PATHS)
    depth, source, selection = _structured_scene()
    source_path = output_dir / "source.png"
    selection_path = output_dir / "selection.png"
    Image.fromarray(source).save(source_path)
    Image.fromarray(selection.astype(np.uint8) * 255).save(selection_path)

    input_pitch_mm = PHYSICAL_SIZE_MM / max(max(depth.shape) - 1, 1)
    composed, compose_stats = compose_selection_depth_with_context(
        depth,
        selection,
        value_transform="linear",
        relief_height_mm=RELIEF_HEIGHT_MM,
        sample_pitch_mm=input_pitch_mm,
        max_slope_mm_per_mm=2.0,
        background_depth_ratio=BACKGROUND_DEPTH_RATIO,
        background_feather_mm=1.5,
        background_smoothing_mm=0.6,
    )
    composed_path = output_dir / "composed_depth.npy"
    np.save(composed_path, composed.astype(np.float32, copy=False))

    emitted = []
    for detail_mm in DETAIL_LEVELS_MM:
        started = time.perf_counter()
        row_dir = output_dir / f"detail_{detail_mm:.2f}".replace(".", "p")
        row_dir.mkdir(parents=True, exist_ok=True)
        stl_path = row_dir / "relief.stl"
        surface_path = row_dir / "surface.npy"
        reference_path = row_dir / "reference_surface.npy"
        postprocess = depth_data_to_3d_model(
            composed_path,
            output_stl_path=str(stl_path),
            target_dimension=512,
            z_scale=RELIEF_HEIGHT_MM,
            invert=False,
            sigma=0.35,
            max_xy_size=PHYSICAL_SIZE_MM,
            relief_gamma=0.75,
            detail_boost=0.8,
            background_detail_boost=2.4,
            low_percentile=1.0,
            high_percentile=99.0,
            base_border_px=2,
            value_transform="linear",
            minimum_feature_mm=0.8,
            max_relief_slope=2.0,
            selection_region_mask=selection_path,
            selection_background_depth_ratio=BACKGROUND_DEPTH_RATIO,
            source_image=source_path,
            background_photo_detail_mm=detail_mm,
            trim_top_background=False,
            printable_feature_depth_mm=0.8,
            feature_bridge_depth_mm=0.8,
            surface_output_path=surface_path,
            reference_surface_output_path=reference_path,
        )
        (row_dir / "postprocess.json").write_text(
            json.dumps(postprocess, indent=2), encoding="utf-8"
        )
        surface = np.load(surface_path)
        transform = postprocess["surface_grid_transform"]
        transformed_selection = _transform_mask_to_surface_grid(selection, transform)
        topology = _mesh_topology(trimesh.load_mesh(stl_path, process=True))
        shell = _stl_heightfield_agreement(
            stl_path,
            surface_path,
            expected_max_xy_size_mm=PHYSICAL_SIZE_MM,
        )
        emitted.append(
            {
                "detail_mm": detail_mm,
                "runtime_seconds": float(time.perf_counter() - started),
                "surface": surface,
                "selection": transformed_selection,
                "postprocess": postprocess,
                "topology": topology,
                "shell": shell,
            }
        )

    baseline, candidate = emitted
    transform = baseline["postprocess"]["surface_grid_transform"]
    transform_match = transform == candidate["postprocess"]["surface_grid_transform"]
    selection_match = np.array_equal(baseline["selection"], candidate["selection"])
    pitch_mm = float(baseline["postprocess"]["mesh_sample_pitch_mm"])
    source_detail = _photo_detail_signal(
        source_path,
        baseline["surface"].shape,
        surface_grid_transform=transform,
    )
    detail_metrics = _detail_metrics(
        baseline["surface"],
        candidate["surface"],
        baseline["selection"],
        source_path,
        0.60,
        protection_halo_px=BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM / pitch_mm,
        surface_grid_transform=transform,
    )
    halo_metrics = halo_continuity_metrics(
        baseline["surface"],
        candidate["surface"],
        baseline["selection"],
        _photo_detail_template(
            source_path,
            transform,
            selection_mask=selection,
            physical_size_mm=PHYSICAL_SIZE_MM,
            requested_detail_mm=0.60,
        ),
        sample_pitch_mm=pitch_mm,
    )
    baseline_telemetry = _detail_telemetry_checks(
        baseline["postprocess"]["background_photo_detail"], 0.0
    )
    candidate_telemetry = _detail_telemetry_checks(
        candidate["postprocess"]["background_photo_detail"], 0.60
    )
    halo_telemetry = _halo_telemetry_checks(
        candidate["postprocess"]["background_photo_detail"]
    )
    checks = {
        "implementation_provenance_clean": bool(
            provenance.get("available") and provenance.get("clean")
        ),
        "surface_grid_match": bool(transform_match),
        "selection_mask_match": bool(selection_match),
        "detail_metrics": bool(detail_metrics["checks"]["passed"]),
        "halo_continuity": bool(halo_metrics["checks"]["passed"]),
        "baseline_telemetry": bool(baseline_telemetry["passed"]),
        "candidate_telemetry": bool(candidate_telemetry["passed"]),
        "candidate_halo_telemetry": bool(halo_telemetry["passed"]),
        "baseline_background_preservation": bool(
            baseline["postprocess"]["background_depth_preservation"].get(
                "passed", False
            )
        ),
        "candidate_background_preservation": bool(
            candidate["postprocess"]["background_depth_preservation"].get(
                "passed", False
            )
        ),
        "baseline_physical_cap": bool(
            baseline["postprocess"]["selection_background_physical_cap"].get(
                "emission_passed", False
            )
        ),
        "candidate_physical_cap": bool(
            candidate["postprocess"]["selection_background_physical_cap"].get(
                "emission_passed", False
            )
        ),
        "baseline_printable": bool(baseline["topology"]["printable"]),
        "candidate_printable": bool(candidate["topology"]["printable"]),
        "baseline_shell": bool(baseline["shell"]["passed"]),
        "candidate_shell": bool(candidate["shell"]["passed"]),
    }
    summary = {
        "schema_version": 1,
        "run_kind": "privacy_safe_background_halo_continuity_smoke",
        "privacy": "deterministic analytic non-face scene; no personal input",
        "implementation_provenance": provenance,
        "research_basis": {
            "finding": (
                "halo artifacts are treated as low-frequency overshoot or gradient "
                "reversal near an edge, while overprotection is treated as missing "
                "source-aligned detail immediately outside the declared halo"
            ),
            "sources": [
                "https://proceedings.iclr.cc/paper_files/paper/2025/hash/22f5d8e689d2a011cd8ead552ed59052-Abstract-Conference.html",
                "https://cg.cs.tsinghua.edu.cn/papers/TIP_2013_Edge-Aware.pdf",
            ],
        },
        "matrix": {
            "scene": "analytic_vase_shelf_window_v1",
            "detail_levels_mm": list(DETAIL_LEVELS_MM),
            "relief_height_mm": RELIEF_HEIGHT_MM,
            "physical_size_mm": PHYSICAL_SIZE_MM,
            "background_depth_ratio": BACKGROUND_DEPTH_RATIO,
            "protection_halo_mm": BACKGROUND_PHOTO_DETAIL_PROTECTION_HALO_MM,
            "protection_zero_guard_mm": BACKGROUND_PHOTO_DETAIL_ZERO_GUARD_MM,
            "recovery_outer_mm": RECOVERY_OUTER_MM,
        },
        "gates": HALO_GATES,
        "compose": {
            "background_context_enabled": bool(
                compose_stats.get("background_context_enabled", False)
            ),
            "background_output_span_ratio": _finite(
                compose_stats.get("background_output_span_ratio")
            ),
            "background_context_correlation": _finite(
                compose_stats.get("background_context_correlation")
            ),
        },
        "detail_metrics": detail_metrics,
        "halo_metrics": halo_metrics,
        "telemetry_checks": {
            "baseline": baseline_telemetry,
            "candidate": candidate_telemetry,
            "candidate_halo": halo_telemetry,
        },
        "rows": [
            {
                "detail_mm": row["detail_mm"],
                "runtime_seconds": row["runtime_seconds"],
                "topology": row["topology"],
                "shell": _compact_shell(row["shell"]),
                "physical_cap": _compact_cap(
                    row["postprocess"]["selection_background_physical_cap"]
                ),
                "background_preservation": {
                    key: row["postprocess"]["background_depth_preservation"].get(key)
                    for key in (
                        "passed",
                        "correlation",
                        "rms_retention",
                        "gradient_correlation",
                        "candidate_coverage_ratio",
                    )
                },
            }
            for row in emitted
        ],
        "checks": {
            **checks,
            "passed": bool(
                all(value for key, value in checks.items() if key != "implementation_provenance_clean")
                and (allow_dirty or checks["implementation_provenance_clean"])
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not summary["checks"]["passed"] and not allow_failures:
        raise RuntimeError("Background halo-continuity smoke failed its gates")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.output_dir,
        allow_dirty=args.allow_dirty,
        allow_failures=args.allow_failures,
    )
    print(json.dumps(summary["checks"], indent=2))
    return 0 if summary["checks"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
