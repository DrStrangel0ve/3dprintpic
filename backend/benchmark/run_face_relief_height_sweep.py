"""Compare legacy face attachment with screened gradient reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    label,
    laplace,
)

from backend.pic_to_3d import (
    _align_stabilized_head_to_reference_boundary,
    _compress_relief_gradients,
    _expand_face_region_to_depth_connected_head,
    _face_translation_core_mask,
    _limit_positive_relief_slope,
    _stabilize_face_relief_height,
)


HEIGHTS_MM = (12.0, 20.0, 30.0, 40.0, 50.0)
SLOPE_BUDGET_MM_PER_MM = 2.0


def _synthetic_portrait():
    shape = (121, 121)
    rows, cols = np.indices(shape, dtype=np.float64)
    x = (cols - 60.0) / 24.0
    y = (rows - 44.0) / 30.0
    face_mask = np.square(x) + np.square(y) <= 1.0
    face_core = np.square(x / 0.72) + np.square(y / 0.80) <= 1.0

    neck_half_width = 11.0 - 0.10 * np.clip(rows - 69.0, 0.0, 22.0)
    neck_mask = (
        (rows >= 66.0)
        & (rows <= 92.0)
        & (np.abs(cols - 60.0) <= neck_half_width)
    )
    torso_mask = (
        np.square((cols - 60.0) / 47.0)
        + np.square((rows - 113.0) / 38.0)
        <= 1.0
    )

    relief = 0.08 + 0.0008 * cols + 0.00035 * np.square(rows - 57.0) / 121.0
    relief[torso_mask] = 0.36 + 0.0009 * cols[torso_mask]
    relief[neck_mask] = 0.49 + 0.0006 * cols[neck_mask]

    facial_shape = 0.040 * np.exp(-(np.square(x / 0.72) + np.square(y / 0.88)))
    facial_shape += 0.077 * np.exp(
        -(np.square(x / 0.17) + np.square((y - 0.02) / 0.28))
    )
    facial_shape -= 0.014 * np.exp(
        -(np.square((x - 0.30) / 0.13) + np.square((y + 0.20) / 0.08))
    )
    facial_shape -= 0.014 * np.exp(
        -(np.square((x + 0.30) / 0.13) + np.square((y + 0.20) / 0.08))
    )
    facial_shape += 0.013 * np.exp(
        -(np.square(x / 0.29) + np.square((y - 0.44) / 0.07))
    )
    relief[face_mask] = 0.58 + facial_shape[face_mask]
    return relief.astype(np.float32), face_mask, face_core, neck_mask, torso_mask


def _legacy_expand_head(face_region_mask, reference_values, depth_tolerance_mm=4.0):
    face_region = np.asarray(face_region_mask, dtype=bool) & np.isfinite(reference_values)
    components, component_count = label(face_region)
    rows, cols = np.indices(face_region.shape, dtype=np.float32)
    head_region = face_region.copy()
    for component_index in range(1, component_count + 1):
        component = components == component_index
        component_rows, component_cols = np.where(component)
        if component_rows.size < 16:
            continue
        width = float(component_cols.max() - component_cols.min() + 1)
        height = float(component_rows.max() - component_rows.min() + 1)
        center_col = float(component_cols.min() + component_cols.max()) * 0.5
        center_row = float(component_rows.min() + component_rows.max()) * 0.5 - 0.12 * height
        ellipse = (
            np.square((cols - center_col) / max(2.0, 0.60 * width))
            + np.square((rows - center_row) / max(2.0, 0.67 * height))
            <= 1.0
        )
        level = float(np.median(reference_values[component]))
        candidate = (
            ellipse
            & np.isfinite(reference_values)
            & (np.abs(reference_values - level) <= float(depth_tolerance_mm))
        )
        candidate_labels, _ = label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
        connected_ids = np.unique(candidate_labels[component])
        connected_ids = connected_ids[connected_ids > 0]
        if connected_ids.size:
            head_region |= np.isin(candidate_labels, connected_ids)
    return head_region


def _legacy_quadratic_align(stabilized, reference, processed, head_region, width_px=10.0):
    valid = np.isfinite(stabilized) & np.isfinite(reference) & np.isfinite(processed)
    component = np.asarray(head_region, dtype=bool) & valid
    boundary = component & ~binary_erosion(
        component,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    _, outside_indices = distance_transform_edt(component | ~valid, return_indices=True)
    desired = processed[tuple(outside_indices)] + (
        reference - reference[tuple(outside_indices)]
    )
    boundary_shift = desired - stabilized

    seeds = np.where(boundary, boundary_shift, 0.0)
    _, boundary_indices = distance_transform_edt(~boundary, return_indices=True)
    edge_field = seeds[tuple(boundary_indices)]
    numerator = gaussian_filter(np.where(component, edge_field, 0.0), sigma=3.0)
    denominator = gaussian_filter(component.astype(np.float32), sigma=3.0)
    edge_field = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-6,
    )
    edge_field[boundary] = boundary_shift[boundary]

    rows, cols = np.indices(component.shape, dtype=np.float32)
    component_rows, component_cols = np.where(component)
    center_col = float(component_cols.min() + component_cols.max()) * 0.5
    center_row = float(component_rows.min() + component_rows.max()) * 0.5
    radius_col = max(1.0, float(component_cols.max() - component_cols.min()) * 0.5)
    radius_row = max(1.0, float(component_rows.max() - component_rows.min()) * 0.5)
    normalized_col = (cols - center_col) / radius_col
    normalized_row = (rows - center_row) / radius_row
    targets = boundary_shift[boundary]
    design = np.column_stack(
        (
            np.ones(np.count_nonzero(boundary), dtype=np.float32),
            normalized_col[boundary],
            normalized_row[boundary],
            np.square(normalized_col[boundary]),
            normalized_col[boundary] * normalized_row[boundary],
            np.square(normalized_row[boundary]),
        )
    )
    low, high = np.percentile(targets, [5.0, 95.0])
    inliers = (targets >= low) & (targets <= high)
    coefficients = np.linalg.lstsq(design[inliers], targets[inliers], rcond=None)[0]
    interior = (
        coefficients[0]
        + coefficients[1] * normalized_col
        + coefficients[2] * normalized_row
        + coefficients[3] * np.square(normalized_col)
        + coefficients[4] * normalized_col * normalized_row
        + coefficients[5] * np.square(normalized_row)
    )
    interior = np.clip(interior, low, high)
    inward_distance = distance_transform_edt(component)
    blend = np.clip((inward_distance - 1.0) / max(float(width_px), 1.0), 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    shift = edge_field * (1.0 - blend) + interior * blend
    output = processed.copy()
    output[component] = stabilized[component] + shift[component]
    return output, shift


def _gradient_values(values, mask):
    horizontal = mask[:, :-1] & mask[:, 1:]
    vertical = mask[:-1, :] & mask[1:, :]
    return np.concatenate(
        (
            np.diff(values, axis=1)[horizontal],
            np.diff(values, axis=0)[vertical],
        )
    )


def _metrics(values, reference, requested, face_core, neck_mask, face_mask, sample_pitch_mm):
    centered = values[face_core] - np.median(values[face_core])
    reference_centered = reference[face_core] - np.median(reference[face_core])
    gradients = _gradient_values(values, face_core)
    reference_gradients = _gradient_values(reference, face_core)
    curvature_region = binary_erosion(face_core, iterations=2)
    curvature_error = laplace(values) - laplace(reference)
    requested_curvature = laplace(requested)[curvature_region]
    output_curvature = laplace(values)[curvature_region]
    requested_curvature_centered = requested_curvature - np.mean(requested_curvature)
    output_curvature_centered = output_curvature - np.mean(output_curvature)
    curvature_correlation = float(
        np.dot(requested_curvature_centered, output_curvature_centered)
        / max(
            np.linalg.norm(requested_curvature_centered)
            * np.linalg.norm(output_curvature_centered),
            1e-12,
        )
    )
    jaw_pairs = face_mask[:-1, :] & neck_mask[1:, :]
    jaw_steps = np.abs(np.diff(values, axis=0)[jaw_pairs])
    all_steps = np.abs(_gradient_values(values, np.isfinite(values)))
    valid = np.isfinite(values)
    diagonal_steps = np.concatenate(
        (
            np.abs(values[1:, 1:] - values[:-1, :-1])[
                valid[1:, 1:] & valid[:-1, :-1]
            ],
            np.abs(values[1:, :-1] - values[:-1, 1:])[
                valid[1:, :-1] & valid[:-1, 1:]
            ],
        )
    )
    return {
        "face_translation_aligned_rmse_mm": float(
            np.sqrt(np.mean(np.square(centered - reference_centered)))
        ),
        "face_span_mm": float(np.percentile(values[face_core], 95.0) - np.percentile(values[face_core], 5.0)),
        "face_span_ratio": float(
            (np.percentile(values[face_core], 95.0) - np.percentile(values[face_core], 5.0))
            / max(np.percentile(reference[face_core], 95.0) - np.percentile(reference[face_core], 5.0), 1e-8)
        ),
        "face_gradient_rmse_mm_per_px": float(
            np.sqrt(np.mean(np.square(gradients - reference_gradients)))
        ),
        "face_curvature_rmse_mm_per_px2": float(
            np.sqrt(np.mean(np.square(curvature_error[curvature_region])))
        ),
        "requested_face_curvature_correlation": curvature_correlation,
        "requested_face_curvature_rms_retention": float(
            np.sqrt(np.mean(np.square(output_curvature_centered)))
            / max(np.sqrt(np.mean(np.square(requested_curvature_centered))), 1e-12)
        ),
        "jaw_neck_step_p95_mm": float(np.percentile(jaw_steps, 95.0)) if jaw_steps.size else None,
        "surface_step_p99_mm": float(np.percentile(all_steps, 99.0)),
        "surface_step_max_mm": float(np.max(all_steps)),
        "surface_slope_p99_mm_per_mm": float(np.percentile(all_steps, 99.0) / sample_pitch_mm),
        "surface_slope_max_mm_per_mm": float(np.max(all_steps) / sample_pitch_mm),
        "diagonal_slope_p99_mm_per_mm": float(
            np.percentile(diagonal_steps, 99.0) / (sample_pitch_mm * np.sqrt(2.0))
        ),
        "diagonal_slope_max_mm_per_mm": float(
            np.max(diagonal_steps) / (sample_pitch_mm * np.sqrt(2.0))
        ),
    }


def _render_relief(values, title, size=240):
    gy, gx = np.gradient(values.astype(np.float64))
    normal = np.stack((-gx, -gy, np.ones_like(values)), axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    lights = np.array(([-0.55, -0.35, 0.76], [0.48, -0.15, 0.86]), dtype=np.float64)
    lights /= np.linalg.norm(lights, axis=1, keepdims=True)
    shade = 0.56 * np.maximum(normal @ lights[0], 0.0)
    shade += 0.34 * np.maximum(normal @ lights[1], 0.0) + 0.10
    shade = np.clip(shade, 0.0, 1.0)
    rgb = np.stack((shade * 0.96, shade * 0.98, shade), axis=-1)
    image = Image.fromarray(np.uint8(np.round(rgb * 255.0))).resize((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 24), "white")
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((7, 6), title, fill=(18, 18, 18))
    return canvas


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    relief, face_mask, face_core, neck_mask, _ = _synthetic_portrait()
    reference = relief * 12.0 + 0.01
    sample_pitch_mm = 0.4
    rows = []
    renders = []
    for height_mm in HEIGHTS_MM:
        original = relief * height_mm + 0.01
        if height_mm <= 12.0:
            variants = {
                "legacy_quadratic": original,
                "screened_poisson_neck": original,
                "screened_gradient_reconstruction": original,
            }
            variant_stats = {
                "legacy_quadratic": {"enabled": False, "reason": "reference_height"},
                "screened_poisson_neck": {"enabled": False, "reason": "reference_height"},
                "screened_gradient_reconstruction": {
                    "enabled": False,
                    "reason": "reference_height",
                },
            }
        else:
            legacy_head = _legacy_expand_head(face_mask, reference)
            legacy_stable, _ = _stabilize_face_relief_height(
                original,
                legacy_head,
                relief_height_mm=height_mm,
                reference_face_height_mm=12.0,
            )
            legacy_scene, _ = _limit_positive_relief_slope(
                legacy_stable,
                sample_pitch_mm=sample_pitch_mm,
                max_slope_mm_per_mm=SLOPE_BUDGET_MM_PER_MM,
                structural_region_mask=legacy_head,
            )
            legacy_output, legacy_shift = _legacy_quadratic_align(
                legacy_stable,
                reference,
                legacy_scene,
                legacy_head,
            )

            protected_head, _ = _expand_face_region_to_depth_connected_head(
                face_mask,
                reference,
            )
            protected_core, _ = _face_translation_core_mask(face_mask, face_mask.shape)
            protected_stable, _ = _stabilize_face_relief_height(
                original,
                protected_head,
                relief_height_mm=height_mm,
                reference_face_height_mm=12.0,
            )
            protected_scene, _ = _limit_positive_relief_slope(
                protected_stable,
                sample_pitch_mm=sample_pitch_mm,
                max_slope_mm_per_mm=SLOPE_BUDGET_MM_PER_MM,
                structural_region_mask=protected_head,
            )
            protected_output, protected_attachment_stats = _align_stabilized_head_to_reference_boundary(
                protected_stable,
                reference,
                protected_scene,
                protected_head,
                protected_core_mask=protected_core,
                protected_face_mask=face_mask,
                max_neighbor_step_mm=sample_pitch_mm * SLOPE_BUDGET_MM_PER_MM,
            )
            gradient_output, gradient_stats = _compress_relief_gradients(
                original,
                sample_pitch_mm=sample_pitch_mm,
                max_slope_mm_per_mm=SLOPE_BUDGET_MM_PER_MM,
                structural_region_mask=protected_head,
                detail_region_mask=face_mask,
            )
            variants = {
                "legacy_quadratic": legacy_output,
                "screened_poisson_neck": protected_output,
                "screened_gradient_reconstruction": gradient_output,
            }
            variant_stats = {
                "legacy_quadratic": {
                    "enabled": True,
                    "method": "legacy_quadratic",
                    "shift_span_mm": float(np.ptp(legacy_shift[legacy_head])),
                },
                "screened_poisson_neck": protected_attachment_stats,
                "screened_gradient_reconstruction": gradient_stats,
            }

        for method, values in variants.items():
            row = {
                "height_mm": height_mm,
                "method": method,
                "attachment": variant_stats[method],
                **_metrics(
                    values,
                    reference,
                    original,
                    face_core,
                    neck_mask,
                    face_mask,
                    sample_pitch_mm,
                ),
            }
            rows.append(row)
            renders.append(_render_relief(values, f"{height_mm:.0f} mm | {method}"))

    width = max(image.width for image in renders) * 3
    height = max(image.height for image in renders) * len(HEIGHTS_MM)
    contact_sheet = Image.new("RGB", (width, height), (238, 238, 238))
    for index, image in enumerate(renders):
        row_index, col_index = divmod(index, 3)
        contact_sheet.paste(image, (col_index * image.width, row_index * image.height))
    contact_sheet.save(output_dir / "height_sweep_shaded.png")

    challenger_rows = [
        row
        for row in rows
        if row["method"] == "screened_gradient_reconstruction"
        and row["height_mm"] >= 30.0
    ]
    checks = {
        "challenger_rows_present": len(challenger_rows) == 3,
        "all_reconstructions_accepted": all(
            row["attachment"].get("enabled", False) for row in challenger_rows
        ),
        "all_quality_gates_passed": all(
            row["attachment"].get("quality_gates", {}).get("passed", False)
            for row in challenger_rows
        ),
        "curvature_correlation_at_least_0_95": all(
            row["requested_face_curvature_correlation"] >= 0.95
            for row in challenger_rows
        ),
        "curvature_rms_retention_at_least_0_85": all(
            row["requested_face_curvature_rms_retention"] >= 0.85
            for row in challenger_rows
        ),
        "cardinal_edge_ratios_within_declared_gate": all(
            row["attachment"].get("output_edge_ratio_p99", np.inf)
            <= row["attachment"].get("quality_gates", {}).get(
                "maximum_output_edge_p99_ratio", -np.inf
            )
            and row["attachment"].get("output_edge_ratio_max", np.inf)
            <= row["attachment"].get("quality_gates", {}).get(
                "maximum_output_edge_ratio", -np.inf
            )
            for row in challenger_rows
        ),
        "diagonal_edge_ratios_within_declared_gate": all(
            row["attachment"].get("diagonal_metrics_available", False)
            and row["attachment"].get("diagonal_edge_ratio_p99", np.inf)
            <= row["attachment"].get("quality_gates", {}).get(
                "maximum_output_edge_p99_ratio", -np.inf
            )
            and row["attachment"].get("diagonal_edge_ratio_max", np.inf)
            <= row["attachment"].get("quality_gates", {}).get(
                "maximum_output_edge_ratio", -np.inf
            )
            for row in challenger_rows
        ),
    }
    summary = {
        "fixture": "analytic_face_head_neck_torso_v1",
        "heights_mm": list(HEIGHTS_MM),
        "reference_face_height_mm": 12.0,
        "sample_pitch_mm": sample_pitch_mm,
        "slope_budget_mm_per_mm": SLOPE_BUDGET_MM_PER_MM,
        "checks": {**checks, "passed": all(checks.values())},
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not summary["checks"]["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Face relief benchmark failed: {', '.join(failed)}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="docs/benchmark-evidence/face_relief_gradient_domain_local",
    )
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
