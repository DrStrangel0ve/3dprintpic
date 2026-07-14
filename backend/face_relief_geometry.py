"""Geometry-domain protection for face-aware printable reliefs."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg


_GRID_EDGE_VIEWS = (
    ((slice(None), slice(None, -1)), (slice(None), slice(1, None)), 1.0),
    ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), 1.0),
    ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None)), np.sqrt(2.0)),
    ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1)), np.sqrt(2.0)),
)


def _conjugate_gradient(matrix, rhs, tolerance, max_iterations):
    """Support both the current and older SciPy conjugate-gradient APIs."""
    try:
        return cg(
            matrix,
            rhs,
            rtol=float(tolerance),
            atol=0.0,
            maxiter=int(max_iterations),
        )
    except TypeError:  # pragma: no cover - exercised only by older SciPy releases
        return cg(
            matrix,
            rhs,
            tol=float(tolerance),
            maxiter=int(max_iterations),
        )


def _solve_screened_shift(
    component,
    boundary,
    boundary_shift,
    screening_length_px,
    solver_tolerance,
    max_iterations,
    hard_core=None,
    soft_core_weight=None,
    soft_core_screening_weight=0.25,
):
    """Interpolate boundary offsets without injecting a polynomial face warp."""
    shift = np.zeros(component.shape, dtype=np.float64)
    shift[boundary] = boundary_shift[boundary]
    screening_length = max(float(screening_length_px), 1.0)
    anchor_shift = float(np.median(boundary_shift[boundary]))
    core = np.zeros(component.shape, dtype=bool)
    if hard_core is not None:
        core = component & np.asarray(hard_core, dtype=bool)
    if not np.any(core):
        inward_distance = distance_transform_edt(component)
        core = component & (inward_distance >= screening_length + 1.0)
        if not np.any(core):
            core = component & (
                inward_distance >= max(float(np.max(inward_distance)) - 1.0, 2.0)
            )
    core &= ~boundary
    shift[core] = anchor_shift
    interior = component & ~boundary & ~core
    interior_rows, interior_cols = np.where(interior)
    if interior_rows.size == 0:
        return shift, 0, False, int(np.count_nonzero(core))

    variable_ids = np.full(component.shape, -1, dtype=np.int64)
    variable_ids[interior_rows, interior_cols] = np.arange(interior_rows.size)
    equation_ids = np.arange(interior_rows.size, dtype=np.int64)
    rhs = np.zeros(interior_rows.size, dtype=np.float64)
    degree = np.zeros(interior_rows.size, dtype=np.float64)
    matrix_rows = []
    matrix_cols = []
    matrix_values = []

    for row_offset, col_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor_rows = interior_rows + row_offset
        neighbor_cols = interior_cols + col_offset
        neighbor_in_component = component[neighbor_rows, neighbor_cols]
        degree += neighbor_in_component
        neighbor_ids = variable_ids[neighbor_rows, neighbor_cols]
        variable_neighbors = neighbor_ids >= 0
        if np.any(variable_neighbors):
            matrix_rows.append(equation_ids[variable_neighbors])
            matrix_cols.append(neighbor_ids[variable_neighbors])
            matrix_values.append(-np.ones(np.count_nonzero(variable_neighbors)))
        fixed_neighbors = neighbor_in_component & ~variable_neighbors
        if np.any(fixed_neighbors):
            rhs[fixed_neighbors] += shift[
                neighbor_rows[fixed_neighbors],
                neighbor_cols[fixed_neighbors],
            ]

    screening_weight = np.full(
        interior_rows.size,
        1.0 / np.square(screening_length),
        dtype=np.float64,
    )
    if soft_core_weight is not None:
        soft_weight = np.asarray(soft_core_weight, dtype=np.float64)[
            interior_rows,
            interior_cols,
        ]
        screening_weight += (
            np.clip(soft_weight, 0.0, 1.0)
            * max(float(soft_core_screening_weight), 0.0)
        )
    rhs += screening_weight * anchor_shift
    matrix_rows.append(equation_ids)
    matrix_cols.append(equation_ids)
    matrix_values.append(degree + screening_weight)
    system = coo_matrix(
        (
            np.concatenate(matrix_values),
            (np.concatenate(matrix_rows), np.concatenate(matrix_cols)),
        ),
        shape=(interior_rows.size, interior_rows.size),
    ).tocsr()

    solution, solver_info = _conjugate_gradient(
        system,
        rhs,
        tolerance=solver_tolerance,
        max_iterations=max_iterations,
    )
    used_fallback = bool(solver_info != 0 or not np.all(np.isfinite(solution)))
    if used_fallback:
        return shift, int(solver_info), True, int(np.count_nonzero(core))
    shift[interior_rows, interior_cols] = solution
    return shift, int(solver_info), used_fallback, int(np.count_nonzero(core))


def _component_gradient_distortion(shift, component):
    horizontal = component[:, :-1] & component[:, 1:]
    vertical = component[:-1, :] & component[1:, :]
    changes = []
    if np.any(horizontal):
        changes.append(np.abs(np.diff(shift, axis=1)[horizontal]))
    if np.any(vertical):
        changes.append(np.abs(np.diff(shift, axis=0)[vertical]))
    return np.concatenate(changes) if changes else np.empty(0, dtype=np.float64)


def _attachment_slope_ratios(
    values,
    reference_values,
    baseline_values,
    component,
    protected_core,
    max_neighbor_step_mm,
):
    core = (
        np.asarray(protected_core, dtype=bool) & component
        if protected_core is not None
        else np.zeros(component.shape, dtype=bool)
    )
    steps = []
    allowed_steps = []
    core_boundary_steps = []
    core_boundary_allowed = []
    baseline = np.asarray(baseline_values, dtype=np.float64)
    for first, second, edge_length in _GRID_EDGE_VIEWS:
        first_component = component[first]
        second_component = component[second]
        internal = first_component & second_component
        crossing = first_component ^ second_component
        first_core = core[first]
        second_core = core[second]
        audited = (internal & ~(first_core & second_core)) | crossing
        if not np.any(audited):
            continue

        edge_steps = np.abs(values[second] - values[first])
        reference_steps = np.abs(reference_values[second] - reference_values[first])
        baseline_steps = np.abs(baseline[second] - baseline[first])
        physical_step = float(max_neighbor_step_mm) * float(edge_length)
        allowed = np.maximum(physical_step, reference_steps)
        # The component boundary is fixed to the already slope-limited scene.
        # Audit crossing edges against that scene rather than an unrelated
        # low-height reference edge.
        allowed = np.where(crossing, np.maximum(physical_step, baseline_steps), allowed)
        steps.append(edge_steps[audited])
        allowed_steps.append(allowed[audited])

        core_crossing = internal & (first_core ^ second_core)
        if np.any(core_crossing):
            core_boundary_steps.append(edge_steps[core_crossing])
            core_boundary_allowed.append(allowed[core_crossing])
    if not steps:
        return np.zeros(1), np.zeros(1)
    ratios = np.concatenate(steps) / np.maximum(np.concatenate(allowed_steps), 1e-8)
    if core_boundary_steps:
        core_ratios = np.concatenate(core_boundary_steps) / np.maximum(
            np.concatenate(core_boundary_allowed),
            1e-8,
        )
    else:
        core_ratios = np.zeros(1)
    return ratios, core_ratios


def _project_attachment_outliers(
    candidate_values,
    reference_values,
    baseline_values,
    component,
    boundary,
    protected_core,
    max_neighbor_step_mm,
    max_attachment_slope_ratio,
    max_core_boundary_slope_ratio,
    max_correction_mm=3.0,
    max_iterations=96,
):
    """Project isolated outer-band cliffs while keeping core and boundary fixed."""
    candidate = np.asarray(candidate_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    baseline = np.asarray(baseline_values, dtype=np.float64)
    output = candidate.copy()
    core = (
        np.asarray(protected_core, dtype=bool) & component
        if protected_core is not None
        else np.zeros(component.shape, dtype=bool)
    )
    movable = component & ~boundary & ~core
    if not np.any(movable):
        return candidate_values, {
            "enabled": False,
            "reason": "no_attachment_band",
        }

    iterations = 0
    for iterations in range(1, max(1, int(max_iterations)) + 1):
        adjusted_edges = 0
        for first, second, edge_length in _GRID_EDGE_VIEWS:
            first_component = component[first]
            second_component = component[second]
            internal = first_component & second_component
            crossing = first_component ^ second_component
            first_core = core[first]
            second_core = core[second]
            audited = (internal & ~(first_core & second_core)) | crossing
            if not np.any(audited):
                continue

            first_values = output[first]
            second_values = output[second]
            finite = (
                np.isfinite(first_values)
                & np.isfinite(second_values)
                & np.isfinite(reference[first])
                & np.isfinite(reference[second])
                & np.isfinite(baseline[first])
                & np.isfinite(baseline[second])
            )
            reference_steps = np.abs(reference[second] - reference[first])
            baseline_steps = np.abs(baseline[second] - baseline[first])
            physical_step = float(max_neighbor_step_mm) * float(edge_length)
            allowed = np.maximum(physical_step, reference_steps)
            allowed = np.where(crossing, np.maximum(physical_step, baseline_steps), allowed)
            core_crossing = internal & (first_core ^ second_core)
            ratio_limit = np.where(
                core_crossing,
                float(max_core_boundary_slope_ratio),
                float(max_attachment_slope_ratio),
            )
            limit = allowed * ratio_limit
            difference = second_values - first_values
            excess = np.abs(difference) - limit
            first_movable = movable[first]
            second_movable = movable[second]
            adjustable_count = first_movable.astype(np.int8) + second_movable.astype(np.int8)
            violations = (
                audited
                & finite
                & (excess > 1e-8)
                & (adjustable_count > 0)
            )
            if not np.any(violations):
                continue
            direction = np.sign(difference[violations])
            correction = excess[violations] / adjustable_count[violations]
            first_values[violations] += (
                direction * correction * first_movable[violations]
            )
            second_values[violations] -= (
                direction * correction * second_movable[violations]
            )
            adjusted_edges += int(np.count_nonzero(violations))
        if adjusted_edges == 0:
            break

    correction = np.abs(output - candidate)
    correction_values = correction[movable]
    correction_max = float(np.max(correction_values, initial=0.0))
    if correction_max > float(max_correction_mm) + 1e-8:
        return candidate_values, {
            "enabled": False,
            "reason": "correction_limit",
            "iterations": int(iterations),
            "correction_max_mm": correction_max,
            "max_correction_mm": float(max_correction_mm),
        }
    return output, {
        "enabled": True,
        "method": "bounded_local_edge_projection",
        "iterations": int(iterations),
        "corrected_pixels": int(np.count_nonzero(correction_values > 1e-6)),
        "correction_p95_mm": float(np.percentile(correction_values, 95.0)),
        "correction_max_mm": correction_max,
        "max_correction_mm": float(max_correction_mm),
    }


def align_face_to_scene_gradient_domain(
    stabilized_values,
    reference_values,
    processed_values,
    head_region_mask,
    screening_length_px=100.0,
    solver_tolerance=1e-6,
    max_iterations=1200,
    protected_core_mask=None,
    protected_face_mask=None,
    soft_core_screening_weight=0.25,
    max_neighbor_step_mm=None,
    max_attachment_slope_ratio=3.0,
    max_attachment_slope_p99_ratio=2.25,
    max_core_boundary_slope_ratio=2.5,
    max_core_boundary_slope_p99_ratio=2.25,
):
    """Attach a stabilized face using a screened Poisson shift field.

    The stabilized surface already carries facial gradients at a printable
    physical scale. Solving for an additive shift field preserves those
    gradients while meeting the already slope-limited scene at the head
    boundary. Weak screening makes the attachment band nearly harmonic while
    the protected core remains one robust component translation.
    """
    stabilized = np.asarray(stabilized_values, dtype=np.float64)
    reference = np.asarray(reference_values, dtype=np.float64)
    processed = np.asarray(processed_values, dtype=np.float64)
    region = np.asarray(head_region_mask, dtype=bool) if head_region_mask is not None else None
    if region is None:
        return processed_values, {"enabled": False, "reason": "no_head_region"}
    if stabilized.shape != reference.shape or stabilized.shape != processed.shape:
        raise ValueError("Face attachment surfaces must have identical shapes")
    if region.shape != processed.shape:
        raise ValueError("Face attachment mask must match the surface shape")
    protected_core = None
    if protected_core_mask is not None:
        protected_core = np.asarray(protected_core_mask, dtype=bool)
        if protected_core.shape != processed.shape:
            raise ValueError("Protected face core mask must match the surface shape")
    protected_face = None
    if protected_face_mask is not None:
        protected_face = np.asarray(protected_face_mask, dtype=bool)
        if protected_face.shape != processed.shape:
            raise ValueError("Protected face mask must match the surface shape")
    slope_step = None
    if max_neighbor_step_mm is not None:
        slope_step = float(max_neighbor_step_mm)
        if not np.isfinite(slope_step) or slope_step <= 0:
            raise ValueError("Maximum attachment neighbor step must be finite and positive")

    valid = np.isfinite(stabilized) & np.isfinite(reference) & np.isfinite(processed)
    region &= valid
    components, component_count = label(region)
    if component_count == 0:
        return processed_values, {"enabled": False, "reason": "empty_head_region"}

    aligned = processed.copy()
    residuals = []
    shape_distortions = []
    gradient_distortions = []
    applied_shifts = []
    protected_core_shape_distortions = []
    protected_core_gradient_distortions = []
    protected_core_shifts = []
    attachment_slope_ratios = []
    core_boundary_slope_ratios = []
    attempted_attachment_slope_ratios = []
    attempted_core_boundary_slope_ratios = []
    solver_info_values = []
    fallback_components = 0
    slope_rejected_components = 0
    core_slope_rejected_components = 0
    slope_projected_components = 0
    slope_projection_records = []
    generated_core_components = 0
    hard_core_pixels = 0
    soft_core_pixels = 0
    aligned_components = 0
    attempted_components = 0
    skipped_small_components = 0
    skipped_boundary_components = 0
    for component_index in range(1, component_count + 1):
        component = components == component_index
        attempted_components += 1
        if np.count_nonzero(component) < 16:
            skipped_small_components += 1
            continue
        boundary = component & ~binary_erosion(
            component,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        )
        if np.count_nonzero(boundary) < 8:
            skipped_boundary_components += 1
            continue
        boundary_shift = processed - stabilized
        component_soft_weight = None
        if protected_face is not None:
            component_face = component & protected_face
            component_core = (
                component & protected_core
                if protected_core is not None
                else np.zeros(component.shape, dtype=bool)
            )
            if np.any(component_face) and not np.any(component_core):
                face_distance = distance_transform_edt(component_face)
                max_face_distance = float(np.max(face_distance))
                margin = max(2.0, min(12.0, round(max_face_distance * 0.12)))
                component_core = component_face & (face_distance >= margin)
                if np.count_nonzero(component_core) < 16:
                    component_core = component_face & (
                        face_distance >= max(2.0, max_face_distance * 0.65)
                    )
                generated_core_components += int(np.any(component_core))
            if np.any(component_face) and np.any(component_core):
                face_distance = distance_transform_edt(component_face)
                core_entry_distance = max(
                    float(np.percentile(face_distance[component_core], 5.0)),
                    2.0,
                )
                component_soft_weight = np.clip(
                    (face_distance - 1.0) / max(core_entry_distance - 1.0, 1.0),
                    0.0,
                    1.0,
                )
                component_soft_weight = (
                    component_soft_weight
                    * component_soft_weight
                    * (3.0 - 2.0 * component_soft_weight)
                )
                soft_core_pixels += int(
                    np.count_nonzero(component_soft_weight > 1e-4)
                )
        shift, solver_info, used_fallback, component_core_pixels = _solve_screened_shift(
            component,
            boundary,
            boundary_shift,
            screening_length_px=screening_length_px,
            solver_tolerance=solver_tolerance,
            max_iterations=max_iterations,
            hard_core=component_core if protected_face is not None else protected_core,
            soft_core_weight=component_soft_weight,
            soft_core_screening_weight=soft_core_screening_weight,
        )
        solver_info_values.append(solver_info)
        fallback_components += int(used_fallback)
        hard_core_pixels += int(component_core_pixels)
        if used_fallback:
            continue

        candidate = processed.copy()
        candidate[component] = stabilized[component] + shift[component]
        if slope_step is not None:
            slope_ratios, core_ratios = _attachment_slope_ratios(
                candidate,
                reference,
                processed,
                component,
                component_core if protected_face is not None else protected_core,
                slope_step,
            )
            slope_p99 = float(np.percentile(slope_ratios, 99.0))
            slope_max = float(np.max(slope_ratios))
            core_slope_p99 = float(np.percentile(core_ratios, 99.0))
            core_slope_max = float(np.max(core_ratios))
            attempted_attachment_slope_ratios.append(slope_ratios)
            attempted_core_boundary_slope_ratios.append(core_ratios)
            projection_stats = {"enabled": False, "reason": "not_needed"}
            if (
                slope_p99 <= float(max_attachment_slope_p99_ratio)
                and slope_max > float(max_attachment_slope_ratio)
                and core_slope_p99 <= float(max_core_boundary_slope_p99_ratio)
                and core_slope_max <= float(max_core_boundary_slope_ratio)
            ):
                projected, projection_stats = _project_attachment_outliers(
                    candidate,
                    reference,
                    processed,
                    component,
                    boundary,
                    component_core if protected_face is not None else protected_core,
                    slope_step,
                    max_attachment_slope_ratio,
                    max_core_boundary_slope_ratio,
                )
                if projection_stats.get("enabled", False):
                    projected_ratios, projected_core_ratios = _attachment_slope_ratios(
                        projected,
                        reference,
                        processed,
                        component,
                        component_core if protected_face is not None else protected_core,
                        slope_step,
                    )
                    projected_p99 = float(np.percentile(projected_ratios, 99.0))
                    projected_max = float(np.max(projected_ratios))
                    projected_core_p99 = float(np.percentile(projected_core_ratios, 99.0))
                    projected_core_max = float(np.max(projected_core_ratios))
                    projection_stats.update(
                        {
                            "slope_ratio_p99": projected_p99,
                            "slope_ratio_max": projected_max,
                            "core_slope_ratio_p99": projected_core_p99,
                            "core_slope_ratio_max": projected_core_max,
                        }
                    )
                    if (
                        projected_p99 <= float(max_attachment_slope_p99_ratio)
                        and projected_max <= float(max_attachment_slope_ratio) + 1e-6
                        and projected_core_p99 <= float(max_core_boundary_slope_p99_ratio)
                        and projected_core_max <= float(max_core_boundary_slope_ratio) + 1e-6
                    ):
                        candidate = projected
                        slope_ratios = projected_ratios
                        core_ratios = projected_core_ratios
                        slope_p99 = projected_p99
                        slope_max = projected_max
                        core_slope_p99 = projected_core_p99
                        core_slope_max = projected_core_max
                        slope_projected_components += 1
                    else:
                        projection_stats["enabled"] = False
                        projection_stats["reason"] = "post_projection_slope_gate"
            slope_projection_records.append(projection_stats)
            if (
                slope_p99 > float(max_attachment_slope_p99_ratio)
                or slope_max > float(max_attachment_slope_ratio) + 1e-6
                or core_slope_p99 > float(max_core_boundary_slope_p99_ratio)
                or core_slope_max > float(max_core_boundary_slope_ratio) + 1e-6
            ):
                slope_rejected_components += 1
                core_slope_rejected_components += int(
                    core_slope_p99 > float(max_core_boundary_slope_p99_ratio)
                    or core_slope_max > float(max_core_boundary_slope_ratio)
                )
                continue
            attachment_slope_ratios.append(slope_ratios)
            core_boundary_slope_ratios.append(core_ratios)
        aligned[component] = candidate[component]

        component_shift = shift[component]
        centered_shift = component_shift - np.median(component_shift)
        residuals.append(np.abs(aligned[boundary] - processed[boundary]))
        shape_distortions.append(np.abs(centered_shift))
        gradient_distortions.append(_component_gradient_distortion(shift, component))
        applied_shifts.append(np.abs(component_shift))
        if protected_core is not None:
            component_core = component & protected_core & ~boundary
            if np.any(component_core):
                core_shift = shift[component_core]
                protected_core_shape_distortions.append(
                    np.abs(core_shift - np.median(core_shift))
                )
                protected_core_gradient_distortions.append(
                    _component_gradient_distortion(shift, component_core)
                )
                protected_core_shifts.append(core_shift)
        aligned_components += 1

    if not residuals or aligned_components != attempted_components:
        reason = "no_aligned_components"
        if fallback_components:
            reason = "solver_nonconvergence"
        elif slope_rejected_components:
            reason = "attachment_slope_gate"
        elif aligned_components != attempted_components:
            reason = "partial_component_alignment"
        failed_stats = {
            "enabled": False,
            "reason": reason,
            "method": "screened_poisson_shift",
            "attempted_components": int(attempted_components),
            "aligned_components": int(aligned_components),
            "skipped_small_components": int(skipped_small_components),
            "skipped_boundary_components": int(skipped_boundary_components),
            "solver_info": [int(value) for value in solver_info_values],
            "solver_fallback_components": int(fallback_components),
            "slope_rejected_components": int(slope_rejected_components),
            "core_slope_rejected_components": int(core_slope_rejected_components),
            "slope_projected_components": int(slope_projected_components),
            "slope_projection": slope_projection_records,
            "generated_core_components": int(generated_core_components),
        }
        if attempted_attachment_slope_ratios:
            attempted_slope_values = np.concatenate(attempted_attachment_slope_ratios)
            attempted_core_values = np.concatenate(attempted_core_boundary_slope_ratios)
            failed_stats.update(
                {
                    "attempted_attachment_slope_ratio_p99": float(
                        np.percentile(attempted_slope_values, 99.0)
                    ),
                    "attempted_attachment_slope_ratio_max": float(
                        np.max(attempted_slope_values)
                    ),
                    "attempted_core_boundary_slope_ratio_p99": float(
                        np.percentile(attempted_core_values, 99.0)
                    ),
                    "attempted_core_boundary_slope_ratio_max": float(
                        np.max(attempted_core_values)
                    ),
                }
            )
        return processed_values, failed_stats

    residual_values = np.concatenate(residuals)
    shape_values = np.concatenate(shape_distortions)
    gradient_values = np.concatenate(
        [values for values in gradient_distortions if values.size]
    ) if any(values.size for values in gradient_distortions) else np.zeros(1)
    shift_values = np.concatenate(applied_shifts)
    output = np.where(np.isfinite(processed), aligned, np.nan).astype(
        np.asarray(processed_values).dtype,
        copy=False,
    )
    stats = {
        "enabled": True,
        "method": "screened_poisson_shift",
        "boundary_target": "processed_scene",
        "attempted_components": int(attempted_components),
        "aligned_components": int(aligned_components),
        "skipped_small_components": int(skipped_small_components),
        "skipped_boundary_components": int(skipped_boundary_components),
        "screening_length_px": float(max(float(screening_length_px), 1.0)),
        "screening_weight": float(1.0 / np.square(max(float(screening_length_px), 1.0))),
        "solver_tolerance": float(solver_tolerance),
        "solver_info": [int(value) for value in solver_info_values],
        "solver_fallback_components": int(fallback_components),
        "slope_rejected_components": int(slope_rejected_components),
        "core_slope_rejected_components": int(core_slope_rejected_components),
        "slope_projected_components": int(slope_projected_components),
        "slope_projection": slope_projection_records,
        "generated_core_components": int(generated_core_components),
        "hard_translation_core_pixels": int(hard_core_pixels),
        "soft_translation_support_pixels": int(soft_core_pixels),
        "soft_core_screening_weight": float(soft_core_screening_weight),
        "core_anchor_source": "boundary_shift_median",
        "boundary_residual_p95_mm": float(np.percentile(residual_values, 95.0)),
        "boundary_residual_max_mm": float(np.max(residual_values)),
        "translation_aligned_shape_distortion_rmse_mm": float(
            np.sqrt(np.mean(np.square(shape_values)))
        ),
        "translation_aligned_shape_distortion_p95_mm": float(
            np.percentile(shape_values, 95.0)
        ),
        "gradient_distortion_p95_mm_per_px": float(
            np.percentile(gradient_values, 95.0)
        ),
        "gradient_distortion_max_mm_per_px": float(np.max(gradient_values)),
        "applied_shift_p95_mm": float(np.percentile(shift_values, 95.0)),
        "applied_shift_max_mm": float(np.max(shift_values)),
    }
    if attachment_slope_ratios:
        slope_ratio_values = np.concatenate(attachment_slope_ratios)
        core_boundary_ratio_values = np.concatenate(core_boundary_slope_ratios)
        stats.update(
            {
                "max_neighbor_step_mm": float(slope_step),
                "max_attachment_slope_ratio": float(max_attachment_slope_ratio),
                "max_attachment_slope_p99_ratio": float(max_attachment_slope_p99_ratio),
                "max_core_boundary_slope_ratio": float(max_core_boundary_slope_ratio),
                "max_core_boundary_slope_p99_ratio": float(
                    max_core_boundary_slope_p99_ratio
                ),
                "attachment_slope_ratio_p95": float(np.percentile(slope_ratio_values, 95.0)),
                "attachment_slope_ratio_p99": float(np.percentile(slope_ratio_values, 99.0)),
                "attachment_slope_ratio_max": float(np.max(slope_ratio_values)),
                "core_boundary_slope_ratio_p99": float(
                    np.percentile(core_boundary_ratio_values, 99.0)
                ),
                "core_boundary_slope_ratio_max": float(np.max(core_boundary_ratio_values)),
            }
        )
    if protected_core_shape_distortions:
        core_shape_values = np.concatenate(protected_core_shape_distortions)
        core_gradient_values = np.concatenate(
            [values for values in protected_core_gradient_distortions if values.size]
        ) if any(values.size for values in protected_core_gradient_distortions) else np.zeros(1)
        core_shift_values = np.concatenate(protected_core_shifts)
        stats.update(
            {
                "protected_core_pixels": int(core_shape_values.size),
                "protected_core_translation_aligned_rmse_mm": float(
                    np.sqrt(np.mean(np.square(core_shape_values)))
                ),
                "protected_core_translation_aligned_p95_mm": float(
                    np.percentile(core_shape_values, 95.0)
                ),
                "protected_core_gradient_distortion_p95_mm_per_px": float(
                    np.percentile(core_gradient_values, 95.0)
                ),
                "protected_core_shift_median_mm": float(
                    np.median(core_shift_values)
                ),
                "protected_core_shift_span_mm": float(np.ptp(core_shift_values)),
            }
        )
    return output, stats
