"""Localized physical diagnostics for face relief heightfields."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

from backend.benchmark.face_part_metrics import (
    FACE_PART_AFFINE_MM_GATES,
    FACE_PART_GATES,
    face_part_affine_surface_error_metrics,
    face_part_cross_height_metrics,
)
from backend.benchmark.run_relief_visual_sweep import (
    FACE_APPEARANCE_GATES,
    _appearance_checks,
)
from backend.pic_to_3d import _surface_lighting_agreement_metrics


LOCAL_FACE_METRIC_SCHEMA_VERSION = 1
DERIVED_REGION_NAMES = (
    "forehead",
    "image_left_cheek",
    "image_right_cheek",
    "jaw",
    "silhouette_band",
)


def globally_align_face_surface(
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    face_mask: np.ndarray,
    *,
    expected_scale_sign: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit one affine transform on the complete face and apply it globally."""
    reference = np.asarray(reference_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    face = np.asarray(face_mask, dtype=bool)
    if reference.shape != candidate.shape or reference.shape != face.shape:
        raise ValueError("Face alignment arrays must have identical shapes")
    valid = face & np.isfinite(reference) & np.isfinite(candidate)
    samples = int(np.count_nonzero(valid))
    if samples < 64:
        raise ValueError("Face alignment requires at least 64 finite samples")
    design = np.column_stack((candidate[valid], np.ones(samples, dtype=np.float64)))
    scale, shift = np.linalg.lstsq(design, reference[valid], rcond=None)[0]
    if expected_scale_sign is not None and (
        not np.isfinite(scale) or scale * float(expected_scale_sign) <= 0
    ):
        raise ValueError("Face alignment violates the expected depth orientation")
    aligned = candidate * float(scale) + float(shift)
    return aligned, {
        "method": "single-global-face-affine-fit",
        "samples": samples,
        "scale": float(scale),
        "shift": float(shift),
    }


def derived_face_region_masks(
    face_mask: np.ndarray,
    named_part_masks: dict[str, np.ndarray],
    *,
    sample_pitch_mm: float,
) -> dict[str, np.ndarray]:
    """Add broad face zones without treating smooth features as discontinuities."""
    face = np.asarray(face_mask, dtype=bool)
    if face.ndim != 2 or not np.any(face):
        raise ValueError("A non-empty 2D face mask is required")
    pitch = float(sample_pitch_mm)
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("sample_pitch_mm must be positive and finite")
    parts = {
        str(name): np.asarray(mask, dtype=bool) & face
        for name, mask in named_part_masks.items()
    }
    if any(mask.shape != face.shape for mask in parts.values()):
        raise ValueError("Named face-part masks must match the face mask")

    rows, columns = np.where(face)
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    height = max(bottom - top, 1)
    width = max(right - left, 1)
    yy, xx = np.indices(face.shape, dtype=np.float64)
    y = (yy - top) / height
    x = (xx - left) / width

    occupied = np.zeros_like(face)
    for mask in parts.values():
        occupied |= mask
    occupied = binary_dilation(
        occupied,
        structure=np.ones((3, 3), dtype=bool),
        iterations=max(1, int(round(0.5 / pitch))),
    )
    interior = face & ~occupied
    zones = {
        "forehead": interior & (y >= 0.08) & (y <= 0.36) & (x >= 0.20) & (x <= 0.80),
        "image_left_cheek": interior
        & (y >= 0.38)
        & (y <= 0.72)
        & (x >= 0.04)
        & (x <= 0.45),
        "image_right_cheek": interior
        & (y >= 0.38)
        & (y <= 0.72)
        & (x >= 0.55)
        & (x <= 0.96),
        "jaw": interior & (y >= 0.70) & (y <= 0.97) & (x >= 0.16) & (x <= 0.84),
    }
    silhouette_width_px = max(2, int(round(2.0 / pitch)))
    eroded = binary_erosion(
        face,
        structure=np.ones((3, 3), dtype=bool),
        iterations=silhouette_width_px,
        border_value=0,
    )
    zones["silhouette_band"] = face & ~eroded
    return {**parts, **zones}


def transform_mask_to_emitted_grid(
    mask_values: np.ndarray,
    surface_grid_transform: dict,
    emitted_shape: tuple[int, int],
) -> np.ndarray:
    """Apply the endpoint's recorded nearest-neighbor mask transform."""
    required = {
        "input_depth_shape",
        "target_depth_shape",
        "mesh_shape_before_crop",
        "crop_bbox_rc",
        "emitted_shape",
        "flip_x",
    }
    missing = sorted(required - set(surface_grid_transform))
    if missing:
        raise ValueError("Surface-grid transform is missing: " + ", ".join(missing))
    input_rows, input_columns = (
        int(value) for value in surface_grid_transform["input_depth_shape"]
    )
    target_rows, target_columns = (
        int(value) for value in surface_grid_transform["target_depth_shape"]
    )
    mask = np.asarray(mask_values, dtype=bool)
    mask = (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                (input_columns, input_rows), Image.Resampling.NEAREST
            )
        )
        >= 128
    )
    mask = (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                (target_columns, target_rows), Image.Resampling.NEAREST
            )
        )
        >= 128
    )
    if bool(surface_grid_transform["flip_x"]):
        mask = np.flip(mask, axis=1)
    mesh_rows, mesh_columns = (
        int(value) for value in surface_grid_transform["mesh_shape_before_crop"]
    )
    mask = (
        np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(
                (mesh_columns, mesh_rows), Image.Resampling.NEAREST
            )
        )
        >= 128
    )
    top, left, bottom, right = (
        int(value) for value in surface_grid_transform["crop_bbox_rc"]
    )
    mask = mask[top:bottom, left:right]
    transform_shape = tuple(
        int(value) for value in surface_grid_transform["emitted_shape"]
    )
    if mask.shape != transform_shape or mask.shape != tuple(emitted_shape):
        raise ValueError(
            f"Transformed mask has shape {mask.shape}, expected {transform_shape} "
            f"and {tuple(emitted_shape)}"
        )
    return mask


def localized_face_surface_metrics(
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    face_mask: np.ndarray,
    part_masks: dict[str, np.ndarray],
    *,
    sample_pitch_mm: float,
) -> dict:
    """Measure shape, physical error, normals, and matte relighting per region."""
    reference = np.asarray(reference_values, dtype=np.float64)
    candidate = np.asarray(candidate_values, dtype=np.float64)
    face = np.asarray(face_mask, dtype=bool)
    regions = derived_face_region_masks(
        face,
        part_masks,
        sample_pitch_mm=sample_pitch_mm,
    )
    named_regions = {
        name: regions[name]
        for name in sorted(part_masks)
    }
    derived_regions = {
        name: regions[name]
        for name in DERIVED_REGION_NAMES
    }
    cross_height = face_part_cross_height_metrics(
        reference,
        candidate,
        face,
        named_regions,
        sample_pitch_mm=sample_pitch_mm,
    )
    affine_mm = face_part_affine_surface_error_metrics(
        reference,
        candidate,
        face,
        named_regions,
    )
    appearance_parts = []
    for name in sorted(named_regions):
        metrics = _surface_lighting_agreement_metrics(
            reference,
            candidate,
            named_regions[name],
            sample_pitch_mm=sample_pitch_mm,
            minimum_samples=12,
            component_metrics=False,
        )
        checks = _appearance_checks(metrics, FACE_APPEARANCE_GATES)
        appearance_parts.append(
            {
                "name": name,
                "metrics": metrics,
                "checks": checks,
                "passed": bool(checks.get("passed", False)),
            }
        )
    failed_appearance = [
        record["name"] for record in appearance_parts if not record["passed"]
    ]
    derived_diagnostics = []
    for name in DERIVED_REGION_NAMES:
        metrics = _surface_lighting_agreement_metrics(
            reference,
            candidate,
            derived_regions[name],
            sample_pitch_mm=sample_pitch_mm,
            minimum_samples=12,
            component_metrics=False,
        )
        derived_diagnostics.append(
            {
                "name": name,
                "metrics": metrics,
                "checks": _appearance_checks(metrics, FACE_APPEARANCE_GATES),
            }
        )
    checks = {
        "cross_height": bool(cross_height.get("passed", False)),
        "affine_mm": bool(affine_mm.get("passed", False)),
        "multi_light_appearance": not failed_appearance,
    }
    checks["passed"] = bool(all(checks.values()))
    return {
        "schema_version": LOCAL_FACE_METRIC_SCHEMA_VERSION,
        "sample_pitch_mm": float(sample_pitch_mm),
        "region_names": sorted(regions),
        "gates": {
            "cross_height": FACE_PART_GATES,
            "affine_mm": FACE_PART_AFFINE_MM_GATES,
            "appearance": FACE_APPEARANCE_GATES,
        },
        "cross_height": cross_height,
        "affine_mm": affine_mm,
        "multi_light_appearance": {
            "parts": appearance_parts,
            "failed_parts": failed_appearance,
            "passed": not failed_appearance,
        },
        "derived_region_diagnostics": {
            "hard_gate": False,
            "parts": derived_diagnostics,
        },
        "checks": checks,
    }
