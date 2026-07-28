"""Gate landmark-aligned SMIRK face geometry on privacy-safe exact depth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.evaluate_vggheads_face_training_generalization import (
    _load_cached_inputs,
    _quality_for_array,
    select_generalization_suite,
)
from backend.benchmark.evaluate_vggheads_small_face_exact_gate import (
    SUBJECT_BOUNDARY_PIXELS,
    _summary,
    build_small_face_candidate,
)
from backend.benchmark.smirk_depth_provider import SMIRKProvider
from backend.benchmark.vggheads_depth_provider import (
    rasterize_projected_mesh_depth,
)
from backend.face_depth_refinement import (
    _detect_faces_mediapipe,
    _resolve_face_landmarker_model,
    _upgrade_yunet_regions_with_mediapipe,
)


METHOD = "smirk-landmark-aligned-small-face-smoke"
SMOKE_ROW_IDS = (
    "mh_african_female__mouth_open_02",
    "mh_african_male__asymmetric_05",
    "mh_asian_female__smile_03",
    "mh_caucasian_male__mouth_open_06",
    "mh_mixed_female__asymmetric_00",
    "mh_mixed_female__mouth_open_02",
    "mh_mixed_male__mouth_open_01",
    "mh_mixed_male__neutral_00",
)
VARIANTS = (
    {
        "variant_id": "global_a0.10_s1",
        "provider_alpha": 0.10,
        "sigma_ratio": 1.0 / 53.0,
    },
    {
        "variant_id": "global_a0.25_s1",
        "provider_alpha": 0.25,
        "sigma_ratio": 1.0 / 53.0,
    },
    {
        "variant_id": "global_a0.50_s1",
        "provider_alpha": 0.50,
        "sigma_ratio": 1.0 / 53.0,
    },
    {
        "variant_id": "global_a0.25_s2",
        "provider_alpha": 0.25,
        "sigma_ratio": 2.0 / 53.0,
    },
    {
        "variant_id": "jaw075_a0.10_a0.25_s1",
        "provider_alpha": 0.25,
        "low_provider_alpha": 0.10,
        "minimum_jaw_open": 0.075,
        "sigma_ratio": 1.0 / 53.0,
    },
    {
        "variant_id": "pose045_jaw075_a0_a0.10_a0.25_s1",
        "provider_alpha": 0.25,
        "low_provider_alpha": 0.10,
        "outside_pose_provider_alpha": 0.0,
        "minimum_jaw_open": 0.075,
        "maximum_absolute_pose_y_radians": 0.45,
        "sigma_ratio": 1.0 / 53.0,
    },
    {
        "variant_id": "registered_global_a0.10_s1",
        "provider_alpha": 0.10,
        "sigma_ratio": 1.0 / 53.0,
        "registration_mode": "similarity",
    },
    {
        "variant_id": "registered_global_a0.25_s1",
        "provider_alpha": 0.25,
        "sigma_ratio": 1.0 / 53.0,
        "registration_mode": "similarity",
    },
    {
        "variant_id": "registered_pose060_jaw075_a0_a0.10_a0.25_s1",
        "provider_alpha": 0.25,
        "low_provider_alpha": 0.10,
        "outside_pose_provider_alpha": 0.0,
        "minimum_jaw_open": 0.075,
        "maximum_absolute_pose_y_radians": 0.60,
        "sigma_ratio": 1.0 / 53.0,
        "registration_mode": "similarity",
    },
    {
        "variant_id": "registered_pose075_jaw075_a0_a0.10_a0.25_s1",
        "provider_alpha": 0.25,
        "low_provider_alpha": 0.10,
        "outside_pose_provider_alpha": 0.0,
        "minimum_jaw_open": 0.075,
        "maximum_absolute_pose_y_radians": 0.75,
        "sigma_ratio": 1.0 / 53.0,
        "registration_mode": "similarity",
    },
)
MINIMUM_LANDMARKS = 100
MINIMUM_LANDMARK_BBOX_IOU = 0.15
MINIMUM_PROVIDER_CROP_COVERAGE = 0.50
MAXIMUM_MEDIAN_REGRESSION = 2e-4
MINIMUM_REGISTRATION_INLIER_RATIO = 0.60
MINIMUM_REGISTRATION_SCALE = 0.60
MAXIMUM_REGISTRATION_SCALE = 1.60
MAXIMUM_REGISTRATION_ROTATION_DEG = 20.0
MAXIMUM_REGISTRATION_RMS_RATIO = 0.95


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_smoke_rows(rows: list[dict]) -> list[dict]:
    by_id = {str(row["row_id"]): row for row in rows}
    missing = [row_id for row_id in SMOKE_ROW_IDS if row_id not in by_id]
    if missing:
        raise ValueError(
            "SMIRK smoke rows are missing: " + ", ".join(missing)
        )
    selected = [by_id[row_id] for row_id in SMOKE_ROW_IDS]
    split_counts = {
        split: sum(row["split"] == split for row in selected)
        for split in ("train", "validation", "sealed")
    }
    if split_counts != {"train": 4, "validation": 2, "sealed": 2}:
        raise ValueError(f"SMIRK smoke split drifted: {split_counts}")
    return selected


def select_evaluation_rows(rows: list[dict], suite: str) -> list[dict]:
    suite = str(suite)
    if suite == "smoke":
        return select_smoke_rows(rows)
    if suite == "challenge":
        return select_generalization_suite(rows)["challenge"]
    raise ValueError(f"Unknown SMIRK evaluation suite: {suite}")


def _bbox_iou(
    first: list[float] | tuple[float, ...],
    second: list[float] | tuple[float, ...],
) -> float:
    ax0, ay0, ax1, ay1 = (float(value) for value in first)
    bx0, by0, bx1, by1 = (float(value) for value in second)
    intersection_width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    intersection_height = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    second_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def select_landmark_region(
    regions: list[dict],
    target_bbox_xyxy: list[float] | tuple[float, ...],
) -> tuple[dict, float]:
    usable = [
        region
        for region in regions
        if len(region.get("bbox") or ()) == 4
        and int(region.get("landmark_count") or 0) >= MINIMUM_LANDMARKS
    ]
    if not usable:
        raise ValueError("No usable MediaPipe landmark face was detected")
    ranked = sorted(
        (
            (_bbox_iou(region["bbox"], target_bbox_xyxy), region)
            for region in usable
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    overlap, region = ranked[0]
    if overlap < MINIMUM_LANDMARK_BBOX_IOU:
        raise ValueError(
            "Best MediaPipe face does not overlap the refinement crop "
            f"enough: {overlap:.6f}"
        )
    return region, float(overlap)


def detect_landmark_regions(
    image_rgb: np.ndarray,
    refinement_bbox_xyxy: list[int] | tuple[int, ...],
    refinement_support: np.ndarray,
) -> tuple[list[dict], dict]:
    direct = _detect_faces_mediapipe(image_rgb, 3, 16)
    direct_usable = [
        region
        for region in direct
        if int(region.get("landmark_count") or 0) >= MINIMUM_LANDMARKS
        and _bbox_iou(region["bbox"], refinement_bbox_xyxy)
        >= MINIMUM_LANDMARK_BBOX_IOU
    ]
    if direct_usable:
        return direct, {
            "method": "full-frame-mediapipe",
            "direct_regions": len(direct),
            "guided_regions": 0,
        }
    support = np.asarray(refinement_support, dtype=np.uint8)
    if support.shape != image_rgb.shape[:2]:
        raise ValueError("Refinement support must use the image grid")
    guided = _upgrade_yunet_regions_with_mediapipe(
        image_rgb,
        [
            {
                "bbox": [
                    int(value) for value in refinement_bbox_xyxy
                ],
                "face_mask": support * 255,
                "detector": "cached-production-face",
                "confidence": None,
            }
        ],
        max_faces=1,
    )
    return direct + guided, {
        "method": "cached-box-guided-upscaled-mediapipe",
        "direct_regions": len(direct),
        "guided_regions": len(guided),
    }


def _similarity_matrix(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
) -> np.ndarray:
    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if source_xy.shape != target_xy.shape or (
        source_xy.ndim != 2 or source_xy.shape[1] != 2
    ):
        raise ValueError("Similarity points must have matching [N, 2] shapes")
    if len(source_xy) < 3:
        raise ValueError("Similarity fit requires at least three points")
    if not np.all(np.isfinite(source_xy)) or not np.all(
        np.isfinite(target_xy)
    ):
        raise ValueError("Similarity points contain non-finite values")
    source_mean = np.mean(source_xy, axis=0)
    target_mean = np.mean(target_xy, axis=0)
    source_centered = source_xy - source_mean
    target_centered = target_xy - target_mean
    source_energy = float(np.sum(source_centered**2))
    if source_energy <= 1e-12:
        raise ValueError("Similarity source points have no usable span")
    covariance = target_centered.T @ source_centered
    left, singular_values, right_t = np.linalg.svd(covariance)
    orientation = np.ones(2, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0:
        orientation[-1] = -1.0
    rotation = left @ np.diag(orientation) @ right_t
    scale = float(
        np.sum(singular_values * orientation) / source_energy
    )
    if scale <= 1e-12:
        raise ValueError("Similarity fit produced an invalid scale")
    linear = scale * rotation
    translation = target_mean - linear @ source_mean
    return np.column_stack((linear, translation))


def apply_xy_registration(
    points_xyz: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 2:
        raise ValueError("Registered points must have shape [N, >=2]")
    if matrix.shape != (2, 3):
        raise ValueError("Registration matrix must have shape [2, 3]")
    registered = points_xyz.copy()
    registered[:, :2] = (
        points_xyz[:, :2].astype(np.float64) @ matrix[:, :2].T
        + matrix[:, 2]
    ).astype(np.float32)
    return registered


def fit_landmark_registration(
    projected_landmarks_xyz: np.ndarray,
    detected_landmarks_xyz: np.ndarray,
    landmark_indices: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    iterations: int = 3,
) -> tuple[np.ndarray, dict]:
    projected = np.asarray(
        projected_landmarks_xyz,
        dtype=np.float64,
    )
    detected = np.asarray(detected_landmarks_xyz, dtype=np.float64)
    indices = np.asarray(landmark_indices, dtype=np.int64).reshape(-1)
    if projected.ndim != 2 or projected.shape[1] < 2:
        raise ValueError("Projected SMIRK landmarks must have shape [N, >=2]")
    if detected.ndim != 2 or detected.shape[1] < 2:
        raise ValueError("Detected MediaPipe landmarks must have shape [N, >=2]")
    if len(projected) != len(indices):
        raise ValueError("SMIRK landmarks and MediaPipe indices differ")
    if len(indices) < MINIMUM_LANDMARKS:
        raise ValueError("Insufficient SMIRK landmarks for registration")
    if np.min(indices) < 0 or np.max(indices) >= len(detected):
        raise ValueError("SMIRK MediaPipe landmark index is out of bounds")
    source = projected[:, :2]
    target = detected[indices, :2].copy()
    target[:, 0] *= max(int(image_width) - 1, 1)
    target[:, 1] *= max(int(image_height) - 1, 1)
    finite = np.all(np.isfinite(source), axis=1) & np.all(
        np.isfinite(target),
        axis=1,
    )
    minimum_inliers = max(
        3,
        int(np.ceil(len(source) * MINIMUM_REGISTRATION_INLIER_RATIO)),
    )
    if np.count_nonzero(finite) < minimum_inliers:
        raise ValueError("Insufficient finite landmarks for registration")
    active = finite.copy()
    for _ in range(max(1, int(iterations))):
        matrix = _similarity_matrix(source[active], target[active])
        transformed = (
            source @ matrix[:, :2].T + matrix[:, 2]
        )
        residual = np.linalg.norm(transformed - target, axis=1)
        finite_residual = residual[finite]
        median = float(np.median(finite_residual))
        mad = float(np.median(np.abs(finite_residual - median)))
        threshold = max(1.25, median + 2.5 * 1.4826 * mad)
        next_active = finite & (residual <= threshold)
        if np.count_nonzero(next_active) < minimum_inliers:
            ordered = np.flatnonzero(finite)[
                np.argsort(residual[finite], kind="stable")
            ]
            next_active = np.zeros_like(active)
            next_active[ordered[:minimum_inliers]] = True
        if np.array_equal(next_active, active):
            break
        active = next_active
    matrix = _similarity_matrix(source[active], target[active])
    transformed = source @ matrix[:, :2].T + matrix[:, 2]
    residual_before = np.linalg.norm(source - target, axis=1)
    residual_after = np.linalg.norm(transformed - target, axis=1)
    linear = matrix[:, :2]
    scale = float(np.sqrt(max(np.linalg.det(linear), 0.0)))
    rotation_deg = float(
        np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))
    )
    before_rms = float(
        np.sqrt(np.mean(residual_before[finite] ** 2))
    )
    after_rms = float(np.sqrt(np.mean(residual_after[finite] ** 2)))
    after_inlier_rms = float(
        np.sqrt(np.mean(residual_after[active] ** 2))
    )
    checks = {
        "inlier_coverage": bool(
            np.count_nonzero(active) >= minimum_inliers
        ),
        "scale": bool(
            MINIMUM_REGISTRATION_SCALE
            <= scale
            <= MAXIMUM_REGISTRATION_SCALE
        ),
        "rotation": abs(rotation_deg) <= MAXIMUM_REGISTRATION_ROTATION_DEG,
        "rms_improves": bool(
            after_rms
            <= before_rms * MAXIMUM_REGISTRATION_RMS_RATIO
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    return matrix.astype(np.float32), {
        "method": "robust-105-point-mediapipe-similarity",
        "landmarks": int(len(source)),
        "inliers": int(np.count_nonzero(active)),
        "inlier_ratio": float(np.count_nonzero(active) / len(source)),
        "scale": scale,
        "rotation_deg": rotation_deg,
        "translation_xy": [
            float(matrix[0, 2]),
            float(matrix[1, 2]),
        ],
        "before_rms_pixels": before_rms,
        "after_rms_pixels": after_rms,
        "after_inlier_rms_pixels": after_inlier_rms,
        "maximum_residual_pixels": float(
            np.max(residual_after[finite])
        ),
        "matrix": matrix.tolist(),
        "checks": checks,
    }


def gate_provider_alpha_by_registration(
    effective_alpha: float,
    provider_policy: dict,
    registration_mode: str,
    registration: dict,
) -> tuple[float, dict]:
    policy = {
        **provider_policy,
        "registration_mode": str(registration_mode),
        "registration_passed": bool(
            registration.get("checks", {}).get("passed", False)
        ),
    }
    if registration_mode == "none":
        return float(effective_alpha), policy
    if registration_mode != "similarity":
        raise ValueError(
            f"Unknown SMIRK registration mode: {registration_mode}"
        )
    if policy["registration_passed"]:
        return float(effective_alpha), policy
    policy.update(
        {
            "branch_before_registration": policy.get("branch"),
            "branch": "registration-bypass",
            "effective_provider_alpha": 0.0,
        }
    )
    return 0.0, policy


def effective_provider_alpha(
    variant: dict,
    inference_metadata: dict,
) -> tuple[float, dict]:
    high_alpha = float(variant["provider_alpha"])
    pose_threshold = variant.get("maximum_absolute_pose_y_radians")
    pose_params = inference_metadata.get("pose_params") or ()
    if pose_threshold is not None:
        if len(pose_params) < 2:
            raise ValueError(
                "Pose-gated SMIRK variant requires pose parameters"
            )
        pose_y = float(pose_params[1])
        pose_threshold = float(pose_threshold)
        if abs(pose_y) > pose_threshold:
            outside_alpha = float(
                variant.get("outside_pose_provider_alpha", 0.0)
            )
            return outside_alpha, {
                "method": "smirk-provider-pose-and-jaw",
                "pose_y_radians": pose_y,
                "maximum_absolute_pose_y_radians": pose_threshold,
                "branch": "pose-bypass",
                "effective_provider_alpha": outside_alpha,
            }
    threshold = variant.get("minimum_jaw_open")
    if threshold is None:
        return high_alpha, {
            "method": "global",
            "effective_provider_alpha": high_alpha,
        }
    jaw_params = inference_metadata.get("jaw_params") or ()
    if not jaw_params:
        raise ValueError("Jaw-gated SMIRK variant requires jaw parameters")
    jaw_open = float(jaw_params[0])
    threshold = float(threshold)
    low_alpha = float(variant["low_provider_alpha"])
    high_branch = jaw_open >= threshold
    effective_alpha = high_alpha if high_branch else low_alpha
    return effective_alpha, {
        "method": (
            "smirk-provider-pose-and-jaw"
            if pose_threshold is not None
            else "smirk-provider-jaw-opening"
        ),
        "pose_y_radians": (
            float(pose_params[1]) if pose_threshold is not None else None
        ),
        "maximum_absolute_pose_y_radians": pose_threshold,
        "jaw_open": jaw_open,
        "minimum_jaw_open": threshold,
        "low_provider_alpha": low_alpha,
        "high_provider_alpha": high_alpha,
        "branch": "high" if high_branch else "low",
        "effective_provider_alpha": effective_alpha,
    }


def _boundary_mask(selection: np.ndarray) -> np.ndarray:
    distance = cv2.distanceTransform(
        selection.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    return (
        (distance > 0)
        & (distance <= float(SUBJECT_BOUNDARY_PIXELS))
    )


def _quality_record(row: dict, quality: dict) -> dict:
    return {
        "row_id": str(row["row_id"]),
        "split": str(row["split"]),
        "identity_group": str(row["identity_group"]),
        "expression": str(row["expression"]),
        "face_height_pixels": int(
            row["render"]["face_bbox_height_pixels"]
        ),
        "camera_yaw_deg": float(row["spec"]["camera_yaw_deg"]),
        "target_dimension": int(row["spec"]["target_dimension"]),
        "background_profile": str(
            row["spec"]["background_profile"]
        ),
        "lighting_profile": str(row["spec"]["lighting_profile"]),
        **quality,
    }


def _paired_counts(baseline: dict, candidate: dict) -> dict:
    baseline_by_id = {row["row_id"]: row for row in baseline["rows"]}
    candidate_by_id = {row["row_id"]: row for row in candidate["rows"]}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("SMIRK baseline and candidate row sets differ")
    deltas = {
        row_id: (
            int(candidate_by_id[row_id]["combined_part_failures"])
            - int(baseline_by_id[row_id]["combined_part_failures"])
        )
        for row_id in baseline_by_id
    }
    return {
        "improved_rows": int(sum(value < 0 for value in deltas.values())),
        "tied_rows": int(sum(value == 0 for value in deltas.values())),
        "regressed_rows": int(sum(value > 0 for value in deltas.values())),
        "maximum_failure_regression": int(max(deltas.values(), default=0)),
        "failure_deltas": dict(sorted(deltas.items())),
    }


def _median_checks(baseline: dict, candidate: dict) -> dict:
    return {
        "shape": bool(
            candidate["median_shape_correlation"]
            >= baseline["median_shape_correlation"]
            - MAXIMUM_MEDIAN_REGRESSION
        ),
        "gradient": bool(
            candidate["median_gradient_correlation"]
            >= baseline["median_gradient_correlation"]
            - MAXIMUM_MEDIAN_REGRESSION
        ),
        "rmse": bool(
            candidate["median_normalized_rmse"]
            <= baseline["median_normalized_rmse"]
            + MAXIMUM_MEDIAN_REGRESSION
        ),
    }


def _partition(summary_rows: list[dict], split: str) -> dict:
    selected = [row for row in summary_rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No SMIRK rows are available for split {split}")
    return _summary(selected)


def select_train_variant(
    baseline_train: dict,
    variants: list[dict],
) -> dict:
    evaluated = []
    for variant in variants:
        candidate = variant["train"]
        paired = _paired_counts(baseline_train, candidate)
        medians = _median_checks(baseline_train, candidate)
        checks = {
            "aggregate_failures_improve": bool(
                candidate["combined_part_failures"]
                < baseline_train["combined_part_failures"]
            ),
            "at_least_one_row_improves": paired["improved_rows"] > 0,
            "no_row_regresses": paired["regressed_rows"] == 0,
            "median_shape_non_regression": medians["shape"],
            "median_gradient_non_regression": medians["gradient"],
            "median_rmse_non_regression": medians["rmse"],
        }
        checks["passed"] = bool(all(checks.values()))
        evaluated.append(
            {
                **variant,
                "train_checks": checks,
                "train_paired": paired,
            }
        )
    eligible = [item for item in evaluated if item["train_checks"]["passed"]]
    pool = eligible or evaluated
    selected = min(
        pool,
        key=lambda item: (
            int(item["train"]["combined_part_failures"]),
            -float(item["train"]["median_shape_correlation"]),
            -float(item["train"]["median_gradient_correlation"]),
            float(item["train"]["median_normalized_rmse"]),
            float(item["provider_alpha"]),
            float(item["sigma_ratio"]),
        ),
    )
    return {
        "selected_variant_id": str(selected["variant_id"]),
        "selected_is_train_eligible": bool(
            selected["train_checks"]["passed"]
        ),
        "evaluated": evaluated,
    }


def final_decision(
    baseline_by_split: dict,
    candidate_by_split: dict,
    selected_rows: list[dict],
    selection: dict,
) -> dict:
    all_paired = _paired_counts(
        baseline_by_split["all"],
        candidate_by_split["all"],
    )
    validation_paired = _paired_counts(
        baseline_by_split["validation"],
        candidate_by_split["validation"],
    )
    sealed_paired = _paired_counts(
        baseline_by_split["sealed"],
        candidate_by_split["sealed"],
    )
    medians = _median_checks(
        baseline_by_split["all"],
        candidate_by_split["all"],
    )
    checks = {
        "train_selection_passes": bool(
            selection["selected_is_train_eligible"]
        ),
        "aggregate_failures_improve": bool(
            candidate_by_split["all"]["combined_part_failures"]
            < baseline_by_split["all"]["combined_part_failures"]
        ),
        "at_least_two_rows_improve": all_paired["improved_rows"] >= 2,
        "no_row_regresses": all_paired["regressed_rows"] == 0,
        "validation_non_regression": (
            validation_paired["regressed_rows"] == 0
        ),
        "sealed_non_regression": sealed_paired["regressed_rows"] == 0,
        "median_shape_non_regression": medians["shape"],
        "median_gradient_non_regression": medians["gradient"],
        "median_rmse_non_regression": medians["rmse"],
        "background_exact": bool(
            all(row["background_value_exact"] for row in selected_rows)
        ),
        "attachment_boundary_exact": bool(
            all(row["boundary_value_exact"] for row in selected_rows)
        ),
        "landmark_registration_passes": bool(
            all(
                row["provider"]["landmark_bbox_iou"]
                >= MINIMUM_LANDMARK_BBOX_IOU
                for row in selected_rows
            )
        ),
        "provider_coverage_passes": bool(
            all(
                row["provider"]["refinement_crop_coverage"]
                >= MINIMUM_PROVIDER_CROP_COVERAGE
                for row in selected_rows
            )
        ),
        "provider_registration_passes": bool(
            all(
                row["provider"]["registration"]["checks"]["passed"]
                for row in selected_rows
            )
        ),
        "rasterization_passes": bool(
            all(
                row["provider"]["raster"]["finite_pixels"] > 0
                and row["provider"]["raster"]["degenerate_faces"] == 0
                for row in selected_rows
            )
        ),
    }
    checks["eligible_for_30mm_replay"] = bool(all(checks.values()))
    return {
        "checks": checks,
        "paired": {
            "all": all_paired,
            "validation": validation_paired,
            "sealed": sealed_paired,
        },
    }


def evaluate(
    corpus_root: str | Path,
    cache_root: str | Path,
    provider_root: str | Path,
    checkpoint_path: str | Path,
    flame_model_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    suite: str = "smoke",
) -> dict:
    started = time.perf_counter()
    corpus_root = Path(corpus_root).resolve()
    cache_root = Path(cache_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "selected_candidate_depth"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    summary_path = corpus_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = select_evaluation_rows(summary["rows"], suite)
    provider = SMIRKProvider(
        provider_root,
        checkpoint_path,
        flame_model_path,
        device=device,
    )

    baseline_rows = []
    variant_rows = {
        str(variant["variant_id"]): [] for variant in VARIANTS
    }
    variant_values: dict[str, dict[str, np.ndarray]] = {
        variant_id: {} for variant_id in variant_rows
    }
    provider_records = {}
    with tempfile.TemporaryDirectory() as directory:
        temporary_path = Path(directory) / "candidate.npy"
        for index, row in enumerate(rows, start=1):
            row_id = str(row["row_id"])
            inputs = _load_cached_inputs(cache_root, corpus_root, row)
            source_path = corpus_root / row["source"]["path"]
            image = np.asarray(Image.open(source_path).convert("RGB"))
            regions, landmark_detection = detect_landmark_regions(
                image,
                inputs["bbox"],
                inputs["support"],
            )
            region, landmark_bbox_iou = select_landmark_region(
                regions,
                inputs["bbox"],
            )
            inference = provider.infer(
                image,
                target_bbox_xyxy=region["bbox"],
            )
            height, width = image.shape[:2]
            x0, y0, x1, y1 = inputs["bbox"]
            projected = np.asarray(
                inference["vertices"],
                dtype=np.float32,
            )
            identity_registration = {
                "method": "none",
                "checks": {"passed": True},
            }
            try:
                registration_matrix, registration = (
                    fit_landmark_registration(
                        inference["projected_landmarks_mp"],
                        region["landmarks_xyz"],
                        inference["mediapipe_landmark_indices"],
                        image_width=width,
                        image_height=height,
                    )
                )
                registered_projected = apply_xy_registration(
                    projected,
                    registration_matrix,
                )
            except (KeyError, TypeError, ValueError) as error:
                registration = {
                    "method": "robust-105-point-mediapipe-similarity",
                    "error": f"{type(error).__name__}:{error}",
                    "checks": {"passed": False},
                }
                registered_projected = projected.copy()

            provider_modes = {}
            for registration_mode, mode_projected, diagnostics in (
                ("none", projected, identity_registration),
                ("similarity", registered_projected, registration),
            ):
                mode_depth, raster = rasterize_projected_mesh_depth(
                    mode_projected,
                    inference["faces"],
                    height=height,
                    width=width,
                    front_surface="minimum-z",
                )
                provider_crop = mode_depth[y0:y1, x0:x1]
                refinement_crop_coverage = float(
                    np.count_nonzero(np.isfinite(provider_crop))
                    / provider_crop.size
                )
                provider_record = {
                    "registration_mode": registration_mode,
                    "registration": diagnostics,
                    "landmark_bbox_xyxy": [
                        int(value) for value in region["bbox"]
                    ],
                    "landmark_count": int(region["landmark_count"]),
                    "landmark_detector": str(region["detector"]),
                    "landmark_detection": landmark_detection,
                    "landmark_bbox_iou": float(landmark_bbox_iou),
                    "refinement_bbox_xyxy": [
                        int(value) for value in inputs["bbox"]
                    ],
                    "refinement_crop_coverage": (
                        refinement_crop_coverage
                    ),
                    "projected_bounds_xyxy": [
                        float(np.min(mode_projected[:, 0])),
                        float(np.min(mode_projected[:, 1])),
                        float(np.max(mode_projected[:, 0])),
                        float(np.max(mode_projected[:, 1])),
                    ],
                    "raster": raster,
                    "inference": inference["metadata"],
                }
                provider_modes[registration_mode] = {
                    "depth": mode_depth,
                    "record": provider_record,
                }
            provider_records[row_id] = {
                mode: value["record"]
                for mode, value in provider_modes.items()
            }

            baseline_quality = _quality_for_array(
                inputs["baseline"],
                row,
                corpus_root,
                temporary_path,
            )
            baseline_rows.append(_quality_record(row, baseline_quality))
            boundary = _boundary_mask(inputs["selection"])
            for variant in VARIANTS:
                variant_id = str(variant["variant_id"])
                registration_mode = str(
                    variant.get("registration_mode", "none")
                )
                provider_mode = provider_modes[registration_mode]
                provider_depth = provider_mode["depth"]
                provider_record = provider_mode["record"]
                effective_alpha, provider_policy = (
                    effective_provider_alpha(
                        variant,
                        inference["metadata"],
                    )
                )
                effective_alpha, provider_policy = (
                    gate_provider_alpha_by_registration(
                        effective_alpha,
                        provider_policy,
                        registration_mode,
                        provider_record["registration"],
                    )
                )
                if effective_alpha == 0.0:
                    candidate = inputs["baseline"].copy()
                    fusion = {
                        "method": "provider-policy-bypass",
                        "provider_alpha": 0.0,
                        "sigma_ratio": float(variant["sigma_ratio"]),
                        "maximum_abs_change": 0.0,
                    }
                else:
                    candidate, fusion = build_small_face_candidate(
                        inputs["baseline"],
                        inputs["local"],
                        provider_depth,
                        inputs["support"],
                        inputs["selection"],
                        inputs["bbox"],
                        provider_alpha=effective_alpha,
                        sigma_ratio=float(variant["sigma_ratio"]),
                    )
                quality = _quality_for_array(
                    candidate,
                    row,
                    corpus_root,
                    temporary_path,
                )
                record = _quality_record(row, quality)
                record.update(
                    {
                        "background_value_exact": bool(
                            np.array_equal(
                                candidate[~inputs["selection"]],
                                inputs["baseline"][~inputs["selection"]],
                            )
                        ),
                        "boundary_value_exact": bool(
                            not np.any(boundary)
                            or np.array_equal(
                                candidate[boundary],
                                inputs["baseline"][boundary],
                            )
                        ),
                        "maximum_abs_change": float(
                            np.max(
                                np.abs(candidate - inputs["baseline"])
                            )
                        ),
                        "provider": provider_record,
                        "provider_policy": provider_policy,
                        "fusion": fusion,
                    }
                )
                variant_rows[variant_id].append(record)
                variant_values[variant_id][row_id] = candidate
            print(
                json.dumps(
                    {
                        "row": index,
                        "total": len(rows),
                        "row_id": row_id,
                        "landmark_bbox_iou": landmark_bbox_iou,
                        "provider_coverage": provider_modes["none"][
                            "record"
                        ]["refinement_crop_coverage"],
                        "registered_provider_coverage": provider_modes[
                            "similarity"
                        ]["record"]["refinement_crop_coverage"],
                        "registration_before_rms": registration.get(
                            "before_rms_pixels"
                        ),
                        "registration_after_rms": registration.get(
                            "after_rms_pixels"
                        ),
                        "registration_passed": registration.get(
                            "checks", {}
                        ).get("passed", False),
                    }
                ),
                flush=True,
            )

    baseline_by_split = {
        split: _partition(baseline_rows, split)
        for split in ("train", "validation", "sealed")
    }
    baseline_by_split["all"] = _summary(baseline_rows)
    variants = []
    for variant in VARIANTS:
        variant_id = str(variant["variant_id"])
        records = variant_rows[variant_id]
        variants.append(
            {
                **variant,
                "train": _partition(records, "train"),
                "validation": _partition(records, "validation"),
                "sealed": _partition(records, "sealed"),
                "all": _summary(records),
            }
        )
    selection = select_train_variant(
        baseline_by_split["train"],
        variants,
    )
    selected_variant_id = selection["selected_variant_id"]
    selected_variant = next(
        variant
        for variant in variants
        if variant["variant_id"] == selected_variant_id
    )
    selected_rows = variant_rows[selected_variant_id]
    selected_paths = {}
    for row in rows:
        row_id = str(row["row_id"])
        path = candidate_dir / f"{row_id}.npy"
        np.save(path, variant_values[selected_variant_id][row_id])
        selected_paths[row_id] = {
            "path": str(path),
            "sha256": _sha256(path),
        }
    candidate_by_split = {
        split: selected_variant[split]
        for split in ("train", "validation", "sealed", "all")
    }
    decision = final_decision(
        baseline_by_split,
        candidate_by_split,
        selected_rows,
        selection,
    )
    landmarker_path = _resolve_face_landmarker_model()
    evidence = {
        "schema_version": 1,
        "status": (
            "eligible-for-30mm-replay"
            if decision["checks"]["eligible_for_30mm_replay"]
            else "hold"
        ),
        "method": METHOD,
        "privacy": summary.get("privacy"),
        "production_changed": False,
        "production_eligible": False,
        "configuration": {
            "suite": str(suite),
            "row_ids": [str(row["row_id"]) for row in rows],
            "variants": [
                dict(variant) for variant in VARIANTS
            ],
            "variant_selection_scope": "train-identities-only",
            "held_out_scope": "validation-and-sealed-identities",
            "front_surface": "minimum-z",
            "minimum_landmark_bbox_iou": (
                MINIMUM_LANDMARK_BBOX_IOU
            ),
            "minimum_provider_crop_coverage": (
                MINIMUM_PROVIDER_CROP_COVERAGE
            ),
        },
        "inputs": {
            "corpus_summary_sha256": _sha256(summary_path),
            "face_landmarker_sha256": _sha256(landmarker_path),
            "rows": len(rows),
        },
        "provider": provider.provenance(),
        "provider_rows": provider_records,
        "baseline": baseline_by_split,
        "variants": variants,
        "selection": selection,
        "candidate": {
            **candidate_by_split,
            "selected_depth_artifacts": selected_paths,
        },
        "decision": decision,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--provider-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--flame-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--suite",
        choices=("smoke", "challenge"),
        default="smoke",
    )
    args = parser.parse_args()
    evidence = evaluate(
        args.corpus_root,
        args.cache_root,
        args.provider_root,
        args.checkpoint_path,
        args.flame_model_path,
        args.output_dir,
        device=args.device,
        suite=args.suite,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["checks"]["eligible_for_30mm_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
