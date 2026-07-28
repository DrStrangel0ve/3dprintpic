"""Evaluate a fail-closed VGGHeads correction on the exact CC0 face rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from backend.benchmark.makehuman_face_fixture import (
    load_makehuman_face_fixture,
)
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    DEFAULT_ASSET_DIR,
    VARIED_CONTEXT_MATRIX,
    FaceSceneSpec,
    _exact_face_depth_quality,
    _stage_scene,
)
from backend.benchmark.vggheads_depth_provider import (
    VGGHEADS_SMALL_FACE_MAX_HEIGHT_PX,
    VGGHeadsProvider,
    fill_depth_nearest,
    normalize_front_depth,
    rasterize_projected_mesh_depth,
    small_face_policy,
    subject_interior_taper,
)
from backend.face_depth_refinement import fuse_face_surface_residual


METHOD = "vggheads-small-face-low-frequency-subject-taper"
HARD_SMALL_FACE_ROW = "small_side_lit_shelves_256"
LOW_FREQUENCY_SIGMA_RATIO = 3.0 / 53.0
PROVIDER_BLEND_ALPHA = 0.50
SUBJECT_TAPER_PIXELS = 4.0
SUBJECT_BOUNDARY_PIXELS = 1.0
MAXIMUM_HARD_ROW_GRADIENT_REGRESSION = 5e-5
MAXIMUM_POSE_MAGNITUDE_ERROR_DEG = 5.0
MINIMUM_PROVIDER_CONFIDENCE = 0.90
MINIMUM_PROVIDER_CROP_COVERAGE = 0.50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quality(
    candidate_path: Path,
    exact_depth_path: Path,
    selection_mask_path: Path,
    part_mask_paths: dict[str, Path],
) -> dict:
    metrics = _exact_face_depth_quality(
        candidate_path,
        exact_depth_path,
        selection_mask_path,
        expected_scale_sign=-1.0,
        part_mask_paths=part_mask_paths,
    )
    check_failures = Counter()
    for part in metrics["named_part_shape"]["parts"]:
        check_failures.update(
            name
            for name, passed in part.get("checks", {}).items()
            if name != "passed" and not passed
        )
    shape_failed = metrics["named_part_shape"]["failed_parts"]
    affine_failed = metrics["named_part_affine_mm"]["failed_parts"]
    return {
        "shape_correlation": float(metrics["shape_correlation"]),
        "gradient_correlation": float(metrics["gradient_correlation"]),
        "normalized_rmse": float(metrics["normalized_rmse"]),
        "shape_failed_parts": shape_failed,
        "affine_failed_parts": affine_failed,
        "combined_part_failures": int(
            len(shape_failed) + len(affine_failed)
        ),
        "shape_check_failures": dict(sorted(check_failures.items())),
        "checks": metrics["checks"],
    }


def _summary(rows: list[dict]) -> dict:
    check_failures = Counter()
    for row in rows:
        check_failures.update(row.get("shape_check_failures", {}))
    return {
        "row_count": len(rows),
        "combined_part_failures": int(
            sum(row["combined_part_failures"] for row in rows)
        ),
        "median_shape_correlation": float(
            np.median([row["shape_correlation"] for row in rows])
        ),
        "median_gradient_correlation": float(
            np.median([row["gradient_correlation"] for row in rows])
        ),
        "median_normalized_rmse": float(
            np.median([row["normalized_rmse"] for row in rows])
        ),
        "shape_check_failure_counts": dict(sorted(check_failures.items())),
        "rows": rows,
    }


def _historical_baseline(baseline_root: Path) -> dict:
    evidence_path = baseline_root / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    baseline = evidence.get("reproduced_current_baseline")
    if not isinstance(baseline, dict):
        raise ValueError("Baseline evidence has no reproduced current baseline")
    return baseline


def _baseline_matches_historical(
    baseline: dict,
    historical: dict,
) -> bool:
    baseline_by_id = {row["row_id"]: row for row in baseline["rows"]}
    historical_by_id = {row["row_id"]: row for row in historical["rows"]}
    if set(baseline_by_id) != set(historical_by_id):
        return False
    for row_id, row in baseline_by_id.items():
        expected = historical_by_id[row_id]
        if (
            row["combined_part_failures"]
            != expected["combined_part_failures"]
            or row["shape_failed_parts"] != expected["shape_failed_parts"]
            or row["affine_failed_parts"] != expected["affine_failed_parts"]
        ):
            return False
        for name in (
            "shape_correlation",
            "gradient_correlation",
            "normalized_rmse",
        ):
            if not math.isclose(
                float(row[name]),
                float(expected[name]),
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                return False
    return True


def build_small_face_candidate(
    baseline_depth: np.ndarray,
    local_face_depth: np.ndarray,
    provider_depth: np.ndarray,
    face_mask: np.ndarray,
    selection_mask: np.ndarray,
    crop_bbox_xyxy: list[int] | tuple[int, int, int, int],
    *,
    provider_alpha: float = PROVIDER_BLEND_ALPHA,
    sigma_ratio: float = LOW_FREQUENCY_SIGMA_RATIO,
    taper_pixels: float = SUBJECT_TAPER_PIXELS,
) -> tuple[np.ndarray, dict]:
    baseline_depth = np.asarray(baseline_depth, dtype=np.float32)
    local_face_depth = np.asarray(local_face_depth, dtype=np.float32)
    provider_depth = np.asarray(provider_depth, dtype=np.float32)
    face_mask = np.asarray(face_mask) > 0
    selection_mask = np.asarray(selection_mask) > 0
    if baseline_depth.ndim != 2 or provider_depth.shape != baseline_depth.shape:
        raise ValueError("Provider and baseline depth grids must match")
    if face_mask.shape != baseline_depth.shape:
        raise ValueError("Face mask must use the baseline depth grid")
    if selection_mask.shape != baseline_depth.shape:
        raise ValueError("Selection mask must use the baseline depth grid")
    x0, y0, x1, y1 = (int(value) for value in crop_bbox_xyxy)
    if (
        x0 < 0
        or y0 < 0
        or x1 > baseline_depth.shape[1]
        or y1 > baseline_depth.shape[0]
        or x1 <= x0
        or y1 <= y0
    ):
        raise ValueError("Face crop bbox is invalid")
    crop_shape = (y1 - y0, x1 - x0)
    if local_face_depth.shape != crop_shape:
        local_face_depth = cv2.resize(
            local_face_depth,
            (crop_shape[1], crop_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
    local_face_mask = face_mask[y0:y1, x0:x1]
    if np.count_nonzero(local_face_mask) < 24:
        raise ValueError("Face mask has insufficient local support")

    provider_crop = provider_depth[y0:y1, x0:x1]
    provider_filled, fill_stats = fill_depth_nearest(provider_crop)
    provider_normalized, normalization = normalize_front_depth(
        provider_filled,
        lower_percentile=1.0,
        upper_percentile=99.0,
    )
    fit_mask = local_face_mask & np.isfinite(local_face_depth)
    design = np.column_stack(
        (
            provider_normalized[fit_mask].astype(np.float64),
            np.ones(np.count_nonzero(fit_mask), dtype=np.float64),
        )
    )
    provider_scale, provider_offset = np.linalg.lstsq(
        design,
        local_face_depth[fit_mask].astype(np.float64),
        rcond=None,
    )[0]
    provider_aligned = (
        provider_normalized * float(provider_scale)
        + float(provider_offset)
    ).astype(np.float32)
    sigma_pixels = max(
        0.75,
        float(min(crop_shape)) * float(sigma_ratio),
    )
    provider_low_frequency = gaussian_filter(
        provider_aligned,
        sigma=sigma_pixels,
    )
    local_low_frequency = gaussian_filter(
        local_face_depth,
        sigma=sigma_pixels,
    )
    surface_candidate = (
        local_face_depth
        + float(provider_alpha)
        * (provider_low_frequency - local_low_frequency)
    ).astype(np.float32)
    raw_crop, _raw_weight, fusion_stats = fuse_face_surface_residual(
        baseline_depth[y0:y1, x0:x1],
        local_face_depth,
        surface_candidate,
        local_face_mask.astype(np.uint8) * 255,
    )
    raw_full = baseline_depth.copy()
    raw_full[y0:y1, x0:x1] = raw_crop

    taper = subject_interior_taper(
        selection_mask,
        taper_pixels=taper_pixels,
        boundary_pixels=SUBJECT_BOUNDARY_PIXELS,
    )
    correction = raw_full - baseline_depth
    candidate = baseline_depth.copy()
    active = taper > 0
    candidate[active] = (
        baseline_depth[active] + correction[active] * taper[active]
    )
    selection_distance = cv2.distanceTransform(
        selection_mask.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    boundary = (
        (selection_distance > 0)
        & (selection_distance <= SUBJECT_BOUNDARY_PIXELS)
    )
    provider_finite = np.isfinite(provider_crop)
    stats = {
        "method": METHOD,
        "provider_alpha": float(provider_alpha),
        "sigma_ratio": float(sigma_ratio),
        "sigma_pixels": float(sigma_pixels),
        "subject_taper_pixels": float(taper_pixels),
        "subject_boundary_pixels": float(SUBJECT_BOUNDARY_PIXELS),
        "provider_scale": float(provider_scale),
        "provider_offset": float(provider_offset),
        "provider_crop_finite_pixels": int(
            np.count_nonzero(provider_finite)
        ),
        "provider_crop_pixels": int(provider_crop.size),
        "provider_crop_coverage": float(
            np.count_nonzero(provider_finite) / provider_crop.size
        ),
        "fill": fill_stats,
        "normalization": normalization,
        "fusion": fusion_stats,
        "outside_selection_max_abs_change": float(
            np.max(
                np.abs(
                    candidate[~selection_mask]
                    - baseline_depth[~selection_mask]
                )
            )
        ),
        "selection_boundary_max_abs_change": (
            float(
                np.max(
                    np.abs(
                        candidate[boundary] - baseline_depth[boundary]
                    )
                )
            )
            if np.any(boundary)
            else 0.0
        ),
        "selection_background_value_exact": bool(
            np.array_equal(
                candidate[~selection_mask],
                baseline_depth[~selection_mask],
            )
        ),
        "selection_boundary_value_exact": bool(
            not np.any(boundary)
            or np.array_equal(
                candidate[boundary],
                baseline_depth[boundary],
            )
        ),
        "maximum_abs_change": float(
            np.max(np.abs(candidate - baseline_depth))
        ),
    }
    return candidate.astype(np.float32), stats


def _decision(
    baseline: dict,
    candidate: dict,
    row_contracts: list[dict],
    *,
    baseline_matches_historical: bool,
    provider_runnable: bool,
) -> dict:
    baseline_by_id = {row["row_id"]: row for row in baseline["rows"]}
    candidate_by_id = {row["row_id"]: row for row in candidate["rows"]}
    contract_by_id = {
        contract["row_id"]: contract for contract in row_contracts
    }
    hard_baseline = baseline_by_id.get(HARD_SMALL_FACE_ROW, {})
    hard_candidate = candidate_by_id.get(HARD_SMALL_FACE_ROW, {})
    hard_contract = contract_by_id.get(HARD_SMALL_FACE_ROW, {})
    bypass_contracts = [
        contract
        for contract in row_contracts
        if not contract["policy"]["eligible"]
    ]
    checks = {
        "provider_preflight_runnable": bool(provider_runnable),
        "baseline_matches_historical": bool(baseline_matches_historical),
        "three_exact_face_rows": bool(
            set(baseline_by_id)
            == set(candidate_by_id)
            == {
                "small_side_lit_shelves_256",
                "eyewear_overhead_panel_384",
                "strong_turn_layered_384",
            }
        ),
        "aggregate_named_parts_improve": bool(
            candidate["combined_part_failures"]
            < baseline["combined_part_failures"]
        ),
        "per_row_named_parts_do_not_regress": bool(
            set(baseline_by_id) == set(candidate_by_id)
            and all(
                candidate_by_id[row_id]["combined_part_failures"]
                <= baseline_by_id[row_id]["combined_part_failures"]
                for row_id in baseline_by_id
            )
        ),
        "hard_small_face_improves": bool(
            hard_candidate.get("combined_part_failures", math.inf)
            < hard_baseline.get("combined_part_failures", -math.inf)
        ),
        "hard_shape_improves": bool(
            hard_candidate.get("shape_correlation", -math.inf)
            > hard_baseline.get("shape_correlation", math.inf)
        ),
        "hard_rmse_improves": bool(
            hard_candidate.get("normalized_rmse", math.inf)
            < hard_baseline.get("normalized_rmse", -math.inf)
        ),
        "hard_gradient_within_tolerance": bool(
            hard_candidate.get("gradient_correlation", -math.inf)
            >= hard_baseline.get("gradient_correlation", math.inf)
            - MAXIMUM_HARD_ROW_GRADIENT_REGRESSION
        ),
        "hard_provider_pose_aligned": bool(
            hard_contract.get("provider", {}).get(
                "absolute_yaw_magnitude_error_deg",
                math.inf,
            )
            <= MAXIMUM_POSE_MAGNITUDE_ERROR_DEG
        ),
        "hard_provider_confident": bool(
            hard_contract.get("provider", {}).get(
                "confidence",
                -math.inf,
            )
            >= MINIMUM_PROVIDER_CONFIDENCE
        ),
        "hard_provider_crop_covered": bool(
            hard_contract.get("fusion", {}).get(
                "provider_crop_coverage",
                -math.inf,
            )
            >= MINIMUM_PROVIDER_CROP_COVERAGE
        ),
        "selection_background_exact": bool(
            row_contracts
            and all(
                contract["background_value_exact"]
                for contract in row_contracts
            )
        ),
        "selection_attachment_exact": bool(
            row_contracts
            and all(
                contract["boundary_value_exact"]
                for contract in row_contracts
            )
        ),
        "bypassed_rows_value_exact": bool(
            bypass_contracts
            and all(
                contract["candidate_maximum_absolute_difference"] == 0.0
                for contract in bypass_contracts
            )
        ),
    }
    checks["eligible_for_30mm_stl_replay"] = bool(all(checks.values()))
    return checks


def evaluate(
    baseline_root: str | Path,
    provider_root: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    *,
    asset_dir: str | Path = DEFAULT_ASSET_DIR,
    device: str = "cuda",
    maximum_face_height_pixels: int = VGGHEADS_SMALL_FACE_MAX_HEIGHT_PX,
) -> dict:
    baseline_root = Path(baseline_root)
    provider_root = Path(provider_root)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = load_makehuman_face_fixture(asset_dir)
    provider = VGGHeadsProvider(
        provider_root,
        model_path,
        device=device,
    )
    historical = _historical_baseline(baseline_root)
    baseline_rows = []
    candidate_rows = []
    row_contracts = []
    started = time.perf_counter()

    face_specs = [
        spec
        for spec in VARIED_CONTEXT_MATRIX
        if isinstance(spec, FaceSceneSpec)
    ]
    for spec in face_specs:
        row_dir = output_dir / "rows" / spec.row_id
        exact_dir = row_dir / "exact"
        exact_dir.mkdir(parents=True, exist_ok=True)
        (
            source_path,
            selection_mask_path,
            exact_depth_path,
            part_mask_paths,
            render,
        ) = _stage_scene(exact_dir, spec, fixture)
        if part_mask_paths is None:
            raise ValueError("VGGHeads exact gate requires face-part masks")

        baseline_dir = baseline_root / spec.row_id / "baseline"
        baseline_path = baseline_dir / "output_depth_data_face_refined.npy"
        metadata_path = (
            baseline_dir / "output_face_refinement_metadata.json"
        )
        local_depth_path = (
            baseline_dir
            / "face_refinement"
            / "face_00_depth"
            / "output_depth_data.npy"
        )
        face_mask_path = (
            baseline_dir
            / "face_refinement"
            / "face_00_parts"
            / "face.png"
        )
        required = (
            baseline_path,
            metadata_path,
            local_depth_path,
            face_mask_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "VGGHeads baseline artifacts are unavailable: "
                + ", ".join(missing)
            )

        baseline_depth = np.load(baseline_path).astype(np.float32)
        baseline_quality = _quality(
            baseline_path,
            exact_depth_path,
            selection_mask_path,
            part_mask_paths,
        )
        baseline_rows.append(
            {
                "row_id": spec.row_id,
                "face_height_pixels": int(
                    render["face_bbox_height_pixels"]
                ),
                **baseline_quality,
            }
        )
        policy = small_face_policy(
            render["face_bbox_height_pixels"],
            occluded=spec.occluder is not None,
            maximum_height_pixels=maximum_face_height_pixels,
        )
        candidate_path = row_dir / "candidate_depth.npy"
        provider_record = None
        fusion_stats = None
        if policy["eligible"]:
            inference = provider.infer(
                source_path,
                target_bbox_xyxy=render["face_bbox_xyxy"],
            )
            provider_depth, raster_stats = rasterize_projected_mesh_depth(
                inference["vertices"],
                inference["faces"],
                height=baseline_depth.shape[0],
                width=baseline_depth.shape[1],
                front_surface="minimum-z",
            )
            np.save(row_dir / "provider_depth_min.npy", provider_depth)
            np.save(row_dir / "provider_vertices.npy", inference["vertices"])
            np.save(row_dir / "provider_faces.npy", inference["faces"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            face_record = metadata["faces"][0]
            crop_bbox = face_record["crop_bbox"]
            local_depth = np.load(local_depth_path).astype(np.float32)
            with Image.open(face_mask_path) as loaded:
                face_mask = np.asarray(loaded.convert("L"))
            with Image.open(selection_mask_path) as loaded:
                selection_mask = np.asarray(loaded.convert("L"))
            candidate_depth, fusion_stats = build_small_face_candidate(
                baseline_depth,
                local_depth,
                provider_depth,
                face_mask,
                selection_mask,
                crop_bbox,
            )
            known_yaw = float(spec.camera_yaw_deg)
            inferred_yaw = float(
                inference["metadata"]["head_pose_deg"]["yaw"]
            )
            provider_record = {
                **inference["metadata"],
                "known_yaw_deg": known_yaw,
                "absolute_yaw_magnitude_error_deg": abs(
                    abs(inferred_yaw) - abs(known_yaw)
                ),
                "raster": raster_stats,
            }
        else:
            candidate_depth = baseline_depth.copy()
        np.save(candidate_path, candidate_depth)
        candidate_quality = _quality(
            candidate_path,
            exact_depth_path,
            selection_mask_path,
            part_mask_paths,
        )
        candidate_rows.append(
            {
                "row_id": spec.row_id,
                "face_height_pixels": int(
                    render["face_bbox_height_pixels"]
                ),
                **candidate_quality,
            }
        )

        with Image.open(selection_mask_path) as loaded:
            selection = np.asarray(loaded.convert("L")) > 0
        distance = cv2.distanceTransform(
            selection.astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        boundary = (
            (distance > 0)
            & (distance <= SUBJECT_BOUNDARY_PIXELS)
        )
        row_contracts.append(
            {
                "row_id": spec.row_id,
                "policy": policy,
                "source_sha256": _sha256(source_path),
                "selection_mask_sha256": _sha256(selection_mask_path),
                "exact_depth_sha256": _sha256(exact_depth_path),
                "baseline_sha256": _sha256(baseline_path),
                "candidate_sha256": _sha256(candidate_path),
                "candidate_maximum_absolute_difference": float(
                    np.max(np.abs(candidate_depth - baseline_depth))
                ),
                "background_maximum_absolute_difference": float(
                    np.max(
                        np.abs(
                            candidate_depth[~selection]
                            - baseline_depth[~selection]
                        )
                    )
                ),
                "boundary_maximum_absolute_difference": (
                    float(
                        np.max(
                            np.abs(
                                candidate_depth[boundary]
                                - baseline_depth[boundary]
                            )
                        )
                    )
                    if np.any(boundary)
                    else 0.0
                ),
                "background_value_exact": bool(
                    np.array_equal(
                        candidate_depth[~selection],
                        baseline_depth[~selection],
                    )
                ),
                "boundary_value_exact": bool(
                    not np.any(boundary)
                    or np.array_equal(
                        candidate_depth[boundary],
                        baseline_depth[boundary],
                    )
                ),
                "provider": provider_record,
                "fusion": fusion_stats,
            }
        )

    baseline = _summary(baseline_rows)
    candidate = _summary(candidate_rows)
    baseline_matches_historical = _baseline_matches_historical(
        baseline,
        historical,
    )
    decision = _decision(
        baseline,
        candidate,
        row_contracts,
        baseline_matches_historical=baseline_matches_historical,
        provider_runnable=provider.preflight["runnable"],
    )
    evidence = {
        "schema_version": 1,
        "status": (
            "eligible-for-30mm-stl-replay"
            if decision["eligible_for_30mm_stl_replay"]
            else "hold"
        ),
        "method": METHOD,
        "privacy": (
            "CC0 MakeHuman synthetic heads and deterministic procedural "
            "scenes only"
        ),
        "production_changed": False,
        "production_eligible": False,
        "configuration": {
            "maximum_face_height_pixels": int(
                maximum_face_height_pixels
            ),
            "provider_blend_alpha": PROVIDER_BLEND_ALPHA,
            "low_frequency_sigma_ratio": LOW_FREQUENCY_SIGMA_RATIO,
            "subject_taper_pixels": SUBJECT_TAPER_PIXELS,
            "subject_boundary_pixels": SUBJECT_BOUNDARY_PIXELS,
            "front_surface": "minimum-z",
        },
        "provider": provider.provenance(),
        "baseline": baseline,
        "candidate": candidate,
        "row_contracts": row_contracts,
        "decision": decision,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--provider-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--maximum-face-height-pixels",
        type=int,
        default=VGGHEADS_SMALL_FACE_MAX_HEIGHT_PX,
    )
    args = parser.parse_args()
    evidence = evaluate(
        args.baseline_root,
        args.provider_root,
        args.model_path,
        args.output_dir,
        asset_dir=args.asset_dir,
        device=args.device,
        maximum_face_height_pixels=args.maximum_face_height_pixels,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["eligible_for_30mm_stl_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
