"""Gate a bounded TRG camera-depth prior on the hardest exact face row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.evaluate_vggheads_small_face_exact_gate import (
    SUBJECT_BOUNDARY_PIXELS,
    build_small_face_candidate,
)
from backend.benchmark.makehuman_face_fixture import load_makehuman_face_fixture
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    DEFAULT_ASSET_DIR,
    VARIED_CONTEXT_MATRIX,
    FaceSceneSpec,
    _exact_face_depth_quality,
    _stage_scene,
)
from backend.benchmark.trg_face_depth_provider import (
    TRGFaceDepthProvider,
)
from backend.benchmark.vggheads_depth_provider import (
    rasterize_projected_mesh_depth,
)
from backend.face_depth_refinement import FACE_PART_NAMES


METHOD = "trg-camera-depth-low-frequency-structural-prior"
HARD_ROW_ID = "small_side_lit_shelves_256"
PROVIDER_ALPHAS = (0.125, 0.25, 0.50)
MINIMUM_PROJECTED_BBOX_IOU = 0.50
MINIMUM_PROVIDER_FACE_COVERAGE = 0.90
COMPARISON_TOLERANCE = 1e-8
EXACT_ROW_CONTRACT = {
    "source_sha256": "2aca3775162f1b2320a47dbfead27e49e2d7b29654d80151d8ebc17315a8c074",
    "selection_mask_sha256": "853d871ac859dc0904cd866bc8e552a55efbc3af8ba59d2da3ee4aac3583d0ef",
    "exact_depth_sha256": "a21f6396c57abd850786e00d40cf2776e048f7b8d0664170e86cdca8fa27a21b",
    "baseline_sha256": "a9b2b8fd8c20efe731331d0b4f2a8f252acde7a9f1b187e79918217cc3bc4114",
    "baseline_metadata_sha256": "d737082ef65852d6363003f9b20746079bed3feb6e513b0caeaaa88d8b1bef5e",
    "baseline_face_mask_sha256": "c5d85b18b0cc7f2aedca799637b3b1e3cc98fc1a5537c51a09eb891dbeb8636a",
    "target_bbox_xyxy": [160, 102, 193, 139],
    "crop_bbox_xyxy": [148, 89, 205, 152],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_summary(metrics: dict) -> dict:
    shape = metrics["named_part_shape"]
    affine = metrics["named_part_affine_mm"]
    return {
        "shape_correlation": float(metrics["shape_correlation"]),
        "gradient_correlation": float(metrics["gradient_correlation"]),
        "normalized_rmse": float(metrics["normalized_rmse"]),
        "shape_failed_parts": list(shape["failed_parts"]),
        "affine_failed_parts": list(affine["failed_parts"]),
        "combined_part_failures": int(
            len(shape["failed_parts"]) + len(affine["failed_parts"])
        ),
        "checks": metrics["checks"],
    }


def _part_records(metrics: dict, family: str) -> tuple[dict[str, dict], dict]:
    family_record = metrics.get(family, {})
    parts = family_record.get("parts", [])
    if not isinstance(parts, list):
        parts = []
    names = [record.get("name") for record in parts if isinstance(record, dict)]
    expected = set(FACE_PART_NAMES)
    exact = bool(
        len(parts) == len(FACE_PART_NAMES)
        and len(names) == len(parts)
        and len(set(names)) == len(names)
        and set(names) == expected
    )
    return (
        {record["name"]: record for record in parts if isinstance(record, dict)},
        {
            "exact_names": exact,
            "names": names,
            "expected_names": sorted(expected),
            "top_level_passed": family_record.get("passed") is True,
            "every_record_passed": bool(
                exact and all(record.get("passed") is True for record in parts)
            ),
        },
    )


def _retention_error(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return math.inf
    return abs(float(value) - 1.0)


def _per_part_no_regression(baseline: dict, candidate: dict) -> dict:
    baseline_shape, baseline_shape_contract = _part_records(
        baseline, "named_part_shape"
    )
    candidate_shape, candidate_shape_contract = _part_records(
        candidate, "named_part_shape"
    )
    baseline_affine, baseline_affine_contract = _part_records(
        baseline, "named_part_affine_mm"
    )
    candidate_affine, candidate_affine_contract = _part_records(
        candidate, "named_part_affine_mm"
    )
    names_match = bool(
        baseline_shape_contract["exact_names"]
        and candidate_shape_contract["exact_names"]
        and baseline_affine_contract["exact_names"]
        and candidate_affine_contract["exact_names"]
        and set(baseline_shape)
        == set(candidate_shape)
        == set(baseline_affine)
        == set(candidate_affine)
    )
    records = []
    for name in sorted(baseline_shape) if names_match else []:
        old_shape = baseline_shape[name]
        new_shape = candidate_shape[name]
        old_affine = baseline_affine[name]
        new_affine = candidate_affine[name]
        checks = {
            "shape_correlation": bool(
                float(new_shape["shape_correlation"])
                >= float(old_shape["shape_correlation"])
                - COMPARISON_TOLERANCE
            ),
            "raw_gradient_correlation": bool(
                float(new_shape["minimum_raw_gradient_correlation"])
                >= float(old_shape["minimum_raw_gradient_correlation"])
                - COMPARISON_TOLERANCE
            ),
            "rmse_mm": bool(
                float(new_affine["rmse_mm"])
                <= float(old_affine["rmse_mm"]) + COMPARISON_TOLERANCE
            ),
            "p95_absolute_error_mm": bool(
                float(new_affine["p95_absolute_error_mm"])
                <= float(old_affine["p95_absolute_error_mm"])
                + COMPARISON_TOLERANCE
            ),
            "absolute_bias_mm": bool(
                abs(float(new_affine["bias_mm"]))
                <= abs(float(old_affine["bias_mm"])) + COMPARISON_TOLERANCE
            ),
            "span_retention": bool(
                _retention_error(new_affine.get("span_retention"))
                <= _retention_error(old_affine.get("span_retention"))
                + COMPARISON_TOLERANCE
            ),
        }
        records.append(
            {
                "name": name,
                "checks": checks,
                "passed": bool(all(checks.values())),
            }
        )
    return {
        "names_match": names_match,
        "telemetry_contract": {
            "baseline_shape": baseline_shape_contract,
            "candidate_shape": candidate_shape_contract,
            "baseline_affine_mm": baseline_affine_contract,
            "candidate_affine_mm": candidate_affine_contract,
        },
        "parts": records,
        "passed": bool(names_match and records and all(r["passed"] for r in records)),
    }


def promotion_decision(
    baseline_metrics: dict,
    candidate_metrics: dict,
    *,
    provider_runnable: bool,
    projected_bbox_iou: float,
    provider_face_coverage: float,
    background_value_exact: bool,
    boundary_value_exact: bool,
) -> dict:
    per_part = _per_part_no_regression(baseline_metrics, candidate_metrics)
    baseline_summary = _metric_summary(baseline_metrics)
    candidate_summary = _metric_summary(candidate_metrics)
    telemetry = per_part["telemetry_contract"]
    candidate_shape = telemetry["candidate_shape"]
    candidate_affine = telemetry["candidate_affine_mm"]
    checks = {
        "provider_preflight_runnable": bool(provider_runnable),
        "projected_bbox_registered": bool(
            math.isfinite(float(projected_bbox_iou))
            and float(projected_bbox_iou) >= MINIMUM_PROJECTED_BBOX_IOU
        ),
        "provider_face_covered": bool(
            math.isfinite(float(provider_face_coverage))
            and float(provider_face_coverage) >= MINIMUM_PROVIDER_FACE_COVERAGE
        ),
        "selection_background_exact": bool(background_value_exact),
        "selection_attachment_exact": bool(boundary_value_exact),
        "aggregate_shape_no_regression": bool(
            candidate_summary["shape_correlation"]
            >= baseline_summary["shape_correlation"] - COMPARISON_TOLERANCE
        ),
        "aggregate_gradient_no_regression": bool(
            candidate_summary["gradient_correlation"]
            >= baseline_summary["gradient_correlation"] - COMPARISON_TOLERANCE
        ),
        "aggregate_rmse_no_regression": bool(
            candidate_summary["normalized_rmse"]
            <= baseline_summary["normalized_rmse"] + COMPARISON_TOLERANCE
        ),
        "combined_named_part_failures_improve": bool(
            candidate_summary["combined_part_failures"]
            < baseline_summary["combined_part_failures"]
        ),
        "all_six_shape_parts_pass": bool(
            candidate_shape["exact_names"]
            and candidate_shape["top_level_passed"]
            and candidate_shape["every_record_passed"]
        ),
        "all_six_affine_mm_parts_pass": bool(
            candidate_affine["exact_names"]
            and candidate_affine["top_level_passed"]
            and candidate_affine["every_record_passed"]
        ),
        "every_part_metric_no_regression": bool(per_part["passed"]),
    }
    checks["eligible_for_30mm_stl_replay"] = bool(all(checks.values()))
    return {"checks": checks, "per_part_comparison": per_part}


def _selected_candidate(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("Candidate sweep is empty")
    return min(
        candidates,
        key=lambda record: (
            not record.get("decision", {})
            .get("checks", {})
            .get("eligible_for_30mm_stl_replay", False),
            record["summary"]["combined_part_failures"],
            -record["summary"]["shape_correlation"],
            -record["summary"]["gradient_correlation"],
            record["summary"]["normalized_rmse"],
            record["alpha"],
        ),
    )


def exact_row_contract_checks(
    input_hashes: dict,
    *,
    target_bbox_xyxy: list[int],
    crop_bbox_xyxy: list[int],
) -> dict:
    checks = {
        name: input_hashes.get(name) == expected
        for name, expected in EXACT_ROW_CONTRACT.items()
        if name.endswith("_sha256")
    }
    checks["target_bbox_xyxy"] = list(target_bbox_xyxy) == EXACT_ROW_CONTRACT[
        "target_bbox_xyxy"
    ]
    checks["crop_bbox_xyxy"] = list(crop_bbox_xyxy) == EXACT_ROW_CONTRACT[
        "crop_bbox_xyxy"
    ]
    return {
        "expected": EXACT_ROW_CONTRACT,
        "actual": {
            **input_hashes,
            "target_bbox_xyxy": list(target_bbox_xyxy),
            "crop_bbox_xyxy": list(crop_bbox_xyxy),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def evaluate(
    baseline_dir: str | Path,
    provider_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    asset_dir: str | Path = DEFAULT_ASSET_DIR,
    device: str = "cuda",
) -> dict:
    baseline_dir = Path(baseline_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = load_makehuman_face_fixture(asset_dir)
    spec = next(
        spec
        for spec in VARIED_CONTEXT_MATRIX
        if isinstance(spec, FaceSceneSpec) and spec.row_id == HARD_ROW_ID
    )
    exact_dir = output_dir / "exact"
    exact_dir.mkdir(parents=True, exist_ok=True)
    (
        source_path,
        selection_mask_path,
        exact_depth_path,
        part_mask_paths,
        render,
    ) = _stage_scene(exact_dir, spec, fixture)
    if part_mask_paths is None:
        raise ValueError("TRG exact gate requires all six face-part masks")

    baseline_path = baseline_dir / "output_depth_data_face_refined.npy"
    metadata_path = baseline_dir / "output_face_refinement_metadata.json"
    face_mask_path = baseline_dir / "face_refinement/face_00_parts/face.png"
    missing = [
        str(path)
        for path in (baseline_path, metadata_path, face_mask_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Current GNM baseline artifacts are unavailable: "
            + ", ".join(missing)
        )

    baseline_depth = np.load(baseline_path).astype(np.float32)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(metadata.get("faces", [])) != 1:
        raise ValueError("Exact TRG row requires exactly one baseline face")
    face_record = metadata["faces"][0]
    target_bbox = face_record["bbox"]
    crop_bbox = face_record["crop_bbox"]
    with Image.open(face_mask_path) as loaded:
        face_mask = np.asarray(loaded.convert("L"))
    with Image.open(selection_mask_path) as loaded:
        selection_mask = np.asarray(loaded.convert("L"))
    exact_depth = np.load(exact_depth_path)
    if (
        baseline_depth.shape != exact_depth.shape
        or face_mask.shape != exact_depth.shape
        or selection_mask.shape != exact_depth.shape
    ):
        raise ValueError("Exact TRG row artifacts do not share one depth grid")
    input_hashes = {
        "source_sha256": _sha256(source_path),
        "selection_mask_sha256": _sha256(selection_mask_path),
        "exact_depth_sha256": _sha256(exact_depth_path),
        "baseline_sha256": _sha256(baseline_path),
        "baseline_metadata_sha256": _sha256(metadata_path),
        "baseline_face_mask_sha256": _sha256(face_mask_path),
    }
    exact_contract = exact_row_contract_checks(
        input_hashes,
        target_bbox_xyxy=target_bbox,
        crop_bbox_xyxy=crop_bbox,
    )
    if not exact_contract["passed"]:
        failed = [
            name
            for name, passed in exact_contract["checks"].items()
            if not passed
        ]
        raise ValueError("Exact TRG row contract failed: " + ", ".join(failed))
    selection = selection_mask > 0
    selection_distance = cv2.distanceTransform(
        selection.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    boundary = (
        (selection_distance > 0)
        & (selection_distance <= SUBJECT_BOUNDARY_PIXELS)
    )
    baseline_metrics = _exact_face_depth_quality(
        baseline_path,
        exact_depth_path,
        selection_mask_path,
        expected_scale_sign=-1.0,
        part_mask_paths=part_mask_paths,
    )

    provider = TRGFaceDepthProvider(
        provider_root,
        checkpoint_path,
        device=device,
    )
    started = time.perf_counter()
    inference = provider.infer(
        source_path,
        target_bbox_xyxy=target_bbox,
    )
    provider_depth, raster_stats = rasterize_projected_mesh_depth(
        inference["vertices"],
        inference["faces"],
        height=baseline_depth.shape[0],
        width=baseline_depth.shape[1],
        front_surface="minimum-z",
    )
    np.save(output_dir / "provider_depth_min.npy", provider_depth)
    np.save(output_dir / "provider_camera_vertices.npy", inference["camera_vertices"])
    np.save(output_dir / "provider_faces.npy", inference["faces"])

    x0, y0, x1, y1 = (int(value) for value in crop_bbox)
    local_baseline = baseline_depth[y0:y1, x0:x1]
    local_face_mask = face_mask[y0:y1, x0:x1] > 0
    provider_face_samples = int(np.count_nonzero(local_face_mask))
    provider_face_finite_samples = int(
        np.count_nonzero(
            np.isfinite(provider_depth[y0:y1, x0:x1]) & local_face_mask
        )
    )
    provider_face_coverage = float(
        provider_face_finite_samples / max(provider_face_samples, 1)
    )
    provider_metadata = inference["metadata"]
    candidates = []
    for alpha in PROVIDER_ALPHAS:
        candidate_depth, fusion = build_small_face_candidate(
            baseline_depth,
            local_baseline,
            provider_depth,
            face_mask,
            selection_mask,
            crop_bbox,
            provider_alpha=alpha,
        )
        fusion["method"] = METHOD
        fusion["provider_face_finite_pixels"] = provider_face_finite_samples
        fusion["provider_face_pixels"] = provider_face_samples
        fusion["provider_face_coverage"] = provider_face_coverage
        alpha_name = str(alpha).replace(".", "p")
        candidate_path = output_dir / f"candidate_alpha_{alpha_name}.npy"
        np.save(candidate_path, candidate_depth)
        metrics = _exact_face_depth_quality(
            candidate_path,
            exact_depth_path,
            selection_mask_path,
            expected_scale_sign=-1.0,
            part_mask_paths=part_mask_paths,
        )
        summary = _metric_summary(metrics)
        background_exact = bool(
            np.array_equal(
                candidate_depth[~selection],
                baseline_depth[~selection],
            )
        )
        boundary_exact = bool(
            not np.any(boundary)
            or np.array_equal(
                candidate_depth[boundary],
                baseline_depth[boundary],
            )
        )
        candidate_decision = promotion_decision(
            baseline_metrics,
            metrics,
            provider_runnable=provider.preflight["runnable"],
            projected_bbox_iou=provider_metadata[
                "projected_target_bbox_iou"
            ],
            provider_face_coverage=provider_face_coverage,
            background_value_exact=background_exact,
            boundary_value_exact=boundary_exact,
        )
        candidates.append(
            {
                "alpha": float(alpha),
                "path": str(candidate_path),
                "sha256": _sha256(candidate_path),
                "summary": summary,
                "metrics": metrics,
                "fusion": fusion,
                "background_value_exact": background_exact,
                "boundary_value_exact": boundary_exact,
                "decision": candidate_decision,
                "maximum_background_abs_change": float(
                    np.max(
                        np.abs(
                            candidate_depth[~selection]
                            - baseline_depth[~selection]
                        )
                    )
                ),
                "maximum_boundary_abs_change": (
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
            }
        )

    selected = _selected_candidate(candidates)
    decision = selected["decision"]
    evidence = {
        "schema_version": 1,
        "status": (
            "eligible-for-30mm-stl-replay"
            if decision["checks"]["eligible_for_30mm_stl_replay"]
            else "hold"
        ),
        "method": METHOD,
        "privacy": "CC0 MakeHuman synthetic face and procedural background only",
        "production_changed": False,
        "production_eligible": False,
        "row_id": HARD_ROW_ID,
        "configuration": {
            "provider_alphas": list(PROVIDER_ALPHAS),
            "front_surface": "minimum-z",
            "minimum_projected_bbox_iou": MINIMUM_PROJECTED_BBOX_IOU,
            "minimum_provider_face_coverage": MINIMUM_PROVIDER_FACE_COVERAGE,
            "face_bbox_xyxy": target_bbox,
            "face_crop_bbox_xyxy": crop_bbox,
            "selection_bbox_xyxy": render["selection_bbox_xyxy"],
        },
        "inputs": input_hashes,
        "exact_row_contract": exact_contract,
        "provider": provider.provenance(),
        "inference": provider_metadata,
        "raster": raster_stats,
        "baseline": {
            "summary": _metric_summary(baseline_metrics),
            "metrics": baseline_metrics,
        },
        "candidates": candidates,
        "selected_alpha": selected["alpha"],
        "selected_summary": selected["summary"],
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
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--provider-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evidence = evaluate(
        args.baseline_dir,
        args.provider_root,
        args.checkpoint_path,
        args.output_dir,
        asset_dir=args.asset_dir,
        device=args.device,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["checks"]["eligible_for_30mm_stl_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
