"""Replay the production GNM face foundation on a held-out synthetic corpus."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    _exact_face_depth_quality,
)
from backend.benchmark.train_face_depth_head import (
    MODEL_ID,
    MODEL_REVISION,
    _padded_box,
)
from backend.benchmark.train_face_surface_fusion_adapter import (
    _detect_production_region,
    _infer_depth_pair,
    _sha256,
    selected_image_from_exact_mask,
)
from backend.face_depth_refinement import refine_depth_for_faces
from backend.gnm_face_foundation import (
    GNM_CENTRAL_CORRECTION_DILATION_PIXELS,
    GNM_CENTRAL_CORRECTION_FEATHER_SIGMA_PIXELS,
    GNM_GUARDED_CORRECTION_STRENGTH,
    GNM_HIGH_CONFIDENCE_ALIGNMENT_CORRELATION,
    GNM_HIGH_CONFIDENCE_ALIGNMENT_NORMALIZED_RMSE,
    GNM_LANDMARKS_SHA256,
    GNM_LICENSE,
    GNM_MAXIMUM_ALIGNMENT_NORMALIZED_RMSE,
    GNM_MAX_FACE_SUPPORT_PIXELS,
    GNM_MINIMUM_ALIGNMENT_CORRELATION,
    GNM_MODEL_SHA256,
    GNM_REVISION,
)


SMOKE_ROW_IDS = (
    "gnm_female_middle_eastern_v03__surprise_04",
    "gnm_female_asian_v03__mouth_right_04",
    "gnm_female_white_v03__smile_wide_02",
    "gnm_female_black_v03__wink_left_02",
    "gnm_male_middle_eastern_v03__happy_00",
    "gnm_male_asian_v03__corners_down_00",
    "gnm_male_white_v03__mouth_left_03",
    "gnm_male_black_v03__wink_right_03",
)
ROW_SETS = ("smoke", "all-small", "all")
MAXIMUM_PER_ROW_GRADIENT_REGRESSION = 0.01
MAXIMUM_ABSOLUTE_PART_FAILURE_RATE = 0.50


def _select_rows(summary: dict, row_set: str) -> list[dict]:
    rows = list(summary.get("rows") or [])
    if row_set == "smoke":
        by_id = {row["row_id"]: row for row in rows}
        missing = [row_id for row_id in SMOKE_ROW_IDS if row_id not in by_id]
        if missing:
            raise ValueError(
                "GNM smoke corpus is missing pinned rows: " + ", ".join(missing)
            )
        selected = [by_id[row_id] for row_id in SMOKE_ROW_IDS]
    elif row_set == "all-small":
        selected = [
            row
            for row in rows
            if int(row["render"]["face_bbox_height_pixels"]) <= 90
        ]
    elif row_set == "all":
        selected = rows
    else:
        raise ValueError(f"Unsupported row set {row_set!r}; expected {ROW_SETS}")
    if not selected:
        raise ValueError("GNM corpus gate selected no rows")
    if any(row.get("split") != "validation" for row in selected):
        raise ValueError("GNM corpus gate accepts validation rows only")
    return selected


def _identity_disjointness(
    selected_rows: list[dict],
    reference_summary: dict,
) -> dict:
    selected_identities = {
        str(row["identity_group"])
        for row in selected_rows
    }
    reference_rows = list(reference_summary.get("rows") or [])
    training_identities = {
        str(row["identity_group"])
        for row in reference_rows
        if row.get("split") == "train"
    }
    sealed_identities = {
        str(row["identity_group"])
        for row in reference_rows
        if row.get("split") == "sealed"
    }
    training_overlap = sorted(selected_identities & training_identities)
    sealed_overlap = sorted(selected_identities & sealed_identities)
    passed = not training_overlap and not sealed_overlap
    return {
        "passed": passed,
        "selected_identity_count": len(selected_identities),
        "reference_training_identity_count": len(training_identities),
        "reference_sealed_identity_count": len(sealed_identities),
        "training_overlap": training_overlap,
        "sealed_overlap": sealed_overlap,
    }


def _quality(corpus_root: Path, row: dict, candidate_path: Path) -> dict:
    metrics = _exact_face_depth_quality(
        candidate_path,
        corpus_root / row["exact_depth"]["path"],
        corpus_root / row["selection_mask"]["path"],
        expected_scale_sign=-1.0,
        part_mask_paths={
            name: corpus_root / row["exact_face_parts"][name]["path"]
            for name in FACE_PART_NAMES
        },
    )
    shape_failed = metrics["named_part_shape"]["failed_parts"]
    affine_failed = metrics["named_part_affine_mm"]["failed_parts"]
    return {
        "shape_correlation": metrics["shape_correlation"],
        "gradient_correlation": metrics["gradient_correlation"],
        "normalized_rmse": metrics["normalized_rmse"],
        "shape_failed_parts": shape_failed,
        "affine_failed_parts": affine_failed,
        "combined_part_failures": len(shape_failed) + len(affine_failed),
    }


def _summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty GNM gate")
    return {
        "row_count": len(rows),
        "combined_part_failures": int(
            sum(row["combined_part_failures"] for row in rows)
        ),
        "part_failure_opportunities": len(rows) * len(FACE_PART_NAMES) * 2,
        "part_failure_rate": float(
            sum(row["combined_part_failures"] for row in rows)
            / max(len(rows) * len(FACE_PART_NAMES) * 2, 1)
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
        "rows": rows,
    }


def _decision(
    control: dict,
    candidate: dict,
    *,
    selected_rows: list[dict],
) -> dict:
    control_by_id = {row["row_id"]: row for row in control["rows"]}
    candidate_by_id = {row["row_id"]: row for row in candidate["rows"]}
    applied = [
        row
        for row in candidate["rows"]
        if row["parametric_foundation"].get("enabled") is True
    ]
    skipped = [
        row
        for row in candidate["rows"]
        if row["parametric_foundation"].get("enabled") is not True
    ]
    checks = {
        "paired_failure_count_no_regression": (
            candidate["combined_part_failures"]
            <= control["combined_part_failures"]
        ),
        "paired_median_shape_no_regression": (
            candidate["median_shape_correlation"]
            >= control["median_shape_correlation"] - 1e-8
        ),
        "paired_median_gradient_no_regression": (
            candidate["median_gradient_correlation"]
            >= control["median_gradient_correlation"] - 1e-8
        ),
        "paired_median_rmse_no_regression": (
            candidate["median_normalized_rmse"]
            <= control["median_normalized_rmse"] + 1e-8
        ),
        "paired_per_row_failure_no_regression": all(
            row["combined_part_failures"]
            <= control_by_id[row["row_id"]]["combined_part_failures"]
            for row in candidate["rows"]
        ),
        "paired_per_row_gradient_regression_bounded": all(
            row["gradient_delta"] >= -MAXIMUM_PER_ROW_GRADIENT_REGRESSION
            for row in candidate["rows"]
        ),
        "absolute_part_failure_rate_within_limit": (
            candidate["part_failure_rate"]
            <= MAXIMUM_ABSOLUTE_PART_FAILURE_RATE
        ),
        "at_least_one_measured_improvement": any(
            row["combined_part_failures"]
            < control_by_id[row["row_id"]]["combined_part_failures"]
            or row["shape_delta"] > 1e-6
            or row["gradient_delta"] > 1e-6
            or row["normalized_rmse_delta"] < -1e-6
            for row in candidate["rows"]
        ),
        "at_least_half_faces_apply_foundation": (
            len(applied) >= max(1, int(np.ceil(len(candidate["rows"]) * 0.5)))
        ),
        "skips_are_reliability_gated_or_size_bypass": all(
            row["parametric_foundation"].get("reason")
            in {
                "alignment_reliability_gate",
                "face_support_above_small_face_gate",
            }
            for row in skipped
        ),
        "non_face_exact_control": all(
            row["maximum_difference_outside_face_region"] == 0.0
            for row in candidate["rows"]
        ),
        "face_boundary_zero": all(
            row["parametric_foundation"].get(
                "boundary_max_abs_correction",
                0.0,
            )
            <= 1e-7
            for row in applied
        ),
        "all_candidate_depth_finite": all(
            row["finite"] for row in candidate["rows"]
        ),
        "all_shared_inputs_identical": all(
            row["shared_input_equal"] for row in candidate["rows"]
        ),
        "all_cache_bboxes_reproduced": all(
            row["cache_bbox_equal"] for row in candidate["rows"]
        ),
        "all_selected_faces_small": all(
            int(row["render"]["face_bbox_height_pixels"]) <= 90
            for row in selected_rows
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    checks["applied_rows"] = len(applied)
    checks["skipped_rows"] = len(skipped)
    checks["candidate_rows"] = len(candidate_by_id)
    return checks


def _run_variant(
    *,
    selected_path: Path,
    global_depth_path: Path,
    local_depth: np.ndarray,
    expected_crop: np.ndarray,
    region: dict,
    output_dir: Path,
    enable_gnm_foundation: bool,
) -> tuple[Path, dict]:
    def cached_local(crop_path: Path, inference_dir: Path) -> Path:
        actual_crop = np.asarray(Image.open(crop_path).convert("RGB"))
        if not np.array_equal(actual_crop, expected_crop):
            raise ValueError("Production GNM gate crop differs from shared input")
        inference_dir = Path(inference_dir)
        inference_dir.mkdir(parents=True, exist_ok=True)
        path = inference_dir / "output_depth_data.npy"
        np.save(path, local_depth)
        return path

    output_path, metadata = refine_depth_for_faces(
        selected_path,
        global_depth_path,
        output_dir,
        infer_depth=cached_local,
        detector=lambda _image: [copy.deepcopy(region)],
        mode="on",
        enable_gnm_foundation=enable_gnm_foundation,
    )
    return Path(output_path), metadata


def evaluate(
    corpus_root: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    *,
    reference_corpus_summary: str | Path,
    row_set: str = "smoke",
    device: str = "cuda",
) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    corpus_root = Path(corpus_root)
    cache_root = Path(cache_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = corpus_root / "summary.json"
    cache_manifest_path = cache_root / "manifest.json"
    reference_summary_path = Path(reference_corpus_summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    reference_summary = json.loads(
        reference_summary_path.read_text(encoding="utf-8")
    )
    if cache_manifest.get("corpus_summary_sha256") != _sha256(summary_path):
        raise ValueError("Production cache does not match the GNM gate corpus")
    selected_rows = _select_rows(summary, row_set)
    identity_disjointness = _identity_disjointness(
        selected_rows,
        reference_summary,
    )
    if not identity_disjointness["passed"]:
        raise ValueError(
            "GNM corpus gate identities overlap reference train/sealed splits: "
            f"train={identity_disjointness['training_overlap']}, "
            f"sealed={identity_disjointness['sealed_overlap']}"
        )
    missing_cache_rows = [
        row["row_id"]
        for row in selected_rows
        if row["row_id"] not in set(cache_manifest.get("row_ids") or [])
    ]
    if missing_cache_rows:
        raise ValueError(
            "Production cache is missing selected rows: "
            + ", ".join(missing_cache_rows)
        )

    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    processor = AutoImageProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        use_fast=False,
    )
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForDepthEstimation.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    control_rows = []
    candidate_rows = []
    detection_scopes: dict[str, int] = {}
    started = time.perf_counter()
    for index, row in enumerate(selected_rows, start=1):
        row_id = row["row_id"]
        row_dir = output_dir / "rows" / row_id
        row_dir.mkdir(parents=True, exist_ok=True)
        source = Image.open(corpus_root / row["source"]["path"]).convert("RGB")
        selection_image = Image.open(
            corpus_root / row["selection_mask"]["path"]
        ).convert("L")
        selection = np.asarray(selection_image) >= 128
        selected = selected_image_from_exact_mask(source, selection_image)
        selected_rgb = np.asarray(selected)
        region, detection = _detect_production_region(
            selected_rgb,
            selection.astype(np.uint8) * 255,
        )
        detection_scopes[detection["scope"]] = (
            detection_scopes.get(detection["scope"], 0) + 1
        )
        crop_box = _padded_box(
            region["bbox"],
            selected.width,
            selected.height,
            0.35,
        )
        with np.load(
            cache_root / "rows" / f"{row_id}.npz",
            allow_pickle=False,
        ) as cached:
            cached_bbox = tuple(int(value) for value in cached["bbox"])
        global_depth, local_depth = _infer_depth_pair(
            processor,
            model,
            selected,
            selected.crop(crop_box),
            device=device,
            dtype=dtype,
        )
        selected_path = row_dir / "selected.png"
        global_depth_path = row_dir / "global_depth.npy"
        local_depth_path = row_dir / "local_depth.npy"
        selected.save(selected_path)
        np.save(global_depth_path, global_depth)
        np.save(local_depth_path, local_depth)
        expected_crop = np.asarray(selected.crop(crop_box).convert("RGB"))

        control_path, control_metadata = _run_variant(
            selected_path=selected_path,
            global_depth_path=global_depth_path,
            local_depth=local_depth,
            expected_crop=expected_crop,
            region=region,
            output_dir=row_dir / "control",
            enable_gnm_foundation=False,
        )
        candidate_path, candidate_metadata = _run_variant(
            selected_path=selected_path,
            global_depth_path=global_depth_path,
            local_depth=local_depth,
            expected_crop=expected_crop,
            region=region,
            output_dir=row_dir / "candidate",
            enable_gnm_foundation=True,
        )
        control_values = np.load(control_path).astype(np.float32)
        candidate_values = np.load(candidate_path).astype(np.float32)
        difference = np.abs(candidate_values - control_values)
        candidate_region_path = (
            row_dir / "candidate" / candidate_metadata["region_file"]
        )
        candidate_region = np.asarray(
            Image.open(candidate_region_path).convert("L")
        )
        control_quality = _quality(corpus_root, row, control_path)
        candidate_quality = _quality(corpus_root, row, candidate_path)
        foundation = candidate_metadata["faces"][0][
            "parametric_face_foundation"
        ]
        common = {
            "row_id": row_id,
            "identity_group": row["identity_group"],
            "expression": row["expression"],
            "occlusion": row["spec"].get("occlusion"),
            "camera_yaw_degrees": row["spec"]["camera_yaw_deg"],
            "face_height_pixels": row["render"]["face_bbox_height_pixels"],
            "detection_scope": detection["scope"],
            "detector": region.get("detector"),
        }
        control_rows.append({**common, **control_quality})
        candidate_rows.append(
            {
                **common,
                "parametric_foundation": foundation,
                "shape_delta": float(
                    candidate_quality["shape_correlation"]
                    - control_quality["shape_correlation"]
                ),
                "gradient_delta": float(
                    candidate_quality["gradient_correlation"]
                    - control_quality["gradient_correlation"]
                ),
                "normalized_rmse_delta": float(
                    candidate_quality["normalized_rmse"]
                    - control_quality["normalized_rmse"]
                ),
                "maximum_absolute_difference": float(np.max(difference)),
                "maximum_difference_outside_face_region": (
                    float(np.max(difference[candidate_region == 0]))
                    if np.any(candidate_region == 0)
                    else 0.0
                ),
                "finite": bool(np.all(np.isfinite(candidate_values))),
                "shared_input_equal": bool(
                    control_metadata["faces"][0]["crop_bbox"]
                    == candidate_metadata["faces"][0]["crop_bbox"]
                    and np.array_equal(
                        np.load(local_depth_path),
                        local_depth,
                    )
                ),
                "cache_bbox_equal": crop_box == cached_bbox,
                **candidate_quality,
            }
        )
        print(
            json.dumps(
                {
                    "completed": index,
                    "total": len(selected_rows),
                    "row_id": row_id,
                    "foundation_enabled": foundation.get("enabled"),
                    "foundation_tier": foundation.get("reliability_tier"),
                    "control_failures": control_quality[
                        "combined_part_failures"
                    ],
                    "candidate_failures": candidate_quality[
                        "combined_part_failures"
                    ],
                }
            ),
            flush=True,
        )

    control = _summary(control_rows)
    candidate = _summary(candidate_rows)
    decision = _decision(
        control,
        candidate,
        selected_rows=selected_rows,
    )
    peak_vram_gib = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if str(device).startswith("cuda")
        else 0.0
    )
    evidence = {
        "schema_version": 1,
        "status": "pass" if decision["passed"] else "hold",
        "method": "production-replay-reliability-calibrated-gnm-foundation",
        "row_set": row_set,
        "source_geometry_training_and_evaluation_only": True,
        "privacy": summary.get("privacy"),
        "corpus_summary_sha256": _sha256(summary_path),
        "production_cache_manifest_sha256": _sha256(cache_manifest_path),
        "reference_corpus_summary_sha256": _sha256(reference_summary_path),
        "identity_disjointness": identity_disjointness,
        "provider": {
            "gnm_revision": GNM_REVISION,
            "gnm_license": GNM_LICENSE,
            "gnm_model_sha256": GNM_MODEL_SHA256,
            "gnm_landmarks_sha256": GNM_LANDMARKS_SHA256,
            "maximum_face_support_pixels": GNM_MAX_FACE_SUPPORT_PIXELS,
            "minimum_alignment_correlation": (
                GNM_MINIMUM_ALIGNMENT_CORRELATION
            ),
            "maximum_alignment_normalized_rmse": (
                GNM_MAXIMUM_ALIGNMENT_NORMALIZED_RMSE
            ),
            "high_confidence_alignment_correlation": (
                GNM_HIGH_CONFIDENCE_ALIGNMENT_CORRELATION
            ),
            "high_confidence_alignment_normalized_rmse": (
                GNM_HIGH_CONFIDENCE_ALIGNMENT_NORMALIZED_RMSE
            ),
            "guarded_correction_strength": GNM_GUARDED_CORRECTION_STRENGTH,
            "central_correction_parts": [
                "nose",
                "left_eye",
                "right_eye",
            ],
            "central_correction_dilation_pixels": (
                GNM_CENTRAL_CORRECTION_DILATION_PIXELS
            ),
            "central_correction_feather_sigma_pixels": (
                GNM_CENTRAL_CORRECTION_FEATHER_SIGMA_PIXELS
            ),
            "maximum_per_row_gradient_regression": (
                MAXIMUM_PER_ROW_GRADIENT_REGRESSION
            ),
            "maximum_absolute_part_failure_rate": (
                MAXIMUM_ABSOLUTE_PART_FAILURE_RATE
            ),
        },
        "depth_provider": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_sha256": _sha256(snapshot / "model.safetensors"),
            "processor_sha256": _sha256(
                snapshot / "preprocessor_config.json"
            ),
            "device": device,
            "peak_vram_gib": peak_vram_gib,
        },
        "matrix": {
            "row_count": len(selected_rows),
            "identity_count": len(
                {row["identity_group"] for row in selected_rows}
            ),
            "expressions": sorted(
                {row["expression"] for row in selected_rows}
            ),
            "occlusions": sorted(
                {str(row["spec"].get("occlusion") or "none") for row in selected_rows}
            ),
            "yaw_degrees": sorted(
                {float(row["spec"]["camera_yaw_deg"]) for row in selected_rows}
            ),
            "detection_scope_counts": dict(sorted(detection_scopes.items())),
        },
        "paired_control": control,
        "candidate": candidate,
        "decision": decision,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-corpus-summary", required=True)
    parser.add_argument("--row-set", choices=ROW_SETS, default="smoke")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evidence = evaluate(
        args.corpus_root,
        args.cache_root,
        args.output_dir,
        reference_corpus_summary=args.reference_corpus_summary,
        row_set=args.row_set,
        device=args.device,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
