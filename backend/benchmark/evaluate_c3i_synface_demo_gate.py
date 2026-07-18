"""Replay the current face-depth path on the pinned C3I-SynFace demo corpus."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    _exact_face_depth_quality,
)
from backend.benchmark.c3i_synface_corpus import (
    DEMO_ASSET_LICENSE,
    verified_corpus_asset,
)
from backend.benchmark.train_face_depth_head import (
    MODEL_ID,
    MODEL_REVISION,
    _padded_box,
)
from backend.benchmark.train_face_surface_fusion_adapter import (
    _detect_production_region,
    _infer_depth_pair,
    _production_baseline,
    selected_image_from_exact_mask,
)
from backend.benchmark.train_face_surface_adapter import _sha256


METHOD = "c3i-synface-demo-current-face-depth-gate"
RAW_C3I_PROVIDER = "c3i-synface-female-part2-raw-exr"
RAP3DF_PROVIDER = "rap3df-v2-kinect-one-raw-depth"
MAXIMUM_ABSOLUTE_PART_FAILURE_RATE = 0.50
MAXIMUM_PER_ROW_METRIC_REGRESSION = 0.01


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as loaded:
        return np.asarray(loaded.convert("L")) >= 128


def _compact_quality(metrics: dict) -> dict:
    shape = metrics.get("named_part_shape") or {}
    affine = metrics.get("named_part_affine_mm") or {}
    shape_failed = list(shape.get("failed_parts") or ())
    affine_failed = list(affine.get("failed_parts") or ())
    return {
        "available": bool(metrics.get("available")),
        "coverage_ratio": float(metrics.get("coverage_ratio", 0.0)),
        "shape_correlation": float(metrics.get("shape_correlation", np.nan)),
        "gradient_correlation": float(
            metrics.get("gradient_correlation", np.nan)
        ),
        "normalized_rmse": float(metrics.get("normalized_rmse", np.nan)),
        "shape_failed_parts": shape_failed,
        "affine_failed_parts": affine_failed,
        "combined_part_failures": len(shape_failed) + len(affine_failed),
        "shape_check_failures": dict(shape.get("check_failures") or {}),
        "checks": dict(metrics.get("checks") or {}),
    }


def _corpus_profile(corpus: dict) -> dict:
    provider = corpus.get("provider")
    if provider == "c3i-synface-official-demo-depth":
        if corpus.get("demo_asset_license") != DEMO_ASSET_LICENSE:
            raise ValueError("C3I gate requires explicit demo-license provenance")
        return {
            "method": METHOD,
            "depth_target_provenance": {
                "representation": "official 8-bit grayscale demo preview",
                "use": "bounded relative-shape screen only",
                "not_claimed": [
                    "raw EXR precision",
                    "metric-scale validation",
                    "identity-disjoint training evidence",
                ],
            },
        }
    if provider == RAW_C3I_PROVIDER:
        if (
            corpus.get("dataset_license") != "CC BY 4.0"
            or not corpus.get("identity_disjoint_splits")
            or not corpus.get("source_geometry_training_and_evaluation_only")
        ):
            raise ValueError("Raw C3I gate requires licensed disjoint provenance")
        return {
            "method": "c3i-synface-raw-exr-current-face-depth-gate",
            "depth_target_provenance": {
                "representation": "official float32 Blender Z-pass EXR",
                "use": "identity-disjoint relative facial-shape supervision",
                "not_claimed": [
                    "direct 30 mm relief coordinates",
                    "absolute facial millimetres without affine fitting",
                ],
            },
        }
    if provider == RAP3DF_PROVIDER:
        if (
            corpus.get("training_eligible") is not False
            or not corpus.get("source_geometry_evaluation_only")
            or (corpus.get("source") or {}).get("license") != "CC BY 4.0"
        ):
            raise ValueError("RAP3DF gate requires evaluation-only provenance")
        return {
            "method": "rap3df-v2-current-face-depth-gate",
            "depth_target_provenance": {
                "representation": "raw Kinect One uint16 depth",
                "use": "real-sensor relative facial-shape diagnostic only",
                "not_claimed": [
                    "training eligibility",
                    "millimetre scale without published calibration",
                    "pixel-perfect RGB/depth registration",
                ],
            },
        }
    raise ValueError("Face-depth gate received an unsupported corpus provider")


def summarize_rows(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    if not values:
        raise ValueError("C3I gate cannot summarize zero rows")
    return {
        "row_count": len(values),
        "combined_part_failures": int(
            sum(value["combined_part_failures"] for value in values)
        ),
        "part_failure_opportunities": len(values) * len(FACE_PART_NAMES) * 2,
        "part_failure_rate": float(
            sum(value["combined_part_failures"] for value in values)
            / (len(values) * len(FACE_PART_NAMES) * 2)
        ),
        "median_shape_correlation": float(
            np.median([value["shape_correlation"] for value in values])
        ),
        "median_gradient_correlation": float(
            np.median([value["gradient_correlation"] for value in values])
        ),
        "median_normalized_rmse": float(
            np.median([value["normalized_rmse"] for value in values])
        ),
        "available_rows": int(sum(value["available"] for value in values)),
    }


def gate_decision(
    global_summary: dict,
    current_summary: dict,
    *,
    rows: list[dict] | None = None,
    expected_row_count: int | None = None,
) -> dict:
    rows = rows or []
    required_rows = (
        int(expected_row_count)
        if expected_row_count is not None
        else int(current_summary["row_count"])
    )
    pairing_complete = bool(
        rows
        and len(rows) == required_rows
        and len(rows) == global_summary["row_count"]
        and len(rows) == current_summary["row_count"]
        and len({row.get("row_id") for row in rows}) == len(rows)
        and None not in {row.get("row_id") for row in rows}
    )

    def candidate_check(row: dict, name: str) -> bool:
        return bool((row["current_refined"].get("checks") or {}).get(name))

    def finite_candidate(row: dict) -> bool:
        candidate = row["current_refined"]
        return bool(
            np.isfinite(candidate.get("shape_correlation", np.nan))
            and np.isfinite(candidate.get("gradient_correlation", np.nan))
            and np.isfinite(candidate.get("normalized_rmse", np.nan))
        )

    checks = {
        "paired_rows_complete": pairing_complete,
        "full_corpus_evaluated": len(rows) == required_rows,
        "all_current_rows_available": (
            current_summary["available_rows"] == current_summary["row_count"]
        ),
        "all_current_rows_have_full_coverage": (
            pairing_complete
            and all(candidate_check(row, "coverage") for row in rows)
        ),
        "all_current_rows_have_expected_orientation": (
            pairing_complete
            and all(
                candidate_check(row, "depth_semantics_orientation")
                for row in rows
            )
        ),
        "all_current_metrics_finite": (
            pairing_complete and all(finite_candidate(row) for row in rows)
        ),
        "part_failures_no_regression": (
            current_summary["combined_part_failures"]
            <= global_summary["combined_part_failures"]
        ),
        "median_shape_no_regression": (
            current_summary["median_shape_correlation"]
            >= global_summary["median_shape_correlation"] - 1e-8
        ),
        "median_gradient_no_regression": (
            current_summary["median_gradient_correlation"]
            >= global_summary["median_gradient_correlation"] - 1e-8
        ),
        "median_rmse_no_regression": (
            current_summary["median_normalized_rmse"]
            <= global_summary["median_normalized_rmse"] + 1e-8
        ),
        "per_row_part_failures_no_regression": pairing_complete and all(
            row["current_refined"]["combined_part_failures"]
            <= row["global_depth"]["combined_part_failures"]
            for row in rows
        ),
        "per_row_shape_regression_bounded": pairing_complete and all(
            row["current_refined"]["shape_correlation"]
            >= row["global_depth"]["shape_correlation"]
            - MAXIMUM_PER_ROW_METRIC_REGRESSION
            for row in rows
        ),
        "per_row_gradient_regression_bounded": pairing_complete and all(
            row["current_refined"]["gradient_correlation"]
            >= row["global_depth"]["gradient_correlation"]
            - MAXIMUM_PER_ROW_METRIC_REGRESSION
            for row in rows
        ),
        "per_row_rmse_regression_bounded": pairing_complete and all(
            row["current_refined"]["normalized_rmse"]
            <= row["global_depth"]["normalized_rmse"]
            + MAXIMUM_PER_ROW_METRIC_REGRESSION
            for row in rows
        ),
        "absolute_part_failure_rate_within_limit": (
            current_summary["part_failure_rate"]
            <= MAXIMUM_ABSOLUTE_PART_FAILURE_RATE
        ),
        "at_least_one_strict_improvement": (
            current_summary["combined_part_failures"]
            < global_summary["combined_part_failures"]
            or current_summary["median_shape_correlation"]
            > global_summary["median_shape_correlation"] + 1e-5
            or current_summary["median_gradient_correlation"]
            > global_summary["median_gradient_correlation"] + 1e-5
            or current_summary["median_normalized_rmse"]
            < global_summary["median_normalized_rmse"] - 1e-5
        ),
    }
    return {
        "status": "advance-current-path" if all(checks.values()) else "hold",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def evaluate_c3i_demo(
    corpus_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    limit: int | None = None,
) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    corpus_root = Path(corpus_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = corpus_root / "summary.json"
    corpus = json.loads(summary_path.read_text(encoding="utf-8"))
    profile = _corpus_profile(corpus)
    rows = list(corpus["rows"])
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("C3I gate selected no rows")

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

    started = time.perf_counter()
    measured = []
    for index, row in enumerate(rows, start=1):
        source_path = verified_corpus_asset(corpus_root, row["source"])
        mask_path = verified_corpus_asset(corpus_root, row["selection_mask"])
        exact_path = verified_corpus_asset(corpus_root, row["exact_depth"])
        source = Image.open(source_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        selected = selected_image_from_exact_mask(source, mask)
        selection = np.asarray(mask) >= 128
        region, detection = _detect_production_region(
            np.asarray(selected),
            selection,
        )
        crop_box = _padded_box(
            region["bbox"],
            selected.width,
            selected.height,
            0.35,
        )
        global_depth, local_depth = _infer_depth_pair(
            processor,
            model,
            selected,
            selected.crop(crop_box),
            device=device,
            dtype=dtype,
        )
        current_depth, refinement = _production_baseline(
            selected,
            global_depth,
            local_depth,
            region,
        )
        with tempfile.TemporaryDirectory(prefix="c3i-demo-gate-") as temporary:
            temporary = Path(temporary)
            global_path = temporary / "global.npy"
            current_path = temporary / "current.npy"
            np.save(global_path, global_depth.astype(np.float32))
            np.save(current_path, current_depth.astype(np.float32))
            part_paths = {
                name: verified_corpus_asset(
                    corpus_root,
                    row["exact_face_parts"][name],
                )
                for name in FACE_PART_NAMES
            }
            global_quality = _compact_quality(
                _exact_face_depth_quality(
                    global_path,
                    exact_path,
                    mask_path,
                    expected_scale_sign=-1.0,
                    part_mask_paths=part_paths,
                )
            )
            current_quality = _compact_quality(
                _exact_face_depth_quality(
                    current_path,
                    exact_path,
                    mask_path,
                    expected_scale_sign=-1.0,
                    part_mask_paths=part_paths,
                )
            )
        measured.append(
            {
                "row_id": row["row_id"],
                "face_height_pixels": row["render"]["face_bbox_height_pixels"],
                "detection": detection,
                "refinement": {
                    "refined_faces": refinement.get("refined_faces"),
                    "skipped_faces": refinement.get("skipped_faces"),
                    "detector_face_height_pixels": (
                        region["bbox"][3] - region["bbox"][1]
                    ),
                    "parametric_foundation": (
                        refinement.get("faces", [{}])[0].get(
                            "parametric_foundation"
                        )
                        if refinement.get("faces")
                        else None
                    ),
                },
                "global_depth": global_quality,
                "current_refined": current_quality,
                "part_failure_delta": (
                    current_quality["combined_part_failures"]
                    - global_quality["combined_part_failures"]
                ),
            }
        )
        print(f"measured {index}/{len(rows)} rows", flush=True)

    global_summary = summarize_rows(measured, "global_depth")
    current_summary = summarize_rows(measured, "current_refined")
    peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if str(device).startswith("cuda")
        else 0.0
    )
    results = {
        "schema_version": 1,
        "method": profile["method"],
        "status": "hold",
        "source_geometry_training_and_evaluation_only": True,
        "depth_target_provenance": profile["depth_target_provenance"],
        "corpus": {
            "summary_sha256": _sha256(summary_path),
            "provider": corpus["provider"],
            "dataset_doi": corpus.get("dataset_doi")
            or (corpus.get("source") or {}).get("doi"),
            "dataset_license": corpus.get("dataset_license")
            or (corpus.get("source") or {}).get("license")
            or corpus.get("demo_asset_license"),
            "identity_disjoint_splits": corpus.get(
                "identity_disjoint_splits"
            ),
            "training_eligible": corpus.get("training_eligible"),
            "row_count": len(rows),
            "full_row_count": int(
                corpus.get("full_balanced_row_count", corpus["row_count"])
            ),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "model_sha256": _sha256(snapshot / "model.safetensors"),
            "processor_sha256": _sha256(snapshot / "preprocessor_config.json"),
        },
        "runtime": {
            "device": str(device),
            "seconds": time.perf_counter() - started,
            "peak_torch_allocated_inference_gib": peak_vram,
        },
        "global_depth": global_summary,
        "current_refined": current_summary,
        "decision": gate_decision(
            global_summary,
            current_summary,
            rows=measured,
            expected_row_count=int(
                corpus.get("full_balanced_row_count", corpus["row_count"])
            ),
        ),
        "rows": measured,
    }
    results["status"] = results["decision"]["status"]
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    results = evaluate_c3i_demo(
        args.corpus_root,
        args.output_dir,
        device=args.device,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "status": results["status"],
                "global_failures": results["global_depth"][
                    "combined_part_failures"
                ],
                "current_failures": results["current_refined"][
                    "combined_part_failures"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
