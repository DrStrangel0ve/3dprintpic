"""Compare DAViD face-crop depth with the current GNM production path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.evaluate_gnm_face_foundation_corpus_gate import (
    SMOKE_ROW_IDS,
    _quality,
    _run_variant,
    _summary,
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
from backend.david_face_depth import DAViDFaceDepth


DAVID_SMOKE_ROW_IDS = ("gnm_female_white_v03__smile_wide_02",)
ROW_SETS = ("provider-smoke", "smoke", "all-small")


def _select_rows(summary: dict, row_set: str) -> list[dict]:
    rows = list(summary.get("rows") or [])
    by_id = {row["row_id"]: row for row in rows}
    if row_set == "provider-smoke":
        row_ids = DAVID_SMOKE_ROW_IDS
    elif row_set == "smoke":
        row_ids = SMOKE_ROW_IDS
    elif row_set == "all-small":
        row_ids = tuple(
            row["row_id"]
            for row in rows
            if int(row["render"]["face_bbox_height_pixels"]) <= 90
        )
    else:
        raise ValueError(f"Unsupported DAViD row set {row_set!r}")
    missing = [row_id for row_id in row_ids if row_id not in by_id]
    if missing:
        raise ValueError("DAViD corpus is missing rows: " + ", ".join(missing))
    selected = [by_id[row_id] for row_id in row_ids]
    if any(row.get("split") != "validation" for row in selected):
        raise ValueError("DAViD corpus gate accepts validation rows only")
    return selected


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _decision(current: dict, challenger: dict) -> dict:
    current_by_id = {row["row_id"]: row for row in current["rows"]}
    checks = {
        "paired_failure_count_no_regression": (
            challenger["combined_part_failures"]
            <= current["combined_part_failures"]
        ),
        "paired_median_shape_no_regression": (
            challenger["median_shape_correlation"]
            >= current["median_shape_correlation"] - 1e-8
        ),
        "paired_median_gradient_no_regression": (
            challenger["median_gradient_correlation"]
            >= current["median_gradient_correlation"] - 1e-8
        ),
        "paired_median_rmse_no_regression": (
            challenger["median_normalized_rmse"]
            <= current["median_normalized_rmse"] + 1e-8
        ),
        "paired_per_row_failure_no_regression": all(
            row["combined_part_failures"]
            <= current_by_id[row["row_id"]]["combined_part_failures"]
            for row in challenger["rows"]
        ),
        "at_least_one_measured_improvement": any(
            row["combined_part_failures"]
            < current_by_id[row["row_id"]]["combined_part_failures"]
            or row["shape_delta"] > 1e-6
            or row["gradient_delta"] > 1e-6
            or row["normalized_rmse_delta"] < -1e-6
            for row in challenger["rows"]
        ),
        "non_face_exact_control": all(
            row["maximum_difference_outside_face_region"] == 0.0
            for row in challenger["rows"]
        ),
        "all_candidate_depth_finite": all(
            row["finite"] for row in challenger["rows"]
        ),
        "all_shared_global_inputs_equal": all(
            row["shared_global_input_equal"] for row in challenger["rows"]
        ),
        "all_faces_small": all(
            row["face_height_pixels"] <= 90 for row in challenger["rows"]
        ),
        "all_foundation_outcomes_auditable": all(
            row["parametric_foundation"].get("enabled") is True
            or row["parametric_foundation"].get("reason")
            in {
                "alignment_reliability_gate",
                "face_support_above_small_face_gate",
            }
            for row in challenger["rows"]
        ),
        "all_local_depth_orientation_matches_control": all(
            row["local_depth_correlation_with_dav2"] > 0.0
            and row["local_depth_correlation_with_dav2"]
            > row["local_depth_negated_correlation_with_dav2"]
            for row in challenger["rows"]
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def evaluate(
    corpus_root: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    *,
    row_set: str = "provider-smoke",
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
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("corpus_summary_sha256") != _sha256(summary_path):
        raise ValueError("Production cache does not match the DAViD corpus")
    selected_rows = _select_rows(summary, row_set)

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
        backend="pil",
    )
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=dtype,
    ).to(device)
    depth_model.eval()
    david = DAViDFaceDepth(device=device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    control_rows = []
    current_rows = []
    challenger_rows = []
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
        region, detection = _detect_production_region(
            np.asarray(selected),
            selection.astype(np.uint8) * 255,
        )
        crop_box = _padded_box(
            region["bbox"],
            selected.width,
            selected.height,
            0.35,
        )
        expected_crop = np.asarray(selected.crop(crop_box).convert("RGB"))
        # Measure DAViD's first fixed-shape inference before DAv2 allocates
        # transient inference buffers; both models are already resident.
        local_david = david.infer_array(expected_crop)
        global_depth, local_dav2 = _infer_depth_pair(
            processor,
            depth_model,
            selected,
            selected.crop(crop_box),
            device=device,
            dtype=dtype,
        )
        selected_path = row_dir / "selected.png"
        global_depth_path = row_dir / "global_depth.npy"
        selected.save(selected_path)
        np.save(global_depth_path, global_depth)
        np.save(row_dir / "local_dav2.npy", local_dav2)
        np.save(row_dir / "local_david.npy", local_david)

        control_path, _control_metadata = _run_variant(
            selected_path=selected_path,
            global_depth_path=global_depth_path,
            local_depth=local_dav2,
            expected_crop=expected_crop,
            region=region,
            output_dir=row_dir / "control_no_gnm",
            enable_gnm_foundation=False,
        )
        current_path, current_metadata = _run_variant(
            selected_path=selected_path,
            global_depth_path=global_depth_path,
            local_depth=local_dav2,
            expected_crop=expected_crop,
            region=region,
            output_dir=row_dir / "current_gnm",
            enable_gnm_foundation=True,
        )
        challenger_path, challenger_metadata = _run_variant(
            selected_path=selected_path,
            global_depth_path=global_depth_path,
            local_depth=local_david,
            expected_crop=expected_crop,
            region=region,
            output_dir=row_dir / "david_gnm",
            enable_gnm_foundation=True,
        )
        control_quality = _quality(corpus_root, row, control_path)
        current_quality = _quality(corpus_root, row, current_path)
        challenger_quality = _quality(corpus_root, row, challenger_path)
        current_values = np.load(current_path).astype(np.float32)
        challenger_values = np.load(challenger_path).astype(np.float32)
        difference = np.abs(challenger_values - current_values)
        region_mask = np.asarray(
            Image.open(
                row_dir
                / "david_gnm"
                / challenger_metadata["region_file"]
            ).convert("L")
        )
        foundation = challenger_metadata["faces"][0][
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
        }
        control_rows.append({**common, **control_quality})
        current_rows.append(
            {
                **common,
                "parametric_foundation": current_metadata["faces"][0][
                    "parametric_face_foundation"
                ],
                **current_quality,
            }
        )
        challenger_rows.append(
            {
                **common,
                "parametric_foundation": foundation,
                "local_depth_correlation_with_dav2": _correlation(
                    local_david,
                    local_dav2,
                ),
                "local_depth_negated_correlation_with_dav2": _correlation(
                    -local_david,
                    local_dav2,
                ),
                "david_inference": dict(david.last_inference),
                "shape_delta": float(
                    challenger_quality["shape_correlation"]
                    - current_quality["shape_correlation"]
                ),
                "gradient_delta": float(
                    challenger_quality["gradient_correlation"]
                    - current_quality["gradient_correlation"]
                ),
                "normalized_rmse_delta": float(
                    challenger_quality["normalized_rmse"]
                    - current_quality["normalized_rmse"]
                ),
                "maximum_absolute_difference": float(np.max(difference)),
                "maximum_difference_outside_face_region": (
                    float(np.max(difference[region_mask == 0]))
                    if np.any(region_mask == 0)
                    else 0.0
                ),
                "finite": bool(np.all(np.isfinite(challenger_values))),
                "shared_global_input_equal": bool(
                    np.array_equal(np.load(global_depth_path), global_depth)
                ),
                **challenger_quality,
            }
        )
        print(
            json.dumps(
                {
                    "completed": index,
                    "total": len(selected_rows),
                    "row_id": row_id,
                    "current_failures": current_quality[
                        "combined_part_failures"
                    ],
                    "david_failures": challenger_quality[
                        "combined_part_failures"
                    ],
                    "shape_delta": challenger_rows[-1]["shape_delta"],
                    "gradient_delta": challenger_rows[-1]["gradient_delta"],
                    "rmse_delta": challenger_rows[-1][
                        "normalized_rmse_delta"
                    ],
                }
            ),
            flush=True,
        )

    current = _summary(current_rows)
    challenger = _summary(challenger_rows)
    decision = _decision(current, challenger)
    evidence = {
        "schema_version": 1,
        "status": "pass" if decision["passed"] else "hold",
        "method": "microsoft-david-face-crop-depth-with-gnm-foundation",
        "row_set": row_set,
        "source_geometry_training_and_evaluation_only": True,
        "privacy": summary.get("privacy"),
        "corpus_summary_sha256": _sha256(summary_path),
        "production_cache_manifest_sha256": _sha256(cache_manifest_path),
        "depth_anything_control": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_sha256": _sha256(snapshot / "model.safetensors"),
            "processor_sha256": _sha256(
                snapshot / "preprocessor_config.json"
            ),
        },
        "david": david.provenance(),
        "no_gnm_diagnostic": _summary(control_rows),
        "current_gnm": current,
        "david_gnm": challenger,
        "decision": decision,
        "torch_peak_vram_gib": (
            float(torch.cuda.max_memory_allocated(device) / (1024**3))
            if str(device).startswith("cuda")
            else 0.0
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    del depth_model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--row-set", choices=ROW_SETS, default="provider-smoke")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evidence = evaluate(
        args.corpus_root,
        args.cache_root,
        args.output_dir,
        row_set=args.row_set,
        device=args.device,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
