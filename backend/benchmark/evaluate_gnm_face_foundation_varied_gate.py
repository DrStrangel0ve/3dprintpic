"""Evaluate reliability-calibrated GNM fusion on the varied CC0 face matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from backend.benchmark.evaluate_face_depth_head_exact_gate import (
    _quality,
    _strictly_improves,
    _summary,
)
from backend.face_depth_refinement import (
    detect_face_regions_in_roi,
    refine_depth_for_faces,
)
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
    MEDIAPIPE_DLIB_MAPPING_LICENSE,
    MEDIAPIPE_DLIB_MAPPING_REVISION,
    get_gnm_mean_face_foundation,
)


SMALL_FACE_ROWS = (
    "fusion_asian_small_left_256",
    "fusion_african_small_right_256",
    "fusion_caucasian_small_left_256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_depth_callback(path: Path):
    def callback(_crop_path: Path, _output_dir: Path) -> Path:
        return path

    return callback


def _detector_callback(regions: list[dict]):
    def callback(_image: np.ndarray) -> list[dict]:
        return copy.deepcopy(regions)

    return callback


def evaluate(
    run_root: str | Path,
    output_dir: str | Path,
) -> dict:
    run_root = Path(run_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / "summary.json"
    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provider = get_gnm_mean_face_foundation()
    historical_rows = []
    control_rows = []
    candidate_rows = []
    started = time.perf_counter()

    for row in run_summary["rows"]:
        row_id = row["row_id"]
        job_dir = run_root.parent / row["variants"]["candidate"]["job_id"]
        source_path = run_root / row["source"]["path"]
        selection_path = run_root / row["selection_mask"]["path"]
        historical_path = job_dir / "output_depth_data_face_refined.npy"
        cached_local_path = (
            job_dir
            / "face_refinement"
            / "face_00_depth"
            / "output_depth_data.npy"
        )
        image = np.asarray(Image.open(source_path).convert("RGB"))
        selection = np.asarray(Image.open(selection_path).convert("L"))
        regions, detector_errors, detector_stats = detect_face_regions_in_roi(
            image,
            selection,
            max_faces=1,
            min_face_pixels=96,
        )
        if detector_errors or len(regions) != 1:
            raise RuntimeError(
                f"Expected one clean face for {row_id}; "
                f"regions={len(regions)}, errors={detector_errors}"
            )
        detector = _detector_callback(regions)
        callback = _cached_depth_callback(cached_local_path)
        control_dir = output_dir / row_id / "control"
        candidate_dir = output_dir / row_id / "candidate"
        control_path, _control_metadata = refine_depth_for_faces(
            source_path,
            job_dir / "output_depth_data.npy",
            control_dir,
            infer_depth=callback,
            mode="on",
            detector=detector,
            detection_roi_mask=selection_path,
            enable_gnm_foundation=False,
        )
        candidate_path, candidate_metadata = refine_depth_for_faces(
            source_path,
            job_dir / "output_depth_data.npy",
            candidate_dir,
            infer_depth=callback,
            mode="on",
            detector=detector,
            detection_roi_mask=selection_path,
            enable_gnm_foundation=True,
        )
        historical_quality = _quality(run_root, row, historical_path)
        control_quality = _quality(run_root, row, Path(control_path))
        candidate_quality = _quality(run_root, row, Path(candidate_path))
        control_values = np.load(control_path).astype(np.float32)
        candidate_values = np.load(candidate_path).astype(np.float32)
        region_mask = np.asarray(
            Image.open(candidate_dir / candidate_metadata["region_file"]).convert("L")
        )
        difference = np.abs(candidate_values - control_values)
        foundation = candidate_metadata["faces"][0]["parametric_face_foundation"]
        common = {
            "row_id": row_id,
            "face_height_pixels": int(
                row["render"]["face_bbox_height_pixels"]
            ),
        }
        historical_rows.append({**common, **historical_quality})
        control_rows.append({**common, **control_quality})
        candidate_rows.append(
            {
                **common,
                "detector": regions[0].get("detector"),
                "detector_bbox": regions[0].get("bbox"),
                "detector_stats": detector_stats,
                "parametric_foundation": foundation,
                "parametric_foundation_faces": candidate_metadata[
                    "parametric_foundation_faces"
                ],
                "control_exact_equal": bool(
                    np.array_equal(candidate_values, control_values)
                ),
                "maximum_absolute_difference": float(np.max(difference)),
                "maximum_difference_outside_face_region": (
                    float(np.max(difference[region_mask == 0]))
                    if np.any(region_mask == 0)
                    else 0.0
                ),
                "finite": bool(np.all(np.isfinite(candidate_values))),
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
                **candidate_quality,
            }
        )

    historical = _summary(historical_rows)
    control = _summary(control_rows)
    candidate = _summary(candidate_rows)
    control_by_id = {row["row_id"]: row for row in control_rows}
    candidate_by_id = {row["row_id"]: row for row in candidate_rows}
    small_rows = [candidate_by_id[row_id] for row_id in SMALL_FACE_ROWS]
    other_rows = [
        row
        for row in candidate_rows
        if row["row_id"] not in SMALL_FACE_ROWS
    ]
    decisions = {
        "paired_aggregate_strictly_improves": _strictly_improves(
            candidate,
            control,
        ),
        "paired_per_row_no_failure_regression": all(
            candidate_by_id[row_id]["combined_part_failures"]
            <= control_by_id[row_id]["combined_part_failures"]
            for row_id in candidate_by_id
        ),
        "small_face_shape_no_regression": all(
            row["shape_delta"] >= -1e-8 for row in small_rows
        ),
        "small_face_gradient_no_regression": all(
            row["gradient_delta"] >= -1e-8 for row in small_rows
        ),
        "small_face_rmse_no_regression": all(
            row["normalized_rmse_delta"] <= 1e-8 for row in small_rows
        ),
        "small_faces_use_guarded_tier": all(
            row["parametric_foundation_faces"] == 1
            and row["parametric_foundation"].get("enabled") is True
            and row["parametric_foundation"].get("reliability_tier")
            == "guarded"
            for row in small_rows
        ),
        "at_least_two_small_faces_improve": (
            sum(
                row["combined_part_failures"]
                < control_by_id[row["row_id"]]["combined_part_failures"]
                for row in small_rows
            )
            >= 2
        ),
        "larger_faces_exact_control": all(
            row["control_exact_equal"]
            and row["parametric_foundation_faces"] == 0
            and row["parametric_foundation"].get("reason")
            == "face_support_above_small_face_gate"
            for row in other_rows
        ),
        "face_boundary_zero": all(
            (
                not row["parametric_foundation"].get("enabled")
                or row["parametric_foundation"].get(
                    "boundary_max_abs_correction",
                    float("inf"),
                )
                <= 1e-7
            )
            for row in candidate_rows
        ),
        "non_face_exact_control": all(
            row["maximum_difference_outside_face_region"] == 0.0
            for row in candidate_rows
        ),
        "all_candidate_depth_finite": all(
            row["finite"] for row in candidate_rows
        ),
    }
    decisions["passed"] = bool(all(decisions.values()))
    evidence = {
        "schema_version": 1,
        "status": "pass" if decisions["passed"] else "hold",
        "method": "reliability-calibrated-gnm-mean-face-foundation",
        "source_revision": run_summary["server_provenance"]["revision"],
        "source_summary_sha256": _sha256(summary_path),
        "provider": {
            "gnm_revision": GNM_REVISION,
            "gnm_license": GNM_LICENSE,
            "gnm_model_sha256": GNM_MODEL_SHA256,
            "gnm_landmarks_sha256": GNM_LANDMARKS_SHA256,
            "mapping_revision": MEDIAPIPE_DLIB_MAPPING_REVISION,
            "mapping_license": MEDIAPIPE_DLIB_MAPPING_LICENSE,
            "loaded_provider": type(provider).__name__,
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
            "guarded_correction_strength": (
                GNM_GUARDED_CORRECTION_STRENGTH
            ),
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
        },
        "historical": historical,
        "paired_control": control,
        "candidate": candidate,
        "decision": decisions,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    evidence = evaluate(args.run_root, args.output_dir)
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
