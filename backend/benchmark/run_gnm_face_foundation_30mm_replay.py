"""Emit the exact hard-face control and GNM candidate as 30 mm relief STLs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    MAX_XY_SIZE_MM,
    RELIEF_HEIGHT_MM,
)
from backend.benchmark.run_vggheads_small_face_30mm_replay import (
    HARD_SMALL_FACE_ROW,
    MAXIMUM_RELIEF_HEIGHT_MM,
    _compact_variant,
    _eligible_for_replay,
    _emit_variant,
    _paired_background,
    _sha256,
    _variant_quality,
)
from backend.gnm_face_foundation import (
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
)


METHOD = "gnm-mean-face-foundation-30mm-production-emission-replay"


def _hard_row(run_summary: dict) -> dict:
    for row in run_summary["rows"]:
        if row["row_id"] == HARD_SMALL_FACE_ROW:
            return row
    raise ValueError(f"Exact source summary is missing {HARD_SMALL_FACE_ROW}")


def evaluate(
    source_run_root: str | Path,
    exact_gate_root: str | Path,
    output_dir: str | Path,
) -> dict:
    source_run_root = Path(source_run_root)
    exact_gate_root = Path(exact_gate_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_summary_path = source_run_root / "summary.json"
    exact_evidence_path = exact_gate_root / "evidence.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    exact_evidence = json.loads(exact_evidence_path.read_text(encoding="utf-8"))
    row = _hard_row(source_summary)
    source_path = source_run_root / row["source"]["path"]
    selection_mask_path = source_run_root / row["selection_mask"]["path"]
    exact_depth_path = source_run_root / row["exact_depth"]["path"]
    part_mask_paths = {
        name: (
            source_run_root
            / row["exact_face_part_masks"]["files"][name]["path"]
        )
        for name in FACE_PART_NAMES
    }
    paired_root = exact_gate_root / HARD_SMALL_FACE_ROW
    baseline_dir = paired_root / "control"
    candidate_dir = paired_root / "candidate"
    baseline_depth_path = baseline_dir / "output_depth_data_face_refined.npy"
    candidate_depth_path = candidate_dir / "output_depth_data_face_refined.npy"
    face_region_path = baseline_dir / "output_face_refinement_region.png"
    feature_weight_path = baseline_dir / "output_face_refinement_weight.png"
    feature_exclusion_path = baseline_dir / "output_face_refinement_occlusion.png"
    required = (
        source_summary_path,
        exact_evidence_path,
        source_path,
        selection_mask_path,
        exact_depth_path,
        baseline_depth_path,
        candidate_depth_path,
        face_region_path,
        feature_weight_path,
        feature_exclusion_path,
        *part_mask_paths.values(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "30 mm GNM replay artifacts are unavailable: "
            + ", ".join(missing)
        )

    started = time.perf_counter()
    oracle = _emit_variant(
        "oracle",
        exact_depth_path,
        output_dir,
        invert=True,
        source_path=source_path,
        selection_mask_path=selection_mask_path,
        face_region_path=face_region_path,
        feature_weight_path=feature_weight_path,
        feature_exclusion_path=feature_exclusion_path,
    )
    baseline = _emit_variant(
        "baseline",
        baseline_depth_path,
        output_dir,
        invert=False,
        source_path=source_path,
        selection_mask_path=selection_mask_path,
        face_region_path=face_region_path,
        feature_weight_path=feature_weight_path,
        feature_exclusion_path=feature_exclusion_path,
    )
    candidate = _emit_variant(
        "candidate",
        candidate_depth_path,
        output_dir,
        invert=False,
        source_path=source_path,
        selection_mask_path=selection_mask_path,
        face_region_path=face_region_path,
        feature_weight_path=feature_weight_path,
        feature_exclusion_path=feature_exclusion_path,
        normalization_reference_depth=baseline_depth_path,
    )
    baseline_quality = _variant_quality(
        baseline,
        oracle,
        selection_mask_path,
        part_mask_paths,
    )
    candidate_quality = _variant_quality(
        candidate,
        oracle,
        selection_mask_path,
        part_mask_paths,
    )
    paired_background = _paired_background(
        baseline,
        candidate,
        selection_mask_path,
    )
    checks = {
        "exact_gate_eligible": bool(_eligible_for_replay(exact_evidence)),
        "candidate_named_parts_improve": bool(
            candidate_quality["combined_named_part_failures"]
            < baseline_quality["combined_named_part_failures"]
        ),
        "baseline_variant_passes": bool(
            baseline_quality["checks"]["passed"]
        ),
        "candidate_variant_passes": bool(
            candidate_quality["checks"]["passed"]
        ),
        "oracle_variant_passes": bool(
            oracle["topology"].get("printable", False)
            and oracle["shell"].get("passed", False)
            and oracle["surface_max_mm"] <= MAXIMUM_RELIEF_HEIGHT_MM
        ),
        "paired_background_preserved": bool(
            paired_background["checks"]["passed"]
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    evidence = {
        "schema_version": 1,
        "status": "pass" if checks["passed"] else "hold",
        "method": METHOD,
        "privacy": (
            "CC0 MakeHuman synthetic head and deterministic procedural "
            "scene only"
        ),
        "provider": {
            "gnm_revision": GNM_REVISION,
            "gnm_license": GNM_LICENSE,
            "gnm_model_sha256": GNM_MODEL_SHA256,
            "gnm_landmarks_sha256": GNM_LANDMARKS_SHA256,
            "mapping_revision": MEDIAPIPE_DLIB_MAPPING_REVISION,
            "mapping_license": MEDIAPIPE_DLIB_MAPPING_LICENSE,
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
        },
        "configuration": {
            "relief_height_mm": RELIEF_HEIGHT_MM,
            "max_xy_size_mm": MAX_XY_SIZE_MM,
            "target_dimension": 256,
            "background_depth_ratio": 0.65,
            "background_photo_detail_mm": 0.60,
            "background_detail_boost": 2.4,
            "normalization": "dual-subject-background-reference",
            "maximum_relief_height_mm": MAXIMUM_RELIEF_HEIGHT_MM,
        },
        "artifacts": {
            "source_summary_sha256": _sha256(source_summary_path),
            "exact_gate_evidence_sha256": _sha256(exact_evidence_path),
            "source_sha256": _sha256(source_path),
            "selection_mask_sha256": _sha256(selection_mask_path),
            "exact_depth_sha256": _sha256(exact_depth_path),
            "baseline_depth_sha256": _sha256(baseline_depth_path),
            "candidate_depth_sha256": _sha256(candidate_depth_path),
        },
        "oracle": _compact_variant(oracle),
        "baseline": {
            **_compact_variant(baseline),
            "quality": baseline_quality,
        },
        "candidate": {
            **_compact_variant(candidate),
            "quality": candidate_quality,
        },
        "paired_background": paired_background,
        "checks": checks,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--exact-gate-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    evidence = evaluate(
        args.source_run_root,
        args.exact_gate_root,
        args.output_dir,
    )
    print(json.dumps(evidence["checks"], indent=2))
    if not evidence["checks"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
