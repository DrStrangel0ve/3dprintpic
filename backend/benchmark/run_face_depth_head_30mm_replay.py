"""Emit a trained face-depth head candidate through the exact 30 mm STL gate."""

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
    PRINTABLE_FEATURE_DEPTH_MM,
    _compact_variant,
    _eligible_for_replay,
    _emit_variant,
    _paired_background,
    _sha256,
    _variant_quality,
)


METHOD = "trained-face-depth-head-small-face-residual-30mm-replay"
BASE_RELIEF_HEIGHT_MM = RELIEF_HEIGHT_MM - PRINTABLE_FEATURE_DEPTH_MM
EMITTER_PATHS = (
    "backend/benchmark/run_face_depth_head_30mm_replay.py",
    "backend/benchmark/run_vggheads_small_face_30mm_replay.py",
    "backend/pic_to_3d.py",
    "backend/stl_diagnostics.py",
)


def _hard_row(summary: dict) -> dict:
    for row in summary["rows"]:
        if row["row_id"] == HARD_SMALL_FACE_ROW:
            return row
    raise ValueError(f"Exact source summary is missing {HARD_SMALL_FACE_ROW}")


def _validate_replay_binding(
    source_summary_sha256: str,
    checkpoint_sha256: str,
    exact_evidence: dict,
) -> None:
    if exact_evidence.get("source_summary_sha256") != source_summary_sha256:
        raise ValueError("30 mm replay source summary does not match the exact gate")
    if exact_evidence.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("30 mm replay checkpoint does not match the exact gate")


def evaluate(
    source_run_root: str | Path,
    exact_gate_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict:
    source_run_root = Path(source_run_root).resolve()
    exact_gate_root = Path(exact_gate_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
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
    baseline_dir = (
        source_run_root.parent / row["variants"]["candidate"]["job_id"]
    )
    candidate_dir = exact_gate_root / HARD_SMALL_FACE_ROW
    baseline_depth_path = baseline_dir / "output_depth_data_face_refined.npy"
    candidate_depth_path = candidate_dir / "output_depth_data_face_refined.npy"
    baseline_face_region_path = baseline_dir / "output_face_refinement_region.png"
    baseline_feature_weight_path = baseline_dir / "output_face_refinement_weight.png"
    baseline_feature_exclusion_path = (
        baseline_dir / "output_face_refinement_occlusion.png"
    )
    candidate_face_region_path = candidate_dir / "output_face_refinement_region.png"
    candidate_feature_weight_path = candidate_dir / "output_face_refinement_weight.png"
    candidate_feature_exclusion_path = (
        candidate_dir / "output_face_refinement_occlusion.png"
    )
    required = (
        source_summary_path,
        exact_evidence_path,
        checkpoint_path,
        source_path,
        selection_mask_path,
        exact_depth_path,
        baseline_depth_path,
        candidate_depth_path,
        baseline_face_region_path,
        baseline_feature_weight_path,
        baseline_feature_exclusion_path,
        candidate_face_region_path,
        candidate_feature_weight_path,
        candidate_feature_exclusion_path,
        *part_mask_paths.values(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "30 mm trained-head replay artifacts are unavailable: "
            + ", ".join(missing)
        )
    source_summary_sha256 = _sha256(source_summary_path)
    checkpoint_sha256 = _sha256(checkpoint_path)
    _validate_replay_binding(
        source_summary_sha256,
        checkpoint_sha256,
        exact_evidence,
    )

    started = time.perf_counter()
    oracle = _emit_variant(
        "oracle",
        exact_depth_path,
        output_dir,
        invert=True,
        source_path=source_path,
        selection_mask_path=selection_mask_path,
        face_region_path=baseline_face_region_path,
        feature_weight_path=baseline_feature_weight_path,
        feature_exclusion_path=baseline_feature_exclusion_path,
        relief_height_mm=BASE_RELIEF_HEIGHT_MM,
    )
    baseline = _emit_variant(
        "baseline",
        baseline_depth_path,
        output_dir,
        invert=False,
        source_path=source_path,
        selection_mask_path=selection_mask_path,
        face_region_path=baseline_face_region_path,
        feature_weight_path=baseline_feature_weight_path,
        feature_exclusion_path=baseline_feature_exclusion_path,
        relief_height_mm=BASE_RELIEF_HEIGHT_MM,
    )
    candidate = _emit_variant(
        "candidate",
        candidate_depth_path,
        output_dir,
        invert=False,
        source_path=source_path,
        selection_mask_path=selection_mask_path,
        face_region_path=candidate_face_region_path,
        feature_weight_path=candidate_feature_weight_path,
        feature_exclusion_path=candidate_feature_exclusion_path,
        normalization_reference_depth=baseline_depth_path,
        relief_height_mm=BASE_RELIEF_HEIGHT_MM,
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
        "exact_gate_eligible": _eligible_for_replay(exact_evidence),
        "exact_integration_is_small_face_only": (
            exact_evidence.get("integration_mode")
            == "paired-residual-small-face-only"
        ),
        "candidate_named_parts_improve": (
            candidate_quality["combined_named_part_failures"]
            < baseline_quality["combined_named_part_failures"]
        ),
        "baseline_variant_passes": baseline_quality["checks"]["passed"],
        "candidate_variant_passes": candidate_quality["checks"]["passed"],
        "oracle_variant_passes": bool(
            oracle["topology"].get("printable", False)
            and oracle["shell"].get("passed", False)
            and oracle["surface_max_mm"] <= MAXIMUM_RELIEF_HEIGHT_MM
        ),
        "paired_background_preserved": paired_background["checks"]["passed"],
    }
    checks["passed"] = bool(all(checks.values()))
    evidence = {
        "schema_version": 1,
        "status": "pass" if checks["passed"] else "hold",
        "method": METHOD,
        "privacy": (
            "CC0 MakeHuman synthetic head and deterministic procedural scene only"
        ),
        "production_changed": False,
        "implementation": {
            "files": {
                relative: _sha256(Path(__file__).resolve().parents[2] / relative)
                for relative in EMITTER_PATHS
            },
        },
        "configuration": {
            "relief_height_mm": RELIEF_HEIGHT_MM,
            "base_relief_height_mm": BASE_RELIEF_HEIGHT_MM,
            "printable_feature_depth_budget_mm": PRINTABLE_FEATURE_DEPTH_MM,
            "height_budget_method": "base-relief-plus-feature-emboss",
            "max_xy_size_mm": MAX_XY_SIZE_MM,
            "target_dimension": 256,
            "background_depth_ratio": 0.65,
            "background_photo_detail_mm": 0.60,
            "background_detail_boost": 2.4,
            "normalization": "dual-subject-background-reference",
            "maximum_relief_height_mm": MAXIMUM_RELIEF_HEIGHT_MM,
        },
        "provider": {
            "checkpoint_sha256": checkpoint_sha256,
            "exact_gate_checkpoint_sha256": exact_evidence["checkpoint_sha256"],
            "model": exact_evidence.get("provider"),
            "surface_residual": exact_evidence.get("surface_residual_provider"),
            "small_face_only_contract": exact_evidence.get(
                "small_face_only_contract"
            ),
        },
        "artifacts": {
            "source_summary_sha256": source_summary_sha256,
            "exact_gate_evidence_sha256": _sha256(exact_evidence_path),
            "source_sha256": _sha256(source_path),
            "selection_mask_sha256": _sha256(selection_mask_path),
            "exact_depth_sha256": _sha256(exact_depth_path),
            "baseline_depth_sha256": _sha256(baseline_depth_path),
            "candidate_depth_sha256": _sha256(candidate_depth_path),
            "baseline_feature_weight_sha256": _sha256(
                baseline_feature_weight_path
            ),
            "candidate_feature_weight_sha256": _sha256(
                candidate_feature_weight_path
            ),
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    evidence = evaluate(
        args.source_run_root,
        args.exact_gate_root,
        args.checkpoint,
        args.output_dir,
    )
    print(json.dumps(evidence["checks"], indent=2))
    if not evidence["checks"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
