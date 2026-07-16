"""Replay selected SMIRK face-depth candidates as 30 mm relief STLs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
)
from backend.benchmark.run_vggheads_small_face_30mm_replay import (
    MAXIMUM_RELIEF_HEIGHT_MM,
    _compact_variant,
    _emit_variant,
    _paired_background,
    _variant_quality,
)


METHOD = "smirk-landmark-aligned-face-30mm-smoke"
PHYSICAL_ROW_IDS = (
    "mh_caucasian_male__mouth_open_06",
    "mh_mixed_female__mouth_open_02",
    "mh_mixed_male__neutral_00",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_by_id(rows: list[dict], row_id: str) -> dict:
    matches = [row for row in rows if row["row_id"] == row_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one corpus row for {row_id}")
    return matches[0]


def _stage_emission_inputs(
    row: dict,
    corpus_root: Path,
    cache_root: Path,
    candidate_path: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / "rows" / f"{row['row_id']}.npz"
    with np.load(cache_path) as cached:
        baseline = cached["baseline_full"].astype(np.float32)
        bbox = tuple(int(value) for value in cached["bbox"])
        support_native = cached["support_native"].astype(bool)
    x0, y0, x1, y1 = bbox
    if support_native.shape != (y1 - y0, x1 - x0):
        raise ValueError(f"Cached face support is invalid for {row['row_id']}")
    support = np.zeros_like(baseline, dtype=np.uint8)
    support[y0:y1, x0:x1] = support_native.astype(np.uint8) * 255
    exclusion = np.zeros_like(support, dtype=np.uint8)

    baseline_path = output_dir / "baseline_depth.npy"
    face_region_path = output_dir / "face_region.png"
    feature_weight_path = output_dir / "feature_weight.png"
    feature_exclusion_path = output_dir / "feature_exclusion.png"
    np.save(baseline_path, baseline)
    Image.fromarray(support).save(face_region_path)
    Image.fromarray(support).save(feature_weight_path)
    Image.fromarray(exclusion).save(feature_exclusion_path)

    candidate = np.load(candidate_path).astype(np.float32)
    if candidate.shape != baseline.shape:
        raise ValueError(f"Candidate depth shape differs for {row['row_id']}")
    return {
        "baseline_depth": baseline_path,
        "candidate_depth": candidate_path,
        "face_region": face_region_path,
        "feature_weight": feature_weight_path,
        "feature_exclusion": feature_exclusion_path,
    }


def _row_checks(
    baseline_quality: dict,
    candidate_quality: dict,
    candidate_variant: dict,
    oracle_variant: dict,
    paired_background: dict,
) -> dict:
    checks = {
        "candidate_named_parts_non_regression": bool(
            candidate_quality["combined_named_part_failures"]
            <= baseline_quality["combined_named_part_failures"]
        ),
        "candidate_physical_checks_pass": bool(
            candidate_quality["checks"]["passed"]
        ),
        "candidate_topology_printable": bool(
            candidate_variant["topology"].get("printable", False)
        ),
        "candidate_exact_shell": bool(
            candidate_variant["shell"].get("passed", False)
        ),
        "candidate_height_cap": bool(
            candidate_variant["surface_max_mm"]
            <= MAXIMUM_RELIEF_HEIGHT_MM
        ),
        "oracle_printable_reference": bool(
            oracle_variant["topology"].get("printable", False)
            and oracle_variant["shell"].get("passed", False)
            and oracle_variant["surface_max_mm"]
            <= MAXIMUM_RELIEF_HEIGHT_MM
        ),
        "paired_background_preserved": bool(
            paired_background["checks"]["passed"]
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def physical_decision(rows: list[dict]) -> dict:
    checks = {
        "three_split_rows": bool(
            len(rows) == 3
            and {row["split"] for row in rows}
            == {"train", "validation", "sealed"}
        ),
        "all_rows_pass": bool(
            rows and all(row["checks"]["passed"] for row in rows)
        ),
        "at_least_one_30mm_named_part_improvement": bool(
            any(
                row["candidate_quality"][
                    "combined_named_part_failures"
                ]
                < row["baseline_quality"][
                    "combined_named_part_failures"
                ]
                for row in rows
            )
        ),
        "no_30mm_named_part_regression": bool(
            all(
                row["candidate_quality"][
                    "combined_named_part_failures"
                ]
                <= row["baseline_quality"][
                    "combined_named_part_failures"
                ]
                for row in rows
            )
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def evaluate(
    corpus_root: str | Path,
    cache_root: str | Path,
    smirk_smoke_root: str | Path,
    output_dir: str | Path,
) -> dict:
    started = time.perf_counter()
    corpus_root = Path(corpus_root).resolve()
    cache_root = Path(cache_root).resolve()
    smirk_smoke_root = Path(smirk_smoke_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_summary_path = corpus_root / "summary.json"
    smirk_evidence_path = smirk_smoke_root / "evidence.json"
    corpus = json.loads(corpus_summary_path.read_text(encoding="utf-8"))
    smirk = json.loads(smirk_evidence_path.read_text(encoding="utf-8"))
    if not smirk["decision"]["checks"]["eligible_for_30mm_replay"]:
        raise ValueError("SMIRK raw-depth smoke is not eligible for replay")
    baseline_by_id = {
        row["row_id"]: row for row in smirk["baseline"]["all"]["rows"]
    }
    candidate_by_id = {
        row["row_id"]: row for row in smirk["candidate"]["all"]["rows"]
    }

    row_records = []
    for row_id in PHYSICAL_ROW_IDS:
        row = _row_by_id(corpus["rows"], row_id)
        baseline_record = baseline_by_id[row_id]
        candidate_record = candidate_by_id[row_id]
        if (
            candidate_record["combined_part_failures"]
            > baseline_record["combined_part_failures"]
        ):
            raise ValueError(
                f"SMIRK raw exact depth regressed before replay: {row_id}"
            )
        candidate_path = Path(
            smirk["candidate"]["selected_depth_artifacts"][row_id][
                "path"
            ]
        )
        row_dir = output_dir / row_id
        staged = _stage_emission_inputs(
            row,
            corpus_root,
            cache_root,
            candidate_path,
            row_dir / "staged",
        )
        source_path = corpus_root / row["source"]["path"]
        selection_path = corpus_root / row["selection_mask"]["path"]
        exact_depth_path = corpus_root / row["exact_depth"]["path"]
        part_paths = {
            name: corpus_root / row["exact_face_parts"][name]["path"]
            for name in FACE_PART_NAMES
        }
        oracle = _emit_variant(
            "oracle",
            exact_depth_path,
            row_dir,
            invert=True,
            source_path=source_path,
            selection_mask_path=selection_path,
            face_region_path=staged["face_region"],
            feature_weight_path=staged["feature_weight"],
            feature_exclusion_path=staged["feature_exclusion"],
        )
        baseline = _emit_variant(
            "baseline",
            staged["baseline_depth"],
            row_dir,
            invert=False,
            source_path=source_path,
            selection_mask_path=selection_path,
            face_region_path=staged["face_region"],
            feature_weight_path=staged["feature_weight"],
            feature_exclusion_path=staged["feature_exclusion"],
        )
        candidate = _emit_variant(
            "candidate",
            staged["candidate_depth"],
            row_dir,
            invert=False,
            source_path=source_path,
            selection_mask_path=selection_path,
            face_region_path=staged["face_region"],
            feature_weight_path=staged["feature_weight"],
            feature_exclusion_path=staged["feature_exclusion"],
        )
        baseline_quality = _variant_quality(
            baseline,
            oracle,
            selection_path,
            part_paths,
        )
        candidate_quality = _variant_quality(
            candidate,
            oracle,
            selection_path,
            part_paths,
        )
        paired_background = _paired_background(
            baseline,
            candidate,
            selection_path,
        )
        checks = _row_checks(
            baseline_quality,
            candidate_quality,
            candidate,
            oracle,
            paired_background,
        )
        row_records.append(
            {
                "row_id": row_id,
                "split": str(row["split"]),
                "identity_group": str(row["identity_group"]),
                "expression": str(row["expression"]),
                "face_height_pixels": int(
                    row["render"]["face_bbox_height_pixels"]
                ),
                "camera_yaw_deg": float(
                    row["spec"]["camera_yaw_deg"]
                ),
                "raw_exact_failures": {
                    "baseline": int(
                        baseline_record["combined_part_failures"]
                    ),
                    "candidate": int(
                        candidate_record["combined_part_failures"]
                    ),
                },
                "provider_policy": candidate_record["provider_policy"],
                "artifacts": {
                    "source_sha256": _sha256(source_path),
                    "selection_sha256": _sha256(selection_path),
                    "exact_depth_sha256": _sha256(exact_depth_path),
                    "baseline_depth_sha256": _sha256(
                        staged["baseline_depth"]
                    ),
                    "candidate_depth_sha256": _sha256(
                        staged["candidate_depth"]
                    ),
                },
                "oracle": _compact_variant(oracle),
                "baseline": _compact_variant(baseline),
                "candidate": _compact_variant(candidate),
                "baseline_quality": baseline_quality,
                "candidate_quality": candidate_quality,
                "paired_background": paired_background,
                "checks": checks,
            }
        )
        print(
            json.dumps(
                {
                    "row_id": row_id,
                    "baseline_failures": baseline_quality[
                        "combined_named_part_failures"
                    ],
                    "candidate_failures": candidate_quality[
                        "combined_named_part_failures"
                    ],
                    "passed": checks["passed"],
                }
            ),
            flush=True,
        )

    decision = physical_decision(row_records)
    evidence = {
        "schema_version": 1,
        "status": "pass" if decision["passed"] else "hold",
        "method": METHOD,
        "privacy": corpus.get("privacy"),
        "production_changed": False,
        "production_eligible": False,
        "configuration": {
            "rows": list(PHYSICAL_ROW_IDS),
            "relief_height_mm": 30.0,
            "candidate_normalization": "production-default",
            "face_mask_source": "cached production detector support",
            "selection_background_depth_ratio": 0.65,
        },
        "inputs": {
            "corpus_summary_sha256": _sha256(corpus_summary_path),
            "smirk_raw_evidence_sha256": _sha256(smirk_evidence_path),
        },
        "rows": row_records,
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
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--smirk-smoke-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    evidence = evaluate(
        args.corpus_root,
        args.cache_root,
        args.smirk_smoke_root,
        args.output_dir,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
