"""Evaluate a size-aware GNM DAv2 residual on production-native cached faces."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.evaluate_face_depth_head_exact_gate import (
    BlendedFaceDepth,
    RESIDUAL_FULL_STRENGTH_SUPPORT_HEIGHT_PIXELS,
    _correlation,
    _face_residual_scale,
    _resize_depth,
    _sha256,
)
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    _exact_face_depth_quality,
)
from backend.benchmark.train_face_depth_head import _strictly_improves
from backend.benchmark.train_face_surface_fusion_adapter import (
    selected_image_from_exact_mask,
)
from backend.face_depth_refinement import fuse_face_surface_residual


VARIANTS = ("full-strength", "size-aware")


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
        "rows": rows,
    }


def _per_row_no_failure_regression(
    candidate: dict,
    baseline: dict,
) -> bool:
    baseline_by_id = {row["row_id"]: row for row in baseline["rows"]}
    return bool(
        all(
            row["combined_part_failures"]
            <= baseline_by_id[row["row_id"]]["combined_part_failures"]
            for row in candidate["rows"]
        )
    )


def evaluate(
    corpus_root: str | Path,
    cache_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> dict:
    corpus_root = Path(corpus_root)
    cache_root = Path(cache_root)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = corpus_root / "summary.json"
    cache_manifest_path = cache_root / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads(
        cache_manifest_path.read_text(encoding="utf-8")
    )
    if (
        cache_manifest.get("corpus_summary_sha256")
        != _sha256(summary_path)
    ):
        raise ValueError("Production cache does not match the GNM corpus")

    provider = BlendedFaceDepth(checkpoint_path, device)
    split_records = {
        split: {
            "baseline": [],
            "full-strength": [],
            "size-aware": [],
        }
        for split in ("validation", "sealed")
    }
    equivalence = []
    started = time.perf_counter()
    for row in summary["rows"]:
        split = row["split"]
        if split not in split_records:
            continue
        row_id = row["row_id"]
        with np.load(cache_root / "rows" / f"{row_id}.npz") as cached:
            bbox = tuple(int(value) for value in cached["bbox"])
            baseline_full = cached["baseline_full"].astype(np.float32)
            local_native = cached["local_native"].astype(np.float32)
            support_native = cached["support_native"].astype(np.uint8)

        source = Image.open(corpus_root / row["source"]["path"]).convert("RGB")
        selection = Image.open(
            corpus_root / row["selection_mask"]["path"]
        ).convert("L")
        selected = selected_image_from_exact_mask(source, selection)
        crop_path = output_dir / "crops" / f"{row_id}.png"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        selected.crop(bbox).save(crop_path)
        reproduced = provider.infer(crop_path, alpha=0.0)
        trained = provider.infer(crop_path)
        comparable = _resize_depth(reproduced, local_native.shape)
        equivalence.append(
            {
                "row_id": row_id,
                "split": split,
                "correlation": _correlation(local_native, comparable),
                "maximum_absolute_difference": float(
                    np.max(np.abs(local_native - comparable))
                ),
            }
        )
        residual = trained - reproduced
        baseline_path = output_dir / "rows" / row_id / "baseline.npy"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(baseline_path, baseline_full)
        base_record = {
            "row_id": row_id,
            "face_height_pixels": row["render"]["face_bbox_height_pixels"],
            **_quality(corpus_root, row, baseline_path),
        }
        split_records[split]["baseline"].append(base_record)

        x0, y0, x1, y1 = bbox
        baseline_crop = baseline_full[y0:y1, x0:x1]
        for variant in VARIANTS:
            if variant == "size-aware":
                residual_scale, support_height = _face_residual_scale(
                    support_native
                )
            else:
                residual_scale = 1.0
                support_height = int(
                    np.flatnonzero(np.any(support_native > 0, axis=1))[-1]
                    - np.flatnonzero(
                        np.any(support_native > 0, axis=1)
                    )[0]
                    + 1
                )
            refined_crop, _weight, fusion = fuse_face_surface_residual(
                baseline_crop,
                local_native,
                residual * residual_scale,
                support_native * 255,
                alignment_depth=baseline_crop,
            )
            candidate_full = baseline_full.copy()
            candidate_full[y0:y1, x0:x1] = refined_crop
            candidate_path = (
                output_dir / "rows" / row_id / f"{variant}.npy"
            )
            np.save(candidate_path, candidate_full)
            split_records[split][variant].append(
                {
                    "row_id": row_id,
                    "face_height_pixels": row["render"][
                        "face_bbox_height_pixels"
                    ],
                    "support_height_pixels": support_height,
                    "residual_scale": residual_scale,
                    "max_abs_correction": fusion["max_abs_correction"],
                    "correction_limit": fusion["correction_limit"],
                    **_quality(corpus_root, row, candidate_path),
                }
            )

    results = {
        split: {
            label: _summary(records)
            for label, records in variants.items()
        }
        for split, variants in split_records.items()
    }
    checks = {}
    for split in ("validation", "sealed"):
        baseline = results[split]["baseline"]
        candidate = results[split]["size-aware"]
        checks[f"{split}_strictly_improves"] = _strictly_improves(
            candidate,
            baseline,
        )
        checks[f"{split}_per_row_no_failure_regression"] = (
            _per_row_no_failure_regression(candidate, baseline)
        )
    checks["local_baseline_equivalent"] = bool(
        equivalence
        and all(
            row["correlation"] >= 0.999
            and row["maximum_absolute_difference"] <= 0.02
            for row in equivalence
        )
    )
    checks["eligible_for_exact_photo_replay"] = bool(all(checks.values()))
    evidence = {
        "schema_version": 1,
        "method": "gnm-dav2-size-aware-paired-surface-residual",
        "source_geometry_training_and_evaluation_only": True,
        "corpus_summary_sha256": _sha256(summary_path),
        "production_cache_manifest_sha256": _sha256(cache_manifest_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "schedule": {
            "training_face_height_anchor_pixels": 90.0,
            "measured_detector_support_height_ratio": 0.5,
            "full_strength_support_height_pixels": (
                RESIDUAL_FULL_STRENGTH_SUPPORT_HEIGHT_PIXELS
            ),
            "formula": "min(1, full_strength_support_height / support_height)",
            "validation_selected": False,
            "rationale": (
                "Directly maps the training loss face-height anchor into "
                "production detector-support coordinates."
            ),
        },
        "provider": provider.provenance(),
        "local_baseline_equivalence": equivalence,
        "results": results,
        "decision": checks,
        "runtime_seconds": time.perf_counter() - started,
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evidence = evaluate(
        args.corpus_root,
        args.cache_root,
        args.checkpoint,
        args.output_dir,
        device=args.device,
    )
    print(json.dumps(evidence["decision"], indent=2))
    if not evidence["decision"]["eligible_for_exact_photo_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
