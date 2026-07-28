"""Run a bounded evaluation-only sample sweep of crop-aligned MHR geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from backend.benchmark.c3i_synface_corpus import verified_corpus_asset
from backend.benchmark.evaluate_face_depth_head_exact_gate import (
    _quality,
)
from backend.benchmark.evaluate_vggheads_small_face_exact_gate import (
    build_small_face_candidate,
)
from backend.benchmark.mhr_parametric_geometry import (
    crop_mask,
    load_mhr_geometry_runtime,
    render_mhr_camera_depth,
)


METHOD = "mhr-parametric-family-bounded-sampled-evaluation"
HARD_ROW_ID = "small_side_lit_shelves_256"
DEFAULT_ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
DEFAULT_YAW_OFFSETS = (-8.0, -4.0, 0.0, 4.0, 8.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_by_id(rows: list[dict], row_id: str) -> dict:
    matches = [row for row in rows if row.get("row_id") == row_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for {row_id}")
    return matches[0]


def _quality_rank(record: dict) -> tuple:
    return (
        int(record["combined_part_failures"]),
        -float(record["shape_correlation"]),
        -float(record["gradient_correlation"]),
        float(record["normalized_rmse"]),
    )


def _compact_quality(quality: dict) -> dict:
    if "shape_failed_parts" in quality:
        shape_failed = quality["shape_failed_parts"]
        affine_failed = quality["affine_failed_parts"]
    else:
        shape_failed = quality["named_part_shape"]["failed_parts"]
        affine_failed = quality["named_part_affine_mm"]["failed_parts"]
    return {
        "shape_correlation": float(quality["shape_correlation"]),
        "gradient_correlation": float(quality["gradient_correlation"]),
        "normalized_rmse": float(quality["normalized_rmse"]),
        "shape_failed_parts": shape_failed,
        "affine_failed_parts": affine_failed,
        "combined_part_failures": int(len(shape_failed) + len(affine_failed)),
    }


def _quality_for_values(
    values: np.ndarray,
    row: dict,
    run_root: Path,
    output_path: Path,
) -> dict:
    np.save(output_path, np.asarray(values, dtype=np.float32))
    return _quality(run_root, row, output_path)


def _load_training_identities(
    corpus_root: Path,
    count: int,
    *,
    start: int = 0,
    expression_count: int = 1,
) -> list[dict]:
    manifest = json.loads(
        (corpus_root / "training_supervision.json").read_text(encoding="utf-8")
    )
    by_identity: dict[str, dict] = {}
    for row in manifest["rows"]:
        identity_group = str(row["row_id"]).split("__scene_", 1)[0]
        if identity_group not in by_identity:
            identity_path = verified_corpus_asset(
                corpus_root,
                row["geometry_targets"]["identity"],
            )
            values = np.load(identity_path).astype(np.float32)
            by_identity[identity_group] = {
                "identity_group": identity_group,
                "head_identity": values[20:40],
                "source_sha256": row["geometry_targets"]["identity"]["sha256"],
                "expressions": [],
            }
        expression_record = row["geometry_targets"]["expression"]
        expression_path = verified_corpus_asset(corpus_root, expression_record)
        by_identity[identity_group]["expressions"].append(
            {
                "row_id": row["row_id"],
                "values": np.load(expression_path).astype(np.float32),
                "source_sha256": expression_record["sha256"],
            }
        )
    ordered = list(by_identity.values())
    selected = ordered[int(start) : int(start) + max(1, int(count))]
    if len(selected) < int(count):
        raise ValueError("MHR training supervision has too few identities")
    for identity in selected:
        identity["expressions"] = identity["expressions"][
            : max(1, int(expression_count))
        ]
        if len(identity["expressions"]) < int(expression_count):
            raise ValueError("MHR identity has too few expression targets")
    return selected


def evaluate(
    run_root: str | Path,
    corpus_root: str | Path,
    mhr_root: str | Path,
    output_dir: str | Path,
    *,
    identity_count: int = 10,
    identity_start: int = 0,
    expression_count: int = 1,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    yaw_offsets: tuple[float, ...] = DEFAULT_YAW_OFFSETS,
) -> dict:
    run_root = Path(run_root)
    corpus_root = Path(corpus_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = _row_by_id(summary["rows"], HARD_ROW_ID)
    job_dir = run_root.parent / row["variants"]["candidate"]["job_id"]
    metadata = json.loads(
        (job_dir / "output_face_refinement_metadata.json").read_text(encoding="utf-8")
    )
    face = metadata["faces"][0]
    crop_bbox = tuple(int(value) for value in face["crop_bbox"])
    face_mask_path = job_dir / face["part_masks"]["face_file"]
    support_crop = crop_mask(face_mask_path, crop_bbox)
    x0, y0, x1, y1 = crop_bbox
    source_path = run_root / row["source"]["path"]
    with Image.open(source_path) as loaded:
        image_width, image_height = loaded.size
    baseline_path = job_dir / "output_depth_data_face_refined.npy"
    baseline = np.load(baseline_path).astype(np.float32)
    local = np.load(
        job_dir / "face_refinement/face_00_depth/output_depth_data.npy"
    ).astype(np.float32)
    selection = (
        np.asarray(Image.open(run_root / row["selection_mask"]["path"]).convert("L"))
        >= 128
    )
    support = np.zeros_like(baseline, dtype=bool)
    support[y0:y1, x0:x1] = support_crop
    baseline_quality = _compact_quality(
        _quality_for_values(
            baseline,
            row,
            run_root,
            output_dir / "baseline_depth.npy",
        )
    )

    model, head_mask, runtime = load_mhr_geometry_runtime(mhr_root, device="cpu")
    identities = _load_training_identities(
        corpus_root,
        identity_count,
        start=identity_start,
        expression_count=expression_count,
    )
    known_yaw = float(row["scene"]["camera_yaw_deg"])
    known_elevation = float(row["scene"]["camera_elevation_deg"])
    records = []
    best = None
    strict_best = None
    render_index = 0
    for identity in identities:
        for expression in identity["expressions"]:
            for yaw_offset in yaw_offsets:
                render_index += 1
                crop_depth, geometry = render_mhr_camera_depth(
                    model,
                    head_mask,
                    identity["head_identity"],
                    expression["values"],
                    support_crop,
                    yaw_degrees=known_yaw + float(yaw_offset),
                    elevation_degrees=known_elevation,
                    device="cpu",
                )
                provider = np.full(
                    (image_height, image_width),
                    np.nan,
                    dtype=np.float32,
                )
                provider[y0:y1, x0:x1] = crop_depth
                for alpha in alphas:
                    candidate, fusion = build_small_face_candidate(
                        baseline,
                        local,
                        provider,
                        support,
                        selection,
                        crop_bbox,
                        provider_alpha=float(alpha),
                    )
                    candidate_path = output_dir / "scratch_candidate.npy"
                    quality = _compact_quality(
                        _quality_for_values(
                            candidate,
                            row,
                            run_root,
                            candidate_path,
                        )
                    )
                    record = {
                        "identity_group": identity["identity_group"],
                        "identity_target_sha256": identity["source_sha256"],
                        "expression_row_id": expression["row_id"],
                        "expression_target_sha256": expression["source_sha256"],
                        "yaw_degrees": known_yaw + float(yaw_offset),
                        "yaw_offset_degrees": float(yaw_offset),
                        "elevation_degrees": known_elevation,
                        "alpha": float(alpha),
                        "geometry": geometry,
                        "fusion": fusion,
                        **quality,
                    }
                    records.append(record)
                    if best is None or _quality_rank(record) < _quality_rank(best):
                        best = record
                        np.save(output_dir / "best_candidate_depth.npy", candidate)
                        np.save(output_dir / "best_provider_depth.npy", provider)
                    strict_improvement = bool(
                        record["combined_part_failures"]
                        < baseline_quality["combined_part_failures"]
                        and record["shape_correlation"]
                        > baseline_quality["shape_correlation"]
                        and record["gradient_correlation"]
                        > baseline_quality["gradient_correlation"]
                        and record["normalized_rmse"]
                        < baseline_quality["normalized_rmse"]
                    )
                    if strict_improvement and (
                        strict_best is None
                        or _quality_rank(record) < _quality_rank(strict_best)
                    ):
                        strict_best = record
                        np.save(
                            output_dir / "strict_best_candidate_depth.npy",
                            candidate,
                        )
                        np.save(
                            output_dir / "strict_best_provider_depth.npy",
                            provider,
                        )
                print(
                    json.dumps(
                        {
                            "render": render_index,
                            "total": (
                                len(identities)
                                * int(expression_count)
                                * len(yaw_offsets)
                            ),
                            "best_failures": best["combined_part_failures"],
                        }
                    ),
                    flush=True,
                )
    (output_dir / "scratch_candidate.npy").unlink(missing_ok=True)
    if best is None:
        raise RuntimeError("MHR family search produced no candidates")
    background_exact = bool(
        np.array_equal(
            np.load(output_dir / "best_candidate_depth.npy")[~selection],
            baseline[~selection],
        )
    )
    checks = {
        "diagnostic_only": True,
        "source_geometry_used_only_for_scoring": True,
        "background_bit_exact": background_exact,
        "part_failures_improve": bool(
            best["combined_part_failures"] < baseline_quality["combined_part_failures"]
        ),
        "shape_improves": bool(
            best["shape_correlation"] > baseline_quality["shape_correlation"]
        ),
        "gradient_improves": bool(
            best["gradient_correlation"] > baseline_quality["gradient_correlation"]
        ),
        "rmse_improves": bool(
            best["normalized_rmse"] < baseline_quality["normalized_rmse"]
        ),
    }
    checks["family_has_usable_upper_bound"] = strict_best is not None
    evidence = {
        "schema_version": 1,
        "method": METHOD,
        "status": "diagnostic-only",
        "promotion_eligible": False,
        "source_geometry": "evaluation-only",
        "row_id": HARD_ROW_ID,
        "source_summary_sha256": _sha256(summary_path),
        "source_image_sha256": row["source"]["sha256"],
        "mhr_runtime": runtime,
        "search": {
            "coverage": "finite-sampled-coefficients-camera-offsets-and-fusion-strengths",
            "identities": len(identities),
            "identity_start": int(identity_start),
            "expressions_per_identity": int(expression_count),
            "yaw_offsets_degrees": list(yaw_offsets),
            "alphas": list(alphas),
            "candidate_count": len(records),
            "known_pose_is_evaluation_only": True,
        },
        "baseline": baseline_quality,
        "best": best,
        "strict_best": strict_best,
        "checks": checks,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def _parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or not all(np.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("Expected a finite comma-separated list")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--mhr-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--identity-count", type=int, default=10)
    parser.add_argument("--identity-start", type=int, default=0)
    parser.add_argument("--expression-count", type=int, default=1)
    parser.add_argument(
        "--alphas",
        type=_parse_floats,
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument(
        "--yaw-offsets",
        type=_parse_floats,
        default=DEFAULT_YAW_OFFSETS,
    )
    args = parser.parse_args()
    evidence = evaluate(
        args.run_root,
        args.corpus_root,
        args.mhr_root,
        args.output_dir,
        identity_count=args.identity_count,
        identity_start=args.identity_start,
        expression_count=args.expression_count,
        alphas=args.alphas,
        yaw_offsets=args.yaw_offsets,
    )
    print(json.dumps(evidence["checks"], indent=2))


if __name__ == "__main__":
    main()
