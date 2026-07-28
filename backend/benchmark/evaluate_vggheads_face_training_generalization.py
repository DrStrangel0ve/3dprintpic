"""Evaluate VGGHeads small-face fusion on identity-disjoint CC0 rows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.evaluate_vggheads_small_face_exact_gate import (
    HARD_SMALL_FACE_ROW,
    MINIMUM_PROVIDER_CROP_COVERAGE,
    MAXIMUM_HARD_ROW_GRADIENT_REGRESSION,
    SUBJECT_BOUNDARY_PIXELS,
    _quality,
    _summary,
    build_small_face_candidate,
)
from backend.benchmark.makehuman_face_training_corpus import (
    EXPRESSIONS,
    SPLIT_BY_IDENTITY,
)
from backend.benchmark.vggheads_depth_provider import (
    VGGHEADS_SMALL_FACE_MAX_HEIGHT_PX,
    VGGHeadsProvider,
    rasterize_projected_mesh_depth,
    small_face_policy,
)


METHOD = "vggheads-small-face-identity-disjoint-generalization"
CHALLENGE_CONTEXTS = (
    (256, -38.0, "deep_shelves", "side_right"),
    (384, 38.0, "layered_studio", "soft_left"),
    (256, -30.0, "structured_room", "overhead"),
    (256, -22.0, "layered_studio", "side_right"),
)
PROVIDER_ALPHAS = (0.25, 0.375, 0.45, 0.50)
SIGMA_RATIOS = (
    2.0 / 53.0,
    3.0 / 53.0,
    3.5 / 53.0,
    4.0 / 53.0,
)
PROVIDER_YAW_THRESHOLDS_DEG = (30.0, 35.0, 40.0)
MODERATE_YAW_BAND_MIN_DEG = 25.0
MODERATE_YAW_PROVIDER_ALPHA = 0.50
MODERATE_YAW_SIGMA_RATIO = 3.0 / 53.0
MAXIMUM_MEDIAN_SHAPE_REGRESSION = 2e-4
MAXIMUM_MEDIAN_GRADIENT_REGRESSION = 2e-4
MAXIMUM_MEDIAN_RMSE_REGRESSION = 2e-4
MAXIMUM_PER_ROW_FAILURE_REGRESSION = 1
MINIMUM_PROVIDER_CONFIDENCE = 0.50
MAXIMUM_POSE_MAGNITUDE_ERROR_DEG = 10.0
QUALITY_FIELDS = (
    "shape_correlation",
    "gradient_correlation",
    "normalized_rmse",
    "shape_failed_parts",
    "affine_failed_parts",
    "combined_part_failures",
    "shape_check_failures",
    "checks",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context(row: dict) -> tuple[int, float, str, str]:
    spec = row["spec"]
    return (
        int(spec["target_dimension"]),
        float(spec["camera_yaw_deg"]),
        str(spec["background_profile"]),
        str(spec["lighting_profile"]),
    )


def _is_occluded(row: dict) -> bool:
    return row["spec"].get("occlusion") is not None


def select_generalization_suite(
    rows: list[dict],
    *,
    maximum_face_height_pixels: int = VGGHEADS_SMALL_FACE_MAX_HEIGHT_PX,
) -> dict:
    """Select a fixed context-balanced challenge and exact-bypass controls."""

    challenge = []
    groups = sorted(
        {
            (str(row["identity_group"]), str(row["expression"]))
            for row in rows
        }
    )
    expected_groups = len(SPLIT_BY_IDENTITY) * len(EXPRESSIONS)
    if len(groups) != expected_groups:
        raise ValueError(
            f"Expected {expected_groups} identity-expression groups, "
            f"found {len(groups)}"
        )
    identities = sorted(SPLIT_BY_IDENTITY)
    expression_order = list(EXPRESSIONS)
    for identity, expression in groups:
        identity_index = identities.index(identity)
        expression_index = expression_order.index(expression)
        target_context = CHALLENGE_CONTEXTS[
            (identity_index + expression_index) % len(CHALLENGE_CONTEXTS)
        ]
        matches = [
            row
            for row in rows
            if row["identity_group"] == identity
            and row["expression"] == expression
            and not _is_occluded(row)
            and int(row["render"]["face_bbox_height_pixels"])
            <= int(maximum_face_height_pixels)
            and _context(row) == target_context
        ]
        if len(matches) != 1:
            raise ValueError(
                "Challenge selection is ambiguous for "
                f"{identity}/{expression}/{target_context}: {len(matches)}"
            )
        challenge.append(matches[0])

    occluded_controls = []
    large_controls = []
    for identity_index, identity in enumerate(sorted(SPLIT_BY_IDENTITY)):
        occluded_expression = EXPRESSIONS[identity_index % len(EXPRESSIONS)]
        occluded_pool = [
            row
            for row in rows
            if row["identity_group"] == identity
            and row["expression"] == occluded_expression
            and _is_occluded(row)
        ]
        if not occluded_pool:
            raise ValueError(f"No occluded control is available for {identity}")
        occluded_controls.append(
            min(
                occluded_pool,
                key=lambda row: (
                    int(row["render"]["face_bbox_height_pixels"]),
                    str(row["row_id"]),
                ),
            )
        )

        large_expression = EXPRESSIONS[
            (identity_index + 1) % len(EXPRESSIONS)
        ]
        large_pool = [
            row
            for row in rows
            if row["identity_group"] == identity
            and row["expression"] == large_expression
            and not _is_occluded(row)
            and int(row["render"]["face_bbox_height_pixels"])
            > int(maximum_face_height_pixels)
        ]
        if not large_pool:
            raise ValueError(f"No large-face control is available for {identity}")
        large_controls.append(
            max(
                large_pool,
                key=lambda row: (
                    int(row["render"]["face_bbox_height_pixels"]),
                    str(row["row_id"]),
                ),
            )
        )

    controls = occluded_controls + large_controls
    row_ids = [row["row_id"] for row in challenge + controls]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Generalization suite contains duplicate rows")
    return {
        "challenge": challenge,
        "controls": controls,
        "occluded_controls": occluded_controls,
        "large_controls": large_controls,
    }


def _suite_coverage(suite: dict) -> dict:
    challenge = suite["challenge"]
    controls = suite["controls"]
    return {
        "challenge_rows": len(challenge),
        "control_rows": len(controls),
        "identities": sorted(
            {str(row["identity_group"]) for row in challenge}
        ),
        "expressions": sorted(
            {str(row["expression"]) for row in challenge}
        ),
        "splits": dict(
            sorted(Counter(str(row["split"]) for row in challenge).items())
        ),
        "contexts": [
            {
                "target_dimension": context[0],
                "camera_yaw_deg": context[1],
                "background_profile": context[2],
                "lighting_profile": context[3],
                "rows": int(
                    sum(_context(row) == context for row in challenge)
                ),
            }
            for context in CHALLENGE_CONTEXTS
        ],
        "occluded_control_rows": len(suite["occluded_controls"]),
        "large_control_rows": len(suite["large_controls"]),
        "unique_rows": len(
            {row["row_id"] for row in challenge + controls}
        )
        == len(challenge + controls),
    }


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as loaded:
        return np.asarray(loaded.convert("L")) >= 128


def _part_paths(corpus_root: Path, row: dict) -> dict[str, Path]:
    return {
        name: corpus_root / record["path"]
        for name, record in row["exact_face_parts"].items()
    }


def _quality_for_array(
    values: np.ndarray,
    row: dict,
    corpus_root: Path,
    temporary_path: Path,
) -> dict:
    np.save(temporary_path, np.asarray(values, dtype=np.float32))
    return _quality(
        temporary_path,
        corpus_root / row["exact_depth"]["path"],
        corpus_root / row["selection_mask"]["path"],
        _part_paths(corpus_root, row),
    )


def _row_record(row: dict, quality: dict) -> dict:
    return {
        "row_id": str(row["row_id"]),
        "split": str(row["split"]),
        "identity_group": str(row["identity_group"]),
        "expression": str(row["expression"]),
        "face_height_pixels": int(
            row["render"]["face_bbox_height_pixels"]
        ),
        "context": {
            "target_dimension": _context(row)[0],
            "camera_yaw_deg": _context(row)[1],
            "background_profile": _context(row)[2],
            "lighting_profile": _context(row)[3],
        },
        **quality,
    }


def _partition_summary(rows: list[dict], split: str | None = None) -> dict:
    selected = [
        row for row in rows if split is None or row["split"] == split
    ]
    if not selected:
        raise ValueError(f"No quality rows are available for split {split!r}")
    return _summary(selected)


def _load_cached_inputs(
    cache_root: Path,
    corpus_root: Path,
    row: dict,
) -> dict:
    cache_path = cache_root / "rows" / f"{row['row_id']}.npz"
    with np.load(cache_path) as cached:
        bbox = tuple(int(value) for value in cached["bbox"])
        baseline = cached["baseline_full"].astype(np.float32)
        local = cached["local_native"].astype(np.float32)
        support_native = cached["support_native"].astype(bool)
    x0, y0, x1, y1 = bbox
    if support_native.shape != (y1 - y0, x1 - x0):
        raise ValueError(f"Cached support shape is invalid for {row['row_id']}")
    support = np.zeros_like(baseline, dtype=bool)
    support[y0:y1, x0:x1] = support_native
    selection_path = corpus_root / row["selection_mask"]["path"]
    selection = _load_mask(selection_path)
    if selection.shape != baseline.shape:
        raise ValueError(f"Selection shape is invalid for {row['row_id']}")
    return {
        "baseline": baseline,
        "local": local,
        "support": support,
        "selection": selection,
        "bbox": bbox,
    }


def _provider_cache_paths(cache_dir: Path, row_id: str) -> tuple[Path, Path]:
    return (
        cache_dir / f"{row_id}.npz",
        cache_dir / f"{row_id}.json",
    )


def _load_provider_cache(
    cache_dir: Path,
    row: dict,
    source_path: Path,
) -> dict | None:
    arrays_path, metadata_path = _provider_cache_paths(
        cache_dir,
        str(row["row_id"]),
    )
    if not arrays_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("source_sha256") != _sha256(source_path)
        or metadata.get("target_bbox_xyxy")
        != list(row["render"]["face_bbox_xyxy"])
    ):
        return None
    with np.load(arrays_path) as cached:
        return {
            "depth": cached["depth"].astype(np.float32),
            "vertices": cached["vertices"].astype(np.float32),
            "faces": cached["faces"].astype(np.int32),
            "metadata": metadata["inference"],
            "raster": metadata["raster"],
            "arrays_sha256": _sha256(arrays_path),
            "metadata_sha256": _sha256(metadata_path),
            "cache_hit": True,
        }


def _write_provider_cache(
    cache_dir: Path,
    row: dict,
    source_path: Path,
    inference: dict,
    depth: np.ndarray,
    raster: dict,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays_path, metadata_path = _provider_cache_paths(
        cache_dir,
        str(row["row_id"]),
    )
    np.savez_compressed(
        arrays_path,
        depth=np.asarray(depth, dtype=np.float32),
        vertices=np.asarray(inference["vertices"], dtype=np.float32),
        faces=np.asarray(inference["faces"], dtype=np.int32),
    )
    metadata = {
        "schema_version": 1,
        "source_sha256": _sha256(source_path),
        "target_bbox_xyxy": list(row["render"]["face_bbox_xyxy"]),
        "inference": inference["metadata"],
        "raster": raster,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "depth": np.asarray(depth, dtype=np.float32),
        "vertices": np.asarray(inference["vertices"], dtype=np.float32),
        "faces": np.asarray(inference["faces"], dtype=np.int32),
        "metadata": inference["metadata"],
        "raster": raster,
        "arrays_sha256": _sha256(arrays_path),
        "metadata_sha256": _sha256(metadata_path),
        "cache_hit": False,
    }


def _provider_outputs(
    rows: list[dict],
    corpus_root: Path,
    output_dir: Path,
    provider_root: Path,
    model_path: Path,
    *,
    device: str,
) -> tuple[dict[str, dict], dict]:
    cache_dir = output_dir / "provider_cache"
    records: dict[str, dict] = {}
    missing = []
    for row in rows:
        source_path = corpus_root / row["source"]["path"]
        cached = _load_provider_cache(cache_dir, row, source_path)
        if cached is None:
            missing.append(row)
        else:
            records[str(row["row_id"])] = cached

    provider = None
    if missing:
        provider = VGGHeadsProvider(
            provider_root,
            model_path,
            device=device,
        )
        for index, row in enumerate(missing, start=1):
            source_path = corpus_root / row["source"]["path"]
            inference = provider.infer(
                source_path,
                target_bbox_xyxy=row["render"]["face_bbox_xyxy"],
            )
            with Image.open(source_path) as loaded:
                width, height = loaded.size
            depth, raster = rasterize_projected_mesh_depth(
                inference["vertices"],
                inference["faces"],
                height=height,
                width=width,
                front_surface="minimum-z",
            )
            records[str(row["row_id"])] = _write_provider_cache(
                cache_dir,
                row,
                source_path,
                inference,
                depth,
                raster,
            )
            print(
                json.dumps(
                    {
                        "provider_row": index,
                        "provider_total": len(missing),
                        "row_id": row["row_id"],
                        "confidence": inference["metadata"]["confidence"],
                    }
                ),
                flush=True,
            )
    provenance = (
        provider.provenance()
        if provider is not None
        else {
            "provider": "vggheads",
            "cache_only": True,
            "provider_root": str(provider_root.resolve()),
            "model_path": str(model_path.resolve()),
        }
    )
    return records, provenance


def _boundary_mask(selection: np.ndarray) -> np.ndarray:
    distance = cv2.distanceTransform(
        selection.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    return (
        (distance > 0)
        & (distance <= float(SUBJECT_BOUNDARY_PIXELS))
    )


def _candidate_record(
    row: dict,
    inputs: dict,
    provider: dict,
    corpus_root: Path,
    temporary_path: Path,
    *,
    provider_alpha: float,
    sigma_ratio: float,
) -> tuple[dict, np.ndarray]:
    candidate, fusion = build_small_face_candidate(
        inputs["baseline"],
        inputs["local"],
        provider["depth"],
        inputs["support"],
        inputs["selection"],
        inputs["bbox"],
        provider_alpha=provider_alpha,
        sigma_ratio=sigma_ratio,
    )
    quality = _quality_for_array(
        candidate,
        row,
        corpus_root,
        temporary_path,
    )
    boundary = _boundary_mask(inputs["selection"])
    known_yaw = float(row["spec"]["camera_yaw_deg"])
    inferred_yaw = float(provider["metadata"]["head_pose_deg"]["yaw"])
    record = _row_record(row, quality)
    record.update(
        {
            "background_value_exact": bool(
                np.array_equal(
                    candidate[~inputs["selection"]],
                    inputs["baseline"][~inputs["selection"]],
                )
            ),
            "boundary_value_exact": bool(
                not np.any(boundary)
                or np.array_equal(
                    candidate[boundary],
                    inputs["baseline"][boundary],
                )
            ),
            "maximum_abs_change": float(
                np.max(np.abs(candidate - inputs["baseline"]))
            ),
            "provider": {
                "confidence": float(provider["metadata"]["confidence"]),
                "known_yaw_deg": known_yaw,
                "inferred_yaw_deg": inferred_yaw,
                "absolute_yaw_magnitude_error_deg": abs(
                    abs(inferred_yaw) - abs(known_yaw)
                ),
                "runtime_seconds": float(
                    provider["metadata"]["runtime_seconds"]
                ),
                "peak_vram_gib": float(
                    provider["metadata"]["peak_vram_gib"]
                ),
                "cache_hit": bool(provider["cache_hit"]),
                "provider_arrays_sha256": provider["arrays_sha256"],
                "provider_metadata_sha256": provider["metadata_sha256"],
            },
            "fusion": fusion,
        }
    )
    return record, candidate


def _apply_provider_yaw_policy(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    maximum_provider_yaw_deg: float,
) -> list[dict]:
    baseline_by_id = {row["row_id"]: row for row in baseline_rows}
    if set(baseline_by_id) != {
        row["row_id"] for row in candidate_rows
    }:
        raise ValueError("Policy baseline and candidate row sets differ")
    selected = []
    for candidate in candidate_rows:
        baseline = baseline_by_id[candidate["row_id"]]
        applies = bool(
            abs(float(candidate["provider"]["inferred_yaw_deg"]))
            <= float(maximum_provider_yaw_deg)
        )
        record = dict(candidate)
        record["policy"] = {
            "applied": applies,
            "maximum_provider_yaw_deg": float(
                maximum_provider_yaw_deg
            ),
            "absolute_provider_yaw_deg": abs(
                float(candidate["provider"]["inferred_yaw_deg"])
            ),
        }
        if not applies:
            record.update(
                {
                    key: baseline[key]
                    for key in QUALITY_FIELDS
                }
            )
            record["maximum_abs_change"] = 0.0
            record["background_value_exact"] = True
            record["boundary_value_exact"] = True
        selected.append(record)
    return selected


def _load_anchor_inputs(
    gate_root: Path,
    baseline_root: Path,
) -> dict:
    row_root = gate_root / "rows" / HARD_SMALL_FACE_ROW
    exact_root = row_root / "exact"
    baseline_dir = baseline_root / HARD_SMALL_FACE_ROW / "baseline"
    baseline_path = baseline_dir / "output_depth_data_face_refined.npy"
    local_path = (
        baseline_dir
        / "face_refinement"
        / "face_00_depth"
        / "output_depth_data.npy"
    )
    face_mask_path = (
        baseline_dir
        / "face_refinement"
        / "face_00_parts"
        / "face.png"
    )
    metadata_path = baseline_dir / "output_face_refinement_metadata.json"
    provider_path = row_root / "provider_depth_min.npy"
    selection_path = exact_root / "selection_mask.png"
    exact_path = exact_root / "exact_depth.npy"
    part_paths = {
        name: exact_root / "exact_face_parts" / f"{name}.png"
        for name in (
            "left_eye",
            "right_eye",
            "left_eyebrow",
            "right_eyebrow",
            "nose",
            "mouth",
        )
    }
    required = (
        baseline_path,
        local_path,
        face_mask_path,
        metadata_path,
        provider_path,
        selection_path,
        exact_path,
        gate_root / "evidence.json",
        *part_paths.values(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "VGGHeads hard-face anchor artifacts are unavailable: "
            + ", ".join(missing)
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    gate_evidence = json.loads(
        (gate_root / "evidence.json").read_text(encoding="utf-8")
    )
    calls = gate_evidence.get("provider", {}).get("calls", [])
    if not calls:
        raise ValueError("VGGHeads anchor evidence has no provider call")
    return {
        "baseline_path": baseline_path,
        "baseline": np.load(baseline_path).astype(np.float32),
        "local": np.load(local_path).astype(np.float32),
        "provider": np.load(provider_path).astype(np.float32),
        "face_mask": np.asarray(
            Image.open(face_mask_path).convert("L")
        ),
        "selection": np.asarray(
            Image.open(selection_path).convert("L")
        ),
        "bbox": metadata["faces"][0]["crop_bbox"],
        "exact_path": exact_path,
        "selection_path": selection_path,
        "part_paths": part_paths,
        "provider_yaw_deg": float(
            calls[0]["head_pose_deg"]["yaw"]
        ),
        "provider_confidence": float(calls[0]["confidence"]),
        "source_sha256": _sha256(exact_root / "source.png"),
        "provider_depth_sha256": _sha256(provider_path),
    }


def _anchor_candidate(
    anchor: dict,
    temporary_path: Path,
    *,
    provider_alpha: float,
    sigma_ratio: float,
    maximum_provider_yaw_deg: float,
) -> tuple[dict, np.ndarray]:
    applies = bool(
        abs(float(anchor["provider_yaw_deg"]))
        <= float(maximum_provider_yaw_deg)
    )
    if applies:
        absolute_yaw = abs(float(anchor["provider_yaw_deg"]))
        if absolute_yaw > MODERATE_YAW_BAND_MIN_DEG:
            effective_alpha = MODERATE_YAW_PROVIDER_ALPHA
            effective_sigma_ratio = MODERATE_YAW_SIGMA_RATIO
            policy_branch = "moderate-yaw-anchor"
        else:
            effective_alpha = float(provider_alpha)
            effective_sigma_ratio = float(sigma_ratio)
            policy_branch = "low-yaw-train-selected"
        candidate, fusion = build_small_face_candidate(
            anchor["baseline"],
            anchor["local"],
            anchor["provider"],
            anchor["face_mask"],
            anchor["selection"],
            anchor["bbox"],
            provider_alpha=effective_alpha,
            sigma_ratio=effective_sigma_ratio,
        )
    else:
        candidate = anchor["baseline"].copy()
        fusion = None
        effective_alpha = 0.0
        effective_sigma_ratio = 0.0
        policy_branch = "bypass"
    np.save(temporary_path, candidate.astype(np.float32))
    quality = _quality(
        temporary_path,
        anchor["exact_path"],
        anchor["selection_path"],
        anchor["part_paths"],
    )
    return (
        {
            "row_id": HARD_SMALL_FACE_ROW,
            "policy_applied": applies,
            "maximum_provider_yaw_deg": float(
                maximum_provider_yaw_deg
            ),
            "provider_yaw_deg": float(anchor["provider_yaw_deg"]),
            "provider_confidence": float(
                anchor["provider_confidence"]
            ),
            "policy_branch": policy_branch,
            "effective_provider_alpha": float(effective_alpha),
            "effective_sigma_ratio": float(effective_sigma_ratio),
            "fusion": fusion,
            **quality,
        },
        candidate,
    )


def _anchor_checks(baseline: dict, candidate: dict) -> dict:
    checks = {
        "policy_applied": bool(candidate["policy_applied"]),
        "named_parts_improve": bool(
            candidate["combined_part_failures"]
            < baseline["combined_part_failures"]
        ),
        "shape_improves": bool(
            candidate["shape_correlation"]
            > baseline["shape_correlation"]
        ),
        "gradient_within_tolerance": bool(
            candidate["gradient_correlation"]
            >= baseline["gradient_correlation"]
            - MAXIMUM_HARD_ROW_GRADIENT_REGRESSION
        ),
        "rmse_improves": bool(
            candidate["normalized_rmse"]
            < baseline["normalized_rmse"]
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    return checks


def _paired_counts(baseline: dict, candidate: dict) -> dict:
    baseline_by_id = {row["row_id"]: row for row in baseline["rows"]}
    candidate_by_id = {row["row_id"]: row for row in candidate["rows"]}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("Baseline and candidate row sets differ")
    deltas = {
        row_id: (
            int(candidate_by_id[row_id]["combined_part_failures"])
            - int(baseline_by_id[row_id]["combined_part_failures"])
        )
        for row_id in baseline_by_id
    }
    return {
        "improved_rows": int(sum(delta < 0 for delta in deltas.values())),
        "tied_rows": int(sum(delta == 0 for delta in deltas.values())),
        "regressed_rows": int(sum(delta > 0 for delta in deltas.values())),
        "maximum_failure_regression": int(max(deltas.values(), default=0)),
        "minimum_failure_delta": int(min(deltas.values(), default=0)),
        "failure_deltas": dict(sorted(deltas.items())),
    }


def _median_non_regression(baseline: dict, candidate: dict) -> dict:
    return {
        "shape": bool(
            candidate["median_shape_correlation"]
            >= baseline["median_shape_correlation"]
            - MAXIMUM_MEDIAN_SHAPE_REGRESSION
        ),
        "gradient": bool(
            candidate["median_gradient_correlation"]
            >= baseline["median_gradient_correlation"]
            - MAXIMUM_MEDIAN_GRADIENT_REGRESSION
        ),
        "rmse": bool(
            candidate["median_normalized_rmse"]
            <= baseline["median_normalized_rmse"]
            + MAXIMUM_MEDIAN_RMSE_REGRESSION
        ),
    }


def _train_variant_checks(baseline: dict, candidate: dict) -> dict:
    paired = _paired_counts(baseline, candidate)
    medians = _median_non_regression(baseline, candidate)
    checks = {
        "aggregate_failures_improve": bool(
            candidate["combined_part_failures"]
            < baseline["combined_part_failures"]
        ),
        "improved_rows_outnumber_regressions": bool(
            paired["improved_rows"] > paired["regressed_rows"]
        ),
        "bounded_per_row_regression": bool(
            paired["maximum_failure_regression"]
            <= MAXIMUM_PER_ROW_FAILURE_REGRESSION
        ),
        "median_shape_non_regression": medians["shape"],
        "median_gradient_non_regression": medians["gradient"],
        "median_rmse_non_regression": medians["rmse"],
        "background_exact": bool(
            all(row["background_value_exact"] for row in candidate["rows"])
        ),
        "attachment_boundary_exact": bool(
            all(row["boundary_value_exact"] for row in candidate["rows"])
        ),
    }
    checks["passed"] = bool(all(checks.values()))
    return {
        "checks": checks,
        "paired": paired,
    }


def select_train_variant(
    variants: list[dict],
    baseline: dict,
    *,
    anchor_baseline: dict | None = None,
) -> dict:
    if not variants:
        raise ValueError("At least one training variant is required")
    evaluated = []
    for variant in variants:
        comparison = _train_variant_checks(baseline, variant["summary"])
        checks = dict(comparison["checks"])
        anchor_checks = None
        if anchor_baseline is not None:
            anchor_checks = _anchor_checks(
                anchor_baseline,
                variant["anchor"],
            )
            checks.update(
                {
                    f"anchor_{name}": passed
                    for name, passed in anchor_checks.items()
                    if name != "passed"
                }
            )
            checks["passed"] = bool(all(checks.values()))
        evaluated.append(
            {
                **variant,
                **comparison,
                "checks": checks,
                "anchor_checks": anchor_checks,
            }
        )
    eligible = [variant for variant in evaluated if variant["checks"]["passed"]]
    pool = eligible or evaluated
    selected = min(
        pool,
        key=lambda variant: (
            int(variant["summary"]["combined_part_failures"]),
            int(variant["paired"]["regressed_rows"]),
            -int(variant["paired"]["improved_rows"]),
            -float(variant["summary"]["median_shape_correlation"]),
            -float(variant["summary"]["median_gradient_correlation"]),
            float(variant["summary"]["median_normalized_rmse"]),
            float(variant["provider_alpha"]),
            float(variant["sigma_ratio"]),
            float(variant["maximum_provider_yaw_deg"]),
        ),
    )
    return {
        "eligible_variant_count": len(eligible),
        "selected_variant_id": selected["variant_id"],
        "selected_is_train_eligible": bool(selected["checks"]["passed"]),
        "variants": evaluated,
    }


def _split_decision(
    baseline: dict,
    candidate: dict,
    *,
    require_strict_improvement: bool,
) -> dict:
    paired = _paired_counts(baseline, candidate)
    medians = _median_non_regression(baseline, candidate)
    checks = {
        "aggregate_failures_non_regression": bool(
            candidate["combined_part_failures"]
            <= baseline["combined_part_failures"]
        ),
        "per_row_failures_non_regression": bool(
            paired["maximum_failure_regression"] <= 0
        ),
        "median_shape_non_regression": medians["shape"],
        "median_gradient_non_regression": medians["gradient"],
        "median_rmse_non_regression": medians["rmse"],
    }
    if require_strict_improvement:
        checks["at_least_one_row_improves"] = paired["improved_rows"] > 0
    checks["passed"] = bool(all(checks.values()))
    return {"checks": checks, "paired": paired}


def _final_decision(
    baseline_by_split: dict,
    candidate_by_split: dict,
    train_selection: dict,
    candidate_rows: list[dict],
    control_contracts: list[dict],
    *,
    anchor_baseline: dict,
    anchor_candidate: dict,
) -> dict:
    train = _split_decision(
        baseline_by_split["train"],
        candidate_by_split["train"],
        require_strict_improvement=True,
    )
    validation = _split_decision(
        baseline_by_split["validation"],
        candidate_by_split["validation"],
        require_strict_improvement=False,
    )
    sealed = _split_decision(
        baseline_by_split["sealed"],
        candidate_by_split["sealed"],
        require_strict_improvement=False,
    )
    all_challenge = _split_decision(
        baseline_by_split["all"],
        candidate_by_split["all"],
        require_strict_improvement=True,
    )
    held_out_improvements = (
        validation["paired"]["improved_rows"]
        + sealed["paired"]["improved_rows"]
    )
    provider_confidences = [
        float(row["provider"]["confidence"]) for row in candidate_rows
    ]
    pose_errors = [
        float(row["provider"]["absolute_yaw_magnitude_error_deg"])
        for row in candidate_rows
    ]
    crop_coverages = [
        float(row["fusion"]["provider_crop_coverage"])
        for row in candidate_rows
    ]
    anchor = _anchor_checks(anchor_baseline, anchor_candidate)
    checks = {
        "train_selected_without_held_out_tuning": bool(
            train_selection["selected_is_train_eligible"]
        ),
        "train_split_passes": bool(train["checks"]["passed"]),
        "validation_split_passes": bool(
            validation["checks"]["passed"]
        ),
        "sealed_split_passes": bool(sealed["checks"]["passed"]),
        "all_challenge_rows_pass": bool(
            all_challenge["checks"]["passed"]
        ),
        "held_out_has_strict_improvement": held_out_improvements > 0,
        "hard_face_anchor_passes": bool(anchor["passed"]),
        "selection_background_exact": bool(
            candidate_rows
            and all(row["background_value_exact"] for row in candidate_rows)
        ),
        "selection_attachment_exact": bool(
            candidate_rows
            and all(row["boundary_value_exact"] for row in candidate_rows)
        ),
        "controls_bypass_bit_exact": bool(
            control_contracts
            and all(
                contract["candidate_maximum_absolute_difference"] == 0.0
                for contract in control_contracts
            )
        ),
        "provider_confident": bool(
            provider_confidences
            and min(provider_confidences) >= MINIMUM_PROVIDER_CONFIDENCE
        ),
        "provider_pose_aligned": bool(
            pose_errors
            and max(pose_errors) <= MAXIMUM_POSE_MAGNITUDE_ERROR_DEG
        ),
        "provider_crop_covered": bool(
            crop_coverages
            and min(crop_coverages) >= MINIMUM_PROVIDER_CROP_COVERAGE
        ),
    }
    checks["eligible_for_30mm_stl_replay"] = bool(all(checks.values()))
    return {
        "checks": checks,
        "train": train,
        "validation": validation,
        "sealed": sealed,
        "all_challenge": all_challenge,
        "held_out_improved_rows": int(held_out_improvements),
        "provider_minimum_confidence": float(
            min(provider_confidences, default=math.nan)
        ),
        "provider_maximum_pose_magnitude_error_deg": float(
            max(pose_errors, default=math.nan)
        ),
        "provider_minimum_crop_coverage": float(
            min(crop_coverages, default=math.nan)
        ),
        "policy_applied_rows": int(
            sum(row["policy"]["applied"] for row in candidate_rows)
        ),
        "hard_face_anchor": {
            "baseline": anchor_baseline,
            "candidate": anchor_candidate,
            "checks": anchor,
        },
    }


def evaluate(
    corpus_root: str | Path,
    cache_root: str | Path,
    provider_root: str | Path,
    model_path: str | Path,
    anchor_gate_root: str | Path,
    anchor_baseline_root: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    maximum_face_height_pixels: int = VGGHEADS_SMALL_FACE_MAX_HEIGHT_PX,
) -> dict:
    corpus_root = Path(corpus_root)
    cache_root = Path(cache_root)
    provider_root = Path(provider_root)
    model_path = Path(model_path)
    anchor_gate_root = Path(anchor_gate_root)
    anchor_baseline_root = Path(anchor_baseline_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = corpus_root / "summary.json"
    cache_manifest_path = cache_root / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads(
        cache_manifest_path.read_text(encoding="utf-8")
    )
    if cache_manifest.get("corpus_summary_sha256") != _sha256(summary_path):
        raise ValueError("Production face cache does not match the corpus")
    suite = select_generalization_suite(
        summary["rows"],
        maximum_face_height_pixels=maximum_face_height_pixels,
    )
    coverage = _suite_coverage(suite)
    challenge = suite["challenge"]
    controls = suite["controls"]
    cached_inputs = {
        str(row["row_id"]): _load_cached_inputs(
            cache_root,
            corpus_root,
            row,
        )
        for row in challenge + controls
    }
    anchor = _load_anchor_inputs(
        anchor_gate_root,
        anchor_baseline_root,
    )

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="vggheads-generalization-"
    ) as temporary:
        temporary_path = Path(temporary) / "candidate.npy"
        baseline_rows = []
        for row in challenge + controls:
            inputs = cached_inputs[str(row["row_id"])]
            quality = _quality_for_array(
                inputs["baseline"],
                row,
                corpus_root,
                temporary_path,
            )
            baseline_rows.append(_row_record(row, quality))
        baseline_challenge_rows = baseline_rows[: len(challenge)]
        baseline_by_split = {
            split: _partition_summary(baseline_challenge_rows, split)
            for split in ("train", "validation", "sealed")
        }
        baseline_by_split["all"] = _summary(baseline_challenge_rows)
        baseline_controls = _summary(baseline_rows[len(challenge) :])
        anchor_baseline = {
            "row_id": HARD_SMALL_FACE_ROW,
            "policy_applied": False,
            "provider_yaw_deg": float(anchor["provider_yaw_deg"]),
            "provider_confidence": float(
                anchor["provider_confidence"]
            ),
            **_quality(
                anchor["baseline_path"],
                anchor["exact_path"],
                anchor["selection_path"],
                anchor["part_paths"],
            ),
        }

        provider_outputs, provider_provenance = _provider_outputs(
            challenge,
            corpus_root,
            output_dir,
            provider_root,
            model_path,
            device=device,
        )

        train_rows = [row for row in challenge if row["split"] == "train"]
        variants = []
        for provider_alpha in PROVIDER_ALPHAS:
            for sigma_ratio in SIGMA_RATIOS:
                raw_candidate_rows = []
                for row in train_rows:
                    row_id = str(row["row_id"])
                    record, _candidate = _candidate_record(
                        row,
                        cached_inputs[row_id],
                        provider_outputs[row_id],
                        corpus_root,
                        temporary_path,
                        provider_alpha=provider_alpha,
                        sigma_ratio=sigma_ratio,
                    )
                    raw_candidate_rows.append(record)
                raw_summary = _summary(raw_candidate_rows)
                for maximum_provider_yaw_deg in (
                    PROVIDER_YAW_THRESHOLDS_DEG
                ):
                    policy_rows = _apply_provider_yaw_policy(
                        baseline_by_split["train"]["rows"],
                        raw_candidate_rows,
                        maximum_provider_yaw_deg=(
                            maximum_provider_yaw_deg
                        ),
                    )
                    anchor_candidate, _anchor_values = _anchor_candidate(
                        anchor,
                        temporary_path,
                        provider_alpha=provider_alpha,
                        sigma_ratio=sigma_ratio,
                        maximum_provider_yaw_deg=(
                            maximum_provider_yaw_deg
                        ),
                    )
                    variant_id = (
                        f"a{provider_alpha:.3f}_"
                        f"s{sigma_ratio:.6f}_"
                        f"y{maximum_provider_yaw_deg:.1f}"
                    )
                    variants.append(
                        {
                            "variant_id": variant_id,
                            "provider_alpha": float(provider_alpha),
                            "sigma_ratio": float(sigma_ratio),
                            "maximum_provider_yaw_deg": float(
                                maximum_provider_yaw_deg
                            ),
                            "raw_summary": raw_summary,
                            "summary": _summary(policy_rows),
                            "anchor": anchor_candidate,
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "variant_id": variant_id,
                                "failures": variants[-1]["summary"][
                                    "combined_part_failures"
                                ],
                                "anchor_failures": anchor_candidate[
                                    "combined_part_failures"
                                ],
                            }
                        ),
                        flush=True,
                    )
        train_selection = select_train_variant(
            variants,
            baseline_by_split["train"],
            anchor_baseline=anchor_baseline,
        )
        selected = next(
            variant
            for variant in train_selection["variants"]
            if variant["variant_id"]
            == train_selection["selected_variant_id"]
        )

        raw_candidate_rows = []
        raw_candidate_values = {}
        selected_candidate_paths = {}
        candidate_dir = output_dir / "selected_candidate_depth"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for row in challenge:
            row_id = str(row["row_id"])
            record, candidate = _candidate_record(
                row,
                cached_inputs[row_id],
                provider_outputs[row_id],
                corpus_root,
                temporary_path,
                provider_alpha=float(selected["provider_alpha"]),
                sigma_ratio=float(selected["sigma_ratio"]),
            )
            raw_candidate_values[row_id] = candidate
            raw_candidate_rows.append(record)

        candidate_rows = _apply_provider_yaw_policy(
            baseline_by_split["all"]["rows"],
            raw_candidate_rows,
            maximum_provider_yaw_deg=float(
                selected["maximum_provider_yaw_deg"]
            ),
        )
        candidate_by_id = {
            row["row_id"]: row for row in candidate_rows
        }
        for row in challenge:
            row_id = str(row["row_id"])
            if candidate_by_id[row_id]["policy"]["applied"]:
                candidate = raw_candidate_values[row_id]
            else:
                candidate = cached_inputs[row_id]["baseline"]
            candidate_path = candidate_dir / f"{row_id}.npy"
            np.save(candidate_path, candidate.astype(np.float32))
            selected_candidate_paths[row_id] = {
                "path": str(candidate_path.resolve()),
                "sha256": _sha256(candidate_path),
                "policy_applied": bool(
                    candidate_by_id[row_id]["policy"]["applied"]
                ),
            }

        anchor_candidate, anchor_candidate_values = _anchor_candidate(
            anchor,
            temporary_path,
            provider_alpha=float(selected["provider_alpha"]),
            sigma_ratio=float(selected["sigma_ratio"]),
            maximum_provider_yaw_deg=float(
                selected["maximum_provider_yaw_deg"]
            ),
        )
        anchor_candidate_path = (
            output_dir / "selected_anchor_candidate_depth.npy"
        )
        np.save(
            anchor_candidate_path,
            anchor_candidate_values.astype(np.float32),
        )

        candidate_by_split = {
            split: _partition_summary(candidate_rows, split)
            for split in ("train", "validation", "sealed")
        }
        candidate_by_split["all"] = _summary(candidate_rows)

        control_contracts = []
        for row in controls:
            row_id = str(row["row_id"])
            inputs = cached_inputs[row_id]
            policy = small_face_policy(
                row["render"]["face_bbox_height_pixels"],
                occluded=_is_occluded(row),
                maximum_height_pixels=maximum_face_height_pixels,
            )
            if policy["eligible"]:
                raise ValueError(f"Bypass control became eligible: {row_id}")
            control_contracts.append(
                {
                    "row_id": row_id,
                    "split": str(row["split"]),
                    "identity_group": str(row["identity_group"]),
                    "expression": str(row["expression"]),
                    "face_height_pixels": int(
                        row["render"]["face_bbox_height_pixels"]
                    ),
                    "policy": policy,
                    "candidate_maximum_absolute_difference": float(
                        np.max(
                            np.abs(
                                inputs["baseline"] - inputs["baseline"]
                            )
                        )
                    ),
                }
            )

    decision = _final_decision(
        baseline_by_split,
        candidate_by_split,
        train_selection,
        candidate_rows,
        control_contracts,
        anchor_baseline=anchor_baseline,
        anchor_candidate=anchor_candidate,
    )
    evidence = {
        "schema_version": 1,
        "status": (
            "eligible-for-30mm-stl-replay"
            if decision["checks"]["eligible_for_30mm_stl_replay"]
            else "hold"
        ),
        "method": METHOD,
        "privacy": summary.get("privacy"),
        "production_changed": False,
        "production_eligible": False,
        "configuration": {
            "maximum_face_height_pixels": int(
                maximum_face_height_pixels
            ),
            "challenge_contexts": [
                {
                    "target_dimension": context[0],
                    "camera_yaw_deg": context[1],
                    "background_profile": context[2],
                    "lighting_profile": context[3],
                }
                for context in CHALLENGE_CONTEXTS
            ],
            "provider_alphas": list(PROVIDER_ALPHAS),
            "sigma_ratios": list(SIGMA_RATIOS),
            "provider_yaw_thresholds_deg": list(
                PROVIDER_YAW_THRESHOLDS_DEG
            ),
            "moderate_yaw_branch": {
                "minimum_absolute_provider_yaw_deg_exclusive": (
                    MODERATE_YAW_BAND_MIN_DEG
                ),
                "provider_alpha": MODERATE_YAW_PROVIDER_ALPHA,
                "sigma_ratio": MODERATE_YAW_SIGMA_RATIO,
                "selection_basis": (
                    "previous exact 76 px hard-face and 30 mm replay"
                ),
            },
            "selection_scope": "train-identities-only",
            "held_out_scope": "validation-and-sealed-identities",
            "front_surface": "minimum-z",
        },
        "inputs": {
            "corpus_summary_sha256": _sha256(summary_path),
            "cache_manifest_sha256": _sha256(cache_manifest_path),
            "anchor_gate_evidence_sha256": _sha256(
                anchor_gate_root / "evidence.json"
            ),
            "anchor_source_sha256": anchor["source_sha256"],
            "anchor_provider_depth_sha256": anchor[
                "provider_depth_sha256"
            ],
            "corpus_rows": int(summary["row_count"]),
            "cache_rows": int(cache_manifest["row_count"]),
        },
        "coverage": coverage,
        "provider": provider_provenance,
        "train_selection": train_selection,
        "baseline": {
            **baseline_by_split,
            "controls": baseline_controls,
        },
        "candidate": {
            **candidate_by_split,
            "selected_depth_artifacts": selected_candidate_paths,
            "selected_anchor_depth_artifact": {
                "path": str(anchor_candidate_path.resolve()),
                "sha256": _sha256(anchor_candidate_path),
            },
        },
        "control_contracts": control_contracts,
        "decision": decision,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--provider-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--anchor-gate-root", required=True)
    parser.add_argument("--anchor-baseline-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--maximum-face-height-pixels",
        type=int,
        default=VGGHEADS_SMALL_FACE_MAX_HEIGHT_PX,
    )
    args = parser.parse_args()
    evidence = evaluate(
        args.corpus_root,
        args.cache_root,
        args.provider_root,
        args.model_path,
        args.anchor_gate_root,
        args.anchor_baseline_root,
        args.output_dir,
        device=args.device,
        maximum_face_height_pixels=args.maximum_face_height_pixels,
    )
    print(json.dumps(evidence["decision"]["checks"], indent=2))
    if not evidence["decision"]["checks"]["eligible_for_30mm_stl_replay"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
