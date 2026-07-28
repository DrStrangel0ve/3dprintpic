import argparse
import csv
import math
from pathlib import Path

import numpy as np


DEFAULT_WEIGHTS = {
    "masked_mae_median": -4.0,
    "object_masked_mae_median": -5.0,
    "seam_mae_median": -1.0,
    "object_seam_mae_median": -1.0,
    "visible_mae_pre_preserve_median": -2.0,
    "seam_mae_pre_preserve_median": -0.5,
    "visible_mae_median": -2.0,
    "masked_psnr_median": 0.15,
    "object_masked_psnr_median": 0.2,
    "masked_ssim_median": 1.0,
    "object_masked_ssim_median": 1.25,
    "depth_mae_median": -2.0,
    "object_depth_mae_median": -3.0,
    "object_depth_rmse_median": -1.0,
    "object_depth_corr_median": 0.75,
    "surface_chamfer_l1_median": -1.5,
    "surface_chamfer_rmse_median": -1.0,
    "surface_hausdorff95_median": -0.5,
    "object_surface_chamfer_l1_median": -2.0,
    "object_surface_chamfer_rmse_median": -1.25,
    "object_surface_hausdorff95_median": -0.75,
    "silhouette_iou_masked_median": 1.0,
    "stl_exists_median": 2.0,
    "stl_is_watertight_median": 2.0,
    "stl_positive_volume_median": 1.0,
}

OBJECT_SURFACE_WEIGHTS = {
    "object_depth_mae_median": -1.0,
    "object_depth_rmse_median": -1.0,
    "object_depth_corr_median": 0.5,
    "object_surface_chamfer_l1_median": -4.0,
    "object_surface_chamfer_rmse_median": -3.0,
    "object_surface_hausdorff95_median": -2.0,
    "silhouette_iou_masked_median": 1.0,
    "stl_exists_median": 2.0,
    "stl_is_watertight_median": 2.0,
    "stl_positive_volume_median": 1.0,
}

STL_QUALITY_WEIGHTS = {
    "heldout_view_silhouette_iou_mean_median": 4.0,
    "heldout_view_silhouette_iou_min_median": 1.0,
    "silhouette_iou_masked_median": 1.0,
    "stl_exists_median": 2.0,
    "stl_is_watertight_median": 3.0,
    "stl_is_volume_median": 2.0,
    "stl_is_manifold_median": 2.0,
    "stl_nonmanifold_edge_count_log1p_median": -0.75,
    "stl_degenerate_face_ratio_median": -1.0,
    "stl_winding_consistent_median": 1.0,
    "stl_positive_volume_median": 1.0,
    "stl_single_component_median": 1.0,
    "stl_component_excess_log1p_median": -0.75,
    "stl_bbox_has_volume_median": 1.0,
    "stl_bbox_aspect_ratio_median": -0.5,
}

ZERO_BASELINE_METRICS = frozenset(
    {
        "heldout_view_silhouette_iou_mean_median",
        "heldout_view_silhouette_iou_min_median",
    }
)

SCORE_PROFILES = {
    "default": DEFAULT_WEIGHTS,
    "object-surface": OBJECT_SURFACE_WEIGHTS,
    "stl-quality": STL_QUALITY_WEIGHTS,
}

SCORE_PROFILE_DESCRIPTIONS = {
    "default": "balanced image, depth, surface, silhouette, and STL validity metrics",
    "object-surface": "object depth/surface reconstruction plus silhouette and STL validity metrics",
    "stl-quality": "final STL mesh accuracy, validity, and printable complexity metrics",
}

SCORE_MODES = {"normalized", "baseline-delta"}


def parse_weights(items, profile="default"):
    if profile not in SCORE_PROFILES:
        raise ValueError(f"Unknown score profile `{profile}`. Expected one of: {', '.join(sorted(SCORE_PROFILES))}")
    weights = dict(SCORE_PROFILES[profile])
    for item in items:
        key, value = item.split("=", 1)
        weights[key] = float(value)
    return weights


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _has_value(row: dict, field: str) -> bool:
    return field in row and str(row.get(field, "")).strip() != ""


def uses_repair_fill_schema_v2(row: dict) -> bool:
    if row.get("repair_fill_ratio_metric") == "legacy-signed-volume-fallback":
        return False
    return any(
        str(key).startswith(
            (
                "repair_fill_ratio_",
                "raw_mesh_volume_fill_ratio_reliability_",
            )
        )
        for key in row
    )


def _derive_scale_free_face_density(row: dict, suffix: str) -> None:
    target_field = f"stl_faces_per_normalized_bbox_volume_log1p{suffix}"
    if _has_value(row, target_field):
        return
    faces = parse_float(row.get(f"stl_faces{suffix}"))
    bbox_volume = parse_float(row.get(f"stl_bbox_volume{suffix}"))
    max_dimension = parse_float(row.get(f"stl_bbox_max_dimension{suffix}"))
    if not (
        math.isfinite(faces)
        and math.isfinite(bbox_volume)
        and bbox_volume > 0
        and math.isfinite(max_dimension)
        and max_dimension > 0
    ):
        return
    normalized_bbox_volume = bbox_volume / (max_dimension**3)
    if not math.isfinite(normalized_bbox_volume) or normalized_bbox_volume <= 0:
        return
    faces_per_normalized_volume = faces / normalized_bbox_volume
    if not math.isfinite(faces_per_normalized_volume):
        return
    row[f"stl_normalized_bbox_volume{suffix}"] = normalized_bbox_volume
    row[f"stl_faces_per_normalized_bbox_volume{suffix}"] = faces_per_normalized_volume
    row[target_field] = math.log1p(faces_per_normalized_volume)


def _derive_absolute_repair_fill_change(row: dict, suffix: str) -> None:
    target_field = f"repair_volume_fill_ratio_relative_change_abs{suffix}"
    if _has_value(row, target_field):
        return
    source_field = f"repair_volume_fill_ratio_relative_change{suffix}"
    value = parse_float(row.get(source_field))
    if math.isfinite(value):
        row[target_field] = abs(value)


def _derive_absolute_canonical_repair_fill_change(row: dict, suffix: str) -> None:
    target_field = f"repair_fill_ratio_relative_change_abs{suffix}"
    if _has_value(row, target_field):
        return
    source_field = f"repair_fill_ratio_relative_change{suffix}"
    value = parse_float(row.get(source_field))
    if math.isfinite(value):
        row[target_field] = abs(value)


def _derive_canonical_repair_fill_change(row: dict, suffix: str) -> None:
    target_field = f"repair_fill_ratio_relative_change_abs{suffix}"
    if _has_value(row, target_field):
        return
    if uses_repair_fill_schema_v2(row):
        return
    legacy_field = f"repair_volume_fill_ratio_relative_change_abs{suffix}"
    value = parse_float(row.get(legacy_field))
    if math.isfinite(value):
        row[target_field] = value
        row.setdefault("repair_fill_ratio_metric", "legacy-signed-volume-fallback")


def with_derived_metrics(row: dict) -> dict:
    derived = dict(row)
    for suffix in ("", "_median", "_mean"):
        _derive_scale_free_face_density(derived, suffix)
    _derive_absolute_repair_fill_change(derived, "")
    for suffix in ("", "_median", "_mean"):
        _derive_absolute_canonical_repair_fill_change(derived, suffix)
        _derive_canonical_repair_fill_change(derived, suffix)
    return derived


def normalize(values, higher_is_better=True):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite):
        span = max(1.0, float(np.max(finite) - np.min(finite)))
        values = values.copy()
        values[np.isposinf(values)] = float(np.max(finite) + span)
        values[np.isneginf(values)] = float(np.min(finite) - span)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros_like(values)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi == lo:
        normalized = np.zeros_like(values)
        normalized[np.isfinite(values)] = 0.5
        return normalized
    normalized = (values - lo) / (hi - lo)
    if not higher_is_better:
        normalized = 1.0 - normalized
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized


def score_normalized(rows, weights):
    scores = np.zeros(len(rows), dtype=np.float64)
    used_metrics = []
    field_names = set()
    for row in rows:
        field_names.update(row.keys())

    for metric, weight in weights.items():
        if metric not in field_names:
            continue
        values = [parse_float(row.get(metric)) for row in rows]
        if not any(math.isfinite(value) for value in values):
            continue
        scores += abs(weight) * normalize(values, higher_is_better=weight > 0)
        used_metrics.append(metric)
    return scores, used_metrics


def score_baseline_delta(rows, weights, baseline_method="masked"):
    baseline = next((row for row in rows if row.get("method") == baseline_method), None)
    if baseline is None:
        raise ValueError(f"Baseline method `{baseline_method}` was not found in summary rows")

    scores = np.zeros(len(rows), dtype=np.float64)
    used_metrics = []
    field_names = set()
    for row in rows:
        field_names.update(row.keys())

    for metric, weight in weights.items():
        if metric not in field_names:
            continue
        baseline_value = parse_float(baseline.get(metric))
        if not math.isfinite(baseline_value) and metric in ZERO_BASELINE_METRICS:
            baseline_value = 0.0
        if not math.isfinite(baseline_value):
            continue

        used_metric = False
        higher_is_better = weight > 0
        for index, row in enumerate(rows):
            method_value = parse_float(row.get(metric))
            if not math.isfinite(method_value):
                continue
            raw_delta = method_value - baseline_value
            improvement = raw_delta if higher_is_better else -raw_delta
            scores[index] += abs(weight) * improvement
            used_metric = True
        if used_metric:
            used_metrics.append(metric)
    return scores, used_metrics


def score_rows(rows, weights, score_mode="normalized", baseline_method="masked"):
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unknown score mode `{score_mode}`. Expected one of: {', '.join(sorted(SCORE_MODES))}")
    if score_mode == "baseline-delta":
        return score_baseline_delta(rows, weights, baseline_method=baseline_method)
    return score_normalized(rows, weights)


def rank_summary_rows(rows, weights, score_mode="normalized", baseline_method="masked"):
    if not rows:
        return [], []
    rows = [with_derived_metrics(row) for row in rows]

    scores, used_metrics = score_rows(
        rows,
        weights,
        score_mode=score_mode,
        baseline_method=baseline_method,
    )
    ranked = []
    for row, score in zip(rows, scores):
        ranked_row = {
            "rank_score": float(score),
            "score_mode": score_mode,
            "score_baseline_method": baseline_method if score_mode == "baseline-delta" else "",
        }
        ranked_row.update({key: value for key, value in row.items() if key not in ranked_row})
        ranked.append(ranked_row)
    ranked.sort(key=lambda item: item["rank_score"], reverse=True)
    return ranked, used_metrics


def main():
    parser = argparse.ArgumentParser(description="Rank completion methods by a weighted benchmark objective.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--score-profile",
        choices=sorted(SCORE_PROFILES),
        default="default",
        help="Named metric weight profile. Explicit --weight overrides are applied on top.",
    )
    parser.add_argument("--weight", action="append", default=[], help="Override weight as metric=value")
    parser.add_argument(
        "--score-mode",
        choices=sorted(SCORE_MODES),
        default="normalized",
        help="`normalized` ranks within the candidate set; `baseline-delta` scores weighted improvements over --baseline-method.",
    )
    parser.add_argument(
        "--baseline-method",
        default="masked",
        help="Baseline method used by --score-mode baseline-delta.",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.summary, encoding="utf-8")))
    output = Path(args.output) if args.output else Path(args.summary).with_name("ranked_methods.csv")
    if not rows:
        output.write_text("rank_score\n", encoding="utf-8")
        print("No methods to rank")
        print(output)
        return

    ranked, used_metrics = rank_summary_rows(
        rows,
        parse_weights(args.weight, profile=args.score_profile),
        score_mode=args.score_mode,
        baseline_method=args.baseline_method,
    )

    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(ranked[0].keys()))
        writer.writeheader()
        writer.writerows(ranked)

    print(f"score_mode: {args.score_mode}")
    print(f"score_profile: {args.score_profile}")
    if args.score_mode == "baseline-delta":
        print(f"baseline_method: {args.baseline_method}")
    if used_metrics:
        print("used_metrics:", ",".join(used_metrics))
    for row in ranked:
        print(f"{row['method']}: {row['rank_score']:.4f}")
    print(output)


if __name__ == "__main__":
    main()
