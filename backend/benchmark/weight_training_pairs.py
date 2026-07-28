from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.benchmark.rank_methods import SCORE_PROFILE_DESCRIPTIONS, SCORE_PROFILES, parse_float, parse_weights
from backend.benchmark.report_run import sample_field_for_summary_metric
from backend.benchmark.select_completion_candidate import load_per_sample_rows
from backend.benchmark.train_inpainting_lora import file_sha256


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metadata rows found in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def metadata_sample_id(row: dict) -> str:
    return str(row.get("id") or row.get("sample_id") or "")


def benchmark_sample_id(row: dict) -> str:
    return str(row.get("source_sample_id") or row.get("sample_id") or "")


def finite_metric(row: dict, field: str) -> float:
    value = parse_float(row.get(field))
    return value if math.isfinite(value) else math.nan


def normalize_badness(values: dict[str, float]) -> dict[str, float]:
    finite = [value for value in values.values() if math.isfinite(value)]
    if not finite:
        return {}
    low = min(finite)
    high = max(finite)
    if high == low:
        return {key: 0.0 for key, value in values.items() if math.isfinite(value)}
    return {key: (value - low) / (high - low) for key, value in values.items() if math.isfinite(value)}


def metric_badness(row: dict, field: str, weight: float) -> float:
    value = finite_metric(row, field)
    if not math.isfinite(value):
        return math.nan
    return value if weight < 0 else -value


def benchmark_rows_by_sample(rows: list[dict], method: str) -> dict[str, dict]:
    matched = {}
    for row in rows:
        if row.get("method") != method:
            continue
        sample_id = benchmark_sample_id(row)
        if sample_id:
            matched[sample_id] = row
    return matched


def weighted_score_by_sample(
    benchmark_by_sample: dict[str, dict],
    weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, int], list[str]]:
    raw_badness_by_metric: dict[str, dict[str, float]] = {}
    used_metrics = []
    for summary_metric, weight in weights.items():
        field = sample_field_for_summary_metric(summary_metric)
        values = {
            sample_id: metric_badness(row, field, weight)
            for sample_id, row in benchmark_by_sample.items()
        }
        normalized = normalize_badness(values)
        if not normalized:
            continue
        raw_badness_by_metric[summary_metric] = normalized
        used_metrics.append(summary_metric)

    scores: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    for sample_id in benchmark_by_sample:
        weighted_sum = 0.0
        weight_sum = 0.0
        metric_count = 0
        for metric, normalized in raw_badness_by_metric.items():
            if sample_id not in normalized:
                continue
            metric_weight = abs(weights[metric])
            weighted_sum += metric_weight * normalized[sample_id]
            weight_sum += metric_weight
            metric_count += 1
        if weight_sum:
            scores[sample_id] = weighted_sum / weight_sum
            metric_counts[sample_id] = metric_count
    return scores, metric_counts, used_metrics


def weight_metadata_rows(
    metadata_rows: list[dict],
    benchmark_rows: list[dict],
    *,
    method: str,
    weights: dict[str, float],
    score_profile: str,
    base_weight: float,
    scale: float,
    max_weight: float,
) -> tuple[list[dict], dict]:
    benchmark_by_sample = benchmark_rows_by_sample(benchmark_rows, method)
    scores, metric_counts, used_metrics = weighted_score_by_sample(benchmark_by_sample, weights)

    weighted_rows = []
    matched_rows = 0
    unmatched_rows = 0
    weights_out = []
    detail_rows = []
    for row in metadata_rows:
        row = dict(row)
        sample_id = metadata_sample_id(row)
        score = scores.get(sample_id)
        if score is None:
            sample_weight = base_weight
            metric_count = 0
            unmatched_rows += 1
        else:
            sample_weight = base_weight + scale * score
            if math.isfinite(max_weight) and max_weight > 0:
                sample_weight = min(sample_weight, max_weight)
            metric_count = metric_counts.get(sample_id, 0)
            matched_rows += 1
        row["sample_weight"] = float(sample_weight)
        row["sample_weight_score"] = "" if score is None else float(score)
        row["sample_weight_method"] = method
        row["sample_weight_score_profile"] = score_profile
        row["sample_weight_metric_count"] = metric_count
        weighted_rows.append(row)
        weights_out.append(float(sample_weight))
        detail_rows.append(
            {
                "id": sample_id,
                "sample_weight": float(sample_weight),
                "sample_weight_score": "" if score is None else float(score),
                "sample_weight_metric_count": metric_count,
                "matched_benchmark": bool(score is not None),
            }
        )

    weights_array = np.asarray(weights_out, dtype=np.float64)
    top_rows = sorted(
        detail_rows,
        key=lambda item: float(item["sample_weight"]),
        reverse=True,
    )[:10]
    report = {
        "method": method,
        "score_profile": score_profile,
        "score_profile_description": SCORE_PROFILE_DESCRIPTIONS.get(score_profile, ""),
        "used_metrics": used_metrics,
        "metadata_rows": len(metadata_rows),
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "base_weight": base_weight,
        "scale": scale,
        "max_weight": max_weight,
        "sample_weight_min": float(np.min(weights_array)) if len(weights_array) else base_weight,
        "sample_weight_mean": float(np.mean(weights_array)) if len(weights_array) else base_weight,
        "sample_weight_max": float(np.max(weights_array)) if len(weights_array) else base_weight,
        "top_weighted_rows": top_rows,
        "detail_rows": detail_rows,
    }
    return weighted_rows, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add per-sample LoRA training weights to metadata.jsonl from benchmark metric difficulty."
    )
    parser.add_argument("--metadata", required=True, help="metadata.jsonl from export_training_pairs.")
    parser.add_argument("--benchmark-dir", required=True, help="Run, optimize, or combined directory with per_sample_metrics.csv rows.")
    parser.add_argument("--method", default="mirror", help="Benchmark method whose metric difficulty drives weights.")
    parser.add_argument(
        "--score-profile",
        choices=sorted(SCORE_PROFILES),
        default="object-surface",
        help="Metric profile used to define hard samples.",
    )
    parser.add_argument("--weight", action="append", default=[], help="Override profile metric weight as metric=value.")
    parser.add_argument("--base-weight", type=float, default=1.0, help="Weight assigned to easiest or unmatched rows.")
    parser.add_argument("--scale", type=float, default=2.0, help="Extra weight applied to normalized hard-sample score.")
    parser.add_argument("--max-weight", type=float, default=5.0, help="Cap per-sample weight; use 0 for no cap.")
    parser.add_argument("--output", default=None, help="Weighted metadata JSONL path.")
    parser.add_argument("--report", default=None, help="Weighting report JSON path.")
    parser.add_argument("--details-csv", default=None, help="Optional per-row weight details CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_weight < 0 or args.scale < 0 or args.max_weight < 0:
        raise ValueError("--base-weight, --scale, and --max-weight must be non-negative")

    metadata_path = Path(args.metadata)
    benchmark_dir = Path(args.benchmark_dir)
    output_path = Path(args.output) if args.output else metadata_path.with_name(f"{metadata_path.stem}_weighted_{args.method}_{args.score_profile}.jsonl")
    report_path = Path(args.report) if args.report else output_path.with_suffix(".report.json")
    details_path = Path(args.details_csv) if args.details_csv else output_path.with_suffix(".details.csv")

    metadata_rows = read_jsonl(metadata_path)
    benchmark_rows = load_per_sample_rows(benchmark_dir)
    if not benchmark_rows:
        raise FileNotFoundError(f"No per_sample_metrics.csv rows found in {benchmark_dir}")
    weights = parse_weights(args.weight or [], profile=args.score_profile)
    weighted_rows, report = weight_metadata_rows(
        metadata_rows,
        benchmark_rows,
        method=args.method,
        weights=weights,
        score_profile=args.score_profile,
        base_weight=args.base_weight,
        scale=args.scale,
        max_weight=args.max_weight,
    )

    write_jsonl(output_path, weighted_rows)
    report.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metadata": str(metadata_path),
            "metadata_sha256": file_sha256(metadata_path),
            "benchmark_dir": str(benchmark_dir),
            "output": str(output_path),
            "output_sha256": file_sha256(output_path),
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({key: value for key, value in report.items() if key != "detail_rows"}, indent=2), encoding="utf-8")
    write_csv(details_path, report["detail_rows"])
    print(output_path)
    print(report_path)
    print(details_path)


if __name__ == "__main__":
    main()
