from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from backend.benchmark.rank_methods import SCORE_MODES, SCORE_PROFILES, normalize, parse_float, parse_weights


def method_label(row: dict) -> str:
    return str(row.get("method") or row.get("experiment") or "")


def finite(value) -> bool:
    return math.isfinite(parse_float(value))


def baseline_delta_contributions(rows: list[dict], weights: dict[str, float], baseline_method: str) -> tuple[list[dict], list[str]]:
    baseline = next((row for row in rows if method_label(row) == baseline_method), None)
    if baseline is None:
        raise ValueError(f"Baseline method `{baseline_method}` was not found in summary rows")

    field_names = {field for row in rows for field in row}
    contributions: list[dict] = []
    used_metrics: list[str] = []

    for metric, weight in weights.items():
        if metric not in field_names:
            continue
        baseline_value = parse_float(baseline.get(metric))
        if not math.isfinite(baseline_value):
            continue

        metric_used = False
        higher_is_better = weight > 0
        for row in rows:
            method_value = parse_float(row.get(metric))
            if not math.isfinite(method_value):
                continue
            raw_delta = method_value - baseline_value
            improvement = raw_delta if higher_is_better else -raw_delta
            contribution = abs(weight) * improvement
            contributions.append(
                {
                    "method": method_label(row),
                    "metric": metric,
                    "weight": weight,
                    "baseline_method": baseline_method,
                    "baseline_value": baseline_value,
                    "method_value": method_value,
                    "raw_delta": raw_delta,
                    "improvement": improvement,
                    "contribution": contribution,
                    "score_mode": "baseline-delta",
                }
            )
            metric_used = True
        if metric_used:
            used_metrics.append(metric)
    return contributions, used_metrics


def normalized_contributions(rows: list[dict], weights: dict[str, float]) -> tuple[list[dict], list[str]]:
    field_names = {field for row in rows for field in row}
    contributions: list[dict] = []
    used_metrics: list[str] = []

    for metric, weight in weights.items():
        if metric not in field_names:
            continue
        values = [parse_float(row.get(metric)) for row in rows]
        if not any(math.isfinite(value) for value in values):
            continue
        normalized_values = normalize(values, higher_is_better=weight > 0)
        for row, raw_value, normalized_value in zip(rows, values, normalized_values):
            if not math.isfinite(raw_value):
                continue
            contribution = abs(weight) * float(normalized_value)
            contributions.append(
                {
                    "method": method_label(row),
                    "metric": metric,
                    "weight": weight,
                    "baseline_method": "",
                    "baseline_value": "",
                    "method_value": raw_value,
                    "raw_delta": "",
                    "improvement": float(normalized_value),
                    "contribution": contribution,
                    "score_mode": "normalized",
                }
            )
        used_metrics.append(metric)
    return contributions, used_metrics


def contribution_rows(
    rows: list[dict],
    weights: dict[str, float],
    score_mode: str = "baseline-delta",
    baseline_method: str = "masked",
) -> tuple[list[dict], list[str]]:
    if score_mode not in SCORE_MODES:
        raise ValueError(f"Unknown score mode `{score_mode}`. Expected one of: {', '.join(sorted(SCORE_MODES))}")
    if score_mode == "baseline-delta":
        return baseline_delta_contributions(rows, weights, baseline_method)
    return normalized_contributions(rows, weights)


def summarize_contributions(contributions: list[dict]) -> list[dict]:
    totals: dict[str, float] = {}
    positives: dict[str, float] = {}
    negatives: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in contributions:
        method = row["method"]
        contribution = float(row["contribution"])
        totals[method] = totals.get(method, 0.0) + contribution
        counts[method] = counts.get(method, 0) + 1
        if contribution >= 0:
            positives[method] = positives.get(method, 0.0) + contribution
        else:
            negatives[method] = negatives.get(method, 0.0) + contribution
    summary = [
        {
            "method": method,
            "rank_score": totals[method],
            "positive_contribution": positives.get(method, 0.0),
            "negative_contribution": negatives.get(method, 0.0),
            "metric_count": counts.get(method, 0),
        }
        for method in totals
    ]
    summary.sort(key=lambda row: row["rank_score"], reverse=True)
    return summary


def top_contributions(contributions: list[dict], method: str, limit: int) -> tuple[list[dict], list[dict]]:
    rows = [row for row in contributions if row["method"] == method]
    positive = sorted((row for row in rows if float(row["contribution"]) >= 0), key=lambda row: row["contribution"], reverse=True)
    negative = sorted((row for row in rows if float(row["contribution"]) < 0), key=lambda row: row["contribution"])
    return positive[:limit], negative[:limit]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_number(value) -> str:
    parsed = parse_float(value)
    if math.isfinite(parsed):
        return f"{parsed:.4f}"
    return ""


def markdown_report(
    *,
    summary: list[dict],
    contributions: list[dict],
    used_metrics: list[str],
    score_profile: str,
    score_mode: str,
    baseline_method: str,
    focus_methods: list[str],
    top_n: int,
) -> str:
    lines = [
        "# Rank Score Explanation",
        "",
        f"- Score profile: `{score_profile}`",
        f"- Score mode: `{score_mode}`",
    ]
    if score_mode == "baseline-delta":
        lines.append(f"- Baseline method: `{baseline_method}`")
    lines.append(f"- Used metrics: `{', '.join(used_metrics)}`")
    lines.extend(["", "## Method Totals", "", "| method | rank score | positive | negative | metrics |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in summary:
        lines.append(
            "| {method} | {score} | {positive} | {negative} | {count} |".format(
                method=row["method"],
                score=format_number(row["rank_score"]),
                positive=format_number(row["positive_contribution"]),
                negative=format_number(row["negative_contribution"]),
                count=row["metric_count"],
            )
        )

    focus = focus_methods or [row["method"] for row in summary[:3]]
    for method in focus:
        positive, negative = top_contributions(contributions, method, top_n)
        if not positive and not negative:
            continue
        lines.extend(["", f"## `{method}`", "", "| direction | metric | contribution | method value | baseline value | raw delta |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for row in positive:
            lines.append(
                f"| helps | `{row['metric']}` | {format_number(row['contribution'])} | {format_number(row['method_value'])} | {format_number(row['baseline_value'])} | {format_number(row['raw_delta'])} |"
            )
        for row in negative:
            lines.append(
                f"| hurts | `{row['metric']}` | {format_number(row['contribution'])} | {format_number(row['method_value'])} | {format_number(row['baseline_value'])} | {format_number(row['raw_delta'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def explain(
    summary_path: Path,
    *,
    score_profile: str = "default",
    weight_overrides: list[str] | None = None,
    score_mode: str = "baseline-delta",
    baseline_method: str = "masked",
) -> tuple[list[dict], list[dict], list[str]]:
    with summary_path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    weights = parse_weights(weight_overrides or [], profile=score_profile)
    contributions, used_metrics = contribution_rows(rows, weights, score_mode=score_mode, baseline_method=baseline_method)
    return summarize_contributions(contributions), contributions, used_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain weighted rank score contributions by method and metric.")
    parser.add_argument("--summary", required=True, help="aggregate_summary.csv or summary_metrics.csv to explain")
    parser.add_argument("--output-dir", default=None, help="Directory for contribution CSV, summary CSV, JSON, and Markdown outputs")
    parser.add_argument("--score-profile", choices=sorted(SCORE_PROFILES), default="default")
    parser.add_argument("--weight", action="append", default=[], help="Override weight as metric=value")
    parser.add_argument("--score-mode", choices=sorted(SCORE_MODES), default="baseline-delta")
    parser.add_argument("--baseline-method", default="masked")
    parser.add_argument("--focus-method", action="append", default=[], help="Method to include in Markdown top contribution tables")
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.with_name("rank_score_explanation")
    summary, contributions, used_metrics = explain(
        summary_path,
        score_profile=args.score_profile,
        weight_overrides=args.weight,
        score_mode=args.score_mode,
        baseline_method=args.baseline_method,
    )

    write_csv(output_dir / "rank_score_summary.csv", summary)
    write_csv(output_dir / "rank_score_contributions.csv", contributions)
    payload = {
        "summary": summary,
        "used_metrics": used_metrics,
        "score_profile": args.score_profile,
        "score_mode": args.score_mode,
        "baseline_method": args.baseline_method if args.score_mode == "baseline-delta" else "",
    }
    (output_dir / "rank_score_explanation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output_dir / "rank_score_explanation.md").write_text(
        markdown_report(
            summary=summary,
            contributions=contributions,
            used_metrics=used_metrics,
            score_profile=args.score_profile,
            score_mode=args.score_mode,
            baseline_method=args.baseline_method,
            focus_methods=args.focus_method,
            top_n=args.top_n,
        ),
        encoding="utf-8",
    )

    print(f"score_mode: {args.score_mode}")
    print(f"score_profile: {args.score_profile}")
    if args.score_mode == "baseline-delta":
        print(f"baseline_method: {args.baseline_method}")
    print("used_metrics:", ",".join(used_metrics))
    for row in summary:
        print(f"{row['method']}: {row['rank_score']:.4f}")
    print(output_dir)


if __name__ == "__main__":
    main()
