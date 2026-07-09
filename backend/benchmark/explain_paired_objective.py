from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from backend.benchmark.rank_methods import SCORE_PROFILES, parse_weights
from backend.benchmark.report_run import format_number, markdown_table, parse_score_value, sample_field_for_summary_metric
from backend.benchmark.select_completion_candidate import load_per_sample_rows


PAIR_TIE_EPSILON = 1e-12


KEY_FIELDS = (
    "mesh_surface_chamfer_l1",
    "mesh_surface_chamfer_rmse",
    "mesh_surface_hausdorff95",
    "stl_bbox_aspect_ratio",
    "stl_faces_per_bbox_volume_log1p",
    "stl_faces",
    "stl_is_watertight",
    "stl_is_volume",
    "stl_is_manifold",
)


def finite_number(value) -> float:
    number = parse_score_value(value)
    return number if math.isfinite(number) else math.nan


def index_by_sample_method(rows: list[dict]) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        method = str(row.get("method") or "")
        if sample_id and method:
            indexed[(sample_id, method)] = row
    return indexed


def pair_contributions(
    *,
    sample_id: str,
    candidate_method: str,
    comparison_method: str,
    candidate_row: dict,
    comparison_row: dict,
    weights: dict[str, float],
) -> list[dict]:
    rows: list[dict] = []
    for metric, weight in weights.items():
        sample_field = sample_field_for_summary_metric(metric)
        candidate_value = finite_number(candidate_row.get(sample_field))
        comparison_value = finite_number(comparison_row.get(sample_field))
        if not (math.isfinite(candidate_value) and math.isfinite(comparison_value)):
            continue
        raw_delta = candidate_value - comparison_value
        improvement = raw_delta if weight > 0 else -raw_delta
        contribution = abs(weight) * improvement
        rows.append(
            {
                "sample_id": sample_id,
                "candidate_method": candidate_method,
                "comparison_method": comparison_method,
                "metric": metric,
                "sample_field": sample_field,
                "weight": weight,
                "candidate_value": candidate_value,
                "comparison_value": comparison_value,
                "raw_delta": raw_delta,
                "improvement": improvement,
                "contribution": contribution,
            }
        )
    return rows


def summarize_pair(contributions: list[dict]) -> dict:
    score = sum(float(row["contribution"]) for row in contributions)
    positive = sum(float(row["contribution"]) for row in contributions if float(row["contribution"]) >= 0)
    negative = sum(float(row["contribution"]) for row in contributions if float(row["contribution"]) < 0)
    return {
        "score": score,
        "positive_contribution": positive,
        "negative_contribution": negative,
        "metric_count": len(contributions),
        "win": score > PAIR_TIE_EPSILON,
        "tie": abs(score) <= PAIR_TIE_EPSILON,
    }


def _top_metric_text(contributions: list[dict], *, positive: bool, limit: int) -> str:
    if positive:
        rows = sorted(
            (row for row in contributions if float(row["contribution"]) > 0),
            key=lambda row: float(row["contribution"]),
            reverse=True,
        )
    else:
        rows = sorted(
            (row for row in contributions if float(row["contribution"]) < 0),
            key=lambda row: float(row["contribution"]),
        )
    return "; ".join(f"{row['sample_field']}={format_number(row['contribution'])}" for row in rows[:limit])


def _key_values(row: dict, prefix: str) -> dict:
    return {f"{prefix}_{field}": row.get(field, "") for field in KEY_FIELDS if field in row}


def paired_sample_rows(
    per_sample_rows: list[dict],
    *,
    candidate_method: str,
    baseline_method: str,
    current_method: str | None,
    weights: dict[str, float],
    top_n: int = 3,
) -> tuple[list[dict], list[dict]]:
    indexed = index_by_sample_method(per_sample_rows)
    candidate_samples = {sample for sample, method in indexed if method == candidate_method}
    baseline_samples = {sample for sample, method in indexed if method == baseline_method}
    current_samples = {sample for sample, method in indexed if current_method and method == current_method}
    paired_samples = sorted(candidate_samples & baseline_samples)
    if current_method:
        paired_samples = [sample for sample in paired_samples if sample in current_samples]

    sample_rows: list[dict] = []
    contribution_rows: list[dict] = []
    for sample_id in paired_samples:
        candidate_row = indexed[(sample_id, candidate_method)]
        baseline_row = indexed[(sample_id, baseline_method)]
        baseline_contributions = pair_contributions(
            sample_id=sample_id,
            candidate_method=candidate_method,
            comparison_method=baseline_method,
            candidate_row=candidate_row,
            comparison_row=baseline_row,
            weights=weights,
        )
        baseline_summary = summarize_pair(baseline_contributions)
        contribution_rows.extend(baseline_contributions)

        row = {
            "sample_id": sample_id,
            "candidate_method": candidate_method,
            "baseline_method": baseline_method,
            "score_vs_baseline": baseline_summary["score"],
            "win_vs_baseline": baseline_summary["win"],
            "tie_vs_baseline": baseline_summary["tie"],
            "metric_count_vs_baseline": baseline_summary["metric_count"],
            "top_helps_vs_baseline": _top_metric_text(baseline_contributions, positive=True, limit=top_n),
            "top_hurts_vs_baseline": _top_metric_text(baseline_contributions, positive=False, limit=top_n),
            **_key_values(candidate_row, "candidate"),
            **_key_values(baseline_row, "baseline"),
        }

        if current_method:
            current_row = indexed[(sample_id, current_method)]
            current_contributions = pair_contributions(
                sample_id=sample_id,
                candidate_method=candidate_method,
                comparison_method=current_method,
                candidate_row=candidate_row,
                comparison_row=current_row,
                weights=weights,
            )
            current_summary = summarize_pair(current_contributions)
            contribution_rows.extend(current_contributions)
            row.update(
                {
                    "current_method": current_method,
                    "score_vs_current": current_summary["score"],
                    "win_vs_current": current_summary["win"],
                    "tie_vs_current": current_summary["tie"],
                    "metric_count_vs_current": current_summary["metric_count"],
                    "top_helps_vs_current": _top_metric_text(current_contributions, positive=True, limit=top_n),
                    "top_hurts_vs_current": _top_metric_text(current_contributions, positive=False, limit=top_n),
                    **_key_values(current_row, "current"),
                }
            )
        sample_rows.append(row)

    sort_key = "score_vs_current" if current_method else "score_vs_baseline"
    sample_rows.sort(key=lambda row: float(row.get(sort_key, math.inf)))
    contribution_rows.sort(key=lambda row: (row["sample_id"], row["comparison_method"], float(row["contribution"])))
    return sample_rows, contribution_rows


def aggregate_pair_summary(sample_rows: list[dict], *, current_method: str | None) -> dict:
    def values(key: str) -> list[float]:
        return [
            float(row[key])
            for row in sample_rows
            if key in row and math.isfinite(float(row[key]))
        ]

    baseline_scores = values("score_vs_baseline")
    summary = {
        "paired_n": len(sample_rows),
        "wins_vs_baseline": sum(1 for row in sample_rows if row.get("win_vs_baseline")),
        "ties_vs_baseline": sum(1 for row in sample_rows if row.get("tie_vs_baseline")),
        "mean_score_vs_baseline": sum(baseline_scores) / len(baseline_scores) if baseline_scores else math.nan,
        "median_score_vs_baseline": sorted(baseline_scores)[len(baseline_scores) // 2] if baseline_scores else math.nan,
    }
    if current_method:
        current_scores = values("score_vs_current")
        summary.update(
            {
                "wins_vs_current": sum(1 for row in sample_rows if row.get("win_vs_current")),
                "ties_vs_current": sum(1 for row in sample_rows if row.get("tie_vs_current")),
                "mean_score_vs_current": sum(current_scores) / len(current_scores) if current_scores else math.nan,
                "median_score_vs_current": sorted(current_scores)[len(current_scores) // 2] if current_scores else math.nan,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def markdown_report(
    *,
    input_dir: Path,
    candidate_method: str,
    baseline_method: str,
    current_method: str | None,
    score_profile: str,
    summary: dict,
    sample_rows: list[dict],
    top_n: int,
) -> str:
    rows = []
    for row in sample_rows:
        rows.append(
            [
                row["sample_id"],
                format_number(row.get("score_vs_current")) if current_method else "",
                format_number(row.get("score_vs_baseline")),
                "yes" if row.get("win_vs_current") else "no" if current_method else "",
                row.get("top_hurts_vs_current") if current_method else row.get("top_hurts_vs_baseline"),
                row.get("top_helps_vs_current") if current_method else row.get("top_helps_vs_baseline"),
                format_number(row.get("candidate_mesh_surface_chamfer_l1")),
                format_number(row.get("current_mesh_surface_chamfer_l1")) if current_method else "",
                format_number(row.get("candidate_stl_faces_per_bbox_volume_log1p")),
                format_number(row.get("current_stl_faces_per_bbox_volume_log1p")) if current_method else "",
            ]
        )
    lines = [
        "# Paired Objective Explanation",
        "",
        f"- Input: `{input_dir}`",
        f"- Candidate: `{candidate_method}`",
        f"- Baseline: `{baseline_method}`",
        f"- Current: `{current_method or ''}`",
        f"- Score profile: `{score_profile}`",
        f"- Paired samples: `{summary.get('paired_n', 0)}`",
        f"- Wins vs baseline: `{summary.get('wins_vs_baseline', 0)}/{summary.get('paired_n', 0)}`",
    ]
    if current_method:
        lines.append(f"- Wins vs current: `{summary.get('wins_vs_current', 0)}/{summary.get('paired_n', 0)}`")
    lines.extend(
        [
            "",
            f"Lowest rows are the most important losses. Top metric lists show the {top_n} largest weighted contributions.",
            "",
            markdown_table(
                [
                    "Sample",
                    "Score vs current",
                    "Score vs baseline",
                    "Win vs current",
                    "Top hurts",
                    "Top helps",
                    "Candidate Chamfer",
                    "Current Chamfer",
                    "Candidate Face Density",
                    "Current Face Density",
                ],
                rows,
            ),
            "",
        ]
    )
    return "\n".join(lines)


def explain(
    input_dir: Path,
    *,
    candidate_method: str,
    baseline_method: str = "masked",
    current_method: str | None = None,
    score_profile: str = "default",
    weight_overrides: list[str] | None = None,
    top_n: int = 3,
) -> tuple[dict, list[dict], list[dict]]:
    per_sample_rows = load_per_sample_rows(input_dir)
    if not per_sample_rows:
        raise FileNotFoundError(f"No per_sample_metrics.csv rows found in {input_dir}")
    weights = parse_weights(weight_overrides or [], profile=score_profile)
    sample_rows, contribution_rows = paired_sample_rows(
        per_sample_rows,
        candidate_method=candidate_method,
        baseline_method=baseline_method,
        current_method=current_method,
        weights=weights,
        top_n=top_n,
    )
    if not sample_rows:
        raise ValueError(
            f"No paired rows found for candidate={candidate_method}, baseline={baseline_method}, current={current_method or ''}"
        )
    summary = aggregate_pair_summary(sample_rows, current_method=current_method)
    summary.update(
        {
            "candidate_method": candidate_method,
            "baseline_method": baseline_method,
            "current_method": current_method or "",
            "score_profile": score_profile,
            "used_metrics": list(weights),
        }
    )
    return summary, sample_rows, contribution_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain paired per-sample objective wins and losses.")
    parser.add_argument("input_dir", help="Run or optimize_completion directory.")
    parser.add_argument("--candidate-method", required=True)
    parser.add_argument("--baseline-method", default="masked")
    parser.add_argument("--current-method", default=None)
    parser.add_argument("--score-profile", choices=sorted(SCORE_PROFILES), default="default")
    parser.add_argument("--weight", action="append", default=[])
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "paired_objective_explanation"
    summary, sample_rows, contribution_rows = explain(
        input_dir,
        candidate_method=args.candidate_method,
        baseline_method=args.baseline_method,
        current_method=args.current_method,
        score_profile=args.score_profile,
        weight_overrides=args.weight,
        top_n=args.top_n,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_sample_scores.csv", sample_rows)
    write_csv(output_dir / "paired_metric_contributions.csv", contribution_rows)
    (output_dir / "paired_objective_explanation.json").write_text(
        json.dumps(
            json_safe(
                {
                    "summary": summary,
                    "samples": sample_rows,
                }
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "paired_objective_explanation.md").write_text(
        markdown_report(
            input_dir=input_dir,
            candidate_method=args.candidate_method,
            baseline_method=args.baseline_method,
            current_method=args.current_method,
            score_profile=args.score_profile,
            summary=summary,
            sample_rows=sample_rows,
            top_n=args.top_n,
        ),
        encoding="utf-8",
    )

    print(f"paired_n: {summary['paired_n']}")
    print(f"wins_vs_baseline: {summary['wins_vs_baseline']}/{summary['paired_n']}")
    if args.current_method:
        print(f"wins_vs_current: {summary['wins_vs_current']}/{summary['paired_n']}")
    print(output_dir)


if __name__ == "__main__":
    main()
