from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.benchmark.rank_methods import SCORE_PROFILES, normalize, parse_float, parse_weights


KEY_METRICS = [
    ("masked_mae_median", "Masked MAE"),
    ("object_masked_mae_median", "Object MAE"),
    ("masked_psnr_median", "Masked PSNR"),
    ("object_masked_psnr_median", "Object PSNR"),
    ("masked_ssim_median", "Masked SSIM"),
    ("object_masked_ssim_median", "Object SSIM"),
    ("seam_mae_median", "Seam MAE"),
    ("object_depth_mae_median", "Object Depth MAE"),
    ("object_depth_corr_median", "Object Depth Corr"),
    ("object_surface_chamfer_l1_median", "Object Surface Chamfer"),
    ("object_surface_chamfer_rmse_median", "Object Surface Chamfer RMSE"),
    ("object_surface_hausdorff95_median", "Object Surface Hausdorff95"),
    ("silhouette_iou_masked_median", "Silhouette IoU"),
]

METADATA_FIELDS = {
    "sample_id",
    "method",
    "base_method",
    "prompt",
    "steps",
    "guidance",
    "seed",
    "inpaint_max_dimension",
    "model_name",
    "lora_weights",
    "lora_scale",
    "raw_completed_image",
    "completed_image",
    "source",
    "asset_id",
    "asset_path",
    "asset_category",
    "asset_source_split",
    "asset_key",
    "view_index",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


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


def experiment_names(experiment_dir: Path) -> list[str]:
    resolved = read_json(experiment_dir / "resolved_experiments.json")
    names = [experiment.get("name") for experiment in resolved.get("experiments", []) if experiment.get("name")]
    if names:
        return names
    return sorted(path.name for path in experiment_dir.iterdir() if (path / "per_sample_metrics.csv").exists())


def load_experiment_rows(experiment_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for name in experiment_names(experiment_dir):
        metrics_path = experiment_dir / name / "per_sample_metrics.csv"
        for row in read_csv(metrics_path):
            row["base_method"] = row.get("base_method") or row.get("method", "")
            row["method"] = name
            rows.append(row)
    if not rows:
        raise ValueError(f"No per_sample_metrics.csv rows found under {experiment_dir}")
    return rows


def source_split_from_key(value: str) -> str:
    parts = str(value or "").replace("\\", "/").split("/")
    for part in parts:
        split = part.lower()
        if split == "validation":
            return "val"
        if split in {"train", "test", "val"}:
            return split
    return "unknown"


def normalize_metadata(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        row = dict(row)
        if not row.get("asset_source_split"):
            row["asset_source_split"] = source_split_from_key(
                row.get("asset_key") or row.get("asset_path") or ""
            )
        normalized.append(row)
    return normalized


def common_success_filter(rows: list[dict], allow_unequal_coverage: bool = False) -> tuple[list[dict], list[dict]]:
    methods = sorted({row["method"] for row in rows})
    by_sample: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_sample[row["sample_id"]].add(row["method"])
    common_samples = {
        sample_id
        for sample_id, sample_methods in by_sample.items()
        if set(methods) <= sample_methods
    }
    if allow_unequal_coverage:
        filtered = list(rows)
    else:
        filtered = [row for row in rows if row["sample_id"] in common_samples]

    coverage_rows = []
    for method in methods:
        method_sample_ids = {row["sample_id"] for row in rows if row["method"] == method}
        coverage_rows.append(
            {
                "method": method,
                "successful_sample_count": len(method_sample_ids),
                "common_success_sample_count": len(method_sample_ids & common_samples),
                "dropped_for_common_coverage": 0 if allow_unequal_coverage else len(method_sample_ids - common_samples),
            }
        )
    return filtered, coverage_rows


def sample_metric_names(rows: list[dict], weights: dict[str, float]) -> list[str]:
    names = []
    for weighted_metric in weights:
        if not weighted_metric.endswith("_median"):
            continue
        sample_metric = weighted_metric[: -len("_median")]
        if any(sample_metric in row for row in rows):
            names.append(sample_metric)
    return names


def finite_median(values) -> float:
    parsed = np.asarray([parse_float(value) for value in values], dtype=np.float64)
    finite = parsed[np.isfinite(parsed)]
    if len(finite) == 0:
        return math.nan
    return float(np.median(finite))


def summarize_rows(rows: list[dict], weights: dict[str, float], methods: list[str] | None = None) -> list[dict]:
    methods = methods or sorted({row["method"] for row in rows})
    metric_names = sample_metric_names(rows, weights)
    summaries = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary = {"method": method, "n": len(method_rows)}
        for metric in metric_names:
            summary[f"{metric}_median"] = finite_median(row.get(metric) for row in method_rows)
        summaries.append(summary)
    return summaries


def score_summary_rows(summary_rows: list[dict], weights: dict[str, float]) -> list[dict]:
    if not summary_rows:
        return []
    scores = np.zeros(len(summary_rows), dtype=np.float64)
    for metric, weight in weights.items():
        if not any(metric in row for row in summary_rows):
            continue
        values = [parse_float(row.get(metric)) for row in summary_rows]
        normalized = normalize(values, higher_is_better=weight > 0)
        scores += abs(weight) * normalized
    ranked = [{"rank_score": float(score), **row} for row, score in zip(summary_rows, scores)]
    ranked.sort(key=lambda item: item["rank_score"], reverse=True)
    return ranked


def score_sample_rows(rows: list[dict], weights: dict[str, float]) -> list[dict]:
    if not rows:
        return []
    scores = np.zeros(len(rows), dtype=np.float64)
    for weighted_metric, weight in weights.items():
        if not weighted_metric.endswith("_median"):
            continue
        metric = weighted_metric[: -len("_median")]
        if not any(metric in row for row in rows):
            continue
        values = [parse_float(row.get(metric)) for row in rows]
        normalized = normalize(values, higher_is_better=weight > 0)
        scores += abs(weight) * normalized
    ranked = [{"sample_rank_score": float(score), **row} for row, score in zip(rows, scores)]
    ranked.sort(key=lambda item: item["sample_rank_score"], reverse=True)
    return ranked


def format_number(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if not math.isfinite(number):
        return ""
    if number == 0:
        return "0"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) < 0.001:
        return f"{number:.2e}"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_None._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def make_slice_rows(rows: list[dict], group_fields: list[str], weights: dict[str, float]) -> tuple[list[dict], list[dict]]:
    slice_rows: list[dict] = []
    winner_rows: list[dict] = []
    methods = sorted({row["method"] for row in rows})
    for group_field in group_fields:
        values = sorted({row.get(group_field, "") or "<blank>" for row in rows})
        for value in values:
            group_rows = [row for row in rows if (row.get(group_field, "") or "<blank>") == value]
            ranked = score_summary_rows(summarize_rows(group_rows, weights, methods=methods), weights)
            for rank, summary in enumerate(ranked, start=1):
                slice_rows.append(
                    {
                        "slice_field": group_field,
                        "slice_value": value,
                        "rank": rank,
                        **summary,
                    }
                )
            if ranked:
                winner = ranked[0]
                runner_up = ranked[1] if len(ranked) > 1 else {}
                winner_rows.append(
                    {
                        "slice_field": group_field,
                        "slice_value": value,
                        "winner": winner.get("method", ""),
                        "winner_score": winner.get("rank_score", ""),
                        "winner_n": winner.get("n", ""),
                        "runner_up": runner_up.get("method", ""),
                        "runner_up_score": runner_up.get("rank_score", ""),
                    }
                )
    return slice_rows, winner_rows


def make_oracle_rows(rows: list[dict], weights: dict[str, float]) -> tuple[list[dict], list[dict]]:
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)

    selected = []
    for sample_id, sample_rows in sorted(by_sample.items()):
        ranked = score_sample_rows(sample_rows, weights)
        if ranked:
            winner = dict(ranked[0])
            winner["selection_type"] = "oracle_per_sample"
            winner["sample_id"] = sample_id
            selected.append(winner)

    oracle_summary = summarize_rows([{**row, "method": "oracle_per_sample"} for row in selected], weights, methods=["oracle_per_sample"])
    method_summary = summarize_rows(rows, weights)
    ranked_summary = score_summary_rows([*method_summary, *oracle_summary], weights)
    return selected, ranked_summary


def parse_filter(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    if "=" not in value:
        raise ValueError(f"Expected filter as field=value, got {value!r}")
    field, expected = value.split("=", 1)
    return field, expected


def apply_filter(rows: list[dict], item: tuple[str, str] | None) -> list[dict]:
    if item is None:
        return list(rows)
    field, expected = item
    return [row for row in rows if str(row.get(field, "")) == expected]


def make_validation_selector(
    rows: list[dict],
    weights: dict[str, float],
    selector_field: str,
    train_filter: tuple[str, str] | None,
    eval_filter: tuple[str, str] | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    train_rows = apply_filter(rows, train_filter)
    eval_rows = apply_filter(rows, eval_filter)
    methods = sorted({row["method"] for row in rows})
    if not train_rows or not eval_rows:
        return [], [], []

    global_ranked = score_summary_rows(summarize_rows(train_rows, weights, methods=methods), weights)
    global_winner = global_ranked[0]["method"] if global_ranked else ""
    choices = []
    choice_by_value = {}
    for value in sorted({row.get(selector_field, "") or "<blank>" for row in train_rows}):
        group_rows = [row for row in train_rows if (row.get(selector_field, "") or "<blank>") == value]
        ranked = score_summary_rows(summarize_rows(group_rows, weights, methods=methods), weights)
        if not ranked:
            continue
        choice_by_value[value] = ranked[0]["method"]
        choices.append(
            {
                "selector_field": selector_field,
                "selector_value": value,
                "selected_method": ranked[0]["method"],
                "selection_source": "group_train",
                "rank_score": ranked[0]["rank_score"],
                "n": ranked[0]["n"],
            }
        )

    row_by_sample_method = {(row["sample_id"], row["method"]): row for row in eval_rows}
    selected = []
    for sample_id in sorted({row["sample_id"] for row in eval_rows}):
        sample_rows = [row for row in eval_rows if row["sample_id"] == sample_id]
        if not sample_rows:
            continue
        selector_value = sample_rows[0].get(selector_field, "") or "<blank>"
        selected_method = choice_by_value.get(selector_value, global_winner)
        selected_row = row_by_sample_method.get((sample_id, selected_method))
        if selected_row is None:
            selected_row = row_by_sample_method.get((sample_id, global_winner))
            selected_method = global_winner
        if selected_row is None:
            continue
        selected.append(
            {
                **selected_row,
                "method": f"selector_by_{selector_field}",
                "selected_method": selected_method,
                "selector_field": selector_field,
                "selector_value": selector_value,
                "selection_source": "group_train" if selector_value in choice_by_value else "global_train_fallback",
            }
        )

    selector_summary = summarize_rows(selected, weights, methods=[f"selector_by_{selector_field}"])
    eval_method_summary = summarize_rows(eval_rows, weights, methods=methods)
    ranked_eval = score_summary_rows([*eval_method_summary, *selector_summary], weights)
    return choices, selected, ranked_eval


def report_lines(
    experiment_dir: Path,
    group_fields: list[str],
    slice_winners: list[dict],
    oracle_summary: list[dict],
    oracle_rows: list[dict],
    selector_choices: list[dict],
    selector_summary: list[dict],
    coverage_rows: list[dict],
    allow_unequal_coverage: bool,
    train_filter: str | None,
    eval_filter: str | None,
    score_profile: str,
) -> list[str]:
    oracle_counts = Counter(row["method"] for row in oracle_rows)
    top_slice_rows = [
        [
            row.get("slice_field", ""),
            row.get("slice_value", ""),
            row.get("winner", ""),
            format_number(row.get("winner_score")),
            row.get("runner_up", ""),
        ]
        for row in slice_winners
    ]
    oracle_table = [
        [
            row.get("method", ""),
            format_number(row.get("rank_score")),
            format_number(row.get("n")),
            *[format_number(row.get(metric)) for metric, _ in KEY_METRICS if metric in row],
        ]
        for row in oracle_summary[:8]
    ]
    selector_table = [
        [
            row.get("method", ""),
            format_number(row.get("rank_score")),
            format_number(row.get("n")),
            *[format_number(row.get(metric)) for metric, _ in KEY_METRICS if metric in row],
        ]
        for row in selector_summary[:8]
    ]
    choice_table = [
        [
            row.get("selector_value", ""),
            row.get("selected_method", ""),
            format_number(row.get("rank_score")),
            row.get("n", ""),
        ]
        for row in selector_choices
    ]
    coverage_table = [
        [
            row.get("method", ""),
            format_number(row.get("successful_sample_count")),
            format_number(row.get("common_success_sample_count")),
            format_number(row.get("dropped_for_common_coverage")),
        ]
        for row in coverage_rows
    ]

    metric_headers = [label for metric, label in KEY_METRICS if oracle_summary and metric in oracle_summary[0]]
    selector_metric_headers = [label for metric, label in KEY_METRICS if selector_summary and metric in selector_summary[0]]
    return [
        f"# Slice Analysis: {experiment_dir.name}",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- Experiment directory: `{experiment_dir}`",
        f"- Group fields: `{', '.join(group_fields)}`",
        f"- Score profile: `{score_profile}`",
        f"- Coverage mode: `{'unequal coverage allowed' if allow_unequal_coverage else 'common successful sample IDs only'}`",
        "",
        "## Coverage",
        "",
        markdown_table(["Method", "Successful Samples", "Common Samples", "Dropped"], coverage_table),
        "",
        "## Slice Winners",
        "",
        markdown_table(["Field", "Value", "Winner", "Score", "Runner Up"], top_slice_rows),
        "",
        "## Oracle Upper Bound",
        "",
        "This is diagnostic only: it chooses the best method separately for each sample using ground-truth metrics.",
        "",
        f"- Oracle winner counts: `{dict(sorted(oracle_counts.items()))}`",
        "",
        markdown_table(["Method", "Score", "n", *metric_headers], oracle_table),
        "",
        "## Validation-Style Selector",
        "",
        (
            f"Selector train filter `{train_filter or '<none>'}` and eval filter `{eval_filter or '<none>'}`. "
            "This is a small held-out diagnostic, not a production classifier."
        ),
        "",
        markdown_table(["Selector Value", "Selected Method", "Train Score", "Train n"], choice_table),
        "",
        markdown_table(["Method", "Score", "n", *selector_metric_headers], selector_table),
        "",
        "## Cautions",
        "",
        "- Rank scores are normalized within each table and should not be compared across separate reports.",
        "- Slice and oracle winners use benchmark ground truth; only the validation-style selector avoids using eval labels for method choice.",
        "- Very small slices can flip winners because medians are based on few samples.",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-slice winners and selector diagnostics for an optimize_completion run.")
    parser.add_argument("experiment_dir", help="Directory produced by backend.benchmark.optimize_completion.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <experiment_dir>/slice_analysis.")
    parser.add_argument("--group-field", action="append", default=None, help="Metadata field to summarize, e.g. asset_category.")
    parser.add_argument("--selector-field", default="asset_category", help="Field used for validation-style method selection.")
    parser.add_argument("--selector-train-filter", default=None, help="Optional field=value filter for selector choice rows.")
    parser.add_argument("--selector-eval-filter", default=None, help="Optional field=value filter for selector evaluation rows.")
    parser.add_argument(
        "--allow-unequal-coverage",
        action="store_true",
        help="Analyze all successful rows instead of intersecting common sample IDs across methods.",
    )
    parser.add_argument(
        "--score-profile",
        choices=sorted(SCORE_PROFILES),
        default="default",
        help="Named metric weight profile. Explicit --weight overrides are applied on top.",
    )
    parser.add_argument("--weight", action="append", default=[], help="Override rank weight as metric=value, same syntax as rank_methods.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "slice_analysis"
    group_fields = args.group_field or ["asset_category", "asset_source_split"]
    weights = parse_weights(args.weight, profile=args.score_profile)
    raw_rows = normalize_metadata(load_experiment_rows(experiment_dir))
    rows, coverage_rows = common_success_filter(raw_rows, allow_unequal_coverage=args.allow_unequal_coverage)

    slice_rows, slice_winners = make_slice_rows(rows, group_fields, weights)
    oracle_rows, oracle_summary = make_oracle_rows(rows, weights)
    train_filter = parse_filter(args.selector_train_filter)
    eval_filter = parse_filter(args.selector_eval_filter)
    selector_choices, selector_rows, selector_summary = make_validation_selector(
        rows,
        weights,
        selector_field=args.selector_field,
        train_filter=train_filter,
        eval_filter=eval_filter,
    )

    write_csv(output_dir / "coverage.csv", coverage_rows)
    write_csv(output_dir / "slice_summary.csv", slice_rows)
    write_csv(output_dir / "slice_winners.csv", slice_winners)
    write_csv(output_dir / "oracle_selection.csv", oracle_rows)
    write_csv(output_dir / "oracle_ranked_summary.csv", oracle_summary)
    write_csv(output_dir / "selector_choices.csv", selector_choices)
    write_csv(output_dir / "selector_selected_rows.csv", selector_rows)
    write_csv(output_dir / "selector_eval_summary.csv", selector_summary)

    report = report_lines(
        experiment_dir,
        group_fields,
        slice_winners,
        oracle_summary,
        oracle_rows,
        selector_choices,
        selector_summary,
        coverage_rows,
        args.allow_unequal_coverage,
        args.selector_train_filter,
        args.selector_eval_filter,
        args.score_profile,
    )
    report_path = output_dir / "slice_analysis_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
