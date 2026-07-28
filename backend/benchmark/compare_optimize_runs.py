from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmark.rank_methods import SCORE_PROFILES, parse_weights, rank_summary_rows
from backend.benchmark.report_run import format_number, markdown_table
from backend.benchmark.select_completion_candidate import (
    evaluate_selection,
    json_safe,
    load_per_sample_rows,
    load_split_audit,
    load_summary_rows,
)


KEY_METRICS = [
    "masked_mae_median",
    "object_masked_mae_median",
    "object_depth_mae_median",
    "object_surface_chamfer_l1_median",
    "silhouette_iou_masked_median",
    "stl_is_watertight_median",
    "stl_positive_volume_median",
]


def parse_label_value(item: str, default_label: str | None = None) -> tuple[str, str]:
    if "=" in item:
        label, value = item.split("=", 1)
        label = label.strip()
        value = value.strip()
    else:
        value = item.strip()
        label = default_label or Path(value).name
    if not label or not value:
        raise ValueError(f"Expected label=value, got: {item}")
    return label, value


def label_map(items: list[str]) -> dict[str, str]:
    parsed = {}
    for item in items:
        label, value = parse_label_value(item)
        if label in parsed:
            raise ValueError(f"Duplicate label: {label}")
        parsed[label] = value
    return parsed


def method_row(rows: list[dict], method: str) -> dict:
    return next((row for row in rows if row.get("method") == method), {})


def check_names(checks: list[dict]) -> str:
    return ",".join(check.get("name", "") for check in checks)


def compare_run(
    label: str,
    run_path: Path,
    candidate_method: str,
    *,
    baseline_method: str,
    current_method: str | None,
    weights: dict[str, float],
    min_success_rate: float,
    min_paired_n: int,
    min_win_rate: float,
    min_ci95_low: float,
    min_score_margin: float,
    min_stl_watertight: float,
    min_stl_positive_volume: float,
    max_train_eval_overlap: int,
    allow_missing_split_audit: bool,
    paired_bootstrap_samples: int,
    paired_bootstrap_seed: int,
) -> dict:
    summary_rows, summary_path = load_summary_rows(run_path)
    per_sample_rows = load_per_sample_rows(run_path)
    ranked_rows, used_metrics = rank_summary_rows(
        summary_rows,
        weights,
        score_mode="baseline-delta",
        baseline_method=baseline_method,
    )
    candidate = method_row(ranked_rows, candidate_method)
    if not candidate:
        raise ValueError(f"{label}: candidate method not found: {candidate_method}")

    split_audit = load_split_audit(run_path, candidate_method)
    decision = evaluate_selection(
        summary_rows,
        per_sample_rows,
        baseline_method=baseline_method,
        candidate_method=candidate_method,
        current_method=current_method,
        weights=weights,
        min_success_rate=min_success_rate,
        min_paired_n=min_paired_n,
        min_win_rate=min_win_rate,
        min_ci95_low=min_ci95_low,
        min_score_margin=min_score_margin,
        min_stl_watertight=min_stl_watertight,
        min_stl_positive_volume=min_stl_positive_volume,
        max_train_eval_overlap=max_train_eval_overlap,
        split_audit=split_audit,
        require_split_audit=not allow_missing_split_audit,
        bootstrap_samples=max(paired_bootstrap_samples, 0),
        bootstrap_seed=paired_bootstrap_seed,
    )
    paired = decision.get("paired_objective") or {}
    paired_current = decision.get("paired_objective_vs_current") or {}
    provenance = decision.get("training_provenance") or {}

    row = {
        "run_label": label,
        "run_path": str(run_path),
        "summary_path": str(summary_path),
        "candidate_method": candidate_method,
        "current_method": current_method or "",
        "decision": decision.get("decision", ""),
        "failed_checks": check_names(decision.get("failed_checks", [])),
        "rank_score": decision.get("candidate_rank_score"),
        "current_rank_score": decision.get("current_rank_score"),
        "success_rate": candidate.get("success_rate", ""),
        "paired_n": paired.get("paired_n", ""),
        "paired_win_rate": paired.get("win_rate", ""),
        "paired_mean_improvement": paired.get("mean_improvement", ""),
        "paired_ci95_low": paired.get("ci95_low", ""),
        "paired_ci95_high": paired.get("ci95_high", ""),
        "paired_vs_current_n": paired_current.get("paired_n", ""),
        "paired_vs_current_win_rate": paired_current.get("win_rate", ""),
        "paired_vs_current_mean_improvement": paired_current.get("mean_improvement", ""),
        "paired_vs_current_ci95_low": paired_current.get("ci95_low", ""),
        "paired_vs_current_ci95_high": paired_current.get("ci95_high", ""),
        "lora_weights": candidate.get("lora_weights", "") or provenance.get("lora_weights", ""),
        "lora_scale": candidate.get("lora_scale", ""),
        "train_loss_recipe": provenance.get("train_loss_recipe", ""),
        "train_prompt_family": provenance.get("train_prompt_family", ""),
        "eval_prompt_family": provenance.get("eval_prompt_family", ""),
        "prompt_family_matches_eval": provenance.get("prompt_family_matches_eval", ""),
        "prompt_family_mismatch_detail": provenance.get("prompt_family_mismatch_detail", ""),
        "adapter_sha256": provenance.get("adapter_sha256", ""),
        "used_metrics": ",".join(used_metrics),
    }
    for metric in KEY_METRICS:
        row[metric] = candidate.get(metric, "")
    return row


def add_score_deltas(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    first_score = rows[0].get("rank_score")
    try:
        first_score = float(first_score)
    except (TypeError, ValueError):
        first_score = None
    for row in rows:
        try:
            row["rank_score_delta_vs_first"] = float(row.get("rank_score")) - first_score if first_score is not None else ""
        except (TypeError, ValueError):
            row["rank_score_delta_vs_first"] = ""
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict], args) -> str:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row.get("run_label", ""),
                row.get("candidate_method", ""),
                format_number(row.get("rank_score")),
                format_number(row.get("rank_score_delta_vs_first")),
                row.get("decision", ""),
                row.get("failed_checks", ""),
                format_number(row.get("paired_mean_improvement")),
                format_number(row.get("paired_ci95_low")),
                format_number(row.get("paired_vs_current_mean_improvement")),
                format_number(row.get("paired_vs_current_ci95_low")),
                row.get("train_prompt_family", ""),
                row.get("eval_prompt_family", ""),
                row.get("prompt_family_matches_eval", ""),
                row.get("prompt_family_mismatch_detail", ""),
            ]
        )
    return "\n".join(
        [
            "# Optimize Run Comparison",
            "",
            f"- Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
            f"- Score mode: `baseline-delta`",
            f"- Score profile: `{getattr(args, 'score_profile', 'default')}`",
            f"- Baseline: `{args.baseline_method}`",
            f"- Current: `{args.current_method or '<none>'}`",
            "",
            markdown_table(
                [
                    "Run",
                    "Candidate",
                    "Score",
                    "Delta vs First",
                    "Decision",
                    "Failed Checks",
                    "Mean vs Baseline",
                    "CI Low vs Baseline",
                    "Mean vs Current",
                    "CI Low vs Current",
                    "Train Prompt",
                    "Eval Prompt",
                    "Prompt Match",
                    "Prompt Detail",
                ],
                table_rows,
            ),
            "",
            "_Each run is ranked independently with baseline-delta scoring, so the score values are comparable across runs that share the same baseline and objective weights._",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare candidate evidence across optimize_completion run directories.")
    parser.add_argument("--run", action="append", required=True, help="Run as label=path. Repeat for each run.")
    parser.add_argument("--candidate", action="append", default=[], help="Candidate as label=method. Repeat per run label.")
    parser.add_argument("--candidate-method", default=None, help="Fallback candidate method used for runs without --candidate.")
    parser.add_argument("--baseline-method", default="masked")
    parser.add_argument("--current-method", default="mirror")
    parser.add_argument(
        "--score-profile",
        choices=sorted(SCORE_PROFILES),
        default="default",
        help="Named metric weight profile. Explicit --weight overrides are applied on top.",
    )
    parser.add_argument("--weight", action="append", default=[], help="Override objective weight as metric=value.")
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--min-paired-n", type=int, default=5)
    parser.add_argument("--min-win-rate", type=float, default=0.8)
    parser.add_argument("--min-ci95-low", type=float, default=0.0)
    parser.add_argument("--min-score-margin", type=float, default=0.0)
    parser.add_argument("--min-stl-watertight", type=float, default=1.0)
    parser.add_argument("--min-stl-positive-volume", type=float, default=1.0)
    parser.add_argument("--max-train-eval-overlap", type=int, default=0)
    parser.add_argument("--allow-missing-split-audit", action="store_true")
    parser.add_argument("--paired-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--paired-bootstrap-seed", type=int, default=1234)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [(label, Path(path)) for label, path in (parse_label_value(item) for item in args.run)]
    candidates = label_map(args.candidate)
    weights = parse_weights(args.weight or [], profile=args.score_profile)
    rows = []
    for label, run_path in runs:
        candidate = candidates.get(label) or args.candidate_method
        if not candidate:
            raise ValueError(f"No candidate method supplied for run label `{label}`")
        rows.append(
            compare_run(
                label,
                run_path,
                candidate,
                baseline_method=args.baseline_method,
                current_method=args.current_method,
                weights=weights,
                min_success_rate=args.min_success_rate,
                min_paired_n=args.min_paired_n,
                min_win_rate=args.min_win_rate,
                min_ci95_low=args.min_ci95_low,
                min_score_margin=args.min_score_margin,
                min_stl_watertight=args.min_stl_watertight,
                min_stl_positive_volume=args.min_stl_positive_volume,
                max_train_eval_overlap=args.max_train_eval_overlap,
                allow_missing_split_audit=args.allow_missing_split_audit,
                paired_bootstrap_samples=args.paired_bootstrap_samples,
                paired_bootstrap_seed=args.paired_bootstrap_seed,
            )
        )
    rows = add_score_deltas(rows)
    if args.output_csv:
        write_csv(Path(args.output_csv), rows)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(json_safe(rows), indent=2, allow_nan=False), encoding="utf-8")
    markdown = render_markdown(rows, args)
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
