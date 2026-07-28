from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmark.compare_optimize_runs import parse_label_value
from backend.benchmark.rank_methods import SCORE_MODES, SCORE_PROFILES, parse_weights, rank_summary_rows
from backend.benchmark.report_run import render_report
from backend.benchmark.run_completion_benchmark import summarize
from backend.benchmark.select_completion_candidate import (
    decision_markdown,
    evaluate_selection,
    json_safe,
    load_per_sample_rows,
    load_summary_rows,
)


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


def method_row(rows: list[dict], method: str) -> dict:
    return next((row for row in rows if row.get("method") == method), {})


def load_labeled_run(label: str, run_path: Path) -> tuple[list[dict], list[dict]]:
    summary_rows, _ = load_summary_rows(run_path)
    rows = []
    for row in load_per_sample_rows(run_path):
        row = dict(row)
        original_sample_id = row.get("sample_id", "")
        row["source_run_label"] = label
        row["source_run_path"] = str(run_path)
        row["source_sample_id"] = original_sample_id
        row["sample_id"] = f"{label}::{original_sample_id}"
        rows.append(row)
    return rows, summary_rows


def combine_runs(runs: list[tuple[str, Path]]) -> tuple[list[dict], list[dict]]:
    per_sample_rows: list[dict] = []
    attempted_by_method: dict[str, int] = {}
    methods: list[str] = []
    for label, run_path in runs:
        rows, summary_rows = load_labeled_run(label, run_path)
        per_sample_rows.extend(rows)
        for row in summary_rows:
            method = row.get("method", "")
            if not method:
                continue
            if method not in methods:
                methods.append(method)
            try:
                attempted_by_method[method] = attempted_by_method.get(method, 0) + int(float(row.get("attempted_n") or row.get("n") or 0))
            except (TypeError, ValueError):
                attempted_by_method[method] = attempted_by_method.get(method, 0)

    if not per_sample_rows:
        raise ValueError("No per-sample rows found in input runs")

    summary_rows = []
    for method in methods:
        method_rows = [row for row in per_sample_rows if row.get("method") == method]
        method_summary = summarize(
            method_rows,
            methods=[method],
            attempted_n=attempted_by_method.get(method, len(method_rows)),
        )
        if method_summary:
            summary_rows.append(method_summary[0])
    return per_sample_rows, summary_rows


def write_selection(args, output_dir: Path, summary_rows: list[dict], per_sample_rows: list[dict], weights: dict[str, float]) -> tuple[Path, Path]:
    ranked_rows, _ = rank_summary_rows(
        summary_rows,
        weights,
        score_mode="baseline-delta",
        baseline_method=args.baseline_method,
    )
    candidate_method = args.candidate_method or next(
        (row.get("method") for row in ranked_rows if row.get("method") != args.baseline_method),
        None,
    )
    decision = evaluate_selection(
        summary_rows,
        per_sample_rows,
        baseline_method=args.baseline_method,
        candidate_method=candidate_method,
        current_method=args.current_method,
        weights=weights,
        min_success_rate=args.min_success_rate,
        min_paired_n=args.min_paired_n,
        min_win_rate=args.min_win_rate,
        min_ci95_low=args.min_ci95_low,
        min_score_margin=args.min_score_margin,
        min_stl_watertight=args.min_stl_watertight,
        min_stl_positive_volume=args.min_stl_positive_volume,
        max_mesh_surface_chamfer_ratio_vs_current=(
            args.max_mesh_surface_chamfer_ratio_vs_current
        ),
        max_mesh_surface_hausdorff95_ratio_vs_current=(
            args.max_mesh_surface_hausdorff95_ratio_vs_current
        ),
        split_audit={},
        require_split_audit=False,
        bootstrap_samples=max(args.paired_bootstrap_samples, 0),
        bootstrap_seed=args.paired_bootstrap_seed,
    )
    decision["input_dir"] = str(output_dir)
    decision["summary_path"] = str(output_dir / "aggregate_summary.csv")
    decision["score_profile"] = args.score_profile
    output_json = output_dir / "selection_decision.json"
    output_md = output_dir / "selection_decision.md"
    output_json.write_text(json.dumps(json_safe(decision), indent=2, allow_nan=False), encoding="utf-8")
    output_md.write_text(decision_markdown(decision, output_dir), encoding="utf-8")
    return output_json, output_md


def write_metadata(output_dir: Path, args, runs: list[tuple[str, Path]], used_metrics: list[str]) -> Path:
    path = output_dir / "combined_runs.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": [{"label": label, "path": str(path)} for label, path in runs],
        "score_mode": args.score_mode,
        "score_profile": args.score_profile,
        "baseline_method": args.baseline_method,
        "selection_thresholds": {
            "max_mesh_surface_chamfer_ratio_vs_current": (
                args.max_mesh_surface_chamfer_ratio_vs_current
            ),
            "max_mesh_surface_hausdorff95_ratio_vs_current": (
                args.max_mesh_surface_hausdorff95_ratio_vs_current
            ),
        },
        "weights": args.weight or [],
        "used_metrics": used_metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine multiple optimize_completion outputs into one evidence set.")
    parser.add_argument("--run", action="append", required=True, help="Run as label=path. Sample IDs are prefixed with the label.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--score-mode",
        choices=sorted(SCORE_MODES),
        default="baseline-delta",
        help="Ranking score mode for the combined aggregate summary.",
    )
    parser.add_argument(
        "--score-profile",
        choices=sorted(SCORE_PROFILES),
        default="default",
        help="Named metric weight profile. Explicit --weight overrides are applied on top.",
    )
    parser.add_argument("--baseline-method", default="masked")
    parser.add_argument("--weight", action="append", default=[], help="Override objective weight as metric=value.")
    parser.add_argument("--select-candidate", action="store_true", help="Write selection_decision.json/.md for the combined set.")
    parser.add_argument("--candidate-method", default=None, help="Candidate method. Defaults to top non-baseline method under baseline-delta.")
    parser.add_argument("--current-method", default="mirror")
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--min-paired-n", type=int, default=5)
    parser.add_argument("--min-win-rate", type=float, default=0.8)
    parser.add_argument("--min-ci95-low", type=float, default=0.0)
    parser.add_argument("--min-score-margin", type=float, default=0.0)
    parser.add_argument("--min-stl-watertight", type=float, default=1.0)
    parser.add_argument("--min-stl-positive-volume", type=float, default=1.0)
    parser.add_argument("--max-mesh-surface-chamfer-ratio-vs-current", type=float, default=1.1)
    parser.add_argument("--max-mesh-surface-hausdorff95-ratio-vs-current", type=float, default=1.1)
    parser.add_argument("--paired-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--paired-bootstrap-seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [(label, Path(path)) for label, path in (parse_label_value(item) for item in args.run)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = parse_weights(args.weight or [], profile=args.score_profile)

    per_sample_rows, summary_rows = combine_runs(runs)
    ranked_rows, used_metrics = rank_summary_rows(
        summary_rows,
        weights,
        score_mode=args.score_mode,
        baseline_method=args.baseline_method,
    )

    per_sample_path = output_dir / "per_sample_metrics.csv"
    summary_path = output_dir / "aggregate_summary.csv"
    ranked_path = output_dir / "ranked_experiments.csv"
    write_csv(per_sample_path, per_sample_rows)
    write_csv(summary_path, summary_rows)
    write_csv(ranked_path, ranked_rows)
    metadata_path = write_metadata(output_dir, args, runs, used_metrics)

    report_path = output_dir / "combined_report.md"
    render_report(
        run_dir=output_dir,
        output_path=report_path,
        summary_rows=summary_rows,
        per_sample_rows=per_sample_rows,
        failures=[],
        split_audit={},
        ranked_rows=ranked_rows,
        used_metrics=used_metrics,
        top=20,
        examples=0,
        baseline_method=args.baseline_method,
        score_mode=args.score_mode,
        weights=weights,
        paired_bootstrap_samples=max(args.paired_bootstrap_samples, 0),
        paired_bootstrap_seed=args.paired_bootstrap_seed,
        score_profile=args.score_profile,
    )

    if args.select_candidate:
        selection_json, selection_md = write_selection(args, output_dir, summary_rows, per_sample_rows, weights)
        print(selection_json)
        print(selection_md)
    print(summary_path)
    print(ranked_path)
    print(report_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
