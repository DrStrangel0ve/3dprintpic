import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.benchmark.make_artifact_contact_sheet import make_contact_sheet, parse_csv_arg
from backend.benchmark.rank_methods import (
    SCORE_MODES,
    SCORE_PROFILE_DESCRIPTIONS,
    SCORE_PROFILES,
    parse_float,
    parse_weights,
    rank_summary_rows,
    with_derived_metrics,
)


KEY_METRICS = [
    ("masked_mae_median", "Masked MAE"),
    ("masked_psnr_median", "Masked PSNR"),
    ("masked_ssim_median", "Masked SSIM"),
    ("seam_mae_median", "Seam MAE"),
    ("object_masked_mae_median", "Object MAE"),
    ("object_masked_psnr_median", "Object PSNR"),
    ("object_masked_ssim_median", "Object SSIM"),
    ("object_depth_mae_median", "Object Depth MAE"),
    ("object_depth_corr_median", "Object Depth Corr"),
    ("depth_mae_median", "Depth MAE"),
    ("depth_corr_median", "Depth Corr"),
    ("surface_chamfer_l1_median", "Surface Chamfer"),
    ("surface_chamfer_rmse_median", "Surface Chamfer RMSE"),
    ("surface_hausdorff95_median", "Surface Hausdorff95"),
    ("object_surface_chamfer_l1_median", "Object Surface Chamfer"),
    ("object_surface_chamfer_rmse_median", "Object Surface Chamfer RMSE"),
    ("object_surface_hausdorff95_median", "Object Surface Hausdorff95"),
    ("mesh_surface_chamfer_l1_median", "Mesh Surface Chamfer"),
    ("mesh_surface_chamfer_rmse_median", "Mesh Surface Chamfer RMSE"),
    ("mesh_surface_hausdorff95_median", "Mesh Surface Hausdorff95"),
    ("silhouette_iou_masked_median", "Silhouette IoU"),
    ("stl_exists_median", "STL Exists"),
    ("stl_is_watertight_median", "STL Watertight"),
    ("stl_is_volume_median", "STL Volume Mesh"),
    ("stl_is_manifold_median", "STL Manifold"),
    ("stl_nonmanifold_edge_count_median", "STL Nonmanifold Edges"),
    ("stl_degenerate_face_ratio_median", "STL Degenerate Face Ratio"),
    ("stl_winding_consistent_median", "STL Winding"),
    ("stl_positive_volume_median", "STL Positive Volume"),
    ("stl_single_component_median", "STL Single Body"),
    ("stl_component_count_median", "STL Bodies"),
    ("stl_component_excess_log1p_median", "STL Body Excess log1p"),
    ("stl_bbox_has_volume_median", "STL 3D BBox"),
    ("stl_bbox_aspect_ratio_median", "STL Aspect"),
    ("stl_faces_per_normalized_bbox_volume_log1p_median", "STL Scale-Free Complexity log1p"),
    ("stl_faces_per_bbox_volume_log1p_median", "STL Face Density log1p"),
    ("stl_faces_median", "STL Faces"),
    ("stl_z_range_median", "STL Z Range"),
]


DELTA_METRICS = [
    ("masked_mae_median", "Masked MAE", False),
    ("object_masked_mae_median", "Object MAE", False),
    ("seam_mae_median", "Seam MAE", False),
    ("depth_mae_median", "Depth MAE", False),
    ("object_depth_mae_median", "Object Depth MAE", False),
    ("surface_chamfer_l1_median", "Surface Chamfer", False),
    ("surface_chamfer_rmse_median", "Surface Chamfer RMSE", False),
    ("surface_hausdorff95_median", "Surface Hausdorff95", False),
    ("object_surface_chamfer_l1_median", "Object Surface Chamfer", False),
    ("object_surface_chamfer_rmse_median", "Object Surface Chamfer RMSE", False),
    ("object_surface_hausdorff95_median", "Object Surface Hausdorff95", False),
    ("mesh_surface_chamfer_l1_median", "Mesh Surface Chamfer", False),
    ("mesh_surface_chamfer_rmse_median", "Mesh Surface Chamfer RMSE", False),
    ("mesh_surface_hausdorff95_median", "Mesh Surface Hausdorff95", False),
    ("masked_psnr_median", "Masked PSNR", True),
    ("object_masked_psnr_median", "Object PSNR", True),
    ("masked_ssim_median", "Masked SSIM", True),
    ("object_masked_ssim_median", "Object SSIM", True),
    ("depth_corr_median", "Depth Corr", True),
    ("object_depth_corr_median", "Object Depth Corr", True),
    ("silhouette_iou_masked_median", "Silhouette IoU", True),
]


PAIR_TIE_EPSILON = 1e-12


ARTIFACT_FIELDS = [
    ("completed_image", "Completed"),
    ("raw_completed_image", "Raw Completed"),
    ("stl_model", "STL"),
]


def read_csv_rows(path):
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader), reader.fieldnames or []


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "method": "<jsonl-parse-error>",
                        "error_type": type(exc).__name__,
                        "error": f"{path.name}:{line_number}: {exc}",
                    }
                )
    return rows


def read_json(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def rank_rows(summary_rows, weights, score_mode="normalized", baseline_method="masked"):
    return rank_summary_rows(
        summary_rows,
        weights,
        score_mode=score_mode,
        baseline_method=baseline_method,
    )


def failure_summary(summary_rows, failure_rows):
    grouped_failures = defaultdict(list)
    for row in failure_rows:
        grouped_failures[str(row.get("method") or "<unknown>")].append(row)

    summary_counts = {}
    summary_last = {}
    for row in summary_rows:
        count = parse_float(row.get("error_count"))
        if not math.isfinite(count) or count <= 0:
            continue
        method = str(row.get("method") or "<unknown>")
        summary_counts[method] = int(count)
        summary_last[method] = row

    rows = []
    for method in sorted(set(grouped_failures) | set(summary_counts)):
        failures = grouped_failures.get(method, [])
        last = failures[-1] if failures else summary_last.get(method, {})
        rows.append(
            {
                "method": method,
                "failures": len(failures) if failures else summary_counts.get(method, 0),
                "last_error_type": last.get("error_type") or last.get("last_error_type") or "",
                "last_error": last.get("error") or last.get("last_error") or "",
            }
        )
    return rows


def format_number(value):
    number = parse_float(value)
    if not math.isfinite(number):
        return ""
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if number == 0:
        return "0"
    if abs(number) < 0.001:
        return f"{number:.2e}"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def parse_score_value(value):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower() if value is not None else ""
    if text in {"true", "yes"}:
        return 1.0
    if text in {"false", "no"}:
        return 0.0
    return parse_float(value)


def format_cell(value, max_length=140):
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if max_length is not None and len(text) > max_length:
        text = text[: max_length - 1] + "..."
    return text.replace("|", "\\|")


def code_cell(value, max_length=None):
    text = format_cell(value, max_length=max_length)
    return f"`{text}`" if text else ""


def markdown_table(headers, rows):
    if not rows:
        return "_None._"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def baseline_delta_rows(summary_rows, baseline_method="masked"):
    if not baseline_method:
        return []

    baseline = next((row for row in summary_rows if row.get("method") == baseline_method), None)
    if baseline is None:
        return []

    rows = []
    for row in summary_rows:
        method = row.get("method", "")
        if method == baseline_method:
            continue
        for field, label, higher_is_better in DELTA_METRICS:
            baseline_value = parse_float(baseline.get(field))
            method_value = parse_float(row.get(field))
            if not (math.isfinite(baseline_value) and math.isfinite(method_value)):
                continue
            raw_delta = method_value - baseline_value
            improvement = raw_delta if higher_is_better else -raw_delta
            rows.append(
                {
                    "method": method,
                    "metric": label,
                    "method_value": method_value,
                    "baseline_value": baseline_value,
                    "delta": raw_delta,
                    "improvement": improvement,
                }
            )
    rows.sort(key=lambda item: (item["method"], item["metric"]))
    return rows


def paired_baseline_delta_rows(per_sample_rows, baseline_method="masked"):
    if not baseline_method:
        return []

    by_sample_method = {}
    methods = set()
    for raw_row in per_sample_rows:
        row = with_derived_metrics(raw_row)
        sample_id = row.get("sample_id")
        method = row.get("method")
        if not sample_id or not method:
            continue
        by_sample_method[(sample_id, method)] = row
        if method != baseline_method:
            methods.add(method)

    baseline_sample_ids = {
        sample_id
        for sample_id, method in by_sample_method
        if method == baseline_method
    }
    if not baseline_sample_ids:
        return []

    rows = []
    for method in sorted(methods):
        method_sample_ids = {
            sample_id
            for sample_id, row_method in by_sample_method
            if row_method == method
        }
        paired_sample_ids = sorted(
            sample_id
            for sample_id in baseline_sample_ids
            if (sample_id, method) in by_sample_method
        )
        if not paired_sample_ids:
            continue
        for summary_field, label, higher_is_better in DELTA_METRICS:
            sample_field = summary_field[: -len("_median")] if summary_field.endswith("_median") else summary_field
            improvements = []
            deltas = []
            skipped_nonfinite = 0
            for sample_id in paired_sample_ids:
                baseline_value = parse_float(by_sample_method[(sample_id, baseline_method)].get(sample_field))
                method_value = parse_float(by_sample_method[(sample_id, method)].get(sample_field))
                if not (math.isfinite(baseline_value) and math.isfinite(method_value)):
                    skipped_nonfinite += 1
                    continue
                raw_delta = method_value - baseline_value
                improvement = raw_delta if higher_is_better else -raw_delta
                deltas.append(raw_delta)
                improvements.append(improvement)
            if not improvements:
                continue
            improvement_array = np.asarray(improvements, dtype=np.float64)
            delta_array = np.asarray(deltas, dtype=np.float64)
            win_count = int(np.count_nonzero(improvement_array > PAIR_TIE_EPSILON))
            tie_count = int(np.count_nonzero(np.abs(improvement_array) <= PAIR_TIE_EPSILON))
            paired_n = len(improvement_array)
            rows.append(
                {
                    "method": method,
                    "metric": label,
                    "baseline_n": len(baseline_sample_ids),
                    "method_n": len(method_sample_ids),
                    "common_n": len(paired_sample_ids),
                    "paired_n": paired_n,
                    "skipped_nonfinite": skipped_nonfinite,
                    "win_count": win_count,
                    "tie_count": tie_count,
                    "win_rate": win_count / paired_n if paired_n else math.nan,
                    "median_delta": float(np.median(delta_array)),
                    "median_improvement": float(np.median(improvement_array)),
                    "mean_improvement": float(np.mean(improvement_array)),
                }
            )
    rows.sort(key=lambda item: (item["method"], item["metric"]))
    return rows


def baseline_delta_table(delta_rows, baseline_method):
    if not delta_rows:
        return f"_No comparable finite metrics found against `{baseline_method}`._"

    rows = [
        [
            row["method"],
            row["metric"],
            format_number(row["method_value"]),
            format_number(row["baseline_value"]),
            format_number(row["delta"]),
            format_number(row["improvement"]),
        ]
        for row in delta_rows
    ]
    return markdown_table(
        ["Method", "Metric", "Value", baseline_method, "Delta", "Improvement"],
        rows,
    )


def paired_baseline_delta_table(delta_rows, baseline_method, baseline_present=True):
    if not baseline_present:
        return f"_Baseline method `{baseline_method}` was not found in `per_sample_metrics.csv`._"
    if not delta_rows:
        return "_No paired finite metrics found against the selected baseline._"

    rows = [
        [
            row["method"],
            row["metric"],
            format_number(row["baseline_n"]),
            format_number(row["method_n"]),
            format_number(row["common_n"]),
            format_number(row["paired_n"]),
            format_number(row["skipped_nonfinite"]),
            f"{row['win_count']}/{row['paired_n']}",
            format_number(row["win_rate"]),
            format_number(row["tie_count"]),
            format_number(row["median_delta"]),
            format_number(row["median_improvement"]),
            format_number(row["mean_improvement"]),
        ]
        for row in delta_rows
    ]
    return markdown_table(
        [
            "Method",
            "Metric",
            "Baseline n",
            "Method n",
            "Common n",
            "Paired n",
            "Skipped",
            "Wins",
            "Win Rate",
            "Ties",
            "Median Delta",
            "Median Improvement",
            "Mean Improvement",
        ],
        rows,
    )


def sample_field_for_summary_metric(metric):
    if metric.endswith("_median"):
        return metric[: -len("_median")]
    return metric


def bootstrap_mean_ci(values, samples=1000, seed=1234, confidence=0.95):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan
    if len(array) == 1 or samples <= 0:
        mean = float(np.mean(array))
        return mean, mean

    rng = np.random.default_rng(seed)
    resampled = rng.choice(array, size=(samples, len(array)), replace=True)
    means = np.mean(resampled, axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


def paired_objective_rows(per_sample_rows, weights, baseline_method="masked", bootstrap_samples=1000, bootstrap_seed=1234):
    if not baseline_method:
        return []

    by_sample_method = {}
    methods = set()
    for raw_row in per_sample_rows:
        row = with_derived_metrics(raw_row)
        sample_id = row.get("sample_id")
        method = row.get("method")
        if not sample_id or not method:
            continue
        by_sample_method[(sample_id, method)] = row
        if method != baseline_method:
            methods.add(method)

    baseline_sample_ids = {
        sample_id
        for sample_id, method in by_sample_method
        if method == baseline_method
    }
    if not baseline_sample_ids:
        return []

    rows = []
    for method in sorted(methods):
        method_sample_ids = {
            sample_id
            for sample_id, row_method in by_sample_method
            if row_method == method
        }
        paired_sample_ids = sorted(
            sample_id
            for sample_id in baseline_sample_ids
            if (sample_id, method) in by_sample_method
        )
        if not paired_sample_ids:
            continue

        improvements = []
        metric_counts = []
        skipped_samples = 0
        for sample_id in paired_sample_ids:
            baseline_row = by_sample_method[(sample_id, baseline_method)]
            method_row = by_sample_method[(sample_id, method)]
            improvement = 0.0
            metric_count = 0
            for metric, weight in weights.items():
                sample_field = sample_field_for_summary_metric(metric)
                baseline_value = parse_score_value(baseline_row.get(sample_field))
                method_value = parse_score_value(method_row.get(sample_field))
                if not (math.isfinite(baseline_value) and math.isfinite(method_value)):
                    continue
                raw_delta = method_value - baseline_value
                signed_improvement = raw_delta if weight > 0 else -raw_delta
                improvement += abs(weight) * signed_improvement
                metric_count += 1
            if metric_count:
                improvements.append(improvement)
                metric_counts.append(metric_count)
            else:
                skipped_samples += 1

        if not improvements:
            continue
        improvement_array = np.asarray(improvements, dtype=np.float64)
        win_count = int(np.count_nonzero(improvement_array > PAIR_TIE_EPSILON))
        tie_count = int(np.count_nonzero(np.abs(improvement_array) <= PAIR_TIE_EPSILON))
        ci_low, ci_high = bootstrap_mean_ci(improvement_array, samples=bootstrap_samples, seed=bootstrap_seed)
        rows.append(
            {
                "method": method,
                "baseline_n": len(baseline_sample_ids),
                "method_n": len(method_sample_ids),
                "common_n": len(paired_sample_ids),
                "paired_n": len(improvement_array),
                "skipped_samples": skipped_samples,
                "win_count": win_count,
                "tie_count": tie_count,
                "win_rate": win_count / len(improvement_array),
                "median_improvement": float(np.median(improvement_array)),
                "mean_improvement": float(np.mean(improvement_array)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_metric_count": float(np.mean(metric_counts)),
            }
        )
    rows.sort(key=lambda item: (-item["mean_improvement"], item["method"]))
    return rows


def paired_objective_table(objective_rows, baseline_method, baseline_present=True):
    if not baseline_present:
        return f"_Baseline method `{baseline_method}` was not found in `per_sample_metrics.csv`._"
    if not objective_rows:
        return "_No paired finite objective scores found against the selected baseline._"

    rows = [
        [
            row["method"],
            format_number(row["baseline_n"]),
            format_number(row["method_n"]),
            format_number(row["common_n"]),
            format_number(row["paired_n"]),
            format_number(row["skipped_samples"]),
            f"{row['win_count']}/{row['paired_n']}",
            format_number(row["win_rate"]),
            format_number(row["tie_count"]),
            format_number(row["median_improvement"]),
            format_number(row["mean_improvement"]),
            format_number(row["ci95_low"]),
            format_number(row["ci95_high"]),
            format_number(row["mean_metric_count"]),
        ]
        for row in objective_rows
    ]
    return markdown_table(
        [
            "Method",
            "Baseline n",
            "Method n",
            "Common n",
            "Paired n",
            "Skipped",
            "Wins",
            "Win Rate",
            "Ties",
            "Median Objective Improvement",
            "Mean Objective Improvement",
            "Mean CI95 Low",
            "Mean CI95 High",
            "Mean Metrics/Sample",
        ],
        rows,
    )


def has_method(rows, method):
    return any(row.get("method") == method for row in rows)


def score_profile_note(profile):
    return f"_Score profile: `{profile}` ({SCORE_PROFILE_DESCRIPTIONS.get(profile, 'custom metric weights')})._"


def ranking_table(ranked_rows, top):
    rows = []
    for index, row in enumerate(ranked_rows[:top], start=1):
        rows.append(
            [
                str(index),
                row.get("method", ""),
                format_number(row.get("rank_score")),
                format_number(row.get("n")),
                format_number(row.get("attempted_n")),
                format_number(row.get("success_rate")),
                format_number(row.get("error_count")),
                format_number(row.get("masked_mae_median")),
                format_number(row.get("object_masked_mae_median")),
                format_number(row.get("depth_mae_median")),
                format_number(row.get("object_depth_mae_median")),
                format_number(row.get("object_surface_chamfer_l1_median")),
                format_number(row.get("stl_is_watertight_median")),
            ]
        )
    return markdown_table(
        [
            "Rank",
            "Method",
            "Score",
            "n",
            "Attempted",
            "Success Rate",
            "Errors",
            "Masked MAE",
            "Object MAE",
            "Depth MAE",
            "Object Depth MAE",
            "Object Surface Chamfer",
            "STL Watertight",
        ],
        rows,
    )


def key_metrics_table(ranked_rows):
    available_metrics = [
        (field, label)
        for field, label in KEY_METRICS
        if any(format_number(row.get(field)) for row in ranked_rows)
    ]
    if not ranked_rows or not available_metrics:
        return "_No finite median metrics found in `summary_metrics.csv`._"

    rows = []
    for row in ranked_rows:
        rows.append([row.get("method", "")] + [format_number(row.get(field)) for field, _ in available_metrics])
    return markdown_table(["Method"] + [label for _, label in available_metrics], rows)


def failure_table(failure_rows):
    rows = [
        [row["method"], str(row["failures"]), row["last_error_type"], row["last_error"]]
        for row in failure_rows
    ]
    return markdown_table(["Method", "Failures", "Last Error Type", "Last Error"], rows)


def artifact_examples(per_sample_rows, examples):
    rows = []
    for row in per_sample_rows:
        artifact_values = [row.get(field, "") for field, _ in ARTIFACT_FIELDS]
        if not any(artifact_values):
            continue
        rows.append(
                [
                    row.get("method", ""),
                    row.get("sample_id", ""),
                    *[code_cell(value, max_length=116) for value in artifact_values],
                ]
            )
        if len(rows) >= examples:
            break

    return markdown_table(
        ["Method", "Sample", *[label for _, label in ARTIFACT_FIELDS]],
        rows,
    )


def split_audit_table(audit):
    if not audit:
        return "_None._"
    rows = [
        ["Eval n", format_number(audit.get("eval_n"))],
        ["Eval assets", format_number(audit.get("eval_asset_count"))],
        ["Train assets", format_number(audit.get("train_asset_count"))],
        ["Train/eval overlap", format_number(audit.get("train_eval_asset_overlap_count"))],
        ["Eval categories", ", ".join(f"{key}:{value}" for key, value in sorted(audit.get("eval_category_counts", {}).items()))],
        ["Train categories", ", ".join(f"{key}:{value}" for key, value in sorted(audit.get("train_category_counts", {}).items()))],
        ["Eval source splits", ", ".join(f"{key}:{value}" for key, value in sorted(audit.get("eval_source_split_counts", {}).items()))],
        ["Train source splits", ", ".join(f"{key}:{value}" for key, value in sorted(audit.get("train_source_split_counts", {}).items()))],
    ]
    return markdown_table(["Field", "Value"], rows)


def render_report(
    run_dir,
    output_path,
    summary_rows,
    per_sample_rows,
    failures,
    split_audit,
    ranked_rows,
    used_metrics,
    top,
    examples,
    baseline_method,
    score_mode,
    weights,
    paired_bootstrap_samples,
    paired_bootstrap_seed,
    contact_sheet_path=None,
    score_profile="default",
):
    relative_output = output_path
    try:
        relative_output = output_path.relative_to(Path.cwd())
    except ValueError:
        pass
    relative_contact_sheet = None
    if contact_sheet_path:
        relative_contact_sheet = Path(contact_sheet_path)
        try:
            relative_contact_sheet = relative_contact_sheet.relative_to(Path.cwd())
        except ValueError:
            pass

    lines = [
        f"# Benchmark Report: {run_dir.name}",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Report: `{relative_output}`",
        f"- Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- Summary rows: `{len(summary_rows)}`",
        f"- Per-sample rows: `{len(per_sample_rows)}`",
        f"- Failure rows: `{len(failures)}`",
    ]
    if relative_contact_sheet:
        lines.append(f"- Contact sheet: `{relative_contact_sheet}`")
    lines.extend(
        [
            "",
            "## Ranking",
            "",
        ]
    )

    if not summary_rows:
        lines.append("_No summary rows found. Ranking is unavailable._")
    else:
        if score_mode == "baseline-delta":
            lines.extend(
                [
                    score_profile_note(score_profile),
                    "",
                    f"_Score mode: weighted sign-normalized metric improvement against `{baseline_method}`. The baseline scores `0`; positive scores improve the objective and negative scores regress it. Scores are stable when unrelated candidate methods are added._",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    score_profile_note(score_profile),
                    "",
                    "_Score mode: weighted min/max normalization within this candidate set. Scores are useful for ordering one run, but the numeric value can change when candidates are added or removed._",
                    "",
                ]
            )
        lines.append(ranking_table(ranked_rows, top))
        success_rates = {
            format_number(row.get("success_rate"))
            for row in ranked_rows
            if format_number(row.get("success_rate"))
        }
        if len(success_rates) > 1:
            lines.extend(
                [
                    "",
                    "_Coverage warning: ranked methods have unequal success rates; compare medians with care._",
                ]
            )
        if not used_metrics:
            lines.extend(["", "_No finite weighted metrics were available, so scores default to `0`._"])

    lines.extend(
        [
            "",
            "## Key Metric Medians",
            "",
            key_metrics_table(ranked_rows),
            "",
            f"## Selected Baseline Deltas: `{baseline_method}`",
            "",
            "Raw `Delta` is `method - baseline`; `Improvement` is sign-normalized so positive means better. These are aggregate median deltas for selected diagnostic metrics, not weighted rank points or paired per-sample wins.",
            "",
            baseline_delta_table(baseline_delta_rows(summary_rows, baseline_method), baseline_method),
            "",
            f"## Paired Baseline Wins: `{baseline_method}`",
            "",
            "Each row compares a method against the baseline on the same `sample_id`. `Wins` count strictly positive sign-normalized improvements; ties are reported separately.",
            "",
            paired_baseline_delta_table(
                paired_baseline_delta_rows(per_sample_rows, baseline_method),
                baseline_method,
                baseline_present=has_method(per_sample_rows, baseline_method),
            ),
            "",
            f"## Paired Objective Confidence: `{baseline_method}`",
            "",
            "Each row applies the same metric weights used for ranking to each shared `sample_id`, but as raw sign-normalized improvement over the baseline rather than candidate-normalized aggregate medians. The confidence interval bootstraps paired sample improvements with a fixed seed; if the CI is entirely above `0`, the method has a stronger promotion signal than an aggregate median alone.",
            "",
            paired_objective_table(
                paired_objective_rows(
                    per_sample_rows,
                    weights,
                    baseline_method=baseline_method,
                    bootstrap_samples=paired_bootstrap_samples,
                    bootstrap_seed=paired_bootstrap_seed,
                ),
                baseline_method,
                baseline_present=has_method(per_sample_rows, baseline_method),
            ),
            "",
            "## Failures",
            "",
            failure_table(failure_summary(summary_rows, failures)),
            "",
            "## Split Audit",
            "",
            split_audit_table(split_audit),
            "",
            "## Example Artifacts",
            "",
            artifact_examples(per_sample_rows, examples),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write a concise Markdown report for a completion benchmark run directory. "
            "The directory must contain summary_metrics.csv and may also contain "
            "per_sample_metrics.csv and failures.jsonl."
        )
    )
    parser.add_argument("run_dir", help="Benchmark run directory containing summary_metrics.csv.")
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown report path. Defaults to <run_dir>/benchmark_report.md.",
    )
    parser.add_argument("--top", type=int, default=10, help="Maximum methods to show in the ranking table.")
    parser.add_argument("--examples", type=int, default=6, help="Maximum per-sample artifact examples to show.")
    parser.add_argument("--contact-sheet", action="store_true", help="Also write a PNG artifact contact sheet.")
    parser.add_argument("--contact-sheet-output", default=None, help="PNG contact sheet path. Defaults to <run_dir>/artifact_contact_sheet.png.")
    parser.add_argument("--contact-sheet-methods", default=None, help="Comma-separated method, base-method, or experiment-name list for the contact sheet.")
    parser.add_argument("--contact-sheet-samples", default=None, help="Comma-separated sample_id list for the contact sheet.")
    parser.add_argument("--contact-sheet-max-samples", type=int, default=6, help="Maximum samples in the contact sheet when --contact-sheet-samples is omitted.")
    parser.add_argument("--contact-sheet-thumb-size", type=int, default=150, help="Thumbnail size in pixels for the contact sheet.")
    parser.add_argument("--weight", action="append", default=[], help="Override ranking weight as metric=value.")
    parser.add_argument("--paired-bootstrap-samples", type=int, default=1000, help="Bootstrap resamples for paired objective confidence intervals.")
    parser.add_argument("--paired-bootstrap-seed", type=int, default=1234, help="Random seed for paired objective confidence intervals.")
    parser.add_argument(
        "--score-mode",
        choices=sorted(SCORE_MODES),
        default="normalized",
        help="`normalized` ranks within the candidate set; `baseline-delta` scores weighted improvements over --baseline-method.",
    )
    parser.add_argument(
        "--score-profile",
        choices=sorted(SCORE_PROFILES),
        default="default",
        help="Named metric weight profile. Explicit --weight overrides are applied on top.",
    )
    parser.add_argument(
        "--baseline-method",
        default="masked",
        help="Method used for absolute delta reporting and --score-mode baseline-delta.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing required summary file: {summary_path}")

    output_path = Path(args.output) if args.output else run_dir / "benchmark_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_rows, _ = read_csv_rows(summary_path)
    per_sample_rows, _ = read_csv_rows(run_dir / "per_sample_metrics.csv")
    failures = read_jsonl(run_dir / "failures.jsonl")
    split_audit = read_json(run_dir / "split_audit.json")
    weights = parse_weights(args.weight or [], profile=args.score_profile)
    ranked_rows, used_metrics = rank_rows(
        summary_rows,
        weights,
        score_mode=args.score_mode,
        baseline_method=args.baseline_method,
    )
    if not ranked_rows and summary_rows:
        ranked_rows = [{"rank_score": 0.0, **row} for row in summary_rows]

    contact_sheet_path = None
    if args.contact_sheet:
        contact_sheet_path = Path(args.contact_sheet_output) if args.contact_sheet_output else run_dir / "artifact_contact_sheet.png"
        make_contact_sheet(
            run_dir,
            output_path=contact_sheet_path,
            methods=parse_csv_arg(args.contact_sheet_methods),
            samples=parse_csv_arg(args.contact_sheet_samples),
            max_samples=max(args.contact_sheet_max_samples, 0),
            thumb_size=max(args.contact_sheet_thumb_size, 32),
        )

    render_report(
        run_dir=run_dir,
        output_path=output_path,
        summary_rows=summary_rows,
        per_sample_rows=per_sample_rows,
        failures=failures,
        split_audit=split_audit,
        ranked_rows=ranked_rows,
        used_metrics=used_metrics,
        top=max(args.top, 0),
        examples=max(args.examples, 0),
        baseline_method=args.baseline_method,
        score_mode=args.score_mode,
        weights=weights,
        paired_bootstrap_samples=max(args.paired_bootstrap_samples, 0),
        paired_bootstrap_seed=args.paired_bootstrap_seed,
        contact_sheet_path=contact_sheet_path,
        score_profile=args.score_profile,
    )
    print(output_path)
    if contact_sheet_path:
        print(contact_sheet_path)


if __name__ == "__main__":
    main()
