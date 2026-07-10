from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmark.direct_mesh import direct_mesh_bbox_uses_hidden_source
from backend.benchmark.rank_methods import SCORE_PROFILES, parse_float, parse_weights, rank_summary_rows, with_derived_metrics
from backend.benchmark.report_run import format_number, markdown_table, paired_objective_rows


SCALE_FREE_COMPLEXITY_MEDIAN = "stl_faces_per_normalized_bbox_volume_log1p_median"
SCALE_FREE_COMPLEXITY_SAMPLE = "stl_faces_per_normalized_bbox_volume_log1p"
LEGACY_FACE_DENSITY_MEDIAN = "stl_faces_per_bbox_volume_log1p_median"
LEGACY_FACE_DENSITY_SAMPLE = "stl_faces_per_bbox_volume_log1p"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def load_summary_rows(input_dir: Path) -> tuple[list[dict[str, str]], Path]:
    for name in ("aggregate_summary.csv", "summary_metrics.csv"):
        path = input_dir / name
        if path.exists():
            return read_csv(path), path
    raise FileNotFoundError(f"Expected aggregate_summary.csv or summary_metrics.csv in {input_dir}")


def load_per_sample_rows(input_dir: Path) -> list[dict[str, str]]:
    root_metrics = input_dir / "per_sample_metrics.csv"
    if root_metrics.exists():
        return read_csv(root_metrics)

    rows = []
    for metrics_path in sorted(input_dir.glob("*/per_sample_metrics.csv")):
        method = metrics_path.parent.name
        for row in read_csv(metrics_path):
            row["method"] = row.get("method") or method
            rows.append(row)
    return rows


def load_split_audit(input_dir: Path, method: str) -> dict:
    return read_json(input_dir / method / "split_audit.json") or read_json(input_dir / "split_audit.json")


def finite_number(value, default=math.nan):
    number = parse_float(value)
    return number if math.isfinite(number) else default


def bool_value(value, default=False):
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def is_oracle_diagnostic(row: dict) -> bool:
    if bool_value(row.get("oracle_diagnostic")):
        return True
    bbox_source = str(row.get("direct_mesh_bbox_source") or "").strip().lower()
    if direct_mesh_bbox_uses_hidden_source(bbox_source, row.get("direct_mesh_reference_method")):
        return True
    names = " ".join(str(row.get(field) or "") for field in ("method", "base_method", "stl_mode"))
    names = names.lower().replace("-", "_")
    return "source_bbox" in names or ("source_mesh" in names and "oracle" in names)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def pass_check(name, passed, value, threshold, detail=""):
    return {
        "name": name,
        "passed": bool(passed),
        "value": value,
        "threshold": threshold,
        "detail": detail,
    }


def method_row(rows, method):
    return next((row for row in rows if row.get("method") == method), None)


def _sample_label(row: dict, index: int) -> str:
    return str(row.get("sample_id") or row.get("id") or f"row{index}")


def fallback_metric_field(field: str) -> str | None:
    return {
        SCALE_FREE_COMPLEXITY_MEDIAN: LEGACY_FACE_DENSITY_MEDIAN,
        SCALE_FREE_COMPLEXITY_SAMPLE: LEGACY_FACE_DENSITY_SAMPLE,
    }.get(field)


def _has_cell(row: dict, field: str) -> bool:
    if field in row and str(row.get(field, "")).strip() != "":
        return True
    fallback = fallback_metric_field(field)
    return bool(fallback and fallback in row and str(row.get(fallback, "")).strip() != "")


def gate_cell(row: dict, field: str):
    if field in row and str(row.get(field, "")).strip() != "":
        return row.get(field)
    fallback = fallback_metric_field(field)
    if fallback:
        return row.get(fallback)
    return row.get(field)


def stl_gate_number(value) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "y"}:
        return 1.0
    if text in {"false", "no", "n"}:
        return 0.0
    return finite_number(value)


def per_sample_stl_checks(
    per_sample_rows,
    method: str,
    *,
    minimums,
    maximums,
):
    method_rows = [row for row in per_sample_rows if row.get("method") == method]
    if not method_rows:
        return []

    checks = []
    for field, label, threshold in minimums:
        if not any(_has_cell(row, field) for row in method_rows):
            continue
        failed = []
        for index, row in enumerate(method_rows):
            value = stl_gate_number(gate_cell(row, field))
            if not math.isfinite(value) or value < threshold:
                failed.append(_sample_label(row, index))
        detail = f"failed_samples={','.join(failed[:10])}" if failed else ""
        if len(failed) > 10:
            detail += f",+{len(failed) - 10}"
        checks.append(
            pass_check(
                f"per_sample_{label}",
                not failed,
                len(method_rows) - len(failed),
                f"{len(method_rows)}/{len(method_rows)} samples >= {threshold}",
                detail=detail,
            )
        )
    for field, label, threshold in maximums:
        if not any(_has_cell(row, field) for row in method_rows):
            continue
        failed = []
        for index, row in enumerate(method_rows):
            value = stl_gate_number(gate_cell(row, field))
            if not math.isfinite(value) or value > threshold:
                failed.append(_sample_label(row, index))
        detail = f"failed_samples={','.join(failed[:10])}" if failed else ""
        if len(failed) > 10:
            detail += f",+{len(failed) - 10}"
        checks.append(
            pass_check(
                f"per_sample_{label}",
                not failed,
                len(method_rows) - len(failed),
                f"{len(method_rows)}/{len(method_rows)} samples <= {threshold}",
                detail=detail,
            )
        )
    return checks


def paired_metric_ratio_check(
    per_sample_rows,
    candidate_method: str,
    current_method: str,
    *,
    field: str,
    label: str,
    maximum_ratio: float,
    minimum_pairs: int,
):
    method_rows = [
        row
        for row in per_sample_rows
        if row.get("method") in {candidate_method, current_method}
    ]
    if not any(_has_cell(row, field) for row in method_rows):
        return None

    by_sample_method = {
        (str(row.get("sample_id")), row.get("method")): row
        for row in method_rows
        if row.get("sample_id")
    }
    sample_ids = sorted({sample_id for sample_id, _method in by_sample_method})
    ratios = []
    missing = []
    for sample_id in sample_ids:
        candidate_row = by_sample_method.get((sample_id, candidate_method))
        current_row = by_sample_method.get((sample_id, current_method))
        candidate_value = finite_number(candidate_row.get(field)) if candidate_row else math.nan
        current_value = finite_number(current_row.get(field)) if current_row else math.nan
        if (
            not math.isfinite(candidate_value)
            or not math.isfinite(current_value)
            or candidate_value < 0
            or current_value < 0
        ):
            missing.append(sample_id)
            continue
        if current_value == 0:
            ratio = 1.0 if candidate_value == 0 else math.inf
        else:
            ratio = candidate_value / current_value
        ratios.append((ratio, sample_id, candidate_value, current_value))

    worst = max(ratios, default=(math.nan, "", math.nan, math.nan), key=lambda item: item[0])
    sorted_ratios = sorted(item[0] for item in ratios)
    median_ratio = math.nan
    if sorted_ratios:
        middle = len(sorted_ratios) // 2
        if len(sorted_ratios) % 2:
            median_ratio = sorted_ratios[middle]
        else:
            median_ratio = (sorted_ratios[middle - 1] + sorted_ratios[middle]) / 2.0
    passed = (
        len(ratios) >= minimum_pairs
        and not missing
        and math.isfinite(worst[0])
        and worst[0] <= maximum_ratio
    )
    detail = (
        f"paired_n={len(ratios)}; required_n={minimum_pairs}; "
        f"median_ratio={format_number(median_ratio)}"
    )
    if worst[1]:
        detail += (
            f"; worst_sample={worst[1]}; candidate={format_number(worst[2])}; "
            f"current={format_number(worst[3])}"
        )
    if missing:
        detail += f"; missing_or_nonfinite={','.join(missing[:10])}"
        if len(missing) > 10:
            detail += f",+{len(missing) - 10}"
    return pass_check(
        f"paired_{label}_ratio_vs_current",
        passed,
        worst[0],
        f"<= {maximum_ratio}",
        detail=detail,
    )


def evaluate_selection(
    summary_rows,
    per_sample_rows,
    *,
    baseline_method="masked",
    candidate_method=None,
    current_method=None,
    weights=None,
    min_success_rate=1.0,
    min_paired_n=5,
    min_win_rate=0.8,
    min_ci95_low=0.0,
    min_score_margin=0.0,
    min_stl_watertight=1.0,
    min_stl_is_volume=1.0,
    min_stl_is_manifold=1.0,
    min_stl_winding_consistent=1.0,
    min_stl_positive_volume=1.0,
    min_stl_single_component=1.0,
    min_stl_bbox_has_volume=1.0,
    max_stl_nonmanifold_edge_count_log1p=0.0,
    max_stl_degenerate_face_ratio=0.0,
    max_stl_component_excess_log1p=0.0,
    max_stl_bbox_aspect_ratio=10.0,
    max_stl_faces_per_bbox_volume_log1p=10.0,
    max_mesh_surface_chamfer_ratio_vs_current=1.1,
    max_mesh_surface_hausdorff95_ratio_vs_current=1.1,
    max_train_eval_overlap=0,
    split_audit=None,
    require_split_audit=True,
    bootstrap_samples=1000,
    bootstrap_seed=1234,
):
    summary_rows = [with_derived_metrics(row) for row in summary_rows]
    per_sample_rows = [with_derived_metrics(row) for row in per_sample_rows]
    weights = weights or parse_weights([])
    ranked_rows, used_metrics = rank_summary_rows(
        summary_rows,
        weights,
        score_mode="baseline-delta",
        baseline_method=baseline_method,
    )
    if not ranked_rows:
        raise ValueError("No ranked rows available for selection")

    candidate_method = candidate_method or next(
        (
            row.get("method")
            for row in ranked_rows
            if row.get("method") != baseline_method and not is_oracle_diagnostic(row)
        ),
        None,
    )
    if not candidate_method:
        raise ValueError("No deployable candidate method available after excluding the baseline and oracle diagnostics")

    candidate = method_row(ranked_rows, candidate_method)
    if candidate is None:
        raise ValueError(f"Candidate method `{candidate_method}` was not found in summary rows")

    current = method_row(ranked_rows, current_method) if current_method else None
    objective_rows = paired_objective_rows(
        per_sample_rows,
        weights,
        baseline_method=baseline_method,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    objective = next((row for row in objective_rows if row.get("method") == candidate_method), None)
    objective_vs_current = None
    if current_method and candidate_method != current_method:
        current_objective_rows = paired_objective_rows(
            per_sample_rows,
            weights,
            baseline_method=current_method,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        objective_vs_current = next((row for row in current_objective_rows if row.get("method") == candidate_method), None)

    checks = []
    candidate_is_oracle = is_oracle_diagnostic(candidate)
    checks.append(
        pass_check(
            "deployable_candidate",
            not candidate_is_oracle,
            candidate.get("direct_mesh_bbox_source") or candidate.get("stl_mode") or "deployable",
            "not a hidden-source/oracle diagnostic",
        )
    )
    if current_method and current is None:
        checks.append(
            pass_check(
                "current_method_present",
                False,
                "",
                f"`{current_method}` in summary rows",
            )
        )

    success_rate = finite_number(candidate.get("success_rate"))
    checks.append(pass_check("success_rate", success_rate >= min_success_rate, success_rate, f">= {min_success_rate}"))

    paired_n = objective.get("paired_n", 0) if objective else 0
    win_rate = objective.get("win_rate", math.nan) if objective else math.nan
    ci95_low = objective.get("ci95_low", math.nan) if objective else math.nan
    checks.append(pass_check("paired_n", paired_n >= min_paired_n, paired_n, f">= {min_paired_n}"))
    checks.append(pass_check("paired_win_rate", math.isfinite(win_rate) and win_rate >= min_win_rate, win_rate, f">= {min_win_rate}"))
    checks.append(pass_check("paired_ci95_low", math.isfinite(ci95_low) and ci95_low > min_ci95_low, ci95_low, f"> {min_ci95_low}"))

    candidate_score = finite_number(candidate.get("rank_score"))
    if current and candidate_method != current_method:
        current_score = finite_number(current.get("rank_score"))
        score_margin = candidate_score - current_score
        checks.append(
            pass_check(
                "score_margin_vs_current",
                math.isfinite(score_margin) and score_margin > min_score_margin,
                score_margin,
                f"> {min_score_margin}",
                detail=f"current_method={current_method}",
            )
        )
        current_paired_n = objective_vs_current.get("paired_n", 0) if objective_vs_current else 0
        current_win_rate = objective_vs_current.get("win_rate", math.nan) if objective_vs_current else math.nan
        current_ci95_low = objective_vs_current.get("ci95_low", math.nan) if objective_vs_current else math.nan
        checks.append(
            pass_check(
                "paired_n_vs_current",
                current_paired_n >= min_paired_n,
                current_paired_n,
                f">= {min_paired_n}",
                detail=f"current_method={current_method}",
            )
        )
        checks.append(
            pass_check(
                "paired_win_rate_vs_current",
                math.isfinite(current_win_rate) and current_win_rate >= min_win_rate,
                current_win_rate,
                f">= {min_win_rate}",
                detail=f"current_method={current_method}",
            )
        )
        checks.append(
            pass_check(
                "paired_ci95_low_vs_current",
                math.isfinite(current_ci95_low) and current_ci95_low > min_ci95_low,
                current_ci95_low,
                f"> {min_ci95_low}",
                detail=f"current_method={current_method}",
            )
        )
        for field, label, maximum_ratio in (
            (
                "mesh_surface_chamfer_l1",
                "mesh_surface_chamfer",
                max_mesh_surface_chamfer_ratio_vs_current,
            ),
            (
                "mesh_surface_hausdorff95",
                "mesh_surface_hausdorff95",
                max_mesh_surface_hausdorff95_ratio_vs_current,
            ),
        ):
            check = paired_metric_ratio_check(
                per_sample_rows,
                candidate_method,
                current_method,
                field=field,
                label=label,
                maximum_ratio=maximum_ratio,
                minimum_pairs=min_paired_n,
            )
            if check:
                checks.append(check)
    elif current:
        current_score = finite_number(current.get("rank_score"))
    else:
        current_score = math.nan
        checks.append(pass_check("score_positive_vs_baseline", candidate_score > 0, candidate_score, "> 0"))

    split_audit = split_audit or {}

    for field, label, minimum in (
        ("stl_is_watertight_median", "stl_watertight", min_stl_watertight),
        ("stl_is_volume_median", "stl_volume", min_stl_is_volume),
        ("stl_is_manifold_median", "stl_manifold", min_stl_is_manifold),
        ("stl_winding_consistent_median", "stl_winding_consistent", min_stl_winding_consistent),
        ("stl_positive_volume_median", "stl_positive_volume", min_stl_positive_volume),
        ("stl_single_component_median", "stl_single_component", min_stl_single_component),
        ("stl_bbox_has_volume_median", "stl_bbox_has_volume", min_stl_bbox_has_volume),
    ):
        if field in candidate and str(candidate.get(field, "")).strip() != "":
            value = finite_number(candidate.get(field))
            checks.append(pass_check(label, value >= minimum, value, f">= {minimum}"))
    for field, label, maximum in (
        ("stl_nonmanifold_edge_count_log1p_median", "stl_nonmanifold_edges", max_stl_nonmanifold_edge_count_log1p),
        ("stl_degenerate_face_ratio_median", "stl_degenerate_face_ratio", max_stl_degenerate_face_ratio),
        ("stl_component_excess_log1p_median", "stl_component_excess", max_stl_component_excess_log1p),
        ("stl_bbox_aspect_ratio_median", "stl_bbox_aspect_ratio", max_stl_bbox_aspect_ratio),
        (
            SCALE_FREE_COMPLEXITY_MEDIAN,
            "stl_scale_free_complexity",
            max_stl_faces_per_bbox_volume_log1p,
        ),
    ):
        if _has_cell(candidate, field):
            value = finite_number(gate_cell(candidate, field))
            checks.append(pass_check(label, value <= maximum, value, f"<= {maximum}"))

    checks.extend(
        per_sample_stl_checks(
            per_sample_rows,
            candidate_method,
            minimums=(
                ("stl_is_watertight", "stl_watertight", min_stl_watertight),
                ("stl_is_volume", "stl_volume", min_stl_is_volume),
                ("stl_is_manifold", "stl_manifold", min_stl_is_manifold),
                ("stl_winding_consistent", "stl_winding_consistent", min_stl_winding_consistent),
                ("stl_positive_volume", "stl_positive_volume", min_stl_positive_volume),
                ("stl_single_component", "stl_single_component", min_stl_single_component),
                ("stl_bbox_has_volume", "stl_bbox_has_volume", min_stl_bbox_has_volume),
            ),
            maximums=(
                ("stl_nonmanifold_edge_count_log1p", "stl_nonmanifold_edges", max_stl_nonmanifold_edge_count_log1p),
                ("stl_degenerate_face_ratio", "stl_degenerate_face_ratio", max_stl_degenerate_face_ratio),
                ("stl_component_excess_log1p", "stl_component_excess", max_stl_component_excess_log1p),
                ("stl_bbox_aspect_ratio", "stl_bbox_aspect_ratio", max_stl_bbox_aspect_ratio),
                (
                    SCALE_FREE_COMPLEXITY_SAMPLE,
                    "stl_scale_free_complexity",
                    max_stl_faces_per_bbox_volume_log1p,
                ),
            ),
        )
    )

    if require_split_audit:
        checks.append(pass_check("split_audit_present", bool(split_audit), bool(split_audit), "present"))
    if split_audit:
        overlap = finite_number(split_audit.get("train_eval_asset_overlap_count"))
        checks.append(
            pass_check(
                "train_eval_asset_overlap",
                math.isfinite(overlap) and overlap <= max_train_eval_overlap,
                overlap,
                f"<= {max_train_eval_overlap}",
            )
        )

    lora_weights = str(candidate.get("lora_weights") or split_audit.get("lora_weights") or "").strip()
    training_report = str(candidate.get("training_report") or split_audit.get("training_report") or "").strip()
    if lora_weights:
        checks.append(
            pass_check(
                "training_report_present",
                bool(training_report),
                bool(training_report),
                "present for LoRA candidates",
                detail=lora_weights,
            )
        )
        if split_audit and "prompt_family_matches_eval" in split_audit:
            matches_eval = bool_value(split_audit.get("prompt_family_matches_eval"))
            checks.append(
                pass_check(
                    "prompt_family_matches_eval",
                    matches_eval,
                    matches_eval,
                    "true",
                    detail=split_audit.get("prompt_family_mismatch_detail", ""),
                )
            )

    passed = all(check["passed"] for check in checks)
    if current_method and candidate_method == current_method:
        decision = "keep_current" if passed else "hold"
    elif passed:
        decision = "promote"
    else:
        decision = "hold"

    failed = [check for check in checks if not check["passed"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": decision,
        "candidate_method": candidate_method,
        "current_method": current_method or "",
        "baseline_method": baseline_method,
        "score_mode": "baseline-delta",
        "candidate_rank_score": candidate_score,
        "current_rank_score": current_score if math.isfinite(current_score) else None,
        "used_metrics": used_metrics,
        "checks": checks,
        "failed_checks": failed,
        "paired_objective": objective or {},
        "paired_objective_vs_current": objective_vs_current or {},
        "training_provenance": {
            "lora_weights": lora_weights,
            "training_report": training_report,
            "training_report_sha256": candidate.get("training_report_sha256") or split_audit.get("training_report_sha256", ""),
            "train_loss_recipe": candidate.get("train_loss_recipe") or split_audit.get("train_loss_recipe", ""),
            "train_metadata_sha256": candidate.get("train_metadata_sha256") or split_audit.get("train_metadata_sha256", ""),
            "adapter_sha256": candidate.get("adapter_sha256") or split_audit.get("adapter_sha256", ""),
            "train_prompt_family": candidate.get("train_prompt_family") or split_audit.get("train_prompt_family", ""),
            "eval_prompt_family": split_audit.get("eval_prompt_family", ""),
            "prompt_family_matches_eval": split_audit.get("prompt_family_matches_eval", ""),
            "prompt_family_mismatch_detail": split_audit.get("prompt_family_mismatch_detail", ""),
        },
        "ranked_methods": ranked_rows,
    }


def decision_markdown(decision: dict, input_dir: Path) -> str:
    checks = [
        [
            check["name"],
            "yes" if check["passed"] else "no",
            format_number(check["value"]),
            check["threshold"],
            check.get("detail", ""),
        ]
        for check in decision["checks"]
    ]
    ranked = [
        [row.get("method", ""), format_number(row.get("rank_score")), format_number(row.get("success_rate"))]
        for row in decision["ranked_methods"]
    ]
    paired = decision.get("paired_objective") or {}
    paired_current = decision.get("paired_objective_vs_current") or {}
    training_provenance = decision.get("training_provenance") or {}
    paired_rows = []
    if paired:
        paired_rows.append(
            [
                paired.get("method", ""),
                format_number(paired.get("paired_n")),
                f"{paired.get('win_count', '')}/{paired.get('paired_n', '')}",
                format_number(paired.get("win_rate")),
                format_number(paired.get("mean_improvement")),
                format_number(paired.get("ci95_low")),
                format_number(paired.get("ci95_high")),
            ]
        )
    paired_current_rows = []
    if paired_current:
        paired_current_rows.append(
            [
                paired_current.get("method", ""),
                format_number(paired_current.get("paired_n")),
                f"{paired_current.get('win_count', '')}/{paired_current.get('paired_n', '')}",
                format_number(paired_current.get("win_rate")),
                format_number(paired_current.get("mean_improvement")),
                format_number(paired_current.get("ci95_low")),
                format_number(paired_current.get("ci95_high")),
            ]
        )

    lines = [
        f"# Completion Candidate Decision: {decision['candidate_method']}",
        "",
        f"- Input: `{input_dir}`",
        f"- Decision: `{decision['decision']}`",
        f"- Candidate: `{decision['candidate_method']}`",
        f"- Current: `{decision['current_method'] or '<none>'}`",
        f"- Baseline: `{decision['baseline_method']}`",
        f"- Score profile: `{decision.get('score_profile', 'default')}`",
        f"- Candidate score: `{format_number(decision['candidate_rank_score'])}`",
        f"- Current score: `{format_number(decision['current_rank_score'])}`",
        "",
        "## Gate Checks",
        "",
        markdown_table(["Check", "Passed", "Value", "Threshold", "Detail"], checks),
        "",
        "## Paired Objective",
        "",
        markdown_table(["Method", "Paired n", "Wins", "Win Rate", "Mean Improvement", "CI95 Low", "CI95 High"], paired_rows),
        "",
    ]
    if paired_current_rows:
        lines.extend(
            [
                "## Paired Objective vs Current",
                "",
                markdown_table(["Method", "Paired n", "Wins", "Win Rate", "Mean Improvement", "CI95 Low", "CI95 High"], paired_current_rows),
                "",
            ]
        )
    if training_provenance.get("lora_weights") or training_provenance.get("training_report"):
        lines.extend(
            [
                "",
                "## Training Provenance",
                "",
                markdown_table(
                    ["LoRA", "Report", "Recipe", "Metadata SHA256", "Adapter SHA256", "Train Prompt", "Eval Prompt", "Prompt Match", "Detail"],
                    [
                        [
                            training_provenance.get("lora_weights", ""),
                            training_provenance.get("training_report", ""),
                            training_provenance.get("train_loss_recipe", ""),
                            training_provenance.get("train_metadata_sha256", ""),
                            training_provenance.get("adapter_sha256", ""),
                            training_provenance.get("train_prompt_family", ""),
                            training_provenance.get("eval_prompt_family", ""),
                            training_provenance.get("prompt_family_matches_eval", ""),
                            training_provenance.get("prompt_family_mismatch_detail", ""),
                        ]
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
        "## Ranked Methods",
        "",
        markdown_table(["Method", "Baseline-Delta Score", "Success Rate"], ranked),
        "",
        ]
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Choose whether a completion method should be promoted from benchmark evidence.")
    parser.add_argument("input_dir", help="Run or optimize_completion directory.")
    parser.add_argument("--candidate-method", default=None, help="Method to evaluate. Defaults to the top non-baseline method.")
    parser.add_argument("--current-method", default=None, help="Current/default method to beat before promotion.")
    parser.add_argument("--baseline-method", default="masked")
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
    parser.add_argument("--min-stl-is-volume", type=float, default=1.0)
    parser.add_argument("--min-stl-is-manifold", type=float, default=1.0)
    parser.add_argument("--min-stl-winding-consistent", type=float, default=1.0)
    parser.add_argument("--min-stl-positive-volume", type=float, default=1.0)
    parser.add_argument("--min-stl-single-component", type=float, default=1.0)
    parser.add_argument("--min-stl-bbox-has-volume", type=float, default=1.0)
    parser.add_argument("--max-stl-nonmanifold-edge-count-log1p", type=float, default=0.0)
    parser.add_argument("--max-stl-degenerate-face-ratio", type=float, default=0.0)
    parser.add_argument("--max-stl-component-excess-log1p", type=float, default=0.0)
    parser.add_argument("--max-stl-bbox-aspect-ratio", type=float, default=10.0)
    parser.add_argument("--max-stl-faces-per-bbox-volume-log1p", type=float, default=10.0)
    parser.add_argument("--max-mesh-surface-chamfer-ratio-vs-current", type=float, default=1.1)
    parser.add_argument("--max-mesh-surface-hausdorff95-ratio-vs-current", type=float, default=1.1)
    parser.add_argument("--max-train-eval-overlap", type=int, default=0)
    parser.add_argument("--allow-missing-split-audit", action="store_true", help="Do not fail the promotion gate when split_audit.json is absent.")
    parser.add_argument("--paired-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--paired-bootstrap-seed", type=int, default=1234)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    summary_rows, summary_path = load_summary_rows(input_dir)
    per_sample_rows = load_per_sample_rows(input_dir)
    if not per_sample_rows:
        raise FileNotFoundError(f"No per_sample_metrics.csv rows found in {input_dir}")

    candidate_method = args.candidate_method
    if not candidate_method:
        ranked_rows, _ = rank_summary_rows(
            summary_rows,
            parse_weights(args.weight or [], profile=args.score_profile),
            score_mode="baseline-delta",
            baseline_method=args.baseline_method,
        )
        candidate_method = next(
            (
                row.get("method")
                for row in ranked_rows
                if row.get("method") != args.baseline_method and not is_oracle_diagnostic(row)
            ),
            None,
        )

    split_audit = load_split_audit(input_dir, candidate_method or "")
    decision = evaluate_selection(
        summary_rows,
        per_sample_rows,
        baseline_method=args.baseline_method,
        candidate_method=candidate_method,
        current_method=args.current_method,
        weights=parse_weights(args.weight or [], profile=args.score_profile),
        min_success_rate=args.min_success_rate,
        min_paired_n=args.min_paired_n,
        min_win_rate=args.min_win_rate,
        min_ci95_low=args.min_ci95_low,
        min_score_margin=args.min_score_margin,
        min_stl_watertight=args.min_stl_watertight,
        min_stl_is_volume=args.min_stl_is_volume,
        min_stl_is_manifold=args.min_stl_is_manifold,
        min_stl_winding_consistent=args.min_stl_winding_consistent,
        min_stl_positive_volume=args.min_stl_positive_volume,
        min_stl_single_component=args.min_stl_single_component,
        min_stl_bbox_has_volume=args.min_stl_bbox_has_volume,
        max_stl_nonmanifold_edge_count_log1p=args.max_stl_nonmanifold_edge_count_log1p,
        max_stl_degenerate_face_ratio=args.max_stl_degenerate_face_ratio,
        max_stl_component_excess_log1p=args.max_stl_component_excess_log1p,
        max_stl_bbox_aspect_ratio=args.max_stl_bbox_aspect_ratio,
        max_stl_faces_per_bbox_volume_log1p=args.max_stl_faces_per_bbox_volume_log1p,
        max_mesh_surface_chamfer_ratio_vs_current=args.max_mesh_surface_chamfer_ratio_vs_current,
        max_mesh_surface_hausdorff95_ratio_vs_current=args.max_mesh_surface_hausdorff95_ratio_vs_current,
        max_train_eval_overlap=args.max_train_eval_overlap,
        split_audit=split_audit,
        require_split_audit=not args.allow_missing_split_audit,
        bootstrap_samples=max(args.paired_bootstrap_samples, 0),
        bootstrap_seed=args.paired_bootstrap_seed,
    )
    decision["input_dir"] = str(input_dir)
    decision["summary_path"] = str(summary_path)
    decision["score_profile"] = args.score_profile

    output_json = Path(args.output_json) if args.output_json else input_dir / "selection_decision.json"
    output_md = Path(args.output_md) if args.output_md else input_dir / "selection_decision.md"
    output_json.write_text(json.dumps(json_safe(decision), indent=2, allow_nan=False), encoding="utf-8")
    output_md.write_text(decision_markdown(decision, input_dir), encoding="utf-8")
    print(f"{decision['decision']}: {decision['candidate_method']}")
    if decision["failed_checks"]:
        print("failed_checks:", ",".join(check["name"] for check in decision["failed_checks"]))
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
