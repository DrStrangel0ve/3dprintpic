from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmark.cache_provider import provider_plan
from backend.benchmark.direct_mesh import DIRECT_MESH_INPUT_MODES, MESH_REPAIR_MODES, is_direct_mesh_method
from backend.benchmark.make_artifact_contact_sheet import make_contact_sheet, parse_csv_arg
from backend.benchmark.preflight_image_to_mesh_providers import (
    preflight_experiments as preflight_image_to_mesh_experiments,
    write_preflight_report as write_image_to_mesh_preflight_report,
)
from backend.benchmark.report_run import (
    baseline_delta_rows,
    baseline_delta_table,
    has_method,
    paired_baseline_delta_rows,
    paired_baseline_delta_table,
    paired_objective_rows,
    paired_objective_table,
)
from backend.benchmark.rank_methods import (
    SCORE_MODES,
    SCORE_PROFILE_DESCRIPTIONS,
    SCORE_PROFILES,
    parse_weights,
    rank_summary_rows,
)
from backend.benchmark.stl_modes import experiment_stl_mode, validate_stl_mode
from backend.benchmark.select_completion_candidate import decision_markdown, evaluate_selection, json_safe
from backend.pic_to_3d import MODERN_INPAINT_MODELS


DEFAULT_EXPERIMENTS = [
    {"name": "masked", "method": "masked"},
    {"name": "mirror", "method": "mirror"},
    {"name": "mirror-seam-repair", "method": "mirror-seam-repair"},
    {"name": "biharmonic", "method": "biharmonic"},
]


REPORT_METRICS = [
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
    ("mesh_surface_chamfer_l1_median", "Mesh Surface Chamfer"),
    ("mesh_surface_chamfer_rmse_median", "Mesh Surface Chamfer RMSE"),
    ("mesh_surface_hausdorff95_median", "Mesh Surface Hausdorff95"),
    ("silhouette_iou_masked_median", "Silhouette IoU"),
    ("stl_is_watertight_median", "STL Watertight"),
    ("stl_is_volume_median", "STL Volume Mesh"),
    ("stl_winding_consistent_median", "STL Winding"),
    ("stl_positive_volume_median", "STL Positive Volume"),
    ("stl_single_component_median", "STL Single Body"),
    ("stl_component_count_median", "STL Bodies"),
    ("stl_component_excess_median", "STL Body Excess"),
    ("stl_component_excess_log1p_median", "STL Body Excess log1p"),
    ("stl_bbox_min_dimension_median", "STL Min Dimension"),
    ("stl_bbox_has_volume_median", "STL 3D BBox"),
    ("stl_bbox_aspect_ratio_median", "STL Aspect"),
    ("stl_faces_per_bbox_volume_log1p_median", "STL Face Density log1p"),
]


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.check_call(command)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_experiments(path: str | None, include_baselines: bool = False) -> list[dict]:
    if not path:
        data = list(DEFAULT_EXPERIMENTS)
    else:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Experiment config must be a JSON list")
    if include_baselines:
        existing = {experiment.get("name", experiment.get("method")) for experiment in data}
        baselines = [experiment for experiment in DEFAULT_EXPERIMENTS if experiment["name"] not in existing]
        data = baselines + data
    for index, experiment in enumerate(data):
        if "method" not in experiment:
            raise ValueError(f"Experiment {index} is missing required field 'method'")
        validate_stl_mode(experiment.get("stl_mode", ""))
        experiment.setdefault("name", experiment["method"])
    return data


def append_optional(command: list[str], flag: str, value) -> None:
    if value is not None and value != "":
        command.extend([flag, str(value)])


def bool_arg(args, experiment: dict, name: str) -> bool:
    return bool(experiment.get(name, getattr(args, name)))


def modern_cache_key(args, experiment: dict) -> tuple[str, str, bool, str, str] | None:
    provider = experiment.get("method", "")
    if provider not in MODERN_INPAINT_MODELS:
        return None
    model_name = experiment.get("model_name", args.model_name) or MODERN_INPAINT_MODELS[provider]["model"]
    full = bool(experiment.get("modern_cache_full", args.modern_cache_full))
    revision = args.modern_cache_revision or ""
    local_dir = args.modern_cache_local_dir or ""
    return (provider, model_name, full, revision, local_dir)


def missing_cache_detail(plan: dict) -> str:
    provider = plan.get("provider", "")
    model = plan.get("model", "")
    if plan.get("error"):
        return f"{provider} ({model}) preflight error: {plan.get('error_type', 'error')}: {plan.get('error')}"
    status = plan.get("cache_status", {})
    missing = status.get("missing_files", []) or []
    missing_count = status.get("missing_file_count", len(missing))
    missing_size = status.get("missing_size_gb", "")
    paths = ", ".join(item.get("path", "") for item in missing[:3])
    if len(missing) > 3:
        paths += f", ... (+{len(missing) - 3} more)"
    size_text = f" / {missing_size} GB" if missing_size != "" else ""
    path_text = f": {paths}" if paths else ""
    return f"{provider} ({model}) missing {missing_count} files{size_text}{path_text}"


def write_modern_cache_preflight(args, experiments: list[dict], output_dir: Path, planner=provider_plan) -> Path | None:
    keyed_experiments: dict[tuple[str, str, bool, str, str], list[str]] = {}
    for experiment in experiments:
        key = modern_cache_key(args, experiment)
        if key is None:
            continue
        keyed_experiments.setdefault(key, []).append(experiment.get("name", experiment.get("method", "")))
    if not keyed_experiments:
        return None

    plans = []
    for (provider, model_name, full, revision, local_dir), names in keyed_experiments.items():
        try:
            plan = planner(
                provider,
                full=full,
                revision=revision or None,
                local_dir=local_dir or None,
                model_name=model_name,
            )
        except Exception as exc:
            plan = {
                "provider": provider,
                "model": model_name,
                "full": full,
                "cache_status": {"revision": revision or "main", "local_dir": local_dir, "complete": False},
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        plan["experiment_names"] = names
        plans.append(plan)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "require_complete": True,
        "plans": plans,
    }
    path = output_dir / "modern_cache_preflight.json"
    path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")

    incomplete = [plan for plan in plans if not plan.get("cache_status", {}).get("complete")]
    if incomplete:
        details = "; ".join(missing_cache_detail(plan) for plan in incomplete)
        raise RuntimeError(
            "Modern provider cache incomplete. Run `python -m backend.benchmark.cache_provider "
            "<provider> --download` before a scored sweep, or omit --require-modern-cache. "
            f"Details: {details}. See {path}"
        )
    return path


def write_image_to_mesh_provider_preflight(
    args,
    experiments: list[dict],
    output_dir: Path,
    preflight=preflight_image_to_mesh_experiments,
) -> Path | None:
    rows = preflight(experiments)
    if not rows:
        return None
    path = output_dir / "image_to_mesh_provider_preflight.json"
    write_image_to_mesh_preflight_report(
        path,
        rows,
        require_runnable=bool(getattr(args, "require_image_to_mesh_providers", False)),
    )
    missing = [row for row in rows if not row.get("runnable")]
    if missing and getattr(args, "require_image_to_mesh_providers", False):
        details = "; ".join(
            f"{row.get('provider', '<unknown>')}: {', '.join(row.get('setup_errors') or ['setup incomplete'])}"
            for row in missing
        )
        raise RuntimeError(
            "Image-to-mesh provider setup incomplete. Configure provider repos/Python before a scored direct-mesh sweep, "
            "or omit --require-image-to-mesh-providers. "
            f"Details: {details}. See {path}"
        )
    return path


def run_experiment(args, experiment: dict, output_dir: Path) -> Path:
    experiment_dir = output_dir / experiment["name"]
    command = [
        sys.executable,
        "-m",
        "backend.benchmark.run_completion_benchmark",
        "--manifest",
        args.manifest,
        "--output-dir",
        str(experiment_dir),
        "--methods",
        experiment["method"],
        "--limit",
        str(args.limit),
        "--start-index",
        str(experiment.get("start_index", args.start_index)),
        "--depth-provider",
        args.depth_provider,
        "--depth-model",
        args.depth_model,
        "--device",
        args.device,
    ]
    if bool_arg(args, experiment, "skip_depth"):
        command.append("--skip-depth")
    if bool_arg(args, experiment, "emit_stl"):
        command.append("--emit-stl")
    if bool_arg(args, experiment, "stl_no_invert"):
        command.append("--stl-no-invert")
    if args.resume:
        command.append("--resume")
    if args.continue_on_error:
        command.append("--continue-on-error")

    append_optional(command, "--prompt", experiment.get("prompt", args.prompt))
    append_optional(command, "--steps", experiment.get("steps", args.steps))
    append_optional(command, "--guidance", experiment.get("guidance", args.guidance))
    append_optional(command, "--seed", experiment.get("seed", args.seed))
    append_optional(command, "--inpaint-max-dimension", experiment.get("inpaint_max_dimension", args.inpaint_max_dimension))
    append_optional(command, "--edit-mask-fill", experiment.get("edit_mask_fill", getattr(args, "edit_mask_fill", "input")))
    append_optional(command, "--model-name", experiment.get("model_name", args.model_name))
    append_optional(command, "--lora-weights", experiment.get("lora_weights", args.lora_weights))
    append_optional(command, "--lora-scale", experiment.get("lora_scale", args.lora_scale))
    append_optional(command, "--stl-target-dimension", experiment.get("stl_target_dimension", args.stl_target_dimension))
    append_optional(command, "--stl-z-scale", experiment.get("stl_z_scale", args.stl_z_scale))
    append_optional(command, "--stl-sigma", experiment.get("stl_sigma", args.stl_sigma))
    append_optional(command, "--mesh-surface-max-points", experiment.get("mesh_surface_max_points", args.mesh_surface_max_points))
    append_optional(command, "--direct-mesh-input", experiment.get("direct_mesh_input", args.direct_mesh_input))
    append_optional(command, "--direct-mesh-command", experiment.get("direct_mesh_command", args.direct_mesh_command))
    append_optional(command, "--direct-mesh-output-ext", experiment.get("direct_mesh_output_ext", args.direct_mesh_output_ext))
    append_optional(command, "--direct-mesh-timeout", experiment.get("direct_mesh_timeout", args.direct_mesh_timeout))
    reference_output_dir = experiment.get("direct_mesh_reference_output_dir", args.direct_mesh_reference_output_dir)
    if reference_output_dir is None and is_direct_mesh_method(experiment["method"]):
        reference_output_dir = output_dir
    append_optional(command, "--direct-mesh-reference-output-dir", reference_output_dir)
    append_optional(
        command,
        "--direct-mesh-reference-method",
        experiment.get("direct_mesh_reference_method", args.direct_mesh_reference_method),
    )
    append_optional(command, "--source-mesh-repair", experiment.get("source_mesh_repair", args.source_mesh_repair))
    max_method_failures = experiment.get("max_method_failures", args.max_method_failures)
    if max_method_failures:
        append_optional(command, "--max-method-failures", max_method_failures)

    run(command)
    annotate_per_sample_metrics(
        experiment,
        experiment_dir,
        default_start_index=args.start_index,
        default_emit_stl=args.emit_stl,
    )
    return experiment_dir / "summary_metrics.csv"


def training_report_path(lora_weights: str | None) -> Path | None:
    if not lora_weights:
        return None
    path = Path(lora_weights)
    candidates = []
    if path.is_dir():
        candidates.append(path / "training_report.json")
    else:
        candidates.append(path.with_name("training_report.json"))
        candidates.append(path.parent / "training_report.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def training_metadata(experiment: dict) -> dict[str, object]:
    report_path = training_report_path(experiment.get("lora_weights"))
    if not report_path:
        return {}
    try:
        report = read_json(report_path)
    except (OSError, json.JSONDecodeError):
        return {"training_report": str(report_path)}
    return {
        "training_report": str(report_path),
        "training_report_sha256": file_sha256(report_path),
        "train_loss_recipe": report.get("loss_recipe", ""),
        "train_metadata_sha256": report.get("metadata_sha256", ""),
        "train_pair_export_sha256": report.get("pair_export_sha256", ""),
        "train_prompt_family": report.get("prompt_family", ""),
        "train_prompt_template": report.get("prompt_template", ""),
        "train_rows": report.get("rows", ""),
        "train_asset_count": report.get("asset_count", ""),
        "train_max_steps": report.get("max_train_steps", ""),
        "train_final_step": report.get("final_step", ""),
        "train_resolution": report.get("resolution", ""),
        "train_rank": report.get("rank", ""),
        "train_mask_loss_weight": report.get("mask_loss_weight", ""),
        "train_seam_loss_weight": report.get("seam_loss_weight", ""),
        "train_object_loss_weight": report.get("object_loss_weight", ""),
        "train_final_loss": report.get("final_loss", ""),
        "train_best_loss": report.get("best_loss", ""),
        "train_base_model": report.get("base_model", ""),
        "adapter_sha256": report.get("adapter_sha256", ""),
    }


def experiment_edit_mask_fill(experiment: dict, default_edit_mask_fill="") -> str:
    if experiment.get("method") not in MODERN_INPAINT_MODELS:
        return ""
    value = experiment.get("edit_mask_fill", default_edit_mask_fill)
    return "" if value in (None, "", "input") else value


def experiment_metadata(
    experiment: dict,
    default_start_index=0,
    default_edit_mask_fill="",
    default_emit_stl=False,
) -> dict[str, object]:
    metadata = {
        "method": experiment["name"],
        "base_method": experiment["method"],
        "stl_mode": experiment_stl_mode(experiment, default_emit_stl=default_emit_stl),
        "prompt": experiment.get("prompt", ""),
        "steps": experiment.get("steps", ""),
        "guidance": experiment.get("guidance", ""),
        "seed": experiment.get("seed", ""),
        "inpaint_max_dimension": experiment.get("inpaint_max_dimension", ""),
        "edit_mask_fill": experiment_edit_mask_fill(experiment, default_edit_mask_fill),
        "model_name": experiment.get("model_name", ""),
        "lora_weights": experiment.get("lora_weights", ""),
        "lora_scale": experiment.get("lora_scale", ""),
        "source_mesh_repair": experiment.get("source_mesh_repair", ""),
        "start_index": experiment.get("start_index", default_start_index),
    }
    metadata.update(training_metadata(experiment))
    return metadata


def per_sample_metadata(experiment: dict, default_start_index=0, default_emit_stl=False) -> dict[str, object]:
    return {
        "method": experiment["name"],
        "base_method": experiment["method"],
        "stl_mode": experiment_stl_mode(experiment, default_emit_stl=default_emit_stl),
        "source_mesh_repair": experiment.get("source_mesh_repair", ""),
        "start_index": experiment.get("start_index", default_start_index),
    }


def annotate_per_sample_metrics(
    experiment: dict,
    experiment_dir: Path,
    default_start_index=0,
    default_emit_stl=False,
) -> None:
    per_sample_path = experiment_dir / "per_sample_metrics.csv"
    if not per_sample_path.exists():
        return

    with per_sample_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        return

    metadata = per_sample_metadata(
        experiment,
        default_start_index=default_start_index,
        default_emit_stl=default_emit_stl,
    )
    for row in rows:
        row["base_method"] = row.get("base_method") or row.get("method", "") or experiment["method"]
        row.update(metadata)

    for key in metadata:
        if key not in fieldnames:
            fieldnames.append(key)
    if "base_method" not in fieldnames:
        insert_at = fieldnames.index("method") + 1 if "method" in fieldnames else len(fieldnames)
        fieldnames.insert(insert_at, "base_method")

    with per_sample_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_summaries(
    summaries: list[tuple[dict, Path]],
    output_dir: Path,
    default_start_index=0,
    default_edit_mask_fill="",
    default_emit_stl=False,
) -> Path:
    rows = []
    fieldnames = []
    for experiment, summary_path in summaries:
        with summary_path.open(encoding="utf-8") as file:
            for row in csv.DictReader(file):
                row.update(
                    experiment_metadata(
                        experiment,
                        default_start_index=default_start_index,
                        default_edit_mask_fill=default_edit_mask_fill,
                        default_emit_stl=default_emit_stl,
                    )
                )
                rows.append(row)
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)

    if not rows:
        raise ValueError("No experiment summaries were produced")

    aggregate_path = output_dir / "aggregate_summary.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return aggregate_path


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


def read_experiment_sample_rows(experiments: list[dict], output_dir: Path) -> list[dict]:
    rows = []
    for experiment in experiments:
        name = experiment.get("name", experiment.get("method", ""))
        metrics_path = output_dir / name / "per_sample_metrics.csv"
        for row in read_csv(metrics_path):
            row["base_method"] = row.get("base_method") or row.get("method", "")
            row["method"] = name
            rows.append(row)
    return rows


def format_cell(value, max_length: int | None = 120) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if max_length is not None and len(text) > max_length:
        text = text[: max_length - 1] + "..."
    return text.replace("|", "\\|")


def format_number(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return format_cell(value)
    if number != number:
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
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(item) for item in row) + " |")
    return "\n".join(lines)


def score_profile(args) -> str:
    return getattr(args, "score_profile", "default")


def score_mode_note(score_mode: str, baseline_method: str, profile: str = "default") -> list[str]:
    lines = [
        f"_Score profile: `{profile}` ({SCORE_PROFILE_DESCRIPTIONS.get(profile, 'custom metric weights')})._",
        "",
    ]
    if score_mode == "baseline-delta":
        lines.extend(
            [
            f"_Score mode: weighted sign-normalized metric improvement against `{baseline_method}`. The baseline scores `0`; positive scores improve the objective and negative scores regress it. Scores are stable when unrelated candidate methods are added._",
            "",
            ]
        )
        return lines
    lines.extend(
        [
        "_Score mode: weighted min/max normalization within this candidate set. Scores are useful for ordering one run, but the numeric value can change when candidates are added or removed._",
        "",
        ]
    )
    return lines


def experiment_with_provenance(experiment: dict) -> dict:
    row = dict(experiment)
    metadata = training_metadata(experiment)
    if metadata:
        row["training_provenance"] = metadata
    return row


def write_resolved_config(args, experiments: list[dict], output_dir: Path) -> Path:
    config = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "defaults": {
            "manifest": args.manifest,
            "limit": args.limit,
            "start_index": args.start_index,
            "skip_depth": args.skip_depth,
            "depth_provider": args.depth_provider,
            "depth_model": args.depth_model,
            "device": args.device,
            "emit_stl": args.emit_stl,
            "stl_mode": experiment_stl_mode({"method": "", "emit_stl": args.emit_stl}),
            "stl_target_dimension": args.stl_target_dimension,
            "stl_z_scale": args.stl_z_scale,
            "stl_sigma": args.stl_sigma,
            "stl_no_invert": args.stl_no_invert,
            "mesh_surface_max_points": getattr(args, "mesh_surface_max_points", 4096),
            "direct_mesh_input": getattr(args, "direct_mesh_input", "masked"),
            "direct_mesh_command": getattr(args, "direct_mesh_command", None),
            "direct_mesh_output_ext": getattr(args, "direct_mesh_output_ext", "glb"),
            "direct_mesh_timeout": getattr(args, "direct_mesh_timeout", 1800),
            "direct_mesh_reference_output_dir": getattr(args, "direct_mesh_reference_output_dir", None),
            "direct_mesh_reference_method": getattr(args, "direct_mesh_reference_method", "mirror"),
            "require_image_to_mesh_providers": getattr(args, "require_image_to_mesh_providers", False),
            "source_mesh_repair": getattr(args, "source_mesh_repair", "none"),
            "prompt": args.prompt,
            "steps": args.steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "inpaint_max_dimension": args.inpaint_max_dimension,
            "edit_mask_fill": getattr(args, "edit_mask_fill", "input"),
            "model_name": args.model_name,
            "lora_weights": args.lora_weights,
            "lora_scale": args.lora_scale,
            "score_mode": args.score_mode,
            "score_profile": score_profile(args),
            "baseline_method": args.baseline_method,
            "weights": args.weight or [],
            "paired_bootstrap_samples": args.paired_bootstrap_samples,
            "paired_bootstrap_seed": args.paired_bootstrap_seed,
            "selection": {
                "enabled": args.select_candidate,
                "candidate_method": args.candidate_method,
                "current_method": args.current_method,
                "min_success_rate": args.min_success_rate,
                "min_paired_n": args.min_paired_n,
                "min_win_rate": args.min_win_rate,
                "min_ci95_low": args.min_ci95_low,
                "min_score_margin": args.min_score_margin,
                "min_stl_watertight": args.min_stl_watertight,
                "min_stl_is_volume": args.min_stl_is_volume,
                "min_stl_is_manifold": args.min_stl_is_manifold,
                "min_stl_winding_consistent": args.min_stl_winding_consistent,
                "min_stl_positive_volume": args.min_stl_positive_volume,
                "min_stl_single_component": args.min_stl_single_component,
                "min_stl_bbox_has_volume": args.min_stl_bbox_has_volume,
                "max_stl_nonmanifold_edge_count_log1p": args.max_stl_nonmanifold_edge_count_log1p,
                "max_stl_degenerate_face_ratio": args.max_stl_degenerate_face_ratio,
                "max_stl_component_excess_log1p": args.max_stl_component_excess_log1p,
                "max_stl_bbox_aspect_ratio": args.max_stl_bbox_aspect_ratio,
                "max_stl_faces_per_bbox_volume_log1p": args.max_stl_faces_per_bbox_volume_log1p,
                "max_train_eval_overlap": args.max_train_eval_overlap,
                "allow_missing_split_audit": args.allow_missing_split_audit,
            },
        },
        "experiments": [experiment_with_provenance(experiment) for experiment in experiments],
    }
    path = output_dir / "resolved_experiments.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def write_experiment_report(
    args,
    experiments: list[dict],
    output_dir: Path,
    aggregate_path: Path,
    ranked_path: Path,
    contact_sheet_path: Path | None = None,
) -> Path:
    ranked_rows = read_csv(ranked_path)
    aggregate_rows = read_csv(aggregate_path)
    per_sample_rows = read_experiment_sample_rows(experiments, output_dir)
    available_metrics = [
        (field, label)
        for field, label in REPORT_METRICS
        if any(format_number(row.get(field)) for row in ranked_rows)
    ]

    ranking_rows = []
    for index, row in enumerate(ranked_rows, start=1):
        ranking_rows.append(
            [
                str(index),
                row.get("method", ""),
                format_number(row.get("rank_score")),
                format_number(row.get("n")),
                format_number(row.get("attempted_n")),
                format_number(row.get("success_rate")),
                format_number(row.get("error_count")),
                *[format_number(row.get(field)) for field, _ in available_metrics],
            ]
        )

    config_rows = []
    for experiment in experiments:
        config_rows.append(
            [
                experiment.get("name", experiment.get("method", "")),
                experiment.get("method", ""),
                experiment_stl_mode(experiment, default_emit_stl=args.emit_stl),
                experiment.get("steps", args.steps),
                experiment.get("guidance", args.guidance),
                experiment.get("seed", args.seed),
                experiment.get("inpaint_max_dimension", args.inpaint_max_dimension),
                experiment.get("edit_mask_fill", getattr(args, "edit_mask_fill", "input")),
                experiment.get("model_name", args.model_name),
                experiment.get("lora_weights", args.lora_weights),
                experiment.get("lora_scale", args.lora_scale),
                experiment.get("start_index", args.start_index),
                experiment.get("skip_depth", args.skip_depth),
                experiment.get("emit_stl", args.emit_stl),
                experiment.get("mesh_surface_max_points", getattr(args, "mesh_surface_max_points", 4096)),
                experiment.get("direct_mesh_input", getattr(args, "direct_mesh_input", "masked")),
                experiment.get("direct_mesh_output_ext", getattr(args, "direct_mesh_output_ext", "glb")),
                experiment.get("direct_mesh_timeout", getattr(args, "direct_mesh_timeout", 1800)),
                experiment.get("direct_mesh_reference_output_dir", getattr(args, "direct_mesh_reference_output_dir", None)) or "",
                experiment.get("direct_mesh_reference_method", getattr(args, "direct_mesh_reference_method", "mirror")),
                experiment.get("source_mesh_repair", getattr(args, "source_mesh_repair", "none")),
                experiment.get("direct_mesh_command", getattr(args, "direct_mesh_command", None)) or "",
                experiment.get("prompt", args.prompt if experiment.get("method") not in ("mirror", "biharmonic") else ""),
            ]
        )

    ranked_by_method = {row.get("method", ""): row for row in ranked_rows}
    training_rows = []
    for experiment in experiments:
        name = experiment.get("name", experiment.get("method", ""))
        metadata = training_metadata(experiment)
        if not metadata:
            continue
        ranked = ranked_by_method.get(name, {})
        training_rows.append(
            [
                name,
                format_number(ranked.get("rank_score")),
                metadata.get("train_base_model", ""),
                metadata.get("train_loss_recipe", ""),
                metadata.get("train_prompt_family", ""),
                format_number(metadata.get("train_rows")),
                format_number(metadata.get("train_asset_count")),
                format_number(metadata.get("train_max_steps")),
                format_number(metadata.get("train_final_step")),
                format_number(metadata.get("train_resolution")),
                format_number(metadata.get("train_rank")),
                format_number(metadata.get("train_mask_loss_weight")),
                format_number(metadata.get("train_seam_loss_weight")),
                format_number(metadata.get("train_object_loss_weight")),
                format_number(metadata.get("train_final_loss")),
                format_number(metadata.get("train_best_loss")),
                metadata.get("train_metadata_sha256", ""),
                metadata.get("adapter_sha256", ""),
                metadata.get("training_report", ""),
            ]
        )

    success_rates = {
        format_number(row.get("success_rate"))
        for row in ranked_rows
        if format_number(row.get("success_rate"))
    }
    coverage_note = []
    if len(success_rates) > 1:
        coverage_note = [
            "",
            "_Coverage warning: ranked methods have unequal success rates; compare medians with care._",
        ]

    failure_rows = []
    for row in aggregate_rows:
        error_count = format_number(row.get("error_count"))
        try:
            has_errors = float(row.get("error_count", 0)) > 0
        except (TypeError, ValueError):
            has_errors = False
        if has_errors:
            failure_rows.append([row.get("method", ""), error_count, row.get("last_error_type", ""), row.get("last_error", "")])

    audit_rows = []
    for experiment in experiments:
        name = experiment.get("name", experiment.get("method", ""))
        audit = read_json(output_dir / name / "split_audit.json")
        if not audit:
            continue
        audit_rows.append(
            [
                name,
                format_number(audit.get("eval_n")),
                format_number(audit.get("eval_asset_count")),
                format_number(audit.get("train_asset_count")),
                format_number(audit.get("train_eval_asset_overlap_count")),
                ", ".join(f"{key}:{value}" for key, value in sorted(audit.get("eval_category_counts", {}).items())),
                ", ".join(f"{key}:{value}" for key, value in sorted(audit.get("train_category_counts", {}).items())),
                ", ".join(f"{key}:{value}" for key, value in sorted(audit.get("eval_source_split_counts", {}).items())),
                ", ".join(f"{key}:{value}" for key, value in sorted(audit.get("train_source_split_counts", {}).items())),
            ]
        )

    output_path = output_dir / "experiment_report.md"
    lines = [
        f"# Experiment Report: {output_dir.name}",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- Manifest: `{args.manifest}`",
        f"- Limit: `{args.limit}`",
        f"- Aggregate summary: `{aggregate_path}`",
        f"- Ranked output: `{ranked_path}`",
        f"- Resolved config: `{output_dir / 'resolved_experiments.json'}`",
    ]
    if contact_sheet_path:
        lines.append(f"- Contact sheet: `{contact_sheet_path}`")
    lines.extend(
        [
            "",
            "## Ranking",
            "",
            *score_mode_note(args.score_mode, args.baseline_method, score_profile(args)),
            markdown_table(
                ["Rank", "Method", "Score", "n", "Attempted", "Success Rate", "Errors", *[label for _, label in available_metrics]],
                ranking_rows,
            ),
            *coverage_note,
            "",
            f"## Selected Baseline Deltas: `{args.baseline_method}`",
            "",
            "Raw `Delta` is `method - baseline`; `Improvement` is sign-normalized so positive means better. These are aggregate median deltas for selected diagnostic metrics, not weighted rank points or paired per-sample wins.",
            "",
            baseline_delta_table(baseline_delta_rows(aggregate_rows, baseline_method=args.baseline_method), args.baseline_method),
            "",
            f"## Paired Baseline Wins: `{args.baseline_method}`",
            "",
            "Each row compares a method against the baseline on the same `sample_id`. `Wins` count strictly positive sign-normalized improvements; ties are reported separately.",
            "",
            paired_baseline_delta_table(
                paired_baseline_delta_rows(per_sample_rows, baseline_method=args.baseline_method),
                args.baseline_method,
                baseline_present=has_method(per_sample_rows, args.baseline_method),
            ),
            "",
            f"## Paired Objective Confidence: `{args.baseline_method}`",
            "",
            "Each row applies the same metric weights used for ranking to each shared `sample_id`, but as raw sign-normalized improvement over the baseline rather than candidate-normalized aggregate medians. The confidence interval bootstraps paired sample improvements with a fixed seed; if the CI is entirely above `0`, the method has a stronger promotion signal than an aggregate median alone.",
            "",
            paired_objective_table(
                paired_objective_rows(
                    per_sample_rows,
                    parse_weights(args.weight or [], profile=score_profile(args)),
                    baseline_method=args.baseline_method,
                    bootstrap_samples=max(args.paired_bootstrap_samples, 0),
                    bootstrap_seed=args.paired_bootstrap_seed,
                ),
                args.baseline_method,
                baseline_present=has_method(per_sample_rows, args.baseline_method),
            ),
            "",
            "## Experiment Config",
            "",
            markdown_table(
                [
                    "Name",
                    "Base Method",
                    "STL Mode",
                    "Steps",
                    "Guidance",
                    "Seed",
                    "Max Dim",
                    "Edit Fill",
                    "Model",
                    "LoRA",
                    "LoRA Scale",
                    "Start",
                    "Skip Depth",
                    "Emit STL",
                    "Mesh Samples",
                    "Direct Input",
                    "Direct Ext",
                    "Direct Timeout",
                    "Direct Ref Root",
                    "Direct Ref Method",
                    "Source Repair",
                    "Direct Command",
                    "Prompt",
                ],
                config_rows,
            ),
            "",
            "## Training Recipes",
            "",
            "LoRA candidates with `training_report.json` are expanded here so benchmark scores can be traced back to the actual training recipe instead of only the adapter path.",
            "",
            markdown_table(
                [
                    "Method",
                    "Score",
                    "Base Model",
                    "Recipe",
                    "Prompt Family",
                    "Rows",
                    "Assets",
                    "Steps",
                    "Final Step",
                    "Resolution",
                    "Rank",
                    "Mask W",
                    "Seam W",
                    "Object W",
                    "Final Loss",
                    "Best Loss",
                    "Metadata SHA256",
                    "Adapter SHA256",
                    "Report",
                ],
                training_rows,
            ),
            "",
            "## Failures",
            "",
            markdown_table(["Method", "Failures", "Last Error Type", "Last Error"], failure_rows),
            "",
            "## Split Audit",
            "",
            markdown_table(
                [
                    "Method",
                    "Eval n",
                    "Eval Assets",
                    "Train Assets",
                    "Train/Eval Overlap",
                    "Eval Categories",
                    "Train Categories",
                    "Eval Source Splits",
                    "Train Source Splits",
                ],
                audit_rows,
            ),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def infer_selection_candidate(args, aggregate_rows: list[dict]) -> str | None:
    if args.candidate_method:
        return args.candidate_method
    ranked_rows, _ = rank_summary_rows(
        aggregate_rows,
        parse_weights(args.weight or [], profile=score_profile(args)),
        score_mode="baseline-delta",
        baseline_method=args.baseline_method,
    )
    return next((row.get("method") for row in ranked_rows if row.get("method") != args.baseline_method), None)


def write_selection_decision(args, output_dir: Path, aggregate_rows: list[dict], per_sample_rows: list[dict]) -> tuple[Path, Path]:
    candidate_method = infer_selection_candidate(args, aggregate_rows)
    split_audit = read_json(output_dir / (candidate_method or "") / "split_audit.json") or read_json(output_dir / "split_audit.json")
    decision = evaluate_selection(
        aggregate_rows,
        per_sample_rows,
        baseline_method=args.baseline_method,
        candidate_method=candidate_method,
        current_method=args.current_method,
        weights=parse_weights(args.weight or [], profile=score_profile(args)),
        min_success_rate=args.min_success_rate,
        min_paired_n=args.min_paired_n,
        min_win_rate=args.min_win_rate,
        min_ci95_low=args.min_ci95_low,
        min_score_margin=args.min_score_margin,
        min_stl_watertight=getattr(args, "min_stl_watertight", 1.0),
        min_stl_is_volume=getattr(args, "min_stl_is_volume", 1.0),
        min_stl_is_manifold=getattr(args, "min_stl_is_manifold", 1.0),
        min_stl_winding_consistent=getattr(args, "min_stl_winding_consistent", 1.0),
        min_stl_positive_volume=getattr(args, "min_stl_positive_volume", 1.0),
        min_stl_single_component=getattr(args, "min_stl_single_component", 1.0),
        min_stl_bbox_has_volume=getattr(args, "min_stl_bbox_has_volume", 1.0),
        max_stl_nonmanifold_edge_count_log1p=getattr(args, "max_stl_nonmanifold_edge_count_log1p", 0.0),
        max_stl_degenerate_face_ratio=getattr(args, "max_stl_degenerate_face_ratio", 0.0),
        max_stl_component_excess_log1p=getattr(args, "max_stl_component_excess_log1p", 0.0),
        max_stl_bbox_aspect_ratio=getattr(args, "max_stl_bbox_aspect_ratio", 10.0),
        max_stl_faces_per_bbox_volume_log1p=getattr(args, "max_stl_faces_per_bbox_volume_log1p", 10.0),
        max_train_eval_overlap=args.max_train_eval_overlap,
        split_audit=split_audit,
        require_split_audit=not args.allow_missing_split_audit,
        bootstrap_samples=max(args.paired_bootstrap_samples, 0),
        bootstrap_seed=args.paired_bootstrap_seed,
    )
    decision["input_dir"] = str(output_dir)
    decision["summary_path"] = str(output_dir / "aggregate_summary.csv")
    decision["score_profile"] = score_profile(args)
    output_json = output_dir / "selection_decision.json"
    output_md = output_dir / "selection_decision.md"
    output_json.write_text(json.dumps(json_safe(decision), indent=2, allow_nan=False), encoding="utf-8")
    output_md.write_text(decision_markdown(decision, output_dir), encoding="utf-8")
    return output_json, output_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and rank completion experiments against a benchmark manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="backend/output/completion-benchmark/experiments/default")
    parser.add_argument("--config", help="JSON list of experiments with method/name/prompt/steps/guidance/seed")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based manifest row offset for held-out slices.")
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--depth-provider", default="depth-anything-v2")
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--emit-stl", action="store_true", help="Pass --emit-stl to benchmark runs. Requires depth generation.")
    parser.add_argument("--stl-target-dimension", type=int, default=160)
    parser.add_argument("--stl-z-scale", type=float, default=50.0)
    parser.add_argument("--stl-sigma", type=float, default=4.0)
    parser.add_argument("--mesh-surface-max-points", type=int, default=4096)
    parser.add_argument("--direct-mesh-input", choices=DIRECT_MESH_INPUT_MODES, default="masked")
    parser.add_argument("--direct-mesh-command", default=None)
    parser.add_argument("--direct-mesh-output-ext", default="glb")
    parser.add_argument("--direct-mesh-timeout", type=int, default=1800)
    parser.add_argument(
        "--direct-mesh-reference-output-dir",
        default=None,
        help=(
            "Optional sweep output root used by direct-mesh candidates to resolve reference STL bbox "
            "placeholders. Defaults to --output-dir for direct-mesh experiments."
        ),
    )
    parser.add_argument("--direct-mesh-reference-method", default="mirror")
    parser.add_argument("--source-mesh-repair", choices=MESH_REPAIR_MODES, default="none")
    parser.add_argument("--stl-no-invert", action="store_true")
    parser.add_argument("--prompt", default="Complete the missing half naturally, preserving the same object, lighting, viewpoint, and background.")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--inpaint-max-dimension", type=int, default=768)
    parser.add_argument("--edit-mask-fill", choices=("input", "white", "gray", "checker", "mirror", "biharmonic"), default="input")
    parser.add_argument("--model-name", default=None, help="Optional base model override for modern completion providers.")
    parser.add_argument("--lora-weights", default=None, help="Optional Diffusers LoRA adapter directory or safetensors file.")
    parser.add_argument("--lora-scale", type=float, default=None)
    parser.add_argument("--include-baselines", action="store_true", help="Prepend mirror and biharmonic when a custom config is supplied.")
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
        help="Baseline method used for baseline-delta scoring and report delta tables.",
    )
    parser.add_argument("--weight", action="append", default=[], help="Override ranking weight as metric=value, same syntax as rank_methods.")
    parser.add_argument("--paired-bootstrap-samples", type=int, default=1000, help="Bootstrap resamples for paired objective confidence intervals.")
    parser.add_argument("--paired-bootstrap-seed", type=int, default=1234, help="Random seed for paired objective confidence intervals.")
    parser.add_argument("--contact-sheet", action="store_true", help="Also write a PNG artifact contact sheet for the sweep.")
    parser.add_argument("--contact-sheet-output", default=None, help="PNG contact sheet path. Defaults to <output-dir>/artifact_contact_sheet.png.")
    parser.add_argument("--contact-sheet-methods", default=None, help="Comma-separated experiment, method, or base-method names for the contact sheet.")
    parser.add_argument("--contact-sheet-samples", default=None, help="Comma-separated sample_id list for the contact sheet.")
    parser.add_argument("--contact-sheet-max-samples", type=int, default=6, help="Maximum samples in the contact sheet when --contact-sheet-samples is omitted.")
    parser.add_argument("--contact-sheet-thumb-size", type=int, default=150, help="Thumbnail size in pixels for the contact sheet.")
    parser.add_argument("--require-modern-cache", action="store_true", help="Fail fast if any modern inpainting provider in the sweep has missing planned cache files.")
    parser.add_argument("--modern-cache-full", action="store_true", help="With --require-modern-cache, require every provider repo file instead of the fp16 benchmark subset.")
    parser.add_argument("--modern-cache-revision", default=None, help="Optional provider cache revision checked by --require-modern-cache.")
    parser.add_argument("--modern-cache-local-dir", default=None, help="Optional local cache directory checked by --require-modern-cache.")
    parser.add_argument(
        "--require-image-to-mesh-providers",
        action="store_true",
        help="Fail fast if any run_image_to_mesh_provider direct-mesh command has missing provider setup.",
    )
    parser.add_argument("--select-candidate", action="store_true", help="Write selection_decision.json/.md after ranking the sweep.")
    parser.add_argument("--candidate-method", default=None, help="Candidate method to evaluate for selection. Defaults to the top non-baseline method.")
    parser.add_argument("--current-method", default=None, help="Current/default method that a candidate must beat before promotion.")
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
    parser.add_argument("--max-train-eval-overlap", type=int, default=0)
    parser.add_argument("--allow-missing-split-audit", action="store_true", help="Do not fail candidate selection when split_audit.json is absent.")
    parser.add_argument(
        "--max-method-failures",
        type=int,
        default=0,
        help="Forward a per-method failure cap to run_completion_benchmark. 0 disables the cap.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_method_failures < 0:
        raise ValueError("--max-method-failures must be non-negative")
    if args.mesh_surface_max_points <= 0:
        raise ValueError("--mesh-surface-max-points must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments(args.config, include_baselines=args.include_baselines)
    if args.emit_stl and args.skip_depth and not all(is_direct_mesh_method(experiment["method"]) for experiment in experiments):
        raise ValueError("--emit-stl requires depth generation for non-direct-mesh experiments; remove --skip-depth")
    for experiment in experiments:
        if (
            bool_arg(args, experiment, "emit_stl")
            and bool_arg(args, experiment, "skip_depth")
            and not is_direct_mesh_method(experiment["method"])
        ):
            raise ValueError(f"{experiment['name']}: emit_stl requires depth generation; remove skip_depth")
    write_resolved_config(args, experiments, output_dir)
    write_image_to_mesh_provider_preflight(args, experiments, output_dir)
    if args.require_modern_cache:
        write_modern_cache_preflight(args, experiments, output_dir)
    summaries = [(experiment, run_experiment(args, experiment, output_dir)) for experiment in experiments]
    aggregate_path = aggregate_summaries(
        summaries,
        output_dir,
        default_start_index=args.start_index,
        default_edit_mask_fill=args.edit_mask_fill,
        default_emit_stl=args.emit_stl,
    )

    ranked_path = output_dir / "ranked_experiments.csv"
    rank_command = [
        sys.executable,
        "-m",
        "backend.benchmark.rank_methods",
        "--summary",
        str(aggregate_path),
        "--output",
        str(ranked_path),
        "--score-mode",
        args.score_mode,
        "--score-profile",
        args.score_profile,
        "--baseline-method",
        args.baseline_method,
    ]
    for weight in args.weight or []:
        rank_command.extend(["--weight", weight])
    run(rank_command)
    explanation_dir = output_dir / "rank_score_explanation"
    explanation_command = [
        sys.executable,
        "-m",
        "backend.benchmark.explain_rank_score",
        "--summary",
        str(aggregate_path),
        "--output-dir",
        str(explanation_dir),
        "--score-mode",
        args.score_mode,
        "--score-profile",
        args.score_profile,
        "--baseline-method",
        args.baseline_method,
    ]
    for weight in args.weight or []:
        explanation_command.extend(["--weight", weight])
    run(explanation_command)
    contact_sheet_path = None
    if args.contact_sheet:
        contact_sheet_path = Path(args.contact_sheet_output) if args.contact_sheet_output else output_dir / "artifact_contact_sheet.png"
        make_contact_sheet(
            output_dir,
            output_path=contact_sheet_path,
            methods=parse_csv_arg(args.contact_sheet_methods),
            samples=parse_csv_arg(args.contact_sheet_samples),
            max_samples=max(args.contact_sheet_max_samples, 0),
            thumb_size=max(args.contact_sheet_thumb_size, 32),
        )
    report_path = write_experiment_report(
        args,
        experiments,
        output_dir,
        aggregate_path,
        ranked_path,
        contact_sheet_path=contact_sheet_path,
    )
    if args.select_candidate:
        aggregate_rows = read_csv(aggregate_path)
        per_sample_rows = read_experiment_sample_rows(experiments, output_dir)
        selection_json, selection_md = write_selection_decision(args, output_dir, aggregate_rows, per_sample_rows)
        print(selection_json)
        print(selection_md)
    print(aggregate_path)
    print(report_path)
    print(explanation_dir / "rank_score_explanation.md")
    if contact_sheet_path:
        print(contact_sheet_path)


if __name__ == "__main__":
    main()
