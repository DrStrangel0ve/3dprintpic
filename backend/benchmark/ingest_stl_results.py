from __future__ import annotations

import argparse
import json
import re
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmark.rank_methods import (
    SCORE_MODES,
    SCORE_PROFILES,
    parse_float,
    parse_weights,
    rank_summary_rows,
    with_derived_metrics,
)
from backend.benchmark.direct_mesh import direct_mesh_bbox_uses_hidden_source
from backend.benchmark.report_run import format_number, markdown_table
from backend.benchmark.stl_modes import STL_MODE_DEPTH_RELIEF, STL_MODE_SOURCE_MESH_ORACLE, STL_MODES, infer_stl_mode
from backend.benchmark.select_completion_candidate import (
    bool_value,
    json_safe,
    load_per_sample_rows,
    load_summary_rows,
    repair_metrics_expected,
    with_repair_summary_metrics,
)


SUMMARY_FILES = ("aggregate_summary.csv", "summary_metrics.csv")
DEPLOYABLE_STL_MODES = tuple(mode for mode in STL_MODES if mode != STL_MODE_SOURCE_MESH_ORACLE)
CHALLENGER_STL_MODES = tuple(mode for mode in DEPLOYABLE_STL_MODES if mode != STL_MODE_DEPTH_RELIEF)
MODE_ORDER = {mode: index for index, mode in enumerate(STL_MODES)}
SCALE_FREE_COMPLEXITY_MEDIAN = "stl_faces_per_normalized_bbox_volume_log1p_median"
SCALE_FREE_COMPLEXITY_SAMPLE = "stl_faces_per_normalized_bbox_volume_log1p"
LEGACY_FACE_DENSITY_MEDIAN = "stl_faces_per_bbox_volume_log1p_median"
LEGACY_FACE_DENSITY_SAMPLE = "stl_faces_per_bbox_volume_log1p"
REPAIR_CONVEX_HULL_RATE_MEAN = "repair_convex_hull_used_mean"
REPAIR_CONVEX_HULL_SAMPLE = "repair_convex_hull_used"
REPAIR_FILL_DRIFT_MEDIAN = "repair_volume_fill_ratio_relative_change_abs_median"
REPAIR_FILL_DRIFT_SAMPLE = "repair_volume_fill_ratio_relative_change_abs"
COMPACT_METRICS = (
    "rank_score",
    "method",
    "base_method",
    "stl_mode",
    "direct_mesh_bbox_source",
    "direct_mesh_reference_method",
    "n",
    "attempted_n",
    "success_rate",
    "error_count",
    "mesh_surface_chamfer_l1_median",
    "mesh_surface_hausdorff95_median",
    "provider_inference_runtime_seconds_median",
    "provider_invocation_runtime_seconds_median",
    "direct_mesh_command_runtime_seconds_median",
    "repair_runtime_seconds_median",
    "mesh_postprocess_runtime_seconds_median",
    "provider_peak_cuda_vram_gib_median",
    "repair_preclean_printable_mean",
    "repair_preclean_retriangulation_attempted_mean",
    "repair_preclean_retriangulation_accepted_mean",
    "repair_preclean_retriangulated_printable_mean",
    "repair_cleaning_skipped_mean",
    "repair_cleaned_printable_mean",
    "heldout_view_silhouette_iou_mean_median",
    "heldout_view_silhouette_iou_min_median",
    "raw_mesh_volume_fill_ratio_median",
    "stl_volume_fill_ratio_median",
    "repair_volume_fill_ratio_relative_change_median",
    REPAIR_FILL_DRIFT_MEDIAN,
    "repair_convex_hull_used_median",
    REPAIR_CONVEX_HULL_RATE_MEAN,
    "inferred_bbox_shape_log_mae_median",
    "inferred_bbox_shape_relative_mae_median",
    "inferred_bbox_centered_iou_median",
    "silhouette_iou_masked_median",
    "stl_exists_median",
    "stl_is_watertight_median",
    "stl_is_volume_median",
    "stl_is_manifold_median",
    "stl_positive_volume_median",
    "stl_single_component_median",
    "stl_nonmanifold_edge_count_log1p_median",
    "stl_degenerate_face_ratio_median",
    "stl_component_excess_log1p_median",
    "stl_bbox_aspect_ratio_median",
    SCALE_FREE_COMPLEXITY_MEDIAN,
    LEGACY_FACE_DENSITY_MEDIAN,
    "stl_faces_median",
)

PROMOTION_MINIMUMS = (
    ("success_rate", 1.0, "Success"),
    ("stl_exists_median", 1.0, "STL Exists"),
    ("stl_is_watertight_median", 1.0, "Watertight"),
    ("stl_is_volume_median", 1.0, "Volume Mesh"),
    ("stl_is_manifold_median", 1.0, "Manifold"),
    ("stl_winding_consistent_median", 1.0, "Winding"),
    ("stl_positive_volume_median", 1.0, "Positive Volume"),
    ("stl_single_component_median", 1.0, "Single Body"),
    ("stl_bbox_has_volume_median", 1.0, "3D BBox"),
)

PROMOTION_MAXIMUMS = (
    ("stl_nonmanifold_edge_count_log1p_median", 0.0, "Nonmanifold Edges log1p"),
    ("stl_degenerate_face_ratio_median", 0.0, "Degenerate Face Ratio"),
    ("stl_component_excess_log1p_median", 0.0, "Body Excess log1p"),
    ("stl_bbox_aspect_ratio_median", 10.0, "BBox Aspect"),
    (SCALE_FREE_COMPLEXITY_MEDIAN, 10.0, "Scale-Free Complexity log1p"),
)

OPTIONAL_PROMOTION_MAXIMUMS = (
    (REPAIR_CONVEX_HULL_RATE_MEAN, 0.25, "Convex Hull Fallback"),
    (REPAIR_FILL_DRIFT_MEDIAN, 0.5, "Repair Fill-Ratio Drift"),
)

OPTIONAL_SAMPLE_MAXIMUMS = (
    (REPAIR_FILL_DRIFT_SAMPLE, 4.0, "Repair Fill-Ratio Drift Maximum"),
)


def archive_label(path: Path) -> str:
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return safe_label(name or path.stem)


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return label.strip("._-") or "result"


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, value = spec.split("=", 1)
        label = safe_label(label)
        path = Path(value)
    else:
        path = Path(spec)
        label = archive_label(path)
    if not label or not str(path):
        raise ValueError(f"Expected label=path or path, got: {spec}")
    return label, path


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".tar", ".tar.gz", ".tgz", ".zip"))


def unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not choose a unique extract directory for {path}")


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(f"Refusing to extract link from archive: {member.name}")
            target = destination / member.name
            if Path(member.name).is_absolute() or not is_within(target, destination):
                raise ValueError(f"Refusing to extract path outside destination: {member.name}")
        tar.extractall(destination, members=members)


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = destination / member.filename
            if Path(member.filename).is_absolute() or not is_within(target, destination):
                raise ValueError(f"Refusing to extract path outside destination: {member.filename}")
        zip_file.extractall(destination)


def materialize_input(label: str, source: Path, output_dir: Path) -> dict:
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}")
    if source.is_dir():
        return {"label": label, "source": str(source), "root": str(source), "extracted_to": ""}
    if not is_archive(source):
        raise ValueError(f"Input must be a directory, tar archive, or zip archive: {source}")

    extract_dir = unique_dir(output_dir / "extracted" / label)
    if source.name.lower().endswith(".zip"):
        safe_extract_zip(source, extract_dir)
    else:
        safe_extract_tar(source, extract_dir)
    return {"label": label, "source": str(source), "root": str(extract_dir), "extracted_to": str(extract_dir)}


def discover_result_runs(root: Path) -> list[Path]:
    aggregate_roots = {summary_path.parent for summary_path in root.rglob("aggregate_summary.csv")}
    summary_roots = {summary_path.parent for summary_path in root.rglob("summary_metrics.csv")}
    roots = set(aggregate_roots)
    roots.update(
        summary_root
        for summary_root in summary_roots
        if not any(is_within(summary_root, aggregate_root) for aggregate_root in aggregate_roots)
    )
    return sorted(roots, key=lambda path: path.as_posix())


def method_stl_mode(row: dict) -> str:
    mode = str(row.get("stl_mode") or "").strip()
    if mode:
        return mode
    method = str(row.get("base_method") or row.get("method") or "")
    emit_stl = str(row.get("stl_exists_median") or "").strip() not in {"", "0", "0.0", "False", "false"}
    return infer_stl_mode(method, emit_stl=emit_stl)


def is_oracle_diagnostic(row: dict) -> bool:
    if bool_value(row.get("oracle_diagnostic")):
        return True
    mode = method_stl_mode(row)
    names = " ".join(str(row.get(field) or "") for field in ("method", "base_method", "stl_mode")).lower()
    names = names.replace("-", "_")
    bbox_source = str(row.get("direct_mesh_bbox_source") or "").strip().lower()
    return (
        mode == STL_MODE_SOURCE_MESH_ORACLE
        or direct_mesh_bbox_uses_hidden_source(bbox_source, row.get("direct_mesh_reference_method"))
        or "source_bbox" in names
        or ("source_mesh" in names and "oracle" in names)
    )


def compact_row(row: dict, per_sample_rows: list[dict] | None = None) -> dict:
    row = with_repair_summary_metrics(row, per_sample_rows or [])
    compact = {key: row.get(key, "") for key in COMPACT_METRICS if key in row}
    compact["stl_mode"] = method_stl_mode(row)
    compact["oracle_diagnostic"] = is_oracle_diagnostic(row)
    gates = promotion_gate_results(row, per_sample_rows)
    compact["promotion_eligible"] = all(gate["passed"] for gate in gates)
    compact["failed_promotion_gates"] = ", ".join(gate["name"] for gate in gates if not gate["passed"])
    return compact


def scale_free_complexity(row: dict) -> object:
    value = row.get(SCALE_FREE_COMPLEXITY_MEDIAN)
    if value not in (None, ""):
        return value
    return row.get(LEGACY_FACE_DENSITY_MEDIAN, "")


def fallback_metric_field(field: str) -> str | None:
    return {
        SCALE_FREE_COMPLEXITY_MEDIAN: LEGACY_FACE_DENSITY_MEDIAN,
        SCALE_FREE_COMPLEXITY_SAMPLE: LEGACY_FACE_DENSITY_SAMPLE,
    }.get(field)


def metric_value(row: dict, field: str) -> float:
    if field not in row or str(row.get(field, "")).strip() == "":
        fallback = fallback_metric_field(field)
        if fallback:
            field = fallback
    if field not in row or str(row.get(field, "")).strip() == "":
        return float("nan")
    value = row.get(field)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "yes", "y"}:
        return 1.0
    if text in {"false", "no", "n"}:
        return 0.0
    return parse_float(row.get(field))


def _sample_metric_field(summary_field: str) -> str:
    if summary_field == REPAIR_CONVEX_HULL_RATE_MEAN:
        return REPAIR_CONVEX_HULL_SAMPLE
    if summary_field == REPAIR_FILL_DRIFT_MEDIAN:
        return REPAIR_FILL_DRIFT_SAMPLE
    return summary_field[: -len("_median")] if summary_field.endswith("_median") else summary_field


def _sample_label(row: dict, index: int) -> str:
    return str(row.get("sample_id") or row.get("id") or f"row{index}")


def _has_cell(row: dict, field: str) -> bool:
    if field in row and str(row.get(field, "")).strip() != "":
        return True
    fallback = fallback_metric_field(field)
    return bool(fallback and fallback in row and str(row.get(fallback, "")).strip() != "")


def method_sample_rows(per_sample_rows: list[dict] | None, method: str) -> list[dict]:
    if not per_sample_rows:
        return []
    return [row for row in per_sample_rows if row.get("method") == method]


def per_sample_gate_results(row: dict, per_sample_rows: list[dict] | None = None) -> list[dict]:
    sample_rows = [with_derived_metrics(sample) for sample in method_sample_rows(per_sample_rows, row.get("method", ""))]
    repair_metrics_required = repair_metrics_expected(row)
    if not sample_rows:
        if repair_metrics_required:
            return [
                {
                    "name": "Sample Repair Fill-Ratio Drift Maximum",
                    "field": REPAIR_FILL_DRIFT_SAMPLE,
                    "passed": False,
                    "value": 0,
                    "threshold": "per-sample repair metrics present",
                    "detail": "missing_per_sample_repair_metrics",
                    "failed_sample_count": 0,
                    "failed_samples": [],
                },
                {
                    "name": "Sample Repair Fill-Ratio Drift Coverage",
                    "field": REPAIR_FILL_DRIFT_SAMPLE,
                    "passed": False,
                    "value": 0.0,
                    "threshold": ">= 0.75 samples <= 0.5",
                    "detail": "missing_per_sample_repair_metrics",
                    "failed_sample_count": 0,
                    "failed_samples": [],
                },
            ]
        return []

    gates = []
    for summary_field, minimum, label in PROMOTION_MINIMUMS:
        field = _sample_metric_field(summary_field)
        if not any(_has_cell(sample, field) for sample in sample_rows):
            continue
        failed = []
        for index, sample in enumerate(sample_rows):
            value = metric_value(sample, field)
            if value != value or value < minimum:
                failed.append(_sample_label(sample, index))
        gates.append(
            {
                "name": f"Sample {label}",
                "field": field,
                "passed": not failed,
                "value": len(sample_rows) - len(failed),
                "threshold": f"{len(sample_rows)}/{len(sample_rows)} samples >= {minimum}",
                "detail": f"failed_samples={','.join(failed[:10])}" if failed else "",
                "failed_sample_count": len(failed),
                "failed_samples": failed,
            }
        )
    for summary_field, maximum, label in PROMOTION_MAXIMUMS:
        field = _sample_metric_field(summary_field)
        if not any(_has_cell(sample, field) for sample in sample_rows):
            continue
        failed = []
        for index, sample in enumerate(sample_rows):
            value = metric_value(sample, field)
            if value != value or value > maximum:
                failed.append(_sample_label(sample, index))
        gates.append(
            {
                "name": f"Sample {label}",
                "field": field,
                "passed": not failed,
                "value": len(sample_rows) - len(failed),
                "threshold": f"{len(sample_rows)}/{len(sample_rows)} samples <= {maximum}",
                "detail": f"failed_samples={','.join(failed[:10])}" if failed else "",
                "failed_sample_count": len(failed),
                "failed_samples": failed,
            }
        )
    for field, maximum, label in OPTIONAL_SAMPLE_MAXIMUMS:
        if not repair_metrics_required and not any(_has_cell(sample, field) for sample in sample_rows):
            continue
        failed = []
        for index, sample in enumerate(sample_rows):
            value = metric_value(sample, field)
            if value != value or value > maximum:
                failed.append(_sample_label(sample, index))
        gates.append(
            {
                "name": f"Sample {label}",
                "field": field,
                "passed": not failed,
                "value": len(sample_rows) - len(failed),
                "threshold": f"{len(sample_rows)}/{len(sample_rows)} samples <= {maximum}",
                "detail": f"failed_samples={','.join(failed[:10])}" if failed else "",
                "failed_sample_count": len(failed),
                "failed_samples": failed,
            }
        )
    if repair_metrics_required:
        field = REPAIR_FILL_DRIFT_SAMPLE
        within_limit = []
        failed = []
        for index, sample in enumerate(sample_rows):
            value = metric_value(sample, field)
            sample_id = _sample_label(sample, index)
            if value == value and value <= 0.5:
                within_limit.append(sample_id)
            else:
                failed.append(sample_id)
        rate = len(within_limit) / len(sample_rows)
        gates.append(
            {
                "name": "Sample Repair Fill-Ratio Drift Coverage",
                "field": field,
                "passed": rate >= 0.75,
                "value": rate,
                "threshold": ">= 0.75 samples <= 0.5",
                "detail": f"failed_samples={','.join(failed[:10])}" if failed else "",
                "failed_sample_count": len(failed),
                "failed_samples": failed,
            }
        )
    return gates


def promotion_gate_results(row: dict, per_sample_rows: list[dict] | None = None) -> list[dict]:
    row = with_repair_summary_metrics(row, per_sample_rows or [])
    mode = method_stl_mode(row)
    oracle = is_oracle_diagnostic(row)
    gates = [
        {
            "name": "deployable_stl_mode",
            "passed": mode in DEPLOYABLE_STL_MODES and not oracle,
            "value": mode,
            "threshold": ",".join(DEPLOYABLE_STL_MODES) + " and not source-mesh oracle",
        }
    ]
    for field, minimum, label in PROMOTION_MINIMUMS:
        value = metric_value(row, field)
        gates.append(
            {
                "name": label,
                "field": field,
                "passed": value >= minimum,
                "value": None if value != value else value,
                "threshold": f">= {minimum}",
            }
        )
    for field, maximum, label in PROMOTION_MAXIMUMS:
        value = metric_value(row, field)
        gates.append(
            {
                "name": label,
                "field": field,
                "passed": value <= maximum,
                "value": None if value != value else value,
                "threshold": f"<= {maximum}",
            }
        )
    for field, maximum, label in OPTIONAL_PROMOTION_MAXIMUMS:
        required = repair_metrics_expected(row)
        if not _has_cell(row, field) and not required:
            continue
        value = metric_value(row, field)
        gates.append(
            {
                "name": label,
                "field": field,
                "passed": value == value and value <= maximum,
                "value": None if value != value else value,
                "threshold": f"<= {maximum}",
            }
        )
    gates.extend(per_sample_gate_results(row, per_sample_rows))
    return gates


def promotion_eligible(row: dict, per_sample_rows: list[dict] | None = None) -> bool:
    return all(gate["passed"] for gate in promotion_gate_results(row, per_sample_rows))


def gate_value_label(value) -> str:
    if value is None:
        return ""
    number = parse_float(value)
    if number == number:
        return format_number(number)
    return str(value)


def gate_detail_label(gate: dict) -> str:
    failed_samples = gate.get("failed_samples") or []
    if failed_samples:
        samples = [str(sample) for sample in failed_samples]
        suffix = f", +{len(samples) - 12}" if len(samples) > 12 else ""
        return "failed_samples=" + ",".join(samples[:12]) + suffix
    return str(gate.get("detail") or "")


def gate_failure_rows(ranked_rows: list[dict], per_sample_rows: list[dict] | None = None) -> list[dict]:
    rows = []
    for row in ranked_rows:
        method = row.get("method", "")
        mode = method_stl_mode(row)
        for gate in promotion_gate_results(row, per_sample_rows):
            if gate.get("passed"):
                continue
            rows.append(
                {
                    "method": method,
                    "stl_mode": mode,
                    "rank_score": row.get("rank_score", ""),
                    "gate": gate.get("name", ""),
                    "field": gate.get("field", ""),
                    "value": gate.get("value"),
                    "threshold": gate.get("threshold", ""),
                    "detail": gate_detail_label(gate),
                    "failed_sample_count": gate.get("failed_sample_count", 0),
                    "failed_samples": gate.get("failed_samples", []),
                }
            )
    return rows


def sample_failure_hotspot_rows(gate_failures: list[dict]) -> list[dict]:
    hotspots: dict[str, dict] = {}
    for failure in gate_failures:
        samples = failure.get("failed_samples") or []
        for sample in samples:
            sample_id = str(sample)
            hotspot = hotspots.setdefault(
                sample_id,
                {
                    "sample_id": sample_id,
                    "method_count": 0,
                    "gate_count": 0,
                    "methods": set(),
                    "stl_modes": set(),
                    "gates": set(),
                },
            )
            if failure.get("method"):
                hotspot["methods"].add(str(failure.get("method")))
            if failure.get("stl_mode"):
                hotspot["stl_modes"].add(str(failure.get("stl_mode")))
            if failure.get("gate"):
                hotspot["gates"].add(str(failure.get("gate")))
            hotspot["gate_count"] += 1

    rows = []
    for hotspot in hotspots.values():
        methods = sorted(hotspot["methods"])
        stl_modes = sorted(hotspot["stl_modes"])
        gates = sorted(hotspot["gates"])
        rows.append(
            {
                "sample_id": hotspot["sample_id"],
                "method_count": len(methods),
                "gate_count": hotspot["gate_count"],
                "methods": methods,
                "stl_modes": stl_modes,
                "gates": gates,
            }
        )
    return sorted(rows, key=lambda row: (-row["method_count"], -row["gate_count"], row["sample_id"]))


def best_by_mode(ranked_rows: list[dict], per_sample_rows: list[dict] | None = None) -> list[dict]:
    best: dict[str, dict] = {}
    for row in ranked_rows:
        if is_oracle_diagnostic(row):
            continue
        mode = method_stl_mode(row)
        if not mode:
            continue
        best.setdefault(mode, row)
    return [
        compact_row(best[mode], per_sample_rows)
        for mode in sorted(best, key=lambda item: MODE_ORDER.get(item, 999))
    ]


def first_deployable(ranked_rows: list[dict], per_sample_rows: list[dict] | None = None) -> dict:
    for row in ranked_rows:
        if method_stl_mode(row) in DEPLOYABLE_STL_MODES and not is_oracle_diagnostic(row):
            return compact_row(row, per_sample_rows)
    return {}


def first_promotion_eligible(ranked_rows: list[dict], per_sample_rows: list[dict] | None = None) -> dict:
    for row in ranked_rows:
        if promotion_eligible(row, per_sample_rows):
            return compact_row(row, per_sample_rows)
    return {}


def first_promotion_eligible_mode(
    ranked_rows: list[dict],
    mode: str,
    per_sample_rows: list[dict] | None = None,
) -> dict:
    for row in ranked_rows:
        if method_stl_mode(row) != mode or is_oracle_diagnostic(row):
            continue
        if promotion_eligible(row, per_sample_rows):
            return compact_row(row, per_sample_rows)
    return {}


def first_deployable_challenger(
    ranked_rows: list[dict],
    per_sample_rows: list[dict] | None = None,
) -> dict:
    for row in ranked_rows:
        if method_stl_mode(row) in CHALLENGER_STL_MODES and not is_oracle_diagnostic(row):
            return compact_row(row, per_sample_rows)
    return {}


def first_promotion_eligible_challenger(
    ranked_rows: list[dict],
    per_sample_rows: list[dict] | None = None,
) -> dict:
    for row in ranked_rows:
        if method_stl_mode(row) not in CHALLENGER_STL_MODES or is_oracle_diagnostic(row):
            continue
        if promotion_eligible(row, per_sample_rows):
            return compact_row(row, per_sample_rows)
    return {}


def score_delta(candidate: dict, baseline: dict) -> float | None:
    candidate_score = parse_float(candidate.get("rank_score"))
    baseline_score = parse_float(baseline.get("rank_score"))
    if candidate_score != candidate_score or baseline_score != baseline_score:
        return None
    return candidate_score - baseline_score


def architecture_replacement_decision(
    ranked_rows: list[dict],
    per_sample_rows: list[dict] | None = None,
) -> dict:
    depth_relief = first_promotion_eligible_mode(ranked_rows, STL_MODE_DEPTH_RELIEF, per_sample_rows)
    challenger = first_promotion_eligible_challenger(ranked_rows, per_sample_rows)
    score_leading_challenger = first_deployable_challenger(ranked_rows, per_sample_rows)

    if not challenger:
        reason = "No full-mesh or multiview STL challenger cleared the hard promotion gates."
        if score_leading_challenger and not score_leading_challenger.get("promotion_eligible"):
            reason = "The highest-scoring full-mesh or multiview challenger was blocked by STL promotion gates."
        return {
            "decision": "keep-depth-relief",
            "recommended_method": depth_relief.get("method", ""),
            "recommended_stl_mode": depth_relief.get("stl_mode", ""),
            "reason": reason,
            "score_delta_vs_depth_relief": None,
            "depth_relief_baseline": depth_relief,
            "promotion_eligible_challenger": {},
            "score_leading_challenger": score_leading_challenger,
        }

    delta = score_delta(challenger, depth_relief) if depth_relief else None
    if not depth_relief:
        return {
            "decision": "promote-challenger",
            "recommended_method": challenger.get("method", ""),
            "recommended_stl_mode": challenger.get("stl_mode", ""),
            "reason": "A full-mesh or multiview STL challenger cleared promotion gates and no promotion-eligible depth-relief baseline was found.",
            "score_delta_vs_depth_relief": delta,
            "depth_relief_baseline": {},
            "promotion_eligible_challenger": challenger,
            "score_leading_challenger": score_leading_challenger,
        }

    if delta is not None and delta > 0:
        return {
            "decision": "promote-challenger",
            "recommended_method": challenger.get("method", ""),
            "recommended_stl_mode": challenger.get("stl_mode", ""),
            "reason": "The best promotion-eligible full-mesh or multiview STL challenger beat the best promotion-eligible depth-relief baseline.",
            "score_delta_vs_depth_relief": delta,
            "depth_relief_baseline": depth_relief,
            "promotion_eligible_challenger": challenger,
            "score_leading_challenger": score_leading_challenger,
        }

    return {
        "decision": "keep-depth-relief",
        "recommended_method": depth_relief.get("method", ""),
        "recommended_stl_mode": depth_relief.get("stl_mode", ""),
        "reason": "The best promotion-eligible full-mesh or multiview STL challenger did not beat the best promotion-eligible depth-relief baseline.",
        "score_delta_vs_depth_relief": delta,
        "depth_relief_baseline": depth_relief,
        "promotion_eligible_challenger": challenger,
        "score_leading_challenger": score_leading_challenger,
    }


def first_oracle(ranked_rows: list[dict], per_sample_rows: list[dict] | None = None) -> dict:
    for row in ranked_rows:
        if is_oracle_diagnostic(row):
            return compact_row(row, per_sample_rows)
    return {}


def summarize_run(
    label: str,
    run_dir: Path,
    *,
    score_profile: str,
    score_mode: str,
    baseline_method: str,
    top: int,
) -> dict:
    summary_rows, summary_path = load_summary_rows(run_dir)
    per_sample_rows = load_per_sample_rows(run_dir)
    weights = parse_weights([], profile=score_profile)
    ranked_rows, used_metrics = rank_summary_rows(
        summary_rows,
        weights,
        score_mode=score_mode,
        baseline_method=baseline_method,
    )
    gate_failures = gate_failure_rows(ranked_rows[:top], per_sample_rows)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "summary_rows": len(summary_rows),
        "per_sample_rows": len(per_sample_rows),
        "score_profile": score_profile,
        "score_mode": score_mode,
        "baseline_method": baseline_method,
        "used_metrics": used_metrics,
        "deployable_winner": first_deployable(ranked_rows, per_sample_rows),
        "promotion_eligible_winner": first_promotion_eligible(ranked_rows, per_sample_rows),
        "oracle_diagnostic_winner": first_oracle(ranked_rows, per_sample_rows),
        "best_by_stl_mode": best_by_mode(ranked_rows, per_sample_rows),
        "architecture_replacement_decision": architecture_replacement_decision(ranked_rows, per_sample_rows),
        "ranked_methods": [compact_row(row, per_sample_rows) for row in ranked_rows[:top]],
        "gate_failures": gate_failures,
        "sample_failure_hotspots": sample_failure_hotspot_rows(gate_failures),
    }


def summarize_inputs(
    input_specs: list[str],
    *,
    output_dir: Path,
    score_profile: str = "stl-quality",
    score_mode: str = "baseline-delta",
    baseline_method: str = "masked",
    top: int = 20,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = [materialize_input(*parse_input_spec(spec), output_dir=output_dir) for spec in input_specs]
    runs = []
    for item in inputs:
        label = item["label"]
        root = Path(item["root"])
        discovered = discover_result_runs(root)
        for run_dir in discovered:
            run_label = label if len(discovered) == 1 else f"{label}:{run_dir.name}"
            runs.append(
                summarize_run(
                    run_label,
                    run_dir,
                    score_profile=score_profile,
                    score_mode=score_mode,
                    baseline_method=baseline_method,
                    top=top,
                )
            )
    if not runs:
        searched = ", ".join(item["root"] for item in inputs)
        raise FileNotFoundError(f"No benchmark result directories found under: {searched}")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs,
        "run_count": len(runs),
        "score_profile": score_profile,
        "score_mode": score_mode,
        "baseline_method": baseline_method,
        "deployable_stl_modes": list(DEPLOYABLE_STL_MODES),
        "runs": runs,
    }


def mode_table_rows(rows: list[dict]) -> list[list[str]]:
    table = []
    for row in rows:
        table.append(
            [
                row.get("stl_mode", ""),
                row.get("method", ""),
                format_number(row.get("rank_score")),
                format_number(row.get("n")),
                format_number(row.get("success_rate")),
                format_number(row.get("mesh_surface_chamfer_l1_median")),
                format_number(row.get("mesh_surface_hausdorff95_median")),
                format_number(row.get("stl_is_watertight_median")),
                format_number(row.get("stl_is_volume_median")),
                format_number(row.get("stl_is_manifold_median")),
                format_number(row.get("stl_single_component_median")),
                format_number(scale_free_complexity(row)),
                "yes" if row.get("promotion_eligible") else "no",
                row.get("failed_promotion_gates", ""),
            ]
        )
    return table


def gate_failure_table_rows(rows: list[dict]) -> list[list[str]]:
    table = []
    for row in rows:
        table.append(
            [
                row.get("method", ""),
                row.get("stl_mode", ""),
                format_number(row.get("rank_score")),
                row.get("gate", ""),
                row.get("field", ""),
                gate_value_label(row.get("value")),
                str(row.get("threshold", "")),
                row.get("detail", ""),
            ]
        )
    return table


def sample_failure_hotspot_table_rows(rows: list[dict]) -> list[list[str]]:
    table = []
    for row in rows:
        table.append(
            [
                row.get("sample_id", ""),
                format_number(row.get("method_count")),
                format_number(row.get("gate_count")),
                ", ".join(row.get("methods", [])),
                ", ".join(row.get("stl_modes", [])),
                ", ".join(row.get("gates", [])),
            ]
        )
    return table


def architecture_decision_table_rows(decision: dict) -> list[list[str]]:
    if not decision:
        return []
    depth = decision.get("depth_relief_baseline") or {}
    challenger = decision.get("promotion_eligible_challenger") or {}
    score_leader = decision.get("score_leading_challenger") or {}
    return [
        [
            decision.get("decision", ""),
            decision.get("recommended_method", ""),
            decision.get("recommended_stl_mode", ""),
            depth.get("method", ""),
            challenger.get("method", ""),
            score_leader.get("method", ""),
            format_number(decision.get("score_delta_vs_depth_relief")),
            decision.get("reason", ""),
        ]
    ]


def render_markdown(report: dict) -> str:
    input_rows = [
        [item["label"], item["source"], item["extracted_to"] or item["root"]]
        for item in report.get("inputs", [])
    ]
    lines = [
        "# STL-First Result Ingest",
        "",
        f"- Generated: `{report.get('generated_at', '')}`",
        f"- Score profile: `{report.get('score_profile', '')}`",
        f"- Score mode: `{report.get('score_mode', '')}`",
        f"- Baseline: `{report.get('baseline_method', '')}`",
        f"- Result directories: `{report.get('run_count', 0)}`",
        "",
        "## Inputs",
        "",
        markdown_table(["Label", "Source", "Materialized Root"], input_rows),
        "",
        "Source-mesh oracle rows, including source-mesh bundle oracles, are kept as diagnostics. Only non-oracle `depth-relief`, `single-image-mesh`, and `multiview-mesh` rows are treated as deployable STL architectures.",
        "",
    ]
    for run in report.get("runs", []):
        deployable = run.get("deployable_winner") or {}
        promotion = run.get("promotion_eligible_winner") or {}
        oracle = run.get("oracle_diagnostic_winner") or {}
        lines.extend(
            [
                f"## Run: {run.get('label', '')}",
                "",
                f"- Run directory: `{run.get('run_dir', '')}`",
                f"- Summary: `{run.get('summary_path', '')}`",
                f"- Summary rows: `{run.get('summary_rows', 0)}`",
                f"- Per-sample rows: `{run.get('per_sample_rows', 0)}`",
                f"- Deployable score leader: `{deployable.get('method', '<none>')}` ({deployable.get('stl_mode', '')}) score `{format_number(deployable.get('rank_score'))}`",
                f"- Promotion-eligible winner: `{promotion.get('method', '<none>')}` ({promotion.get('stl_mode', '')}) score `{format_number(promotion.get('rank_score'))}`",
            ]
        )
        if oracle:
            lines.append(
                f"- Oracle diagnostic: `{oracle.get('method', '')}` score `{format_number(oracle.get('rank_score'))}`"
            )
        lines.extend(
            [
                "",
                "### Architecture Decision",
                "",
                markdown_table(
                    [
                        "Decision",
                        "Recommended",
                        "Mode",
                        "Depth-Relief Baseline",
                        "Promotion Challenger",
                        "Score-Leading Challenger",
                        "Delta vs Depth-Relief",
                        "Reason",
                    ],
                    architecture_decision_table_rows(run.get("architecture_replacement_decision", {})),
                ),
                "",
                "### Architecture Leaders",
                "",
                markdown_table(
                    [
                        "STL Mode",
                        "Method",
                        "Score",
                        "n",
                        "Success",
                        "Chamfer",
                        "H95",
                        "Watertight",
                        "Volume",
                        "Manifold",
                        "Single Body",
                        "Scale-Free Complexity",
                        "Promote",
                        "Gate Failures",
                    ],
                    mode_table_rows(run.get("best_by_stl_mode", [])),
                ),
                "",
                "### Top Methods",
                "",
                markdown_table(
                    [
                        "STL Mode",
                        "Method",
                        "Score",
                        "n",
                        "Success",
                        "Chamfer",
                        "H95",
                        "Watertight",
                        "Volume",
                        "Manifold",
                        "Single Body",
                        "Scale-Free Complexity",
                        "Promote",
                        "Gate Failures",
                    ],
                    mode_table_rows(run.get("ranked_methods", [])),
                ),
                "",
                "### Promotion Gate Failures",
                "",
                markdown_table(
                    ["Method", "STL Mode", "Score", "Gate", "Field", "Value", "Threshold", "Detail"],
                    gate_failure_table_rows(run.get("gate_failures", [])),
                ),
                "",
                "### Sample Failure Hotspots",
                "",
                markdown_table(
                    ["Sample", "Methods", "Gates", "Method Names", "STL Modes", "Gate Names"],
                    sample_failure_hotspot_table_rows(run.get("sample_failure_hotspots", [])),
                ),
                "",
            ]
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Colab or local STL-first benchmark results and write an architecture-focused report."
    )
    parser.add_argument("input", nargs="+", help="Result directory/archive, optionally label=path.")
    parser.add_argument("--output-dir", required=True, help="Directory for extracted archives and reports.")
    parser.add_argument("--output-json", default=None, help="Defaults to <output-dir>/stl_first_ingest_report.json.")
    parser.add_argument("--output-md", default=None, help="Defaults to <output-dir>/stl_first_ingest_report.md.")
    parser.add_argument("--top", type=int, default=20, help="Top ranked methods to keep per run.")
    parser.add_argument(
        "--score-profile",
        choices=sorted(SCORE_PROFILES),
        default="stl-quality",
        help="Metric weight profile used to re-rank result summaries.",
    )
    parser.add_argument(
        "--score-mode",
        choices=sorted(SCORE_MODES),
        default="baseline-delta",
        help="Ranking mode used for the ingest report.",
    )
    parser.add_argument("--baseline-method", default="masked")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report = summarize_inputs(
        args.input,
        output_dir=output_dir,
        score_profile=args.score_profile,
        score_mode=args.score_mode,
        baseline_method=args.baseline_method,
        top=max(args.top, 1),
    )
    output_json = Path(args.output_json) if args.output_json else output_dir / "stl_first_ingest_report.json"
    output_md = Path(args.output_md) if args.output_md else output_dir / "stl_first_ingest_report.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
