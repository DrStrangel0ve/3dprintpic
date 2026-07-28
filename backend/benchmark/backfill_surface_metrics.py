from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.make_artifact_contact_sheet import (
    PROJECT_ROOT,
    discover_metric_paths,
    find_manifest_path,
    load_manifest,
    read_json,
    resolve_existing_path,
)
from backend.benchmark.metrics import align_depth, load_bool_mask, load_mask, surface_distance_metrics
from backend.benchmark.run_completion_benchmark import summarize


SURFACE_FIELDS = [
    "surface_chamfer_l1",
    "surface_chamfer_rmse",
    "surface_rmse",
    "surface_hausdorff95",
    "surface_point_count",
    "object_surface_chamfer_l1",
    "object_surface_chamfer_rmse",
    "object_surface_rmse",
    "object_surface_hausdorff95",
    "object_surface_point_count",
]


def read_csv_with_fieldnames(path: Path) -> tuple[list[dict], list[str]]:
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], preferred_fieldnames: list[str] | None = None) -> None:
    fieldnames = list(preferred_fieldnames or [])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.float32) > 127


def _resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth.astype(np.float32)
    image = Image.fromarray(depth.astype(np.float32))
    resized = image.resize((shape[1], shape[0]), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def finite_text(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value or "")
    if not math.isfinite(number):
        return "nan"
    return repr(float(number))


def row_has_surface_metrics(row: dict) -> bool:
    return any(str(row.get(field, "")).strip() for field in SURFACE_FIELDS)


def resolve_reference_path(
    row: dict,
    sample: dict,
    fields: tuple[str, ...],
    anchors: list[Path],
) -> Path | None:
    for field in fields:
        path = resolve_existing_path(row.get(field), anchors)
        if path:
            return path
    for field in fields:
        path = resolve_existing_path(sample.get(field), anchors)
        if path:
            return path
    return None


def infer_depth_data_path(row: dict, anchors: list[Path]) -> Path | None:
    for field in ("depth_data", "output_depth_data"):
        path = resolve_existing_path(row.get(field), anchors)
        if path:
            return path

    for field in ("stl_model", "completed_image", "raw_completed_image"):
        artifact_path = resolve_existing_path(row.get(field), anchors)
        if not artifact_path:
            continue
        candidate = artifact_path.parent / "output_depth_data.npy"
        if candidate.exists():
            return candidate
    return None


def empty_surface_metrics() -> dict[str, float | int]:
    return {
        "surface_chamfer_l1": math.nan,
        "surface_chamfer_rmse": math.nan,
        "surface_rmse": math.nan,
        "surface_hausdorff95": math.nan,
        "surface_point_count": 0,
    }


def compute_surface_metrics_for_row(
    row: dict,
    sample: dict,
    metrics_path: Path,
    manifest_path: Path | None,
    max_points: int = 4096,
) -> tuple[bool, str]:
    anchors = [metrics_path.parent, PROJECT_ROOT, Path.cwd()]
    if manifest_path:
        anchors.insert(0, manifest_path.parent)

    gt_depth_path = resolve_reference_path(row, sample, ("gt_depth",), anchors)
    mask_path = resolve_reference_path(row, sample, ("mask",), anchors)
    silhouette_path = resolve_reference_path(row, sample, ("gt_silhouette", "silhouette"), anchors)
    pred_depth_path = infer_depth_data_path(row, anchors)

    if not gt_depth_path:
        return False, "missing_gt_depth"
    if not mask_path:
        return False, "missing_mask"
    if not pred_depth_path:
        return False, "missing_depth_data"

    gt_depth = np.load(gt_depth_path).astype(np.float32)
    pred_depth = _resize_depth(np.load(pred_depth_path), gt_depth.shape)
    mask = _resize_mask(load_mask(mask_path), gt_depth.shape)
    visible = ~mask

    aligned_depth, _, _ = align_depth(pred_depth, gt_depth, visible)
    row.update({key: finite_text(value) for key, value in surface_distance_metrics(gt_depth, aligned_depth, mask, max_points=max_points).items()})

    if silhouette_path:
        reference_silhouette = _resize_mask(load_bool_mask(silhouette_path), gt_depth.shape)
        object_fit = visible & reference_silhouette
        object_eval = mask & reference_silhouette
        if np.count_nonzero(object_fit) >= 2 and np.count_nonzero(object_eval) > 0:
            object_aligned_depth, _, _ = align_depth(pred_depth, gt_depth, object_fit)
            object_surface = surface_distance_metrics(gt_depth, object_aligned_depth, object_eval, max_points=max_points)
        else:
            object_surface = empty_surface_metrics()
        row.update({f"object_{key}": finite_text(value) for key, value in object_surface.items()})

    row["gt_depth"] = row.get("gt_depth") or str(gt_depth_path)
    row["mask"] = row.get("mask") or str(mask_path)
    row["depth_data"] = row.get("depth_data") or str(pred_depth_path)
    if silhouette_path:
        row["gt_silhouette"] = row.get("gt_silhouette") or str(silhouette_path)
    return True, "updated"


def summary_methods(rows: list[dict], existing_summary_rows: list[dict]) -> list[str]:
    row_methods = []
    for row in rows:
        method = str(row.get("method") or "")
        if method and method not in row_methods:
            row_methods.append(method)

    methods = [
        str(row.get("method") or "")
        for row in existing_summary_rows
        if str(row.get("method") or "").strip() in row_methods
    ]
    for row in rows:
        method = str(row.get("method") or "")
        if method and method not in methods:
            methods.append(method)
    return methods


def write_updated_summary(metrics_path: Path, rows: list[dict]) -> Path:
    summary_path = metrics_path.parent / "summary_metrics.csv"
    existing_summary_rows, existing_fieldnames = read_csv_with_fieldnames(summary_path)
    existing_by_method = {row.get("method"): row for row in existing_summary_rows}
    methods = summary_methods(rows, existing_summary_rows)
    summary_rows = summarize(rows, methods=methods)
    for row in summary_rows:
        existing = existing_by_method.get(row.get("method"), {})
        for field in ("attempted_n", "error_count", "logged_failure_count", "success_rate", "last_error_type", "last_error"):
            if field in existing and str(existing.get(field, "")).strip():
                row[field] = existing[field]
    write_csv(summary_path, summary_rows, existing_fieldnames)
    return summary_path


def write_aggregate_summary(input_dir: Path, metric_paths: list[Path]) -> Path | None:
    if len(metric_paths) <= 1:
        return None

    existing_path = input_dir / "aggregate_summary.csv"
    existing_rows, existing_fieldnames = read_csv_with_fieldnames(existing_path)
    existing_by_method = {row.get("method"): row for row in existing_rows}
    rows = []
    for metrics_path in metric_paths:
        method_name = metrics_path.parent.name
        summary_rows, _ = read_csv_with_fieldnames(metrics_path.parent / "summary_metrics.csv")
        summary_row = next((row for row in summary_rows if row.get("method") == method_name), None)
        summary_row = summary_row or next((row for row in summary_rows if row_count(row) > 0), None)
        summary_row = summary_row or (summary_rows[0] if summary_rows else None)
        if summary_row is None:
            continue
        row = dict(existing_by_method.get(method_name, existing_by_method.get(summary_row.get("method"), {})))
        row.update(summary_row)
        row["method"] = method_name
        rows.append(row)
    if not rows:
        return None
    write_csv(existing_path, rows, existing_fieldnames)
    return existing_path


def row_count(row: dict) -> int:
    try:
        return int(float(row.get("n") or 0))
    except (TypeError, ValueError):
        return 0


def backfill_surface_metrics(
    input_dir: Path,
    *,
    manifest: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    max_points: int = 4096,
) -> dict:
    metric_paths = discover_metric_paths(input_dir)
    if not metric_paths:
        raise FileNotFoundError(f"No per_sample_metrics.csv found in {input_dir}")

    manifest_path = find_manifest_path(input_dir, metric_paths, explicit_manifest=manifest)
    manifest_rows = load_manifest(manifest_path)
    result = {
        "input_dir": str(input_dir),
        "manifest": str(manifest_path) if manifest_path else "",
        "metric_files": [],
        "updated_rows": 0,
        "skipped_existing_rows": 0,
        "skipped_missing_rows": 0,
        "error_rows": 0,
        "dry_run": dry_run,
    }

    for metrics_path in metric_paths:
        rows, fieldnames = read_csv_with_fieldnames(metrics_path)
        file_result = {
            "path": str(metrics_path),
            "rows": len(rows),
            "updated": 0,
            "skipped_existing": 0,
            "skipped_missing": 0,
            "errors": 0,
            "skip_reasons": {},
        }
        for row in rows:
            if row_has_surface_metrics(row) and not force:
                file_result["skipped_existing"] += 1
                continue
            sample = manifest_rows.get(str(row.get("sample_id")), {})
            try:
                updated, reason = compute_surface_metrics_for_row(
                    row,
                    sample,
                    metrics_path,
                    manifest_path,
                    max_points=max_points,
                )
            except Exception as exc:  # noqa: BLE001 - keep backfills moving and report the row-level issue.
                updated, reason = False, f"{type(exc).__name__}: {exc}"
                file_result["errors"] += 1
            if updated:
                file_result["updated"] += 1
            else:
                file_result["skipped_missing"] += 1
                file_result["skip_reasons"][reason] = file_result["skip_reasons"].get(reason, 0) + 1

        if not dry_run and file_result["updated"]:
            write_csv(metrics_path, rows, fieldnames)
            write_updated_summary(metrics_path, rows)

        result["updated_rows"] += file_result["updated"]
        result["skipped_existing_rows"] += file_result["skipped_existing"]
        result["skipped_missing_rows"] += file_result["skipped_missing"]
        result["error_rows"] += file_result["errors"]
        result["metric_files"].append(file_result)

    aggregate_path = None
    if not dry_run and result["updated_rows"]:
        aggregate_path = write_aggregate_summary(input_dir, metric_paths)
    result["aggregate_summary"] = str(aggregate_path) if aggregate_path else ""
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill depth-derived 3D surface metrics into cached benchmark CSVs.")
    parser.add_argument("input_dir", help="Benchmark run directory or optimize_completion experiment directory.")
    parser.add_argument("--manifest", default=None, help="Optional manifest override when split_audit.json is missing.")
    parser.add_argument("--force", action="store_true", help="Recompute rows even when surface metrics already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be updated without writing CSVs.")
    parser.add_argument("--max-points", type=int, default=4096, help="Maximum points sampled from each depth cloud.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = backfill_surface_metrics(
        Path(args.input_dir),
        manifest=args.manifest,
        force=args.force,
        dry_run=args.dry_run,
        max_points=args.max_points,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
