from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmark.run_completion_benchmark import training_provenance_fields, training_report_path
from backend.benchmark.train_inpainting_lora import loss_recipe_label, prompt_family_from_rows


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def resolve_existing_path(value: str | None, base_dir: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    if base_dir:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
    return path


def source_split_from_key(value: str) -> str:
    parts = str(value or "").replace("\\", "/").split("/")
    for part in parts:
        split = part.lower()
        if split == "validation":
            return "val"
        if split in {"train", "test", "val"}:
            return split
    return "unknown"


def row_source_split(row: dict) -> str:
    explicit = str(row.get("asset_source_split") or "")
    if explicit:
        return explicit
    return source_split_from_key(row.get("asset_key") or row.get("asset_path") or "")


def count_values(rows: list[dict], getter) -> dict:
    counts = {}
    for row in rows:
        value = str(getter(row) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def row_asset_key(row: dict) -> str:
    return str(row.get("asset_key") or row.get("asset_path") or row.get("asset_id") or row.get("id") or "")


def manifest_index_range(rows: list[dict]) -> tuple[int | None, int | None]:
    indices = []
    for row in rows:
        try:
            indices.append(int(row.get("manifest_index")))
        except (TypeError, ValueError):
            pass
    if not indices:
        return None, None
    return min(indices), max(indices)


def infer_prompt_mode(rows: list[dict]) -> str:
    templates = {str(row.get("prompt_template") or "") for row in rows if row.get("prompt_template")}
    prompts = {str(row.get("prompt") or "") for row in rows if row.get("prompt")}
    if not prompts:
        return "empty"
    if len(prompts) == 1 and (not templates or prompts == templates):
        return "literal"
    return "templated"


def pair_export_summary(metadata_path: Path, rows: list[dict]) -> dict:
    first_index, last_index = manifest_index_range(rows)
    asset_keys = sorted({key for key in (row_asset_key(row) for row in rows) if key})
    object_mask_rows = sum(1 for row in rows if row.get("object_mask") or row.get("gt_silhouette") or row.get("silhouette"))
    prompt = rows[0].get("prompt_template") or rows[0].get("prompt", "") if rows else ""
    return {
        "generated_at": now_iso(),
        "backfilled": True,
        "metadata": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "output_dir": str(metadata_path.parent),
        "rows": len(rows),
        "manifest_indices": [row.get("manifest_index") for row in rows if row.get("manifest_index") is not None],
        "first_manifest_index": first_index,
        "last_manifest_index": last_index,
        "prompt": prompt,
        "prompt_mode": infer_prompt_mode(rows),
        "literal_prompt": infer_prompt_mode(rows) == "literal",
        "object_mask_rows": object_mask_rows,
        "object_mask_coverage": object_mask_rows / len(rows) if rows else 0.0,
        "category_counts": count_values(rows, lambda row: row.get("category") or row.get("asset_category")),
        "asset_category_counts": count_values(rows, lambda row: row.get("asset_category") or row.get("category")),
        "source_split_counts": count_values(rows, row_source_split),
        "asset_count": len(asset_keys),
        "asset_keys": asset_keys,
    }


def ensure_pair_export_report(metadata_path: Path, rows: list[dict], write: bool = True) -> tuple[Path, dict, bool]:
    report_path = metadata_path.with_name("pair_export_report.json")
    if report_path.exists():
        return report_path, read_json(report_path), False
    report = pair_export_summary(metadata_path, rows)
    if write:
        write_json(report_path, report)
    return report_path, report, True


def read_train_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        return {}

    parsed = []
    for row in rows:
        try:
            parsed.append(
                {
                    "step": int(float(row.get("step", 0))),
                    "loss": float(row.get("loss")),
                    "unweighted_loss": float(row.get("unweighted_loss")),
                    "lr": float(row.get("lr")),
                }
            )
        except (TypeError, ValueError):
            continue
    if not parsed:
        return {}
    finite_loss = [row["loss"] for row in parsed if math.isfinite(row["loss"])]
    finite_unweighted = [row["unweighted_loss"] for row in parsed if math.isfinite(row["unweighted_loss"])]
    last = parsed[-1]
    return {
        "train_metrics": str(path),
        "final_step": last.get("step", 0),
        "final_loss": last.get("loss"),
        "final_unweighted_loss": last.get("unweighted_loss"),
        "best_loss": min(finite_loss) if finite_loss else None,
        "best_unweighted_loss": min(finite_unweighted) if finite_unweighted else None,
    }


def adapter_file_for_lora(lora_dir: Path) -> Path | None:
    for name in ("pytorch_lora_weights.safetensors", "adapter_model.safetensors", "pytorch_lora_weights.bin"):
        candidate = lora_dir / name
        if candidate.exists():
            return candidate
    return None


def lora_args_from_report(report: dict, output_dir: Path) -> dict:
    return {
        "metadata": report.get("metadata", ""),
        "output_dir": str(output_dir),
        "base_model": report.get("base_model", ""),
        "variant": report.get("variant", ""),
        "resolution": report.get("resolution", ""),
        "max_train_steps": report.get("max_train_steps", ""),
        "rank": report.get("rank", ""),
        "lora_alpha": report.get("lora_alpha", ""),
        "mask_loss_weight": report.get("mask_loss_weight", ""),
        "seam_loss_weight": report.get("seam_loss_weight", ""),
        "object_loss_weight": report.get("object_loss_weight", ""),
    }


def backfill_training_report(report_path: Path, write: bool = True) -> dict:
    report = read_json(report_path)
    lora_dir = report_path.parent
    metadata_path = resolve_existing_path(report.get("metadata"), base_dir=Path.cwd())
    rows = read_jsonl(metadata_path, limit=report.get("rows")) if metadata_path and metadata_path.exists() else []
    updates = {
        "backfilled_at": now_iso(),
        "backfilled_provenance_version": 1,
    }

    if metadata_path and metadata_path.exists():
        export_path, export_report, _ = ensure_pair_export_report(metadata_path, rows, write=write)
        updates.update(
            {
                "metadata": str(metadata_path),
                "metadata_sha256": file_sha256(metadata_path),
                "pair_export_report": str(export_path),
                "pair_export_sha256": export_report.get("metadata_sha256", ""),
                "prompt_family": prompt_family_from_rows(rows),
                "prompt_template": rows[0].get("prompt_template", "") if rows else "",
                "object_mask_rows": sum(
                    1 for row in rows if row.get("object_mask") or row.get("gt_silhouette") or row.get("silhouette")
                ),
                "source_split_counts": count_values(rows, row_source_split),
            }
        )

    updates["loss_recipe"] = loss_recipe_label(
        float(report.get("mask_loss_weight", 0) or 0),
        float(report.get("seam_loss_weight", 0) or 0),
        float(report.get("object_loss_weight", 0) or 0),
    )
    updates["args"] = report.get("args") or lora_args_from_report(report, lora_dir)
    updates.update(read_train_metrics(lora_dir / "train_metrics.csv"))

    adapter_file = adapter_file_for_lora(lora_dir)
    if adapter_file:
        updates["adapter_file"] = str(adapter_file)
        updates["adapter_sha256"] = file_sha256(adapter_file)

    report.update({key: value for key, value in updates.items() if value is not None})
    if write:
        write_json(report_path, report)
    return report


def first_prompt_from_metrics(metrics_path: Path) -> str:
    if not metrics_path.exists():
        return ""
    with metrics_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            prompt = str(row.get("prompt") or "").strip()
            if prompt:
                return prompt
    return ""


def backfill_split_audit(audit_path: Path, write: bool = True) -> dict:
    audit = read_json(audit_path)
    report_path = resolve_existing_path(audit.get("training_report"), base_dir=Path.cwd())
    if not report_path or not report_path.exists():
        report_path = training_report_path(audit.get("lora_weights"))
    if not report_path or not report_path.exists():
        return {"path": str(audit_path), "updated": False, "reason": "no_training_report"}

    train_report = read_json(report_path)
    eval_prompt = audit.get("eval_prompt_template") or first_prompt_from_metrics(audit_path.parent / "per_sample_metrics.csv")
    audit.update(training_provenance_fields(train_report, report_path, eval_prompt))
    audit["backfilled_at"] = now_iso()
    audit["backfilled_provenance_version"] = 1
    if write:
        write_json(audit_path, audit)
    return {"path": str(audit_path), "updated": True, "training_report": str(report_path)}


def backfill_lora_root(lora_root: Path, write: bool = True) -> list[dict]:
    rows = []
    if not lora_root.exists():
        return rows
    for report_path in sorted(lora_root.glob("*/training_report.json")):
        before = read_json(report_path)
        after = backfill_training_report(report_path, write=write)
        rows.append(
            {
                "path": str(report_path),
                "updated": before != after,
                "loss_recipe": after.get("loss_recipe", ""),
                "prompt_family": after.get("prompt_family", ""),
                "adapter_sha256": after.get("adapter_sha256", ""),
            }
        )
    return rows


def backfill_experiment_root(experiment_root: Path, write: bool = True) -> list[dict]:
    rows = []
    if not experiment_root.exists():
        return rows
    for audit_path in sorted(experiment_root.glob("**/split_audit.json")):
        rows.append(backfill_split_audit(audit_path, write=write))
    return rows


def write_summary(path: Path, data: dict) -> None:
    write_json(path, data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill LoRA training provenance into cached benchmark artifacts.")
    parser.add_argument("--lora-root", default="backend/output/completion-benchmark/lora")
    parser.add_argument("--experiment-root", default="backend/output/completion-benchmark/experiments")
    parser.add_argument("--output", default=None, help="Optional JSON summary path.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write = not args.dry_run
    summary = {
        "generated_at": now_iso(),
        "dry_run": args.dry_run,
        "lora_root": args.lora_root,
        "experiment_root": args.experiment_root,
        "lora_reports": backfill_lora_root(Path(args.lora_root), write=write),
        "split_audits": backfill_experiment_root(Path(args.experiment_root), write=write),
    }
    summary["updated_lora_reports"] = sum(1 for row in summary["lora_reports"] if row.get("updated"))
    summary["updated_split_audits"] = sum(1 for row in summary["split_audits"] if row.get("updated"))
    output = Path(args.output) if args.output else None
    if output:
        write_summary(output, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
