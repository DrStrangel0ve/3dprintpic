import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROMPT_TEMPLATE = (
    "Complete the missing half of the same {category} with matching geometry, "
    "lighting, viewpoint, and background. The masked half must contain the missing "
    "{category}, not an empty background."
)


def infer_category(sample: dict) -> str:
    asset_category = str(sample.get("asset_category") or "")
    if asset_category:
        return asset_category.replace("_", " ").replace("-", " ")
    asset_id = str(sample.get("asset_id") or "")
    if asset_id:
        return re.sub(r"[_-]\d+$", "", asset_id).replace("_", " ").replace("-", " ")
    source = str(sample.get("source") or "")
    if source == "procedural":
        return "object"
    return "object"


def render_prompt(sample: dict, prompt_template: str) -> str:
    values = {
        "sample_id": sample.get("id", ""),
        "asset_id": sample.get("asset_id", ""),
        "category": infer_category(sample),
        "source": sample.get("source", ""),
        "completion_mode": sample.get("completion_mode", ""),
    }
    try:
        return prompt_template.format(**values)
    except (KeyError, ValueError):
        return prompt_template


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def counted(rows: list[dict], field: str) -> dict:
    counts = {}
    for row in rows:
        value = str(row.get(field) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def source_split_counts(rows: list[dict]) -> dict:
    counts = {}
    for row in rows:
        value = row_source_split(row) or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def row_weight(row: dict) -> float | None:
    value = row.get("sample_weight", row.get("training_weight"))
    if value in (None, ""):
        return None
    number = float(value)
    if number != number or number < 0:
        raise ValueError(f"sample weight must be finite and non-negative, got {value!r}")
    return number


def main():
    parser = argparse.ArgumentParser(description="Export masked/full image pairs for inpainting adapter training.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="backend/output/completion-benchmark/training_pairs")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT_TEMPLATE,
        help=(
            "Prompt template for each pair. Supports {category}, {sample_id}, {asset_id}, "
            "{source}, and {completion_mode} placeholders."
        ),
    )
    parser.add_argument("--literal-prompt", action="store_true", help="Write --prompt literally instead of formatting placeholders.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based manifest row offset for train/held-out splits.")
    args = parser.parse_args()
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    target_dir = output_dir / "targets"
    object_dir = output_dir / "object_masks"
    for directory in (image_dir, mask_dir, target_dir, object_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.jsonl"
    count = 0
    rows = []
    with open(args.manifest, encoding="utf-8") as manifest_file, metadata_path.open("w", encoding="utf-8") as metadata_file:
        for manifest_index, line in enumerate(manifest_file):
            if manifest_index < args.start_index:
                continue
            if count >= args.limit:
                break
            sample = json.loads(line)
            sample_id = sample["id"]
            image_path = image_dir / f"{sample_id}.png"
            mask_path = mask_dir / f"{sample_id}.png"
            target_path = target_dir / f"{sample_id}.png"
            object_path = object_dir / f"{sample_id}.png"
            shutil.copyfile(sample["masked_image"], image_path)
            shutil.copyfile(sample["mask"], mask_path)
            shutil.copyfile(sample["full_image"], target_path)
            silhouette_source = sample.get("gt_silhouette") or sample.get("silhouette")
            has_object_mask = False
            if silhouette_source:
                silhouette_path = Path(silhouette_source)
                if not silhouette_path.exists():
                    raise FileNotFoundError(f"silhouette mask not found for {sample_id}: {silhouette_source}")
                shutil.copyfile(silhouette_path, object_path)
                has_object_mask = True
            metadata = {
                "id": sample_id,
                "manifest_index": manifest_index,
                "image": str(image_path),
                "mask": str(mask_path),
                "target": str(target_path),
                "prompt": args.prompt if args.literal_prompt else render_prompt(sample, args.prompt),
                "prompt_template": args.prompt,
                "category": infer_category(sample),
                "completion_mode": sample.get("completion_mode"),
                "source": sample.get("source"),
                "asset_id": sample.get("asset_id"),
                "asset_category": sample.get("asset_category"),
                "asset_source_split": sample.get("asset_source_split"),
                "asset_key": sample.get("asset_key"),
                "asset_path": sample.get("asset_path"),
            }
            if has_object_mask:
                metadata["object_mask"] = str(object_path)
            weight = row_weight(sample)
            if weight is not None:
                metadata["sample_weight"] = weight
            rows.append(metadata)
            metadata_file.write(
                json.dumps(metadata)
                + "\n"
            )
            count += 1

    asset_keys = sorted({str(row.get("asset_key") or row.get("asset_path") or row.get("asset_id") or row.get("id") or "") for row in rows if row})
    object_mask_rows = sum(1 for row in rows if row.get("object_mask"))
    sample_weights = [float(row["sample_weight"]) for row in rows if "sample_weight" in row]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "manifest": args.manifest,
        "manifest_sha256": file_sha256(Path(args.manifest)),
        "metadata": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "output_dir": str(output_dir),
        "start_index": args.start_index,
        "limit": args.limit,
        "rows": count,
        "manifest_indices": [row["manifest_index"] for row in rows],
        "first_manifest_index": rows[0]["manifest_index"] if rows else None,
        "last_manifest_index": rows[-1]["manifest_index"] if rows else None,
        "prompt": args.prompt,
        "prompt_mode": "literal" if args.literal_prompt else "templated",
        "literal_prompt": bool(args.literal_prompt),
        "object_mask_rows": object_mask_rows,
        "object_mask_coverage": object_mask_rows / count if count else 0.0,
        "sample_weight_rows": len(sample_weights),
        "sample_weight_min": min(sample_weights) if sample_weights else "",
        "sample_weight_mean": sum(sample_weights) / len(sample_weights) if sample_weights else "",
        "sample_weight_max": max(sample_weights) if sample_weights else "",
        "category_counts": counted(rows, "category"),
        "asset_category_counts": counted(rows, "asset_category"),
        "source_split_counts": source_split_counts(rows),
        "asset_count": len(asset_keys),
        "asset_keys": asset_keys,
    }
    report_path = output_dir / "pair_export_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(metadata_path)
    print(report_path)
    print(f"pairs={count}")


if __name__ == "__main__":
    main()
