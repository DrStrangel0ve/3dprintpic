"""Compose disjoint face-relief corpora into one content-verified dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from backend.benchmark.train_dinov2_face_spatial_decoder import (
    _validate_corpus_artifacts,
)
from backend.benchmark.train_face_surface_fusion_adapter import FACE_PART_NAMES


SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "sealed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_records(row: dict) -> list[tuple[str, dict]]:
    return [
        ("source", row["source"]),
        ("exact_depth", row["exact_depth"]),
        ("selection_mask", row["selection_mask"]),
        *[
            (f"exact_face_parts/{name}", row["exact_face_parts"][name])
            for name in FACE_PART_NAMES
        ],
    ]


def _load_source(
    root: Path,
    split: str,
    *,
    expected_summary_sha256: str,
    expected_row_count: int,
) -> tuple[dict, dict]:
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"{split} corpus summary is missing: {summary_path}")
    observed_summary_sha256 = _sha256(summary_path)
    if observed_summary_sha256 != str(expected_summary_sha256).lower():
        raise ValueError(
            f"{split} corpus summary hash mismatch: expected "
            f"{expected_summary_sha256}, observed {observed_summary_sha256}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{split} corpus must contain at least one row")
    if (
        summary.get("row_count") != len(rows)
        or len(rows) != int(expected_row_count)
    ):
        raise ValueError(f"{split} corpus row count does not match its summary")
    mismatched = [
        str(row.get("row_id", ""))
        for row in rows
        if row.get("split") != split
    ]
    if mismatched:
        raise ValueError(
            f"{split} corpus contains rows assigned to another split: "
            + ", ".join(mismatched[:5])
        )
    artifact_binding = _validate_corpus_artifacts(root, rows)
    return summary, {
        "split": split,
        "root_name": root.name,
        "summary_sha256": observed_summary_sha256,
        "row_count": len(rows),
        **artifact_binding,
    }


def _copy_verified_asset(
    source_root: Path,
    destination_root: Path,
    record: dict,
) -> int:
    relative = Path(str(record["path"]))
    source = (source_root / relative).resolve()
    destination = destination_root / relative
    if destination.exists():
        raise ValueError(f"Composite asset path collision: {relative.as_posix()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    expected = str(record["sha256"]).lower()
    observed = _sha256(destination)
    if observed != expected:
        raise ValueError(
            f"Composite asset hash mismatch for {relative.as_posix()}: "
            f"expected {expected}, observed {observed}"
        )
    return destination.stat().st_size


def compose_face_training_corpus(
    sources: Mapping[str, str | Path],
    output_root: str | Path,
    *,
    expected_summary_sha256: Mapping[str, str],
    expected_split_counts: Mapping[str, int],
) -> dict:
    """Build an atomic, deterministic train/validation/sealed corpus."""

    if (
        set(sources) != set(SPLITS)
        or set(expected_summary_sha256) != set(SPLITS)
        or set(expected_split_counts) != set(SPLITS)
    ):
        raise ValueError("Composite corpus requires train, validation, and sealed roots")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Composite output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, tuple[Path, dict, dict]] = {}
    row_owners: dict[str, str] = {}
    identity_owners: dict[str, str] = {}
    combined_rows = []
    for split in SPLITS:
        root = Path(sources[split]).resolve()
        summary, provenance = _load_source(
            root,
            split,
            expected_summary_sha256=expected_summary_sha256[split],
            expected_row_count=expected_split_counts[split],
        )
        loaded[split] = (root, summary, provenance)
        for row in sorted(summary["rows"], key=lambda item: str(item["row_id"])):
            row_id = str(row.get("row_id", ""))
            identity = str(row.get("identity_group", ""))
            if not row_id or not identity:
                raise ValueError(f"{split} corpus rows require row and identity IDs")
            spec = row.get("spec")
            if isinstance(spec, dict) and any(
                str(spec.get(key, "")) != str(row.get(key, ""))
                for key in ("row_id", "identity_group", "split")
            ):
                raise ValueError(f"Row {row_id} disagrees with its rendering spec")
            if row_id in row_owners:
                raise ValueError(
                    f"Row {row_id} appears in {row_owners[row_id]} and {split}"
                )
            if identity in identity_owners and identity_owners[identity] != split:
                raise ValueError(
                    f"Identity {identity} crosses {identity_owners[identity]} "
                    f"and {split}"
                )
            row_owners[row_id] = split
            identity_owners[identity] = split
            combined_rows.append(row)

    common_fields = ("provider", "source_revision", "license", "privacy")
    for field in common_fields:
        values = {str(loaded[split][1].get(field, "")) for split in SPLITS}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f"Composite sources disagree on {field}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        artifact_digest = hashlib.sha256()
        artifact_count = 0
        artifact_bytes = 0
        for split in SPLITS:
            source_root, summary, _provenance = loaded[split]
            for row in sorted(summary["rows"], key=lambda item: str(item["row_id"])):
                for logical_name, record in _artifact_records(row):
                    size = _copy_verified_asset(source_root, temporary, record)
                    artifact_count += 1
                    artifact_bytes += size
                    artifact_digest.update(
                        (
                            f"{row['row_id']}\0{logical_name}\0"
                            f"{Path(record['path']).as_posix()}\0"
                            f"{str(record['sha256']).lower()}\0{size}\n"
                        ).encode("utf-8")
                    )

        source_records = [loaded[split][2] for split in SPLITS]
        source_digest = hashlib.sha256()
        for record in source_records:
            source_digest.update(
                (
                    f"{record['split']}\0{record['root_name']}\0"
                    f"{record['summary_sha256']}\0{record['row_count']}\n"
                ).encode("utf-8")
            )
        first_summary = loaded[SPLITS[0]][1]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "provider": "content-verified-composite",
            "source_provider": first_summary["provider"],
            "source_revision": first_summary["source_revision"],
            "license": first_summary["license"],
            "privacy": first_summary["privacy"],
            "row_count": len(combined_rows),
            "split_counts": {
                split: int(
                    sum(row["split"] == split for row in combined_rows)
                )
                for split in SPLITS
            },
            "identity_counts": {
                split: len(
                    {
                        str(row["identity_group"])
                        for row in combined_rows
                        if row["split"] == split
                    }
                )
                for split in SPLITS
            },
            "source_summary_manifest_sha256": source_digest.hexdigest(),
            "sources": source_records,
            "asset_count": artifact_count,
            "asset_bytes": artifact_bytes,
            "asset_manifest_sha256": artifact_digest.hexdigest(),
            "rows": combined_rows,
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verified = _validate_corpus_artifacts(temporary, combined_rows)
        if (
            verified["artifact_count"] != artifact_count
            or verified["artifact_bytes"] != artifact_bytes
            or verified["ordered_artifact_content_sha256"]
            != artifact_digest.hexdigest()
        ):
            raise ValueError("Composite corpus verification changed after materialization")
        temporary.replace(output)
        return {
            **summary,
            "summary_sha256": _sha256(output / "summary.json"),
            "ordered_artifact_content_sha256": verified[
                "ordered_artifact_content_sha256"
            ],
        }
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--sealed-root", required=True)
    parser.add_argument("--train-summary-sha256", required=True)
    parser.add_argument("--validation-summary-sha256", required=True)
    parser.add_argument("--sealed-summary-sha256", required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--validation-count", type=int, required=True)
    parser.add_argument("--sealed-count", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = compose_face_training_corpus(
        {
            "train": args.train_root,
            "validation": args.validation_root,
            "sealed": args.sealed_root,
        },
        args.output_root,
        expected_summary_sha256={
            "train": args.train_summary_sha256,
            "validation": args.validation_summary_sha256,
            "sealed": args.sealed_summary_sha256,
        },
        expected_split_counts={
            "train": args.train_count,
            "validation": args.validation_count,
            "sealed": args.sealed_count,
        },
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
