"""Compose a content-verified face corpus from multiple split-scoped sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from backend.benchmark.compose_face_training_corpus import (
    SPLITS,
    _artifact_records,
    _copy_verified_asset,
    _sha256,
)
from backend.benchmark.train_dinov2_face_spatial_decoder import (
    _validate_corpus_artifacts,
)


SCHEMA_VERSION = 1


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _load_source(spec: Mapping[str, object]) -> tuple[Path, dict, dict]:
    required = {"root", "summary_sha256", "row_count", "include_splits"}
    missing = required - set(spec)
    if missing:
        raise ValueError(
            "Multisource corpus specification is missing: "
            + ", ".join(sorted(missing))
        )
    root = Path(str(spec["root"])).resolve()
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Corpus summary is missing: {summary_path}")
    observed_summary_sha256 = _sha256(summary_path)
    expected_summary_sha256 = str(spec["summary_sha256"]).lower()
    if observed_summary_sha256 != expected_summary_sha256:
        raise ValueError(
            f"Corpus summary hash mismatch for {root.name}: expected "
            f"{expected_summary_sha256}, observed {observed_summary_sha256}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Corpus source must contain rows: {root}")
    if (
        summary.get("row_count") != len(rows)
        or len(rows) != int(spec["row_count"])
    ):
        raise ValueError(f"Corpus row count does not match its summary: {root}")
    include_splits = tuple(str(value) for value in spec["include_splits"])
    if (
        not include_splits
        or len(include_splits) != len(set(include_splits))
        or any(value not in SPLITS for value in include_splits)
    ):
        raise ValueError(f"Invalid include_splits for corpus source: {root}")
    artifact_binding = _validate_corpus_artifacts(root, rows)
    return root, summary, {
        "root_name": root.name,
        "summary_sha256": observed_summary_sha256,
        "row_count": len(rows),
        "include_splits": list(include_splits),
        **artifact_binding,
    }


def _verify_exclusion_evidence(record: Mapping[str, object]) -> dict:
    if set(record) != {"path", "sha256"}:
        raise ValueError("Exclusion evidence requires path and sha256")
    path = Path(str(record["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Exclusion evidence is missing: {path}")
    observed = _sha256(path)
    expected = str(record["sha256"]).lower()
    if observed != expected:
        raise ValueError(
            "Exclusion evidence hash mismatch: "
            f"expected {expected}, observed {observed}"
        )
    return {
        "name": path.name,
        "sha256": observed,
        "bytes": path.stat().st_size,
    }


def compose_multisource_face_training_corpus(
    source_specs: Sequence[Mapping[str, object]],
    output_root: str | Path,
    *,
    expected_split_counts: Mapping[str, int],
    excluded_rows: Mapping[str, str],
    exclusion_evidence: Mapping[str, object],
    expected_duplicate_rows: int,
) -> dict:
    """Build an atomic, deterministic corpus from verified source summaries."""

    if not source_specs:
        raise ValueError("Multisource corpus requires at least one source")
    if set(expected_split_counts) != set(SPLITS):
        raise ValueError("Expected split counts must cover train/validation/sealed")
    if not excluded_rows or any(not key or not value for key, value in excluded_rows.items()):
        raise ValueError("Every excluded row requires a nonempty reason")
    if int(expected_duplicate_rows) < 0:
        raise ValueError("Expected duplicate row count cannot be negative")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Composite output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    loaded = [_load_source(spec) for spec in source_specs]
    common_fields = ("provider", "source_revision", "license", "privacy")
    for field in common_fields:
        values = {str(summary.get(field, "")) for _, summary, _ in loaded}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f"Multisource corpora disagree on {field}")

    row_owners: dict[str, tuple[Path, dict]] = {}
    identity_owners: dict[str, str] = {}
    exclusion_hits: dict[str, int] = {row_id: 0 for row_id in excluded_rows}
    duplicate_rows = 0
    source_records = []
    for root, summary, provenance in loaded:
        include_splits = set(provenance["include_splits"])
        selected_before_exclusion = 0
        selected_after_exclusion = 0
        duplicate_count = 0
        for row in sorted(summary["rows"], key=lambda item: str(item["row_id"])):
            split = str(row.get("split", ""))
            if split not in include_splits:
                continue
            selected_before_exclusion += 1
            row_id = str(row.get("row_id", ""))
            identity = str(row.get("identity_group", ""))
            if not row_id or not identity:
                raise ValueError("Selected rows require row and identity IDs")
            spec = row.get("spec")
            if isinstance(spec, dict) and any(
                str(spec.get(key, "")) != str(row.get(key, ""))
                for key in ("row_id", "identity_group", "split")
            ):
                raise ValueError(f"Row {row_id} disagrees with its rendering spec")
            if row_id in excluded_rows:
                exclusion_hits[row_id] += 1
                continue
            selected_after_exclusion += 1
            if row_id in row_owners:
                _, prior = row_owners[row_id]
                if _canonical_json(prior) != _canonical_json(row):
                    raise ValueError(
                        f"Duplicate row {row_id} differs across corpus sources"
                    )
                duplicate_rows += 1
                duplicate_count += 1
                continue
            prior_split = identity_owners.get(identity)
            if prior_split is not None and prior_split != split:
                raise ValueError(
                    f"Identity {identity} crosses {prior_split} and {split}"
                )
            identity_owners[identity] = split
            row_owners[row_id] = (root, row)
        source_records.append(
            {
                **provenance,
                "selected_rows_before_exclusion": selected_before_exclusion,
                "selected_rows_after_exclusion": selected_after_exclusion,
                "duplicate_rows": duplicate_count,
            }
        )

    missing_exclusions = [
        row_id for row_id, hits in exclusion_hits.items() if hits != 1
    ]
    if missing_exclusions:
        raise ValueError(
            "Excluded rows must occur exactly once in selected sources: "
            + ", ".join(sorted(missing_exclusions))
        )
    if duplicate_rows != int(expected_duplicate_rows):
        raise ValueError(
            "Duplicate row count mismatch: "
            f"expected {expected_duplicate_rows}, observed {duplicate_rows}"
        )
    split_counts = {
        split: sum(row["split"] == split for _, row in row_owners.values())
        for split in SPLITS
    }
    normalized_expected_counts = {
        split: int(expected_split_counts[split]) for split in SPLITS
    }
    if split_counts != normalized_expected_counts:
        raise ValueError(
            f"Composite split count mismatch: expected "
            f"{normalized_expected_counts}, observed {split_counts}"
        )
    verified_exclusion_evidence = _verify_exclusion_evidence(exclusion_evidence)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        artifact_digest = hashlib.sha256()
        artifact_count = 0
        artifact_bytes = 0
        combined_rows = []
        for row_id in sorted(row_owners):
            source_root, row = row_owners[row_id]
            combined_rows.append(row)
            for logical_name, record in _artifact_records(row):
                size = _copy_verified_asset(
                    source_root,
                    temporary,
                    record,
                )
                artifact_count += 1
                artifact_bytes += size
                artifact_digest.update(
                    (
                        f"{row_id}\0{logical_name}\0"
                        f"{Path(record['path']).as_posix()}\0"
                        f"{str(record['sha256']).lower()}\0{size}\n"
                    ).encode("utf-8")
                )

        source_digest = hashlib.sha256()
        for record in source_records:
            source_digest.update(_canonical_json(record))
            source_digest.update(b"\n")
        first_summary = loaded[0][1]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "provider": "content-verified-multisource-composite",
            "source_provider": first_summary["provider"],
            "source_revision": first_summary["source_revision"],
            "license": first_summary["license"],
            "privacy": first_summary["privacy"],
            "row_count": len(combined_rows),
            "split_counts": split_counts,
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
            "duplicate_rows_removed": duplicate_rows,
            "excluded_rows": [
                {"row_id": row_id, "reason": excluded_rows[row_id]}
                for row_id in sorted(excluded_rows)
            ],
            "exclusion_evidence": verified_exclusion_evidence,
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
            raise ValueError("Composite verification changed after materialization")
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported multisource corpus manifest schema")
    result = compose_multisource_face_training_corpus(
        manifest["sources"],
        args.output_root,
        expected_split_counts=manifest["expected_split_counts"],
        excluded_rows=manifest["excluded_rows"],
        exclusion_evidence=manifest["exclusion_evidence"],
        expected_duplicate_rows=manifest["expected_duplicate_rows"],
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
