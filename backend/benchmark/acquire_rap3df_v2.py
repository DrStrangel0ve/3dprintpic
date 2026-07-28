"""Authenticate and stage a bounded RAP3DF V2 evaluation slice.

The publisher's ZIP is convenient but does not include per-file provenance.
This tool cross-checks selected archive members against the anonymous Mendeley
public API and emits the manifest consumed by ``rap3df_corpus``. Real volunteer
images stay outside version control and remain evaluation-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile, ZipInfo

from backend.benchmark.rap3df_corpus import (
    DATABASE_BYTES,
    DATABASE_FILENAME,
    DATABASE_SHA256,
    DATASET_DOI,
    DATASET_LICENSE,
    DATASET_VERSION,
    DEFAULT_IDENTITY_LIMIT,
    POSES,
    _reject_duplicate_object_pairs,
    _safe_relative_path,
    select_database_rows,
)


DATASET_ID = "kpdkpcs8zb"
PUBLIC_API_BASE = "https://data.mendeley.com/public-api"
ARCHIVE_V2_ROOT = PurePosixPath("RAP3DF V2/V2")
LOGICAL_V2_ROOT = PurePosixPath("rap3df_data_02")
MAX_API_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_AUTHENTICATED_MEMBER_BYTES = 16 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 500.0


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any, *, label: str) -> Any:
    try:
        return json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"RAP3DF {label} is not strict UTF-8 JSON") from exc


def _http_json(url: str) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.mendeley-public-dataset.1+json",
            "User-Agent": "3dprintpic-rap3df-provenance/1",
        },
    )
    with urlopen(request, timeout=30) as response:
        body = response.read(MAX_API_RESPONSE_BYTES + 1)
    if len(body) > MAX_API_RESPONSE_BYTES:
        raise ValueError("RAP3DF publisher API response exceeds the safety limit")
    return _json_bytes(body, label="publisher API response")


def _list_response(value: Any, *, label: str) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"RAP3DF publisher {label} response must be a list of objects")
    return value


def _folder_index(folders: list[dict]) -> tuple[str, dict[str, str]]:
    roots = [
        item
        for item in folders
        if item.get("name") == "V2" and not item.get("parent_id")
    ]
    if len(roots) != 1 or not str(roots[0].get("id") or ""):
        raise ValueError("RAP3DF publisher metadata has no unique V2 root folder")
    root_id = str(roots[0]["id"])
    identities: dict[str, str] = {}
    for item in folders:
        if item.get("parent_id") != root_id:
            continue
        name = str(item.get("name") or "")
        folder_id = str(item.get("id") or "")
        if not name or not folder_id or name in identities:
            raise ValueError("RAP3DF publisher identity folders are ambiguous")
        identities[name] = folder_id
    return root_id, identities


def _files_for_folder(folder_id: str) -> list[dict]:
    endpoint = (
        f"{PUBLIC_API_BASE}/datasets/{DATASET_ID}/files"
        f"?folder_id={quote(folder_id, safe='')}&version={DATASET_VERSION}"
    )
    return _list_response(_http_json(endpoint), label="files")


def _file_index(files: list[dict], *, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    folded: set[str] = set()
    for record in files:
        filename = str(record.get("filename") or "")
        key = filename.casefold()
        if not filename or filename in indexed or key in folded:
            raise ValueError(f"RAP3DF publisher {label} files are ambiguous")
        indexed[filename] = record
        folded.add(key)
    return indexed


def _manifest_record(record: dict, logical_path: PurePosixPath) -> dict:
    details = record.get("content_details")
    if not isinstance(details, dict):
        raise ValueError(f"RAP3DF publisher file lacks content details: {logical_path}")
    filename = str(record.get("filename") or "")
    file_id = str(record.get("id") or "")
    content_id = str(details.get("id") or "")
    digest = str(details.get("sha256_hash") or "").lower()
    size = details.get("size")
    record_size = record.get("size")
    status = record.get("status")
    if (
        filename != logical_path.name
        or not file_id
        or not content_id
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or record_size != size
        or status != "COMPLETED"
    ):
        raise ValueError(f"RAP3DF publisher file metadata is invalid: {logical_path}")
    return {
        "path": logical_path.as_posix(),
        "filename": filename,
        "id": file_id,
        "status": status,
        "content_details": {
            "id": content_id,
            "sha256_hash": digest,
            "size": size,
        },
    }


def _archive_members(archive: ZipFile) -> dict[str, ZipInfo]:
    members: dict[str, ZipInfo] = {}
    folded: set[str] = set()
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename
        relative = _safe_relative_path(name, label="archive member")
        normalized = relative.as_posix()
        key = normalized.casefold()
        if normalized in members or key in folded:
            raise ValueError(f"RAP3DF archive has an ambiguous member: {normalized}")
        members[normalized] = info
        folded.add(key)
    return members


def _verified_member(
    archive: ZipFile,
    members: dict[str, ZipInfo],
    archive_path: PurePosixPath,
    manifest_record: dict,
) -> bytes:
    info = members.get(archive_path.as_posix())
    if info is None:
        raise ValueError(f"RAP3DF archive member is missing: {archive_path}")
    details = manifest_record["content_details"]
    expected_size = details["size"]
    if (
        expected_size > MAX_AUTHENTICATED_MEMBER_BYTES
        or info.file_size != expected_size
        or info.compress_size <= 0
        or info.file_size / info.compress_size > MAX_ZIP_COMPRESSION_RATIO
    ):
        raise ValueError(
            f"RAP3DF archive member has unsafe size metadata: {archive_path}"
        )
    with archive.open(info, "r") as handle:
        value = handle.read(expected_size + 1)
    if len(value) != expected_size or _sha256_bytes(value) != details["sha256_hash"]:
        raise ValueError(f"RAP3DF archive member failed publisher authentication: {archive_path}")
    return value


def _write_verified(root: Path, relative: PurePosixPath, value: bytes) -> Path:
    target = (root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("RAP3DF output path escapes the staging root") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = _sha256_bytes(value)
    if target.exists():
        if not target.is_file() or target.stat().st_size != len(value) or _sha256(target) != digest:
            raise ValueError(f"RAP3DF staging target conflicts with verified data: {relative}")
        return target
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _acquire_rap3df_slice_into(
    archive_path: str | Path,
    output_root: str | Path,
    *,
    identity_limit: int = DEFAULT_IDENTITY_LIMIT,
    poses: tuple[str, ...] = POSES,
) -> dict:
    """Stage selected files and an authenticated importer manifest."""

    archive_path = Path(archive_path).resolve()
    output_root = Path(output_root).resolve()
    if not archive_path.is_file():
        raise ValueError("RAP3DF publisher archive is missing")

    folders_endpoint = (
        f"{PUBLIC_API_BASE}/datasets/{DATASET_ID}/folders/{DATASET_VERSION}"
    )
    folders = _list_response(_http_json(folders_endpoint), label="folders")
    v2_folder_id, identity_folders = _folder_index(folders)
    root_files = _file_index(_files_for_folder(v2_folder_id), label="V2 root")
    database_api = root_files.get(DATABASE_FILENAME)
    if database_api is None:
        raise ValueError("RAP3DF publisher metadata omits database.json")
    database_record = _manifest_record(
        database_api,
        PurePosixPath(DATABASE_FILENAME),
    )

    with ZipFile(archive_path) as archive:
        members = _archive_members(archive)
        database_bytes = _verified_member(
            archive,
            members,
            ARCHIVE_V2_ROOT / DATABASE_FILENAME,
            database_record,
        )
        if len(database_bytes) != DATABASE_BYTES or _sha256_bytes(database_bytes) != DATABASE_SHA256:
            raise ValueError("RAP3DF database.json does not match the pinned V4 release")
        database = _json_bytes(database_bytes, label="database.json")
        if not isinstance(database, dict):
            raise ValueError("RAP3DF database.json root must be an object")
        selected = select_database_rows(
            database,
            identity_limit=identity_limit,
            poses=tuple(poses),
        )

        manifest_records = [database_record]
        staged_records: list[dict] = []
        _write_verified(output_root, PurePosixPath(DATABASE_FILENAME), database_bytes)
        for identity in sorted({row["identity"] for row in selected}):
            folder_id = identity_folders.get(identity)
            if folder_id is None:
                raise ValueError(f"RAP3DF publisher metadata omits identity {identity}")
            publisher_files = _file_index(
                _files_for_folder(folder_id),
                label=f"identity {identity}",
            )
            rows = [row for row in selected if row["identity"] == identity]
            requested = sorted(
                {
                    PurePosixPath(row[kind])
                    for row in rows
                    for kind in ("rgb_path", "depth_path")
                },
                key=lambda path: path.as_posix(),
            )
            for logical_path in requested:
                api_record = publisher_files.get(logical_path.name)
                if api_record is None:
                    raise ValueError(f"RAP3DF publisher metadata omits {logical_path}")
                manifest_record = _manifest_record(api_record, logical_path)
                archive_member = ARCHIVE_V2_ROOT / identity / logical_path.name
                value = _verified_member(
                    archive,
                    members,
                    archive_member,
                    manifest_record,
                )
                _write_verified(output_root, logical_path, value)
                manifest_records.append(manifest_record)
                staged_records.append(
                    {
                        "path": logical_path.as_posix(),
                        "sha256": manifest_record["content_details"]["sha256_hash"],
                        "size": manifest_record["content_details"]["size"],
                    }
                )

    manifest = {
        "dataset_doi": DATASET_DOI,
        "dataset_version": DATASET_VERSION,
        "license": DATASET_LICENSE,
        "files": sorted(manifest_records, key=lambda item: item["path"]),
    }
    manifest_path = output_root / "mendeley-manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_verified(output_root, PurePosixPath(manifest_path.name), manifest_bytes)

    report = {
        "schema_version": 1,
        "dataset_doi": DATASET_DOI,
        "dataset_version": DATASET_VERSION,
        "license": DATASET_LICENSE,
        "evaluation_only_real_volunteer_data": True,
        "training_eligible": False,
        "archive": {
            "size": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        },
        "publisher_api": {
            "base": PUBLIC_API_BASE,
            "v2_folder_id": v2_folder_id,
        },
        "selection": {
            "identity_limit": identity_limit,
            "identities": sorted({row["identity"] for row in selected}),
            "poses": list(poses),
            "row_count": len(selected),
        },
        "manifest": {
            "path": manifest_path.name,
            "sha256": _sha256_bytes(manifest_bytes),
            "file_count": len(manifest_records),
        },
        "staged_assets": staged_records,
    }
    report_path = output_root / "acquisition-report.json"
    report_bytes = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_verified(output_root, PurePosixPath(report_path.name), report_bytes)
    return report


def acquire_rap3df_slice(
    archive_path: str | Path,
    output_root: str | Path,
    *,
    identity_limit: int = DEFAULT_IDENTITY_LIMIT,
    poses: tuple[str, ...] = POSES,
) -> dict:
    """Authenticate a slice and publish it only after every check passes."""

    output_root = Path(output_root).resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        raise ValueError("RAP3DF staging target conflicts with an existing path")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-",
            dir=output_root.parent,
        )
    )
    try:
        report = _acquire_rap3df_slice_into(
            archive_path,
            temporary,
            identity_limit=identity_limit,
            poses=poses,
        )
        os.replace(temporary, output_root)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Authenticate and stage a bounded RAP3DF V2 evaluation slice."
    )
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--identity-limit", type=int, default=DEFAULT_IDENTITY_LIMIT)
    parser.add_argument("--pose", action="append", choices=POSES)
    args = parser.parse_args()
    report = acquire_rap3df_slice(
        args.archive,
        args.output_root,
        identity_limit=args.identity_limit,
        poses=tuple(args.pose or POSES),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
