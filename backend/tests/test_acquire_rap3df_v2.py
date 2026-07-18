import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from backend.benchmark import acquire_rap3df_v2 as acquire
from backend.benchmark.rap3df_corpus import load_mendeley_manifest


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_record(filename: str, value: bytes, suffix: str) -> dict:
    return {
        "filename": filename,
        "id": f"file-{suffix}",
        "status": "COMPLETED",
        "size": len(value),
        "content_details": {
            "id": f"content-{suffix}",
            "sha256_hash": _sha(value),
            "size": len(value),
        },
    }


def _fixture(tmp_path: Path, monkeypatch):
    identity = "IDENTITY1"
    samples = {"front": "AAA", "left": "BBB"}
    database = {
        "_faces": [identity],
        identity: {
            pose: [
                {
                    "rgb": f"rap3df_data_02/{identity}/rgb_{sample}.bmp",
                    "depth_data_with_bg": (
                        f"rap3df_data_02/{identity}/depth_bgRm_{sample}.data"
                    ),
                }
            ]
            for pose, sample in samples.items()
        },
    }
    database_bytes = json.dumps(database, separators=(",", ":")).encode("utf-8")
    monkeypatch.setattr(acquire, "DATABASE_BYTES", len(database_bytes))
    monkeypatch.setattr(acquire, "DATABASE_SHA256", _sha(database_bytes))

    values = {"database.json": database_bytes}
    for sample in samples.values():
        values[f"rgb_{sample}.bmp"] = f"rgb-{sample}".encode()
        values[f"depth_bgRm_{sample}.data"] = f"depth-{sample}".encode()

    archive_path = tmp_path / "publisher.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("RAP3DF V2/V2/database.json", database_bytes)
        for filename, value in values.items():
            if filename == "database.json":
                continue
            archive.writestr(f"RAP3DF V2/V2/{identity}/{filename}", value)

    folders = [
        {"id": "v2-folder", "name": "V2"},
        {"id": "identity-folder", "name": identity, "parent_id": "v2-folder"},
    ]
    root_files = [_file_record("database.json", database_bytes, "database")]
    identity_files = [
        _file_record(filename, value, filename)
        for filename, value in values.items()
        if filename != "database.json"
    ]

    def fake_http(url: str):
        if url.endswith("/folders/4"):
            return folders
        if "folder_id=v2-folder" in url:
            return root_files
        if "folder_id=identity-folder" in url:
            return identity_files
        raise AssertionError(url)

    monkeypatch.setattr(acquire, "_http_json", fake_http)
    return archive_path, identity, samples, identity_files


def test_acquire_stages_authenticated_bounded_slice(tmp_path, monkeypatch):
    archive_path, identity, samples, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "staged"

    report = acquire.acquire_rap3df_slice(
        archive_path,
        output,
        identity_limit=1,
        poses=("front", "left"),
    )

    assert report["selection"] == {
        "identity_limit": 1,
        "identities": [identity],
        "poses": ["front", "left"],
        "row_count": 2,
    }
    assert report["manifest"]["file_count"] == 5
    assert report["training_eligible"] is False
    records, provenance = load_mendeley_manifest(output / "mendeley-manifest.json")
    assert provenance["file_count"] == 5
    assert set(records) == {
        "database.json",
        *{
            f"rap3df_data_02/{identity}/{kind}_{sample}.{extension}"
            for sample in samples.values()
            for kind, extension in (("rgb", "bmp"), ("depth_bgRm", "data"))
        },
    }
    assert (output / "acquisition-report.json").is_file()


def test_acquire_rejects_publisher_hash_mismatch(tmp_path, monkeypatch):
    archive_path, _, _, identity_files = _fixture(tmp_path, monkeypatch)
    identity_files[0]["content_details"]["sha256_hash"] = "0" * 64

    with pytest.raises(ValueError, match="publisher authentication"):
        acquire.acquire_rap3df_slice(
            archive_path,
            tmp_path / "staged",
            identity_limit=1,
            poses=("front", "left"),
        )
    assert not (tmp_path / "staged").exists()


def test_acquire_rejects_member_size_before_decompression(tmp_path, monkeypatch):
    archive_path, _, _, identity_files = _fixture(tmp_path, monkeypatch)
    identity_files[0]["size"] += 1
    identity_files[0]["content_details"]["size"] += 1

    with pytest.raises(ValueError, match="unsafe size metadata"):
        acquire.acquire_rap3df_slice(
            archive_path,
            tmp_path / "staged",
            identity_limit=1,
            poses=("front", "left"),
        )
    assert not (tmp_path / "staged").exists()


def test_acquire_rejects_ambiguous_v2_root(tmp_path, monkeypatch):
    archive_path, _, _, _ = _fixture(tmp_path, monkeypatch)

    def fake_http(url: str):
        if url.endswith("/folders/4"):
            return [{"id": "one", "name": "V2"}, {"id": "two", "name": "V2"}]
        raise AssertionError(url)

    monkeypatch.setattr(acquire, "_http_json", fake_http)
    with pytest.raises(ValueError, match="no unique V2 root"):
        acquire.acquire_rap3df_slice(
            archive_path,
            tmp_path / "staged",
            identity_limit=1,
            poses=("front", "left"),
        )


def test_acquire_refuses_conflicting_existing_target(tmp_path, monkeypatch):
    archive_path, identity, _, _ = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "staged"
    output.mkdir()
    (output / "database.json").write_bytes(b"conflict")

    with pytest.raises(ValueError, match="staging target conflicts"):
        acquire.acquire_rap3df_slice(
            archive_path,
            output,
            identity_limit=1,
            poses=("front", "left"),
        )
    assert not (output / "rap3df_data_02" / identity).exists()
