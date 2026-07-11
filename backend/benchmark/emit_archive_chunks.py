from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
from pathlib import Path


DEFAULT_ARCHIVE_GLOB = "/content/*results_compact.tar.gz"
DEFAULT_CHUNK_SIZE = 140_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_archive(pattern: str = DEFAULT_ARCHIVE_GLOB) -> Path:
    matches = [Path(value) for value in glob.glob(pattern) if Path(value).is_file()]
    if not matches:
        raise FileNotFoundError(f"No compact result archives match: {pattern}")
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, path.name))


def archive_chunk_lines(
    archive: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    prefix: str = "COLAB_ARCHIVE",
) -> tuple[dict, list[str]]:
    archive = Path(archive)
    if not archive.is_file():
        raise FileNotFoundError(f"Archive does not exist: {archive}")
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    prefix = str(prefix).strip()
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("prefix must contain only letters, numbers, and underscores")

    encoded = base64.b64encode(archive.read_bytes()).decode("ascii")
    chunks = [encoded[offset : offset + chunk_size] for offset in range(0, len(encoded), chunk_size)]
    metadata = {
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "base64_bytes": len(encoded),
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
    }
    lines = [f"{prefix}_META:{json.dumps(metadata, sort_keys=True)}"]
    lines.extend(f"{prefix}_CHUNK_{index:03d}:{chunk}" for index, chunk in enumerate(chunks))
    return metadata, lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a compact Colab result archive as verified base64 chunks.")
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--archive-glob", default=DEFAULT_ARCHIVE_GLOB)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--prefix", default="COLAB_ARCHIVE")
    args = parser.parse_args()

    archive = args.archive or latest_archive(args.archive_glob)
    _, lines = archive_chunk_lines(
        archive,
        chunk_size=args.chunk_size,
        prefix=args.prefix,
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
