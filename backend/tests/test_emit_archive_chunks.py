from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from backend.benchmark.emit_archive_chunks import archive_chunk_lines, latest_archive


class EmitArchiveChunksTest(unittest.TestCase):
    def test_chunks_round_trip_with_verified_metadata(self):
        payload = bytes(range(256)) * 17
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "result_compact.tar.gz"
            archive.write_bytes(payload)

            metadata, lines = archive_chunk_lines(
                archive,
                chunk_size=113,
                prefix="G4_RESULT",
            )

        parsed_metadata = json.loads(lines[0].split(":", 1)[1])
        encoded = "".join(line.split(":", 1)[1] for line in lines[1:])
        reconstructed = base64.b64decode(encoded)
        self.assertEqual(metadata, parsed_metadata)
        self.assertEqual(reconstructed, payload)
        self.assertEqual(metadata["archive_bytes"], len(payload))
        self.assertEqual(metadata["archive_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(metadata["chunk_count"], len(lines) - 1)

    def test_latest_archive_is_deterministic_and_validation_is_strict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a_results_compact.tar.gz"
            second = root / "b_results_compact.tar.gz"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            timestamp = 1_700_000_000_000_000_000
            first.touch()
            second.touch()
            os.utime(first, ns=(timestamp, timestamp))
            os.utime(second, ns=(timestamp, timestamp))

            selected = latest_archive(str(root / "*results_compact.tar.gz"))
            with self.assertRaisesRegex(ValueError, "chunk_size"):
                archive_chunk_lines(first, chunk_size=0)
            with self.assertRaisesRegex(ValueError, "prefix"):
                archive_chunk_lines(first, prefix="bad-prefix")

        self.assertEqual(selected.name, "b_results_compact.tar.gz")


if __name__ == "__main__":
    unittest.main()
