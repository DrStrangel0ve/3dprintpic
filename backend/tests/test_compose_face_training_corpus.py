import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from backend.benchmark.compose_face_training_corpus import (
    compose_face_training_corpus,
)
from backend.benchmark.train_face_surface_fusion_adapter import FACE_PART_NAMES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path, relative: str, payload: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": relative, "sha256": _sha256(path)}


def _corpus(root: Path, split: str, row_id: str, identity: str) -> None:
    row_root = f"rows/{row_id}"
    row = {
        "row_id": row_id,
        "identity_group": identity,
        "split": split,
        "expression": "neutral",
        "render": {"face_bbox_height_pixels": 96},
        "source": _artifact(root, f"{row_root}/source.png", b"source-" + split.encode()),
        "exact_depth": _artifact(
            root, f"{row_root}/exact_depth.npy", b"depth-" + split.encode()
        ),
        "selection_mask": _artifact(
            root, f"{row_root}/selection_mask.png", b"mask-" + split.encode()
        ),
        "exact_face_parts": {
            name: _artifact(
                root,
                f"{row_root}/{name}.png",
                f"{split}-{name}".encode(),
            )
            for name in FACE_PART_NAMES
        },
    }
    summary = {
        "provider": "fixture",
        "source_revision": "fixture-revision",
        "license": "Apache-2.0",
        "privacy": "synthetic fixture",
        "row_count": 1,
        "rows": [row],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


class ComposeFaceTrainingCorpusTests(unittest.TestCase):
    def _sources(self, root: Path) -> dict[str, Path]:
        sources = {}
        for split in ("train", "validation", "sealed"):
            source = root / split
            _corpus(source, split, f"{split}-row", f"{split}-identity")
            sources[split] = source
        return sources

    def _compose(self, sources: dict[str, Path], output: Path) -> dict:
        return compose_face_training_corpus(
            sources,
            output,
            expected_summary_sha256={
                split: _sha256(root / "summary.json")
                for split, root in sources.items()
            },
            expected_split_counts={split: 1 for split in sources},
        )

    def test_composes_deterministic_disjoint_content_verified_corpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = self._sources(root)
            first = self._compose(sources, root / "first")
            second = self._compose(sources, root / "second")

            self.assertEqual(first["row_count"], 3)
            self.assertEqual(
                first["split_counts"],
                {"train": 1, "validation": 1, "sealed": 1},
            )
            self.assertEqual(first["asset_count"], 27)
            self.assertEqual(first["summary_sha256"], second["summary_sha256"])
            self.assertEqual(
                (root / "first" / "summary.json").read_bytes(),
                (root / "second" / "summary.json").read_bytes(),
            )
            self.assertEqual(
                (root / "first" / "rows/train-row/source.png").read_bytes(),
                b"source-train",
            )

    def test_rejects_source_asset_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = self._sources(root)
            (sources["validation"] / "rows/validation-row/source.png").write_bytes(
                b"substituted"
            )
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self._compose(sources, root / "output")

    def test_rejects_identity_leakage_across_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = self._sources(root)
            summary_path = sources["sealed"] / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["rows"][0]["identity_group"] = "train-identity"
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "crosses train and sealed"):
                self._compose(sources, root / "output")

    def test_rejects_row_leakage_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = self._sources(root)
            output = root / "output"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                self._compose(sources, output)

            output.rmdir()
            summary_path = sources["sealed"] / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["rows"][0]["row_id"] = "train-row"
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "appears in train and sealed"):
                self._compose(sources, output)

    def test_rejects_summary_substitution_and_wrong_expected_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = self._sources(root)
            hashes = {
                split: _sha256(source / "summary.json")
                for split, source in sources.items()
            }
            hashes["validation"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "summary hash mismatch"):
                compose_face_training_corpus(
                    sources,
                    root / "wrong-hash",
                    expected_summary_sha256=hashes,
                    expected_split_counts={split: 1 for split in sources},
                )
            with self.assertRaisesRegex(ValueError, "row count"):
                compose_face_training_corpus(
                    sources,
                    root / "wrong-count",
                    expected_summary_sha256={
                        split: _sha256(source / "summary.json")
                        for split, source in sources.items()
                    },
                    expected_split_counts={
                        "train": 2,
                        "validation": 1,
                        "sealed": 1,
                    },
                )


if __name__ == "__main__":
    unittest.main()
