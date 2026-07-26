import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from backend.benchmark.compose_multisource_face_training_corpus import (
    compose_multisource_face_training_corpus,
)
from backend.benchmark.train_face_surface_fusion_adapter import FACE_PART_NAMES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path, relative: str, payload: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": relative, "sha256": _sha256(path)}


def _row(root: Path, split: str, row_id: str, identity: str) -> dict:
    row_root = f"rows/{row_id}"
    return {
        "row_id": row_id,
        "identity_group": identity,
        "split": split,
        "expression": "neutral",
        "source": _artifact(root, f"{row_root}/source.png", f"{row_id}-rgb".encode()),
        "exact_depth": _artifact(
            root,
            f"{row_root}/exact_depth.npy",
            f"{row_id}-depth".encode(),
        ),
        "selection_mask": _artifact(
            root,
            f"{row_root}/selection_mask.png",
            f"{row_id}-mask".encode(),
        ),
        "exact_face_parts": {
            name: _artifact(
                root,
                f"{row_root}/{name}.png",
                f"{row_id}-{name}".encode(),
            )
            for name in FACE_PART_NAMES
        },
    }


def _corpus(root: Path, rows: list[dict]) -> dict:
    summary = {
        "provider": "fixture",
        "source_revision": "fixture-revision",
        "license": "Apache-2.0",
        "privacy": "synthetic fixture",
        "row_count": len(rows),
        "rows": rows,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "root": str(root),
        "summary_sha256": _sha256(root / "summary.json"),
        "row_count": len(rows),
    }


class ComposeMultisourceFaceTrainingCorpusTests(unittest.TestCase):
    def _fixture(self, root: Path):
        first = root / "first"
        duplicate = root / "duplicate"
        held_out = root / "held-out"
        excluded = root / "excluded"
        train_a = _row(first, "train", "train-a", "train-id-a")
        train_b = _row(first, "train", "train-b", "train-id-b")
        validation = _row(first, "validation", "validation-a", "validation-id")
        duplicate_b = _row(duplicate, "train", "train-b", "train-id-b")
        train_c = _row(duplicate, "train", "train-c", "train-id-c")
        bad = _row(excluded, "train", "detector-failed", "train-id-d")
        sealed = _row(held_out, "sealed", "sealed-a", "sealed-id")
        validation_only = _row(
            held_out,
            "validation",
            "validation-a",
            "validation-id",
        )
        specs = [
            {
                **_corpus(first, [train_a, train_b, validation]),
                "include_splits": ["train"],
            },
            {
                **_corpus(duplicate, [duplicate_b, train_c]),
                "include_splits": ["train"],
            },
            {
                **_corpus(excluded, [bad]),
                "include_splits": ["train"],
            },
            {
                **_corpus(held_out, [validation_only, sealed]),
                "include_splits": ["validation", "sealed"],
            },
        ]
        evidence = root / "exclusion-evidence.json"
        evidence.write_text('{"detector-failed": true}\n', encoding="utf-8")
        return specs, {
            "path": str(evidence),
            "sha256": _sha256(evidence),
        }

    def _compose(self, root: Path, output_name: str = "output"):
        specs, evidence = self._fixture(root)
        return compose_multisource_face_training_corpus(
            specs,
            root / output_name,
            expected_split_counts={"train": 3, "validation": 1, "sealed": 1},
            excluded_rows={"detector-failed": "production detector failed"},
            exclusion_evidence=evidence,
            expected_duplicate_rows=1,
        )

    def test_composes_deterministic_filtered_deduplicated_corpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._compose(root, "first-output")
            second_root = root / "second-fixture"
            second_root.mkdir()
            second = self._compose(second_root, "second-output")

            self.assertEqual(first["split_counts"], {
                "train": 3,
                "validation": 1,
                "sealed": 1,
            })
            self.assertEqual(first["duplicate_rows_removed"], 1)
            self.assertEqual(first["row_count"], 5)
            self.assertEqual(first["asset_count"], 45)
            self.assertEqual(first["summary_sha256"], second["summary_sha256"])
            self.assertEqual(
                [row["row_id"] for row in first["rows"]],
                [
                    "sealed-a",
                    "train-a",
                    "train-b",
                    "train-c",
                    "validation-a",
                ],
            )

    def test_rejects_nonidentical_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs, evidence = self._fixture(root)
            duplicate_summary = Path(specs[1]["root"]) / "summary.json"
            payload = json.loads(duplicate_summary.read_text(encoding="utf-8"))
            payload["rows"][0]["expression"] = "smile"
            duplicate_summary.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            specs[1]["summary_sha256"] = _sha256(duplicate_summary)
            with self.assertRaisesRegex(ValueError, "differs across"):
                compose_multisource_face_training_corpus(
                    specs,
                    root / "output",
                    expected_split_counts={
                        "train": 3,
                        "validation": 1,
                        "sealed": 1,
                    },
                    excluded_rows={
                        "detector-failed": "production detector failed",
                    },
                    exclusion_evidence=evidence,
                    expected_duplicate_rows=1,
                )

    def test_rejects_unproved_exclusion_and_wrong_duplicate_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs, evidence = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "exactly once"):
                compose_multisource_face_training_corpus(
                    specs,
                    root / "missing-exclusion",
                    expected_split_counts={
                        "train": 3,
                        "validation": 1,
                        "sealed": 1,
                    },
                    excluded_rows={"absent": "not in sources"},
                    exclusion_evidence=evidence,
                    expected_duplicate_rows=1,
                )
            with self.assertRaisesRegex(ValueError, "Duplicate row count"):
                compose_multisource_face_training_corpus(
                    specs,
                    root / "wrong-duplicates",
                    expected_split_counts={
                        "train": 3,
                        "validation": 1,
                        "sealed": 1,
                    },
                    excluded_rows={
                        "detector-failed": "production detector failed",
                    },
                    exclusion_evidence=evidence,
                    expected_duplicate_rows=2,
                )

    def test_rejects_identity_leakage_and_evidence_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs, evidence = self._fixture(root)
            held_out_summary = Path(specs[3]["root"]) / "summary.json"
            payload = json.loads(held_out_summary.read_text(encoding="utf-8"))
            payload["rows"][1]["identity_group"] = "train-id-a"
            held_out_summary.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
            specs[3]["summary_sha256"] = _sha256(held_out_summary)
            with self.assertRaisesRegex(ValueError, "crosses train and sealed"):
                compose_multisource_face_training_corpus(
                    specs,
                    root / "leakage",
                    expected_split_counts={
                        "train": 3,
                        "validation": 1,
                        "sealed": 1,
                    },
                    excluded_rows={
                        "detector-failed": "production detector failed",
                    },
                    exclusion_evidence=evidence,
                    expected_duplicate_rows=1,
                )

            specs, evidence = self._fixture(root / "tamper-fixture")
            Path(evidence["path"]).write_text("substituted\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence hash mismatch"):
                compose_multisource_face_training_corpus(
                    specs,
                    root / "tamper",
                    expected_split_counts={
                        "train": 3,
                        "validation": 1,
                        "sealed": 1,
                    },
                    excluded_rows={
                        "detector-failed": "production detector failed",
                    },
                    exclusion_evidence=evidence,
                    expected_duplicate_rows=1,
                )


if __name__ == "__main__":
    unittest.main()
