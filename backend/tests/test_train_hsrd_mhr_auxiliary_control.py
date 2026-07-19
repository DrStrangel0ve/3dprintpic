import copy
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.benchmark.mhr_face_training_corpus import (
    MHR_LICENSE,
    MHR_PROVIDER,
    MHR_SOURCE_REVISION,
)
from backend.benchmark.train_hsrd_mhr_auxiliary_control import (
    _adapt_mhr_row,
    _matched_training_contract,
    _next_auxiliary_batch,
    _partition_hsrd_rows,
    _prepare_sealed_after_validation,
    _resolve_auxiliary_epochs,
    _select_mhr_training_rows,
    _validate_mhr_auxiliary_corpus,
    _validate_summary_hash,
    _validation_gate,
)


def _mhr_row(row_id: str, identity: str, split: str) -> dict:
    return {
        "row_id": row_id,
        "identity_group": identity,
        "split": split,
        "render": {
            "face_bbox_height_pixels": 75,
            "face_bbox_xyxy": [86, 91, 135, 166],
            "camera": {
                "intrinsics": [
                    [444.0, 0.0, 127.5],
                    [0.0, 444.0, 127.5],
                    [0.0, 0.0, 1.0],
                ]
            },
        },
        "spec": {"camera_yaw_deg": -40.0},
    }


def _mhr_summary() -> dict:
    rows = []
    split_layout = (("train", 30), ("validation", 5), ("sealed", 5))
    identity_index = 0
    for split, identity_count in split_layout:
        for _ in range(identity_count):
            identity = f"identity-{identity_index:02d}"
            rows.extend(
                _mhr_row(f"{identity}-row-{row_index}", identity, split)
                for row_index in range(8)
            )
            identity_index += 1
    return {
        "provider": MHR_PROVIDER,
        "source_revision": MHR_SOURCE_REVISION,
        "source": {"license": MHR_LICENSE},
        "privacy_safe_synthetic": True,
        "matrix_kind": "training",
        "corpus_complete": True,
        "training_eligible": True,
        "production_training_eligible": False,
        "promotion_eligible": False,
        "identity_disjoint_splits": True,
        "source_geometry_training_and_evaluation_only": True,
        "deterministic_generation": {"device_requirement_met": True},
        "depth_target_provenance": {
            "camera_convention": "OpenCV +X right, +Y down, +Z forward",
            "metric_scale_claimed": False,
        },
        "geometry_target_contract": {
            "training_rows_only": True,
            "compact_evidence_must_exclude_raw_targets": True,
            "supervision_manifest": {
                "path": "training_supervision.json",
                "row_count": 240,
                "sha256": "a" * 64,
                "excluded_from_compact_evidence": True,
            },
        },
        "rows": rows,
    }


class TrainHSRDMHRAuxiliaryControlTests(unittest.TestCase):
    def test_mhr_gate_requires_complete_nonpromotion_training_contract(self):
        summary = _mhr_summary()
        contract = _validate_mhr_auxiliary_corpus(summary)
        self.assertTrue(contract["training_contract"]["camera_aligned"])
        self.assertTrue(contract["checks"]["not_promotion_evidence"])

        summary["production_training_eligible"] = True
        with self.assertRaisesRegex(ValueError, "not_production_evidence"):
            _validate_mhr_auxiliary_corpus(summary)

    def test_mhr_gate_rejects_provider_privacy_camera_and_scale_substitution(self):
        mutations = {
            "provider": lambda summary: summary.__setitem__("provider", "substitute"),
            "privacy_safe_synthetic": lambda summary: summary.__setitem__(
                "privacy_safe_synthetic", False
            ),
            "camera_convention": lambda summary: summary[
                "depth_target_provenance"
            ].__setitem__("camera_convention", "unknown"),
            "no_metric_scale_claim": lambda summary: summary[
                "depth_target_provenance"
            ].__setitem__("metric_scale_claimed", True),
            "supervision_manifest": lambda summary: summary["geometry_target_contract"][
                "supervision_manifest"
            ].__setitem__("sha256", "not-a-hash"),
        }
        for expected_check, mutate in mutations.items():
            with self.subTest(expected_check=expected_check):
                summary = _mhr_summary()
                mutate(summary)
                with self.assertRaisesRegex(ValueError, expected_check):
                    _validate_mhr_auxiliary_corpus(summary)

    def test_mhr_adapter_preserves_assets_and_exposes_camera_geometry(self):
        row = _mhr_row("row-one", "identity-one", "train")
        row["source"] = {"path": "source.png", "sha256": "a" * 64}
        adapted = _adapt_mhr_row(row)
        self.assertEqual(adapted["source"], row["source"])
        self.assertEqual(
            adapted["selection_geometry"]["face_bbox_xyxy"],
            [86, 91, 135, 166],
        )
        self.assertEqual(
            adapted["selection_geometry"]["camera"]["intrinsics"][0][0],
            444.0,
        )
        self.assertEqual(adapted["rendering"]["camera_yaw_degrees"], -40.0)
        self.assertEqual(adapted["auxiliary_source_provider"], MHR_PROVIDER)
        self.assertNotIn("selection_geometry", row)

    def test_mhr_adapter_fails_closed_without_calibration(self):
        row = _mhr_row("row-one", "identity-one", "train")
        del row["render"]["camera"]["intrinsics"]
        with self.assertRaisesRegex(ValueError, "camera-aligned geometry"):
            _adapt_mhr_row(row)

    def test_auxiliary_selection_uses_training_rows_only(self):
        rows = [
            _mhr_row("train-1", "a", "train"),
            _mhr_row("validation-1", "b", "validation"),
            _mhr_row("train-2", "c", "train"),
            _mhr_row("sealed-1", "d", "sealed"),
        ]
        selected = _select_mhr_training_rows(rows, 1)
        self.assertEqual([row["row_id"] for row in selected], ["train-1"])
        self.assertTrue(all(row["split"] == "train" for row in selected))
        with self.assertRaisesRegex(ValueError, "row limit"):
            _select_mhr_training_rows(rows, 3)

    def test_hsrd_partition_keeps_sealed_rows_out_of_tuning(self):
        rows = [
            {"row_id": "train", "split": "train"},
            {"row_id": "validation", "split": "validation"},
            {"row_id": "sealed", "split": "sealed"},
        ]
        tuning, sealed = _partition_hsrd_rows(rows)
        self.assertEqual([row["row_id"] for row in tuning], ["train", "validation"])
        self.assertEqual([row["row_id"] for row in sealed], ["sealed"])
        duplicate = rows + [{"row_id": "sealed", "split": "sealed"}]
        with self.assertRaisesRegex(ValueError, "unique"):
            _partition_hsrd_rows(duplicate)

    def test_auxiliary_batcher_cycles_without_short_batches(self):
        order = ["a", "b", "c"]
        generator = random.Random(17)
        first, cursor = _next_auxiliary_batch(order, len(order), 2, generator)
        second, cursor = _next_auxiliary_batch(order, cursor, 2, generator)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertTrue(set(first + second) <= {"a", "b", "c"})
        self.assertGreaterEqual(cursor, 0)

    def test_auxiliary_epoch_schedule_defaults_to_all_or_accepts_fixed_handoff(self):
        self.assertEqual(_resolve_auxiliary_epochs(12, None), 12)
        self.assertEqual(_resolve_auxiliary_epochs(12, 6), 6)
        with self.assertRaisesRegex(ValueError, "must be in"):
            _resolve_auxiliary_epochs(12, 0)
        with self.assertRaisesRegex(ValueError, "must be in"):
            _resolve_auxiliary_epochs(12, 13)

    def test_matched_contract_checks_schedule_optimizer_and_initialization(self):
        control = {
            "initial_state_sha256": "a" * 64,
            "hsrd_schedule_sha256": "b" * 64,
            "hsrd_exposure_count": 120,
            "optimizer_steps": 60,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
            },
            "selected_epoch": 12,
            "checkpoint_selection": "fixed-final-step",
            "auxiliary_exposure_count": 0,
            "auxiliary_epochs": 0,
            "auxiliary_weight": 0.0,
            "auxiliary_schedule_seed": None,
            "auxiliary_schedule_sha256": None,
        }
        candidate = {
            **copy.deepcopy(control),
            "auxiliary_exposure_count": 120,
            "auxiliary_epochs": 6,
            "auxiliary_weight": 0.25,
            "auxiliary_schedule_seed": 20260720,
            "auxiliary_schedule_sha256": "c" * 64,
        }
        self.assertTrue(_matched_training_contract(control, candidate)["passed"])
        candidate["hsrd_schedule_sha256"] = "d" * 64
        contract = _matched_training_contract(control, candidate)
        self.assertFalse(contract["passed"])
        self.assertFalse(contract["checks"]["same_hsrd_schedule"])

    def test_matched_contract_fails_closed_when_required_fields_are_absent(self):
        contract = _matched_training_contract(
            {"auxiliary_exposure_count": 0},
            {"auxiliary_exposure_count": 1},
        )
        self.assertFalse(contract["passed"])
        self.assertFalse(contract["checks"]["valid_initial_state_hashes"])
        self.assertFalse(contract["checks"]["valid_hsrd_schedule_hashes"])
        self.assertFalse(contract["checks"]["same_positive_optimizer_steps"])

    def test_matched_contract_rejects_missing_or_substituted_auxiliary_provenance(self):
        control = {
            "initial_state_sha256": "a" * 64,
            "hsrd_schedule_sha256": "b" * 64,
            "hsrd_exposure_count": 2,
            "optimizer_steps": 1,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
            },
            "selected_epoch": 1,
            "checkpoint_selection": "fixed-final-step",
            "auxiliary_exposure_count": 0,
            "auxiliary_epochs": 0,
            "auxiliary_weight": 0.0,
            "auxiliary_schedule_seed": None,
            "auxiliary_schedule_sha256": None,
        }
        candidate = {
            **copy.deepcopy(control),
            "auxiliary_exposure_count": 2,
            "auxiliary_epochs": 1,
            "auxiliary_weight": 0.25,
            "auxiliary_schedule_seed": 20260720,
            "auxiliary_schedule_sha256": "c" * 64,
        }
        self.assertTrue(_matched_training_contract(control, candidate)["passed"])
        for field, value in (
            ("auxiliary_weight", 0.0),
            ("auxiliary_schedule_seed", None),
            ("auxiliary_schedule_seed", 123),
        ):
            with self.subTest(field=field, value=value):
                substituted = {**candidate, field: value}
                contract = _matched_training_contract(control, substituted)
                self.assertFalse(contract["passed"])
                self.assertFalse(contract["checks"]["candidate_has_valid_auxiliary"])

    def test_failed_validation_gate_leaves_sealed_assets_unopened(self):
        foundation = {
            "combined_part_failures": 2,
            "median_shape_correlation": 0.8,
            "median_gradient_correlation": 0.7,
            "median_normalized_rmse": 0.2,
            "source_background_bit_exact": True,
            "rows": [
                {
                    "row_id": "one",
                    "shape_failed_parts": ["nose"],
                    "affine_failed_parts": ["mouth"],
                }
            ],
        }
        control = copy.deepcopy(foundation)
        candidate = copy.deepcopy(foundation)
        selected = {"hsrd_control": control, "mixed_auxiliary": candidate}
        gate = _validation_gate(selected, foundation)
        self.assertFalse(gate["passed"])
        with patch(
            "backend.benchmark.train_hsrd_mhr_auxiliary_control._prepare_sources"
        ) as prepare:
            result = _prepare_sealed_after_validation(
                gate,
                Path("hsrd"),
                [{"row_id": "sealed", "split": "sealed"}],
                Path("mhr"),
                "cuda",
            )
        self.assertIsNone(result)
        prepare.assert_not_called()

    def test_summary_hash_is_exact_and_labeled(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_bytes(b'{"source":"pinned"}\n')
            expected = (
                "c8cd813d8151638bc825b33f0dfd768c1905e12b1a620592b00994c8fc735ed0"
            )
            self.assertEqual(
                _validate_summary_hash(path, expected, label="MHR"),
                expected,
            )
            path.write_bytes(b'{"source":"changed"}\n')
            with self.assertRaisesRegex(ValueError, "MHR summary SHA256 mismatch"):
                _validate_summary_hash(path, expected, label="MHR")


if __name__ == "__main__":
    unittest.main()
