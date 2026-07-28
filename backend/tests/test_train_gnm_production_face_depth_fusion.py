import unittest
from types import SimpleNamespace

import numpy as np

from backend.benchmark.train_gnm_production_face_depth_fusion import (
    _prepare,
    _selection_record,
    _summaries_equivalent,
    _validate_additional_training_contract,
    normalize_prediction_to_shape,
    residual_application_scale,
    worst_part_physical_non_regression,
)


class TrainGnmProductionFaceDepthFusionTests(unittest.TestCase):
    def test_residual_application_scale_preserves_small_faces(self):
        self.assertEqual(residual_application_scale(76), 1.0)
        self.assertAlmostEqual(residual_application_scale(100), 0.5)
        self.assertEqual(residual_application_scale(180), 0.0)
        self.assertLessEqual(residual_application_scale(1), 1.0)

    def test_normalize_prediction_matches_minmax_contract(self):
        import torch

        prediction = torch.tensor(
            [[2.0, 4.0], [6.0, 10.0]],
            dtype=torch.float32,
        )
        normalized = normalize_prediction_to_shape(
            prediction,
            source_shape=(2, 2),
            target_shape=(2, 2),
        )

        self.assertEqual(tuple(normalized.shape), (2, 2))
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)
        self.assertTrue(np.all(np.isfinite(normalized.numpy())))

    def test_validation_override_paths_must_be_supplied_together(self):
        with self.assertRaisesRegex(ValueError, "supplied together"):
            _prepare(
                None,
                None,
                None,
                device="cuda",
                validation_corpus_root=None,
                validation_cache_root="cache",
            )

    def test_additional_training_paths_must_be_supplied_together(self):
        with self.assertRaisesRegex(ValueError, "supplied together"):
            _prepare(
                None,
                None,
                None,
                device="cuda",
                additional_training_corpus_root="corpus",
                additional_training_cache_root=None,
            )

    def test_additional_training_contract_is_disjoint_and_detector_audited(self):
        def surface(row_id, identity, split):
            return SimpleNamespace(
                row={
                    "row_id": row_id,
                    "identity_group": identity,
                    "split": split,
                }
            )

        primary = [
            surface("train-a", "identity-a", "train"),
            surface("sealed-b", "identity-b", "sealed"),
        ]
        validation = [
            surface("validation-c", "identity-c", "validation"),
        ]
        additional = [
            surface("extra-d", "identity-d", "train"),
            surface("extra-e", "identity-e", "train"),
        ]
        summary = {
            "row_count": 2,
            "production_detector_filter": {
                "method": "unchanged-production-face-detector",
                "input_rows": 3,
                "passed_rows": 2,
                "excluded_rows": ["failed-f"],
                "fallback_geometry_used": False,
            },
        }

        contract = _validate_additional_training_contract(
            primary,
            validation,
            additional,
            summary,
            {"row_count": 2},
        )

        self.assertEqual(contract["row_count"], 2)
        self.assertEqual(contract["identity_count"], 2)
        self.assertEqual(
            contract["identity_groups"],
            ["identity-d", "identity-e"],
        )

    def test_additional_training_contract_rejects_identity_leakage(self):
        surface = SimpleNamespace(
            row={
                "row_id": "extra",
                "identity_group": "held-identity",
                "split": "train",
            }
        )
        held = SimpleNamespace(
            row={
                "row_id": "held",
                "identity_group": "held-identity",
                "split": "sealed",
            }
        )
        summary = {
            "row_count": 1,
            "production_detector_filter": {
                "input_rows": 1,
                "passed_rows": 1,
                "excluded_rows": [],
                "fallback_geometry_used": False,
            },
        }

        with self.assertRaisesRegex(ValueError, "overlaps held"):
            _validate_additional_training_contract(
                [held],
                [],
                [surface],
                summary,
                {"row_count": 1},
            )

    def test_selection_record_requires_strict_and_per_row_checks(self):
        baseline = {
            "combined_part_failures": 2,
            "median_shape_correlation": 0.8,
            "median_gradient_correlation": 0.7,
            "median_normalized_rmse": 0.2,
            "rows": [
                {"row_id": "a", "combined_part_failures": 2},
            ],
        }
        candidate = {
            "combined_part_failures": 1,
            "median_shape_correlation": 0.81,
            "median_gradient_correlation": 0.71,
            "median_normalized_rmse": 0.19,
            "rows": [
                {"row_id": "a", "combined_part_failures": 1},
            ],
        }

        self.assertTrue(_selection_record(candidate, baseline)["eligible"])

    def test_summary_equivalence_requires_metrics_and_rows_to_match(self):
        baseline = {
            "combined_part_failures": 2,
            "median_shape_correlation": 0.8,
            "median_gradient_correlation": 0.7,
            "median_normalized_rmse": 0.2,
            "rows": [
                {"row_id": "a", "combined_part_failures": 2},
            ],
        }

        self.assertTrue(_summaries_equivalent(dict(baseline), baseline))
        changed = {
            **baseline,
            "median_shape_correlation": 0.81,
        }
        self.assertFalse(_summaries_equivalent(changed, baseline))

    def test_worst_part_non_regression_penalizes_local_regression(self):
        import torch

        baseline = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        target = baseline.clone()
        target[:, :, 1:3, 1:3] = 1.0
        parts = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
        parts[:, 0, 1:3, 1:3] = 1.0
        parts[:, 1, 0:2, 0:2] = 1.0
        tensors = {
            "local_depth": baseline.clone(),
            "baseline": baseline,
            "target": target,
            "exact_face": torch.ones_like(baseline),
            "support_face": torch.ones_like(baseline),
            "fusion_weight": torch.ones_like(baseline),
            "parts_individual": parts,
            "alignment_scale": torch.ones((1, 1, 1, 1)),
            "correction_limit": torch.ones((1, 1, 1, 1)),
        }
        improving = target.clone()
        regressing = -target.clone()

        improving_value, improving_gradient = (
            worst_part_physical_non_regression(improving, tensors)
        )
        regressing_value, regressing_gradient = (
            worst_part_physical_non_regression(regressing, tensors)
        )

        self.assertEqual(float(improving_value), 0.0)
        self.assertEqual(float(improving_gradient), 0.0)
        self.assertGreater(float(regressing_value), 0.0)
        self.assertGreater(float(regressing_gradient), 0.0)


if __name__ == "__main__":
    unittest.main()
