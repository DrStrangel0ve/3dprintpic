import unittest

import numpy as np

from backend.benchmark.evaluate_face_surface_adapter_exact_gate import (
    _adapter_input,
    _current_baseline_equivalent,
    _decision,
)
from backend.benchmark.train_face_surface_adapter import GEOMETRY_ONLY_INPUT_MODE


def _summary(failures, shape, gradient, rmse, *, hard_failures):
    return {
        "combined_part_failures": failures,
        "median_shape_correlation": shape,
        "median_gradient_correlation": gradient,
        "median_normalized_rmse": rmse,
        "shape_check_failure_counts": {
            "raw_gradient_correlation": 2,
            "gradient_correlation": 2,
            "slope_retention": 2,
            "curvature_retention": 2,
        },
        "rows": [
            {
                "row_id": "small_side_lit_shelves_256",
                "combined_part_failures": hard_failures,
            },
            {
                "row_id": "another_face",
                "combined_part_failures": failures - hard_failures,
            },
        ],
    }


class EvaluateFaceSurfaceAdapterExactGateTests(unittest.TestCase):
    def test_adapter_input_has_expected_channels_and_ranges(self):
        image = np.full((20, 12, 3), 128, dtype=np.uint8)
        depth = np.arange(240, dtype=np.float32).reshape(20, 12)
        face = np.zeros((20, 12), dtype=np.uint8)
        face[4:16, 3:9] = 255

        values = _adapter_input(
            image,
            depth,
            face,
            network_size=32,
        )

        self.assertEqual(values.shape, (7, 32, 32))
        self.assertGreaterEqual(float(values[:3].min()), 0.0)
        self.assertLessEqual(float(values[:3].max()), 1.0)
        self.assertGreater(float(values[3].max()), float(values[3].min()))
        self.assertEqual(set(np.unique(values[4])), {0.0, 1.0})
        self.assertEqual(float(values[5].min()), -1.0)
        self.assertEqual(float(values[5].max()), 1.0)
        self.assertEqual(float(values[6].min()), -1.0)
        self.assertEqual(float(values[6].max()), 1.0)

        geometry_only = _adapter_input(
            image,
            depth,
            face,
            network_size=32,
            input_mode=GEOMETRY_ONLY_INPUT_MODE,
        )
        self.assertTrue(np.all(geometry_only[:3] == 0.0))
        self.assertTrue(np.array_equal(geometry_only[3:], values[3:]))

    def test_decision_requires_hard_row_and_aggregate_improvement(self):
        baseline = _summary(
            12,
            0.90,
            0.80,
            0.12,
            hard_failures=8,
        )
        candidate = _summary(
            9,
            0.91,
            0.799,
            0.11,
            hard_failures=6,
        )

        decision = _decision(
            baseline,
            candidate,
            baseline_equivalent=True,
            residual_contract_passed=True,
        )

        self.assertTrue(decision["eligible_for_full_stl_replay"])
        candidate["rows"][0]["combined_part_failures"] = 8
        self.assertFalse(
            _decision(
                baseline,
                candidate,
                baseline_equivalent=True,
                residual_contract_passed=True,
            )["eligible_for_full_stl_replay"]
        )

    def test_baseline_equivalence_is_exact_and_metric_bound(self):
        baseline = _summary(
            12,
            0.90,
            0.80,
            0.12,
            hard_failures=8,
        )
        reproduced = _summary(
            12,
            0.90,
            0.80,
            0.12,
            hard_failures=8,
        )

        self.assertTrue(
            _current_baseline_equivalent(
                baseline,
                reproduced,
                [0.0, 0.0],
            )
        )
        self.assertFalse(
            _current_baseline_equivalent(
                baseline,
                reproduced,
                [0.0, 1e-4],
            )
        )


if __name__ == "__main__":
    unittest.main()
