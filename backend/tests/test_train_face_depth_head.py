import unittest

import numpy as np

from backend.benchmark.train_face_depth_head import (
    _near_high_target,
    _padded_box,
    _strictly_improves,
)


class TrainFaceDepthHeadTests(unittest.TestCase):
    def test_padded_box_matches_production_ratio_and_clamps(self):
        self.assertEqual(_padded_box((20, 30, 60, 80), 100, 100), (6, 12, 74, 98))
        self.assertEqual(_padded_box((0, 0, 20, 20), 30, 30), (0, 0, 27, 27))

    def test_near_high_target_reverses_exact_far_high_depth(self):
        exact = np.tile(np.linspace(0.2, 0.8, 16, dtype=np.float32), (16, 1))
        face = np.ones_like(exact, dtype=bool)
        target = _near_high_target(exact, face)

        self.assertGreater(float(target[:, 0].mean()), float(target[:, -1].mean()))
        self.assertTrue(np.all(np.isfinite(target)))
        self.assertGreater(float(np.ptp(target)), 0.9)

    def test_near_high_target_rejects_tiny_support(self):
        with self.assertRaisesRegex(ValueError, "at least 64"):
            _near_high_target(
                np.zeros((8, 8), dtype=np.float32),
                np.eye(8, dtype=bool),
            )

    def test_strict_selector_requires_part_and_all_global_improvements(self):
        baseline = {
            "combined_part_failures": 10,
            "median_shape_correlation": 0.80,
            "median_gradient_correlation": 0.60,
            "median_normalized_rmse": 0.20,
        }
        passing = {
            "combined_part_failures": 9,
            "median_shape_correlation": 0.81,
            "median_gradient_correlation": 0.61,
            "median_normalized_rmse": 0.19,
        }
        self.assertTrue(_strictly_improves(passing, baseline))
        for field, value in (
            ("combined_part_failures", 10),
            ("median_shape_correlation", 0.79),
            ("median_gradient_correlation", 0.59),
            ("median_normalized_rmse", 0.21),
        ):
            candidate = dict(passing)
            candidate[field] = value
            self.assertFalse(_strictly_improves(candidate, baseline))


if __name__ == "__main__":
    unittest.main()
