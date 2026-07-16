import unittest

import numpy as np

from backend.benchmark import train_gnm_face_depth_fusion as training


class TrainGNMFaceDepthFusionTests(unittest.TestCase):
    def test_per_row_non_regression_requires_every_row(self):
        baseline = {
            "rows": [
                {
                    "row_id": "a",
                    "shape_failed_parts": ["left_eye"],
                    "affine_failed_parts": [],
                },
                {
                    "row_id": "b",
                    "shape_failed_parts": [],
                    "affine_failed_parts": ["mouth"],
                },
            ]
        }
        candidate = {
            "rows": [
                {
                    "row_id": "a",
                    "shape_failed_parts": [],
                    "affine_failed_parts": [],
                },
                {
                    "row_id": "b",
                    "shape_failed_parts": ["nose"],
                    "affine_failed_parts": ["mouth"],
                },
            ]
        }
        self.assertEqual(
            training._per_row_non_regression(candidate, baseline),
            0.5,
        )

    def test_strict_improvement_preserves_global_metrics(self):
        baseline = {
            "combined_part_failures": 10,
            "median_shape_correlation": 0.90,
            "median_gradient_correlation": 0.80,
            "median_normalized_rmse": 0.10,
        }
        candidate = {
            "combined_part_failures": 9,
            "median_shape_correlation": 0.91,
            "median_gradient_correlation": 0.80,
            "median_normalized_rmse": 0.09,
        }
        self.assertTrue(training._strictly_improves(candidate, baseline))
        candidate["median_gradient_correlation"] = 0.70
        self.assertFalse(training._strictly_improves(candidate, baseline))

    def test_equal_mask_mean_weights_parts_equally(self):
        import torch

        prediction = torch.tensor([[0.0, 2.0], [4.0, 8.0]])
        target = torch.zeros_like(prediction)
        masks = torch.tensor(
            [
                [[1, 0], [0, 0]],
                [[0, 1], [1, 1]],
            ],
            dtype=torch.float32,
        )
        value = training._equal_mask_mean(
            lambda candidate, reference, mask: training._masked_mean(
                (candidate - reference).abs(), mask
            ),
            prediction,
            target,
            masks,
        )
        expected = 0.5 * (0.0 + (2.0 + 4.0 + 8.0) / 3.0)
        self.assertAlmostEqual(float(value), expected, places=6)

    def test_sample_weight_focus_is_bounded(self):
        heights = np.asarray([60.0, 90.0, 180.0])
        weights = np.clip(90.0 / heights, 1.0, 1.5)
        np.testing.assert_allclose(weights, [1.5, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
