import unittest
from unittest.mock import patch

from backend.benchmark import train_selector_aligned_face_geometry as trainer


class TrainSelectorAlignedFaceGeometryTests(unittest.TestCase):
    def test_selector_loss_is_zero_at_baseline_and_penalizes_harm(self):
        import torch

        height = width = 16
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        target = (
            0.2 * x
            + 0.1 * y
            + 0.3 * torch.exp(-5.0 * (x**2 + y**2))
        ).reshape(1, 1, height, width)
        baseline = target.clone()
        baseline[:, :, 5:11, 5:11] *= 0.7
        face = torch.ones_like(target)
        parts = torch.zeros((1, 6, height, width))
        for index in range(6):
            left = 1 + 2 * index
            parts[:, index, 4:12, left : left + 3] = 1.0

        zero_mean, zero_worst, _details = (
            trainer.selector_aligned_part_non_regression(
                baseline,
                baseline,
                target,
                face,
                parts,
            )
        )
        harmful = baseline.clone()
        harmful[:, :, 4:12, 4:12] += 0.25 * torch.sin(
            8.0 * x[4:12, 4:12]
        )
        bad_mean, bad_worst, details = (
            trainer.selector_aligned_part_non_regression(
                harmful,
                baseline,
                target,
                face,
                parts,
            )
        )
        self.assertAlmostEqual(float(zero_mean), 0.0, places=6)
        self.assertAlmostEqual(float(zero_worst), 0.0, places=6)
        self.assertGreater(float(bad_mean), 0.0)
        self.assertGreater(float(bad_worst), 0.0)
        self.assertEqual(
            set(details),
            {
                "affine_rmse",
                "affine_bias",
                "affine_p95",
                "affine_span",
                "shape_rmse",
                "shape_correlation",
                "raw_gradient_correlation",
            },
        )

    def test_selector_geometry_loss_adds_mean_and_worst_penalties(self):
        import torch

        residual = torch.zeros((1, 1, 4, 4))
        tensors = {
            "baseline": residual,
            "target": residual,
            "exact_face": torch.ones_like(residual),
            "parts_individual": torch.ones((1, 6, 4, 4)),
        }
        with (
            patch.object(
                trainer,
                "_BASE_GEOMETRY_LOSS",
                return_value=(torch.tensor(1.0), {"base": 1.0}),
            ),
            patch.object(
                trainer,
                "apply_training_residual",
                return_value=(residual, residual, {}),
            ),
            patch.object(
                trainer,
                "selector_aligned_part_non_regression",
                return_value=(
                    torch.tensor(0.25),
                    torch.tensor(0.5),
                    {"affine_span": 0.25},
                ),
            ),
        ):
            total, details = trainer.selector_geometry_loss(
                residual,
                residual,
                tensors,
                selector_mean_non_regression_weight=4.0,
                selector_worst_non_regression_weight=8.0,
            )
        self.assertEqual(float(total), 6.0)
        self.assertEqual(details["selector_mean_non_regression"], 0.25)
        self.assertEqual(details["selector_worst_non_regression"], 0.5)

    def test_invalid_part_mask_shape_fails_closed(self):
        import torch

        values = torch.zeros((1, 1, 8, 8))
        with self.assertRaisesRegex(ValueError, "part masks"):
            trainer.selector_aligned_part_non_regression(
                values,
                values,
                values,
                torch.ones_like(values),
                torch.ones((2, 6, 8, 8)),
            )


if __name__ == "__main__":
    unittest.main()
