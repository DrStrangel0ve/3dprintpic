import unittest

import numpy as np
import torch
from PIL import Image

from backend.benchmark.train_face_surface_fusion_adapter import (
    _curvature_non_regression_loss,
    _gradient_non_regression_loss,
    apply_training_residual,
    part_balanced_physical_losses,
    physical_amplitude_losses,
    remove_affine_component,
    selected_image_from_exact_mask,
    small_face_sample_weight,
)


class TrainFaceSurfaceFusionAdapterTests(unittest.TestCase):
    def test_small_face_sample_weight_is_bounded_and_scale_aware(self):
        self.assertAlmostEqual(
            small_face_sample_weight(
                76,
                reference_pixels=120,
                maximum_weight=2.0,
            ),
            120 / 76,
        )
        self.assertEqual(
            small_face_sample_weight(
                120,
                reference_pixels=120,
                maximum_weight=2.0,
            ),
            1.0,
        )
        self.assertEqual(
            small_face_sample_weight(
                40,
                reference_pixels=120,
                maximum_weight=2.0,
            ),
            2.0,
        )

    def test_selected_image_matches_production_neutral_contract(self):
        source = Image.fromarray(
            np.full((32, 32, 3), (20, 40, 60), dtype=np.uint8)
        )
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255

        selected = np.asarray(
            selected_image_from_exact_mask(source, Image.fromarray(mask))
        )

        np.testing.assert_array_equal(selected[0, 0], (245, 245, 245))
        np.testing.assert_array_equal(selected[16, 16], (20, 40, 60))
        self.assertTrue(np.all(selected[7, 16] > (20, 40, 60)))
        self.assertTrue(np.all(selected[7, 16] < (245, 245, 245)))

    def test_affine_residual_component_is_removed(self):
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, 16),
            torch.linspace(0.0, 1.0, 16),
            indexing="ij",
        )
        local = (xx + 0.2 * yy)[None, None]
        residual = 0.4 * local - 0.2
        support = torch.ones_like(local)

        non_affine, scale, offset = remove_affine_component(
            residual,
            local,
            support,
        )

        self.assertLess(float(non_affine.abs().max()), 1e-5)
        self.assertAlmostEqual(float(scale), 0.4, places=5)
        self.assertAlmostEqual(float(offset), -0.2, places=5)

    def test_zero_residual_leaves_training_baseline_unchanged(self):
        shape = (1, 1, 16, 16)
        baseline = torch.rand(shape)
        tensors = {
            "local_depth": torch.rand(shape),
            "support_face": torch.ones(shape),
            "alignment_scale": torch.ones((1, 1, 1, 1)),
            "correction_limit": torch.full((1, 1, 1, 1), 0.1),
            "fusion_weight": torch.rand(shape),
            "baseline": baseline,
        }

        prediction, correction, _stats = apply_training_residual(
            torch.zeros(shape),
            tensors,
        )

        torch.testing.assert_close(prediction, baseline)
        torch.testing.assert_close(correction, torch.zeros_like(correction))

    def test_non_regression_losses_are_zero_for_baseline_and_positive_when_worse(self):
        baseline = torch.zeros((1, 1, 12, 12))
        target = baseline.clone()
        weights = torch.ones_like(baseline)
        worse = baseline.clone()
        worse[..., 5:7, 5:7] = 0.4

        self.assertEqual(
            float(
                _gradient_non_regression_loss(
                    baseline,
                    baseline,
                    target,
                    weights,
                )
            ),
            0.0,
        )
        self.assertEqual(
            float(
                _curvature_non_regression_loss(
                    baseline,
                    baseline,
                    target,
                    weights,
                )
            ),
            0.0,
        )
        self.assertGreater(
            float(
                _gradient_non_regression_loss(
                    worse,
                    baseline,
                    target,
                    weights,
                )
            ),
            0.0,
        )
        self.assertGreater(
            float(
                _curvature_non_regression_loss(
                    worse,
                    baseline,
                    target,
                    weights,
                )
            ),
            0.0,
        )

    def test_physical_amplitude_loss_uses_baseline_affine_calibration(self):
        target = torch.linspace(0.0, 1.0, 16)[None, None, None, :].expand(
            1,
            1,
            16,
            16,
        )
        baseline = target.clone()
        weights = torch.ones_like(target)
        value, gradient, scale, shift = physical_amplitude_losses(
            0.5 * target,
            baseline,
            target,
            weights,
            weights,
        )

        self.assertAlmostEqual(float(scale), 1.0, places=5)
        self.assertAlmostEqual(float(shift), 0.0, places=5)
        self.assertGreater(float(value), 0.0)
        self.assertGreater(float(gradient), 0.0)

    def test_part_balanced_loss_does_not_hide_a_small_failed_part(self):
        target = torch.zeros((1, 1, 8, 8))
        baseline = target.clone()
        prediction = target.clone()
        prediction[..., 1:3, 2] = 1.0
        parts = torch.zeros((1, 2, 8, 8))
        parts[:, 0, 1:3, 1:3] = 1.0
        parts[:, 1, 4:8, 0:8] = 1.0
        sample_weight = torch.ones((1, 1, 1, 1))

        (
            value,
            gradient,
            value_non_regression,
            gradient_non_regression,
        ) = part_balanced_physical_losses(
            prediction,
            baseline,
            target,
            parts,
            sample_weight,
        )

        self.assertAlmostEqual(float(value), 0.25, places=5)
        self.assertGreater(float(gradient), 0.0)
        self.assertAlmostEqual(
            float(value_non_regression),
            float(value),
            places=5,
        )
        self.assertAlmostEqual(
            float(gradient_non_regression),
            float(gradient),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
