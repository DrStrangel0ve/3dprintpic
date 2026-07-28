import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from backend.benchmark.evaluate_face_depth_head_exact_gate import (
    PairedFusionSurfaceResidual,
    RESIDUAL_FULL_STRENGTH_SUPPORT_HEIGHT_PIXELS,
    RESIDUAL_ZERO_STRENGTH_SUPPORT_HEIGHT_PIXELS,
    GNM_FUSION_METHOD,
    GNM_PRODUCTION_FUSION_METHOD,
    _checkpoint_mode_and_alpha,
    _correlation,
    _face_residual_scale,
    _minmax_normalize,
    _resize_depth,
    _small_face_residual_scale,
    _small_face_only_decision_contract,
)


class EvaluateFaceDepthHeadExactGateTests(unittest.TestCase):
    def test_minmax_normalize_matches_production_linear_storage(self):
        values = np.asarray([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32)
        normalized = _minmax_normalize(values)

        self.assertEqual(float(normalized.min()), 0.0)
        self.assertEqual(float(normalized.max()), 1.0)
        self.assertAlmostEqual(float(normalized[0, 1]), 0.25)

    def test_correlation_ignores_nonfinite_values(self):
        left = np.asarray([0.0, 1.0, np.nan, 2.0], dtype=np.float32)
        right = np.asarray([0.0, 2.0, 8.0, 4.0], dtype=np.float32)
        self.assertAlmostEqual(_correlation(left, right), 1.0)

    def test_checkpoint_contract_supports_head_and_fusion_scopes(self):
        self.assertEqual(
            _checkpoint_mode_and_alpha(
                {
                    "head_state_dict": {},
                    "blend_alpha": 0.1,
                }
            ),
            ("head-only", 0.1),
        )
        self.assertEqual(
            _checkpoint_mode_and_alpha(
                {
                    "method": GNM_FUSION_METHOD,
                    "fusion_stage_state_dict": {},
                    "head_state_dict": {},
                    "selected_alpha": 0.75,
                }
            ),
            ("fusion-head", 0.75),
        )
        self.assertEqual(
            _checkpoint_mode_and_alpha(
                {
                    "method": GNM_PRODUCTION_FUSION_METHOD,
                    "fusion_stage_state_dict": {},
                    "head_state_dict": {},
                    "selected_alpha": 0.5,
                }
            ),
            ("fusion-head", 0.5),
        )

    def test_checkpoint_contract_fails_closed_for_missing_fusion_state(self):
        with self.assertRaisesRegex(ValueError, "missing state dictionaries"):
            _checkpoint_mode_and_alpha(
                {
                    "method": GNM_FUSION_METHOD,
                    "head_state_dict": {},
                    "selected_alpha": 0.5,
                }
            )

    def test_resize_depth_preserves_shape_and_finite_values(self):
        values = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        resized = _resize_depth(values, (5, 7))

        self.assertEqual(resized.shape, (5, 7))
        self.assertTrue(np.all(np.isfinite(resized)))

    def test_paired_residual_uses_trained_minus_baseline_depth(self):
        class FakeProvider:
            alpha = 0.75

            def infer(self, _image_path, alpha=None):
                baseline = np.asarray(
                    [[0.0, 0.25], [0.5, 1.0]],
                    dtype=np.float32,
                )
                if alpha == 0.0:
                    return baseline
                return baseline + np.asarray(
                    [[0.0, 0.1], [-0.2, 0.0]],
                    dtype=np.float32,
                )

        provider = PairedFusionSurfaceResidual(FakeProvider())
        local_depth = np.asarray(
            [[0.0, 0.25], [0.5, 1.0]],
            dtype=np.float32,
        )
        with TemporaryDirectory() as temporary:
            residual, stats = provider.callback(
                Path(temporary) / "crop.png",
                local_depth,
                np.ones_like(local_depth),
                np.ones_like(local_depth),
                np.ones_like(local_depth),
                Path(temporary) / "surface",
            )

            np.testing.assert_allclose(
                residual,
                np.asarray(
                    [[0.0, 0.1], [-0.2, 0.0]],
                    dtype=np.float32,
                ),
            )
            self.assertAlmostEqual(stats["local_baseline_correlation"], 1.0)
            self.assertAlmostEqual(
                stats["local_baseline_maximum_absolute_difference"],
                0.0,
            )
            self.assertTrue(
                provider.provenance()["all_local_baselines_equivalent"]
            )

    def test_face_residual_scale_is_full_for_small_faces_and_bounded(self):
        small = np.zeros((80, 40), dtype=np.uint8)
        small[10:50] = 255
        large = np.zeros((140, 80), dtype=np.uint8)
        large[10:110] = 255

        small_scale, small_height = _face_residual_scale(small)
        large_scale, large_height = _face_residual_scale(large)

        self.assertEqual(small_height, 40)
        self.assertEqual(small_scale, 1.0)
        self.assertEqual(large_height, 100)
        self.assertAlmostEqual(
            large_scale,
            RESIDUAL_FULL_STRENGTH_SUPPORT_HEIGHT_PIXELS / 100.0,
        )
        self.assertLessEqual(large_scale, 1.0)

    def test_size_aware_paired_residual_scales_the_delta(self):
        class FakeProvider:
            alpha = 1.0

            def infer(self, _image_path, alpha=None):
                baseline = np.zeros((100, 2), dtype=np.float32)
                return baseline if alpha == 0.0 else np.ones_like(baseline)

        provider = PairedFusionSurfaceResidual(
            FakeProvider(),
            size_aware=True,
        )
        face_mask = np.ones((100, 2), dtype=np.uint8)
        with TemporaryDirectory() as temporary:
            residual, stats = provider.callback(
                Path(temporary) / "crop.png",
                np.zeros((100, 2), dtype=np.float32),
                face_mask,
                face_mask,
                face_mask,
                Path(temporary) / "surface",
            )

        expected = RESIDUAL_FULL_STRENGTH_SUPPORT_HEIGHT_PIXELS / 100.0
        np.testing.assert_allclose(residual, expected)
        self.assertAlmostEqual(stats["residual_scale"], expected)
        self.assertEqual(stats["support_height_pixels"], 100)

    def test_small_face_scale_tapers_to_zero(self):
        small = np.ones((40, 3), dtype=np.uint8)
        transition = np.ones((50, 3), dtype=np.uint8)
        large = np.ones((80, 3), dtype=np.uint8)

        self.assertEqual(_small_face_residual_scale(small), (1.0, 40))
        transition_scale, transition_height = _small_face_residual_scale(
            transition
        )
        self.assertEqual(transition_height, 50)
        self.assertAlmostEqual(
            transition_scale,
            (
                RESIDUAL_ZERO_STRENGTH_SUPPORT_HEIGHT_PIXELS - 50
            )
            / (
                RESIDUAL_ZERO_STRENGTH_SUPPORT_HEIGHT_PIXELS
                - RESIDUAL_FULL_STRENGTH_SUPPORT_HEIGHT_PIXELS
            ),
        )
        self.assertEqual(_small_face_residual_scale(large), (0.0, 80))

    def test_small_face_contract_uses_affected_gain_and_zero_replay_equivalence(self):
        baseline = [
            {
                "row_id": "small",
                "combined_part_failures": 10,
                "shape_correlation": 0.80,
                "gradient_correlation": 0.60,
                "normalized_rmse": 0.20,
            },
            {
                "row_id": "large",
                "combined_part_failures": 5,
                "shape_correlation": 0.90,
                "gradient_correlation": 0.70,
                "normalized_rmse": 0.10,
            },
        ]
        candidate = [
            {
                "row_id": "small",
                "combined_part_failures": 9,
                "shape_correlation": 0.81,
                "gradient_correlation": 0.61,
                "normalized_rmse": 0.19,
                "surface_residual": {
                    "residual_scale": 0.5,
                    "max_abs_correction": 0.02,
                },
            },
            {
                "row_id": "large",
                "combined_part_failures": 5,
                "shape_correlation": 0.8995,
                "gradient_correlation": 0.7005,
                "normalized_rmse": 0.1005,
                "surface_residual": {
                    "residual_scale": 0.0,
                    "max_abs_correction": 0.0,
                },
            },
        ]
        contract = _small_face_only_decision_contract(baseline, candidate)
        self.assertTrue(contract["passes"])
        self.assertEqual(contract["affected_row_ids"], ["small"])
        self.assertTrue(contract["zero_strength_rows"][0]["equivalent"])
        candidate[1]["surface_residual"]["max_abs_correction"] = 1e-4
        rejected = _small_face_only_decision_contract(baseline, candidate)
        self.assertFalse(rejected["passes"])
        self.assertFalse(rejected["checks"]["zero_strength_rows_equivalent"])


if __name__ == "__main__":
    unittest.main()
