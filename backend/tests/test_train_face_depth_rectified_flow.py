import unittest

import numpy as np

from backend.benchmark.train_face_depth_rectified_flow import (
    MAX_PEAK_VRAM_GB,
    PilotConfig,
    _attachment_boundary,
    _calibrated_uncertainty,
    _flow_advance_gate,
    _nearest_face_fill,
    _robust_unit_depth,
    _select_rows,
    _source_resolution_candidate,
)


def _quality_row(row_id: str, failures: int, rmse: float) -> dict:
    failed_parts = [f"part-{index}" for index in range(failures)]
    return {
        "row_id": row_id,
        "shape_failed_parts": failed_parts,
        "affine_failed_parts": [],
        "normalized_rmse": rmse,
        "gradient_correlation": 0.8,
        "shape_correlation": 0.9,
    }


def _summary(rows: list[dict], failures: int) -> dict:
    return {
        "combined_part_failures": failures,
        "median_shape_correlation": 0.9,
        "median_gradient_correlation": 0.8,
        "median_normalized_rmse": 0.1,
        "source_background_bit_exact": True,
        "rows": rows,
    }


class TrainFaceDepthRectifiedFlowTests(unittest.TestCase):
    def test_pilot_limits_cannot_exceed_local_gpu_budget(self):
        self.assertEqual(PilotConfig().max_peak_vram_gb, MAX_PEAK_VRAM_GB)
        with self.assertRaisesRegex(ValueError, "11 GiB"):
            PilotConfig(max_peak_vram_gb=11.1).validated()
        with self.assertRaisesRegex(ValueError, "2 hours"):
            PilotConfig(max_runtime_seconds=7201.0).validated()
        with self.assertRaisesRegex(ValueError, "evaluation reserve"):
            PilotConfig(evaluation_reserve_seconds=29.0).validated()

    def test_camera_depth_normalization_preserves_or_inverts_order(self):
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        face = np.ones_like(values, dtype=bool)
        direct = _robust_unit_depth(values, face, inverse=False)
        inverse = _robust_unit_depth(values, face, inverse=True)
        self.assertGreater(float(direct[-1, -1]), float(direct[0, 0]))
        self.assertLess(float(inverse[-1, -1]), float(inverse[0, 0]))
        np.testing.assert_allclose(direct + inverse, 1.0, atol=1e-6)

    def test_attachment_boundary_is_inside_support(self):
        face = np.zeros((12, 12), dtype=bool)
        face[2:10, 2:10] = True
        boundary = _attachment_boundary(face, width=1)
        self.assertTrue(np.all(boundary[~face] == 0.0))
        self.assertTrue(np.all(boundary[2, 2:10] == 1.0))
        self.assertTrue(np.all(boundary[4:8, 4:8] == 0.0))

    def test_nearest_fill_does_not_mix_zero_into_face_edge(self):
        values = np.full((7, 7), np.nan, dtype=np.float32)
        face = np.zeros((7, 7), dtype=bool)
        face[2:5, 2:5] = True
        values[face] = 3.0
        filled = _nearest_face_fill(values, face)
        self.assertTrue(np.isfinite(filled).all())
        self.assertTrue(np.all(filled == 3.0))

    def test_source_resolution_resize_preserves_background_bit_exactly(self):
        coarse = np.arange(48, dtype=np.float32).reshape(6, 8)
        face = np.zeros((6, 8), dtype=bool)
        face[1:5, 2:6] = True
        candidate, bit_exact = _source_resolution_candidate(
            coarse,
            face,
            np.ones((3, 3), dtype=np.float32),
        )
        self.assertTrue(bit_exact)
        self.assertTrue(np.array_equal(candidate[~face], coarse[~face]))
        self.assertTrue(np.allclose(candidate[face], coarse[face] + 1.0))

    def test_unsupported_low_resolution_values_cannot_bleed_into_emission(self):
        coarse = np.zeros((12, 12), dtype=np.float32)
        face = np.ones((12, 12), dtype=bool)
        residual = np.full((4, 4), 100.0, dtype=np.float32)
        support = np.zeros((4, 4), dtype=np.float32)
        residual[1:3, 1:3] = 1.0
        support[1:3, 1:3] = 1.0
        candidate, _ = _source_resolution_candidate(
            coarse,
            face,
            residual,
            support,
        )
        self.assertLessEqual(float(np.max(candidate)), 1.01)
        self.assertGreater(float(np.max(candidate)), 0.9)

    def test_uncertainty_interval_uses_validation_fitted_scale(self):
        validation = _calibrated_uncertainty(
            np.ones(10),
            np.linspace(0.1, 1.0, 10),
            None,
        )
        self.assertEqual(validation["calibration_mode"], "validation-fit")
        sealed = _calibrated_uncertainty(
            np.ones(4),
            np.full(4, 0.5),
            validation["calibration_scale_90"],
        )
        self.assertEqual(sealed["calibration_mode"], "sealed-using-validation-scale")
        self.assertFalse(sealed["raw_sample_quantile_claimed"])
        zero_spread = _calibrated_uncertainty(
            np.zeros(4),
            np.full(4, 0.5),
            500_000.0,
        )
        self.assertEqual(
            zero_spread["validation_calibrated_90_interval_coverage"],
            1.0,
        )

    def test_smoke_row_selection_keeps_all_splits_without_overlap(self):
        rows = [
            {"row_id": f"{split}-{index}", "split": split}
            for split in ("train", "validation", "sealed")
            for index in range(3)
        ]
        selected = _select_rows(rows, 2)
        self.assertEqual(len(selected), 6)
        self.assertEqual(len({row["row_id"] for row in selected}), 6)
        self.assertEqual(
            {split: sum(row["split"] == split for row in selected) for split in {
                "train",
                "validation",
                "sealed",
            }},
            {"train": 2, "validation": 2, "sealed": 2},
        )

    def test_advance_gate_requires_every_quality_and_uncertainty_check(self):
        control_rows = [_quality_row(f"row-{index}", 1, 0.2) for index in range(5)]
        flow_rows = [_quality_row(f"row-{index}", 0, 0.1) for index in range(5)]
        control_small = _summary(control_rows, 5)
        flow_small = _summary(flow_rows, 0)
        control = {**control_small, "small_turned": control_small}
        flow = {
            **flow_small,
            "small_turned": flow_small,
            "uncertainty": {
                "mad_absolute_error_spearman": 0.5,
                "validation_calibrated_90_interval_coverage": 0.9,
            },
        }
        run_checks = {
            "step_budget_complete": True,
            "runtime_cap_passed": True,
            "vram_cap_passed": True,
            "sealed_evaluation_complete": True,
        }
        passed = _flow_advance_gate(flow, control, run_checks)
        self.assertEqual(passed["decision"], "advance-to-exact-hard-row")
        self.assertTrue(all(passed["checks"].values()))
        flow["uncertainty"]["validation_calibrated_90_interval_coverage"] = 0.5
        held = _flow_advance_gate(flow, control, run_checks)
        self.assertEqual(held["decision"], "hold")
        self.assertFalse(
            held["checks"]["validation_calibrated_90_coverage_in_0_85_0_95"]
        )
        flow["uncertainty"]["validation_calibrated_90_interval_coverage"] = 0.9
        run_checks["runtime_cap_passed"] = False
        cap_held = _flow_advance_gate(flow, control, run_checks)
        self.assertEqual(cap_held["decision"], "hold")
        self.assertFalse(cap_held["checks"]["runtime_cap_passed"])


if __name__ == "__main__":
    unittest.main()
