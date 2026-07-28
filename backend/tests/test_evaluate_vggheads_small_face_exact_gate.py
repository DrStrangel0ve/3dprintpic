import unittest

from backend.benchmark.evaluate_vggheads_small_face_exact_gate import (
    HARD_SMALL_FACE_ROW,
    _decision,
)


def _quality(row_id: str, failures: int, shape: float, gradient: float, rmse: float):
    return {
        "row_id": row_id,
        "combined_part_failures": failures,
        "shape_correlation": shape,
        "gradient_correlation": gradient,
        "normalized_rmse": rmse,
    }


class EvaluateVGGHeadsSmallFaceExactGateTests(unittest.TestCase):
    def _summaries(self):
        baseline_rows = [
            _quality(HARD_SMALL_FACE_ROW, 10, 0.80, 0.62553, 0.1760),
            _quality("eyewear_overhead_panel_384", 3, 0.94, 0.67, 0.102),
            _quality("strong_turn_layered_384", 6, 0.89, 0.50, 0.135),
        ]
        candidate_rows = [
            _quality(HARD_SMALL_FACE_ROW, 9, 0.81, 0.62551, 0.1757),
            baseline_rows[1].copy(),
            baseline_rows[2].copy(),
        ]
        return (
            {"combined_part_failures": 19, "rows": baseline_rows},
            {"combined_part_failures": 18, "rows": candidate_rows},
        )

    def _contracts(self):
        return [
            {
                "row_id": HARD_SMALL_FACE_ROW,
                "policy": {"eligible": True},
                "background_value_exact": True,
                "boundary_value_exact": True,
                "candidate_maximum_absolute_difference": 0.03,
                "provider": {
                    "confidence": 0.94,
                    "absolute_yaw_magnitude_error_deg": 0.08,
                },
                "fusion": {"provider_crop_coverage": 0.59},
            },
            {
                "row_id": "eyewear_overhead_panel_384",
                "policy": {"eligible": False},
                "background_value_exact": True,
                "boundary_value_exact": True,
                "candidate_maximum_absolute_difference": 0.0,
            },
            {
                "row_id": "strong_turn_layered_384",
                "policy": {"eligible": False},
                "background_value_exact": True,
                "boundary_value_exact": True,
                "candidate_maximum_absolute_difference": 0.0,
            },
        ]

    def test_decision_accepts_small_face_improvement_with_exact_bypasses(self):
        baseline, candidate = self._summaries()

        decision = _decision(
            baseline,
            candidate,
            self._contracts(),
            baseline_matches_historical=True,
            provider_runnable=True,
        )

        self.assertTrue(decision["eligible_for_30mm_stl_replay"])

    def test_decision_rejects_background_or_bypass_changes(self):
        baseline, candidate = self._summaries()
        contracts = self._contracts()
        contracts[1]["background_value_exact"] = False
        contracts[1]["candidate_maximum_absolute_difference"] = 1e-4

        decision = _decision(
            baseline,
            candidate,
            contracts,
            baseline_matches_historical=True,
            provider_runnable=True,
        )

        self.assertFalse(decision["selection_background_exact"])
        self.assertFalse(decision["bypassed_rows_value_exact"])
        self.assertFalse(decision["eligible_for_30mm_stl_replay"])


if __name__ == "__main__":
    unittest.main()
