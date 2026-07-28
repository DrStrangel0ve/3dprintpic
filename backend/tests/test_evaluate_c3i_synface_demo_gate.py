import unittest

from backend.benchmark.evaluate_c3i_synface_demo_gate import (
    MHR_PROVIDER,
    RAW_C3I_PROVIDER,
    RAP3DF_PROVIDER,
    _corpus_profile,
    gate_decision,
    summarize_rows,
)


class EvaluateC3ISynFaceDemoGateTests(unittest.TestCase):
    @staticmethod
    def _quality(**overrides):
        return {
            "available": True,
            "combined_part_failures": 4,
            "shape_correlation": 0.8,
            "gradient_correlation": 0.6,
            "normalized_rmse": 0.2,
            "checks": {
                "coverage": True,
                "depth_semantics_orientation": True,
            },
            **overrides,
        }

    def test_summarize_rows_uses_named_part_failures_and_medians(self):
        rows = [
            {
                "candidate": {
                    "available": True,
                    "combined_part_failures": 3,
                    "shape_correlation": 0.8,
                    "gradient_correlation": 0.6,
                    "normalized_rmse": 0.2,
                }
            },
            {
                "candidate": {
                    "available": True,
                    "combined_part_failures": 1,
                    "shape_correlation": 0.9,
                    "gradient_correlation": 0.7,
                    "normalized_rmse": 0.1,
                }
            },
        ]
        summary = summarize_rows(rows, "candidate")
        self.assertEqual(summary["combined_part_failures"], 4)
        self.assertEqual(summary["part_failure_opportunities"], 24)
        self.assertAlmostEqual(summary["part_failure_rate"], 1 / 6)
        self.assertAlmostEqual(summary["median_shape_correlation"], 0.85)
        self.assertEqual(summary["available_rows"], 2)

    def test_decision_advances_only_on_complete_non_regressing_improvement(self):
        global_summary = {
            "row_count": 1,
            "available_rows": 1,
            "combined_part_failures": 4,
            "part_failure_rate": 1 / 3,
            "median_shape_correlation": 0.8,
            "median_gradient_correlation": 0.6,
            "median_normalized_rmse": 0.2,
        }
        current_summary = {
            **global_summary,
            "combined_part_failures": 3,
            "median_shape_correlation": 0.81,
            "median_gradient_correlation": 0.61,
            "median_normalized_rmse": 0.19,
        }
        rows = [
            {
                "row_id": "row-1",
                "global_depth": self._quality(),
                "current_refined": self._quality(
                    combined_part_failures=3,
                    shape_correlation=0.81,
                    gradient_correlation=0.61,
                    normalized_rmse=0.19,
                ),
            }
        ]
        decision = gate_decision(global_summary, current_summary, rows=rows)
        self.assertEqual(decision["status"], "advance-current-path")
        self.assertEqual(decision["failed_checks"], [])

    def test_decision_holds_a_gradient_regression(self):
        global_summary = {
            "row_count": 2,
            "available_rows": 2,
            "combined_part_failures": 8,
            "part_failure_rate": 1 / 3,
            "median_shape_correlation": 0.8,
            "median_gradient_correlation": 0.6,
            "median_normalized_rmse": 0.2,
        }
        current_summary = {
            **global_summary,
            "combined_part_failures": 7,
            "median_gradient_correlation": 0.59,
        }
        decision = gate_decision(global_summary, current_summary)
        self.assertEqual(decision["status"], "hold")
        self.assertIn("median_gradient_no_regression", decision["failed_checks"])

    def test_decision_holds_a_per_row_part_regression(self):
        summary = {
            "row_count": 1,
            "available_rows": 1,
            "combined_part_failures": 4,
            "part_failure_rate": 1 / 3,
            "median_shape_correlation": 0.8,
            "median_gradient_correlation": 0.6,
            "median_normalized_rmse": 0.2,
        }
        rows = [
            {
                "row_id": "row-1",
                "global_depth": self._quality(combined_part_failures=3),
                "current_refined": self._quality(
                    combined_part_failures=4,
                    shape_correlation=0.81,
                    gradient_correlation=0.61,
                    normalized_rmse=0.19,
                ),
            }
        ]
        decision = gate_decision(summary, summary, rows=rows)
        self.assertEqual(decision["status"], "hold")
        self.assertIn(
            "per_row_part_failures_no_regression",
            decision["failed_checks"],
        )

    def test_decision_holds_incomplete_pairing(self):
        summary = {
            "row_count": 2,
            "available_rows": 2,
            "combined_part_failures": 4,
            "part_failure_rate": 1 / 6,
            "median_shape_correlation": 0.8,
            "median_gradient_correlation": 0.6,
            "median_normalized_rmse": 0.2,
        }
        rows = [
            {
                "row_id": "row-1",
                "global_depth": self._quality(),
                "current_refined": self._quality(
                    combined_part_failures=3,
                ),
            }
        ]
        decision = gate_decision(summary, summary, rows=rows)
        self.assertEqual(decision["status"], "hold")
        self.assertIn("paired_rows_complete", decision["failed_checks"])

    def test_decision_holds_a_complete_bounded_slice(self):
        summary = {
            "row_count": 1,
            "available_rows": 1,
            "combined_part_failures": 3,
            "part_failure_rate": 0.25,
            "median_shape_correlation": 0.81,
            "median_gradient_correlation": 0.61,
            "median_normalized_rmse": 0.19,
        }
        rows = [
            {
                "row_id": "row-1",
                "global_depth": self._quality(),
                "current_refined": self._quality(
                    combined_part_failures=3,
                    shape_correlation=0.81,
                    gradient_correlation=0.61,
                    normalized_rmse=0.19,
                ),
            }
        ]
        decision = gate_decision(
            summary,
            summary,
            rows=rows,
            expected_row_count=10,
        )
        self.assertEqual(decision["status"], "hold")
        self.assertIn("full_corpus_evaluated", decision["failed_checks"])

    def test_decision_holds_wrong_orientation(self):
        summary = {
            "row_count": 1,
            "available_rows": 1,
            "combined_part_failures": 4,
            "part_failure_rate": 1 / 3,
            "median_shape_correlation": 0.8,
            "median_gradient_correlation": 0.6,
            "median_normalized_rmse": 0.2,
        }
        rows = [
            {
                "row_id": "row-1",
                "global_depth": self._quality(),
                "current_refined": self._quality(
                    combined_part_failures=3,
                    checks={
                        "coverage": True,
                        "depth_semantics_orientation": False,
                    },
                ),
            }
        ]
        decision = gate_decision(summary, summary, rows=rows)
        self.assertEqual(decision["status"], "hold")
        self.assertIn(
            "all_current_rows_have_expected_orientation",
            decision["failed_checks"],
        )

    def test_raw_c3i_profile_requires_license_and_disjoint_splits(self):
        profile = _corpus_profile(
            {
                "provider": RAW_C3I_PROVIDER,
                "dataset_license": "CC BY 4.0",
                "identity_disjoint_splits": True,
                "source_geometry_training_and_evaluation_only": True,
            }
        )
        self.assertIn("raw-exr", profile["method"])
        with self.assertRaisesRegex(ValueError, "licensed disjoint"):
            _corpus_profile(
                {
                    "provider": RAW_C3I_PROVIDER,
                    "dataset_license": "CC BY 4.0",
                    "identity_disjoint_splits": False,
                    "source_geometry_training_and_evaluation_only": True,
                }
            )

    def test_rap3df_profile_is_evaluation_only(self):
        profile = _corpus_profile(
            {
                "provider": RAP3DF_PROVIDER,
                "training_eligible": False,
                "source_geometry_evaluation_only": True,
                "source": {"license": "CC BY 4.0"},
            }
        )
        self.assertIn("diagnostic only", profile["depth_target_provenance"]["use"])
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            _corpus_profile(
                {
                    "provider": RAP3DF_PROVIDER,
                    "training_eligible": True,
                    "source_geometry_evaluation_only": True,
                    "source": {"license": "CC BY 4.0"},
                }
            )

    def test_mhr_profile_accepts_only_pinned_nonpromotable_pilot_contract(self):
        corpus = {
            "provider": MHR_PROVIDER,
            "source_revision": "4998cec385b1aaa07abdefba71bfba2f83c7db32",
            "privacy_safe_synthetic": True,
            "pilot_corpus": True,
            "experimental_corpus": True,
            "corpus_complete": False,
            "training_eligible": False,
            "promotion_eligible": False,
            "identity_disjoint_splits": False,
            "source_geometry_training_and_evaluation_only": True,
            "source": {"license": "Apache-2.0"},
            "preflight": {
                "runnable": True,
                "checks": {"source_revision_matches": True},
            },
            "face_part_provenance": {
                "stable_across_identity_and_expression": True,
                "not_claimed": "official MHR semantic segmentation",
            },
            "depth_target_provenance": {"metric_scale_claimed": False},
        }
        profile = _corpus_profile(corpus)
        self.assertIn("smoke only", profile["depth_target_provenance"]["use"])
        with self.assertRaisesRegex(ValueError, "pilot-corpus provenance"):
            _corpus_profile({**corpus, "promotion_eligible": True})


if __name__ == "__main__":
    unittest.main()
