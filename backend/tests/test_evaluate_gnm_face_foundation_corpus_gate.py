import unittest

from backend.benchmark.evaluate_gnm_face_foundation_corpus_gate import (
    SMOKE_ROW_IDS,
    _decision,
    _identity_disjointness,
    _select_rows,
    _summary,
)


def _summary_row(row_id, *, failures=2, shape=0.8, gradient=0.6, rmse=0.2):
    return {
        "row_id": row_id,
        "combined_part_failures": failures,
        "shape_correlation": shape,
        "gradient_correlation": gradient,
        "normalized_rmse": rmse,
    }


class GNMFaceFoundationCorpusGateTests(unittest.TestCase):
    def test_smoke_selection_is_pinned_small_validation_matrix(self):
        rows = []
        for index, row_id in enumerate(reversed(SMOKE_ROW_IDS)):
            rows.append(
                {
                    "row_id": row_id,
                    "split": "validation",
                    "render": {"face_bbox_height_pixels": 70 + index},
                }
            )

        selected = _select_rows({"rows": rows}, "smoke")

        self.assertEqual([row["row_id"] for row in selected], list(SMOKE_ROW_IDS))

    def test_smoke_selection_rejects_missing_pinned_row(self):
        with self.assertRaisesRegex(ValueError, "missing pinned rows"):
            _select_rows({"rows": []}, "smoke")

    def test_identity_disjointness_checks_train_and_sealed_splits(self):
        selected = [{"identity_group": "held-out"}]
        reference = {
            "rows": [
                {"identity_group": "train-a", "split": "train"},
                {"identity_group": "sealed-a", "split": "sealed"},
                {"identity_group": "held-out", "split": "validation"},
            ]
        }

        result = _identity_disjointness(selected, reference)

        self.assertTrue(result["passed"])
        self.assertEqual(result["training_overlap"], [])
        self.assertEqual(result["sealed_overlap"], [])

    def test_identity_disjointness_reports_train_overlap(self):
        result = _identity_disjointness(
            [{"identity_group": "shared"}],
            {
                "rows": [
                    {"identity_group": "shared", "split": "train"},
                ]
            },
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["training_overlap"], ["shared"])

    def test_decision_requires_auditable_non_regressing_application(self):
        control_rows = [_summary_row("a"), _summary_row("b")]
        candidate_rows = [
            {
                **_summary_row(
                    "a",
                    failures=1,
                    shape=0.81,
                    gradient=0.61,
                    rmse=0.19,
                ),
                "parametric_foundation": {
                    "enabled": True,
                    "boundary_max_abs_correction": 0.0,
                },
                "shape_delta": 0.01,
                "gradient_delta": 0.01,
                "normalized_rmse_delta": -0.01,
                "maximum_difference_outside_face_region": 0.0,
                "finite": True,
                "shared_input_equal": True,
                "cache_bbox_equal": True,
            },
            {
                **_summary_row("b"),
                "parametric_foundation": {
                    "enabled": False,
                    "reason": "alignment_reliability_gate",
                },
                "shape_delta": 0.0,
                "gradient_delta": 0.0,
                "normalized_rmse_delta": 0.0,
                "maximum_difference_outside_face_region": 0.0,
                "finite": True,
                "shared_input_equal": True,
                "cache_bbox_equal": True,
            },
        ]
        selected_rows = [
            {"render": {"face_bbox_height_pixels": 70}},
            {"render": {"face_bbox_height_pixels": 88}},
        ]

        decision = _decision(
            _summary(control_rows),
            _summary(candidate_rows),
            selected_rows=selected_rows,
        )

        self.assertTrue(decision["passed"])
        self.assertEqual(decision["applied_rows"], 1)
        self.assertEqual(decision["skipped_rows"], 1)

    def test_decision_holds_on_per_row_failure_regression(self):
        control_rows = [_summary_row("a", failures=1)]
        candidate_rows = [
            {
                **_summary_row("a", failures=2),
                "parametric_foundation": {
                    "enabled": True,
                    "boundary_max_abs_correction": 0.0,
                },
                "shape_delta": 0.01,
                "gradient_delta": 0.01,
                "normalized_rmse_delta": -0.01,
                "maximum_difference_outside_face_region": 0.0,
                "finite": True,
                "shared_input_equal": True,
                "cache_bbox_equal": True,
            }
        ]

        decision = _decision(
            _summary(control_rows),
            _summary(candidate_rows),
            selected_rows=[{"render": {"face_bbox_height_pixels": 70}}],
        )

        self.assertFalse(decision["paired_per_row_failure_no_regression"])
        self.assertFalse(decision["passed"])

    def test_decision_holds_on_excessive_per_row_gradient_regression(self):
        control_rows = [_summary_row("a", gradient=0.6)]
        candidate_rows = [
            {
                **_summary_row("a", failures=1, gradient=0.589),
                "parametric_foundation": {
                    "enabled": True,
                    "boundary_max_abs_correction": 0.0,
                },
                "shape_delta": 0.01,
                "gradient_delta": -0.011,
                "normalized_rmse_delta": -0.01,
                "maximum_difference_outside_face_region": 0.0,
                "finite": True,
                "shared_input_equal": True,
                "cache_bbox_equal": True,
            }
        ]

        decision = _decision(
            _summary(control_rows),
            _summary(candidate_rows),
            selected_rows=[{"render": {"face_bbox_height_pixels": 70}}],
        )

        self.assertFalse(
            decision["paired_per_row_gradient_regression_bounded"]
        )
        self.assertFalse(decision["passed"])

    def test_decision_holds_when_absolute_face_quality_is_unacceptable(self):
        control_rows = [_summary_row("a", failures=8)]
        candidate_rows = [
            {
                **_summary_row(
                    "a",
                    failures=7,
                    shape=0.81,
                    gradient=0.61,
                    rmse=0.19,
                ),
                "parametric_foundation": {
                    "enabled": True,
                    "boundary_max_abs_correction": 0.0,
                },
                "shape_delta": 0.01,
                "gradient_delta": 0.01,
                "normalized_rmse_delta": -0.01,
                "maximum_difference_outside_face_region": 0.0,
                "finite": True,
                "shared_input_equal": True,
                "cache_bbox_equal": True,
            }
        ]

        decision = _decision(
            _summary(control_rows),
            _summary(candidate_rows),
            selected_rows=[{"render": {"face_bbox_height_pixels": 70}}],
        )

        self.assertFalse(
            decision["absolute_part_failure_rate_within_limit"]
        )
        self.assertFalse(decision["passed"])


if __name__ == "__main__":
    unittest.main()
