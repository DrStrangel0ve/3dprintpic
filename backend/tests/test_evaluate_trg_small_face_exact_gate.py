import copy
import unittest

from backend.benchmark.evaluate_trg_small_face_exact_gate import (
    EXACT_ROW_CONTRACT,
    _selected_candidate,
    exact_row_contract_checks,
    promotion_decision,
)


PARTS = (
    "left_eye",
    "left_eyebrow",
    "mouth",
    "nose",
    "right_eye",
    "right_eyebrow",
)


def _metrics(*, candidate: bool) -> dict:
    shape_parts = []
    affine_parts = []
    for name in PARTS:
        shape_parts.append(
            {
                "name": name,
                "shape_correlation": 0.93 if candidate else 0.91,
                "minimum_raw_gradient_correlation": 0.31 if candidate else 0.29,
                "passed": candidate,
            }
        )
        affine_parts.append(
            {
                "name": name,
                "rmse_mm": 0.60 if candidate else 0.70,
                "p95_absolute_error_mm": 1.00 if candidate else 1.20,
                "bias_mm": 0.10 if candidate else 0.20,
                "span_retention": 0.95 if candidate else 0.90,
                "passed": candidate,
            }
        )
    return {
        "shape_correlation": 0.82 if candidate else 0.81,
        "gradient_correlation": 0.64 if candidate else 0.63,
        "normalized_rmse": 0.16 if candidate else 0.17,
        "named_part_shape": {
            "passed": candidate,
            "failed_parts": [] if candidate else list(PARTS),
            "parts": shape_parts,
        },
        "named_part_affine_mm": {
            "passed": candidate,
            "failed_parts": [] if candidate else list(PARTS),
            "parts": affine_parts,
        },
        "checks": {"passed": candidate},
    }


class EvaluateTRGSmallFaceExactGateTests(unittest.TestCase):
    def test_exact_row_contract_rejects_stale_baseline_or_bbox(self):
        hashes = {
            name: value
            for name, value in EXACT_ROW_CONTRACT.items()
            if name.endswith("_sha256")
        }
        accepted = exact_row_contract_checks(
            hashes,
            target_bbox_xyxy=EXACT_ROW_CONTRACT["target_bbox_xyxy"],
            crop_bbox_xyxy=EXACT_ROW_CONTRACT["crop_bbox_xyxy"],
        )
        stale = dict(hashes)
        stale["baseline_sha256"] = "0" * 64
        rejected = exact_row_contract_checks(
            stale,
            target_bbox_xyxy=[0, 0, 1, 1],
            crop_bbox_xyxy=EXACT_ROW_CONTRACT["crop_bbox_xyxy"],
        )

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["checks"]["baseline_sha256"])
        self.assertFalse(rejected["checks"]["target_bbox_xyxy"])
        self.assertFalse(rejected["passed"])

    def test_promotion_requires_complete_six_part_improvement(self):
        decision = promotion_decision(
            _metrics(candidate=False),
            _metrics(candidate=True),
            provider_runnable=True,
            projected_bbox_iou=0.75,
            provider_face_coverage=0.95,
            background_value_exact=True,
            boundary_value_exact=True,
        )

        self.assertTrue(decision["checks"]["eligible_for_30mm_stl_replay"])
        self.assertTrue(decision["per_part_comparison"]["passed"])

    def test_promotion_rejects_one_part_regression(self):
        baseline = _metrics(candidate=False)
        candidate = _metrics(candidate=True)
        candidate = copy.deepcopy(candidate)
        candidate["named_part_shape"]["parts"][1][
            "minimum_raw_gradient_correlation"
        ] = 0.20

        decision = promotion_decision(
            baseline,
            candidate,
            provider_runnable=True,
            projected_bbox_iou=0.75,
            provider_face_coverage=0.95,
            background_value_exact=True,
            boundary_value_exact=True,
        )

        self.assertFalse(decision["checks"]["every_part_metric_no_regression"])
        self.assertFalse(decision["checks"]["eligible_for_30mm_stl_replay"])

    def test_promotion_rejects_registration_or_background_failure(self):
        decision = promotion_decision(
            _metrics(candidate=False),
            _metrics(candidate=True),
            provider_runnable=True,
            projected_bbox_iou=0.10,
            provider_face_coverage=0.95,
            background_value_exact=False,
            boundary_value_exact=True,
        )

        self.assertFalse(decision["checks"]["projected_bbox_registered"])
        self.assertFalse(decision["checks"]["selection_background_exact"])
        self.assertFalse(decision["checks"]["eligible_for_30mm_stl_replay"])

    def test_promotion_rejects_incomplete_or_duplicate_part_telemetry(self):
        baseline = _metrics(candidate=False)
        candidate = _metrics(candidate=True)
        candidate["named_part_shape"]["parts"] = [
            candidate["named_part_shape"]["parts"][0]
        ] * len(PARTS)
        candidate["named_part_shape"]["passed"] = True

        decision = promotion_decision(
            baseline,
            candidate,
            provider_runnable=True,
            projected_bbox_iou=0.75,
            provider_face_coverage=0.95,
            background_value_exact=True,
            boundary_value_exact=True,
        )

        self.assertFalse(decision["checks"]["all_six_shape_parts_pass"])
        self.assertFalse(decision["checks"]["every_part_metric_no_regression"])
        self.assertFalse(decision["checks"]["eligible_for_30mm_stl_replay"])

    def test_candidate_selection_prefers_any_replay_eligible_alpha(self):
        ineligible = {
            "alpha": 0.5,
            "summary": {
                "combined_part_failures": 0,
                "shape_correlation": 0.99,
                "gradient_correlation": 0.99,
                "normalized_rmse": 0.01,
            },
            "decision": {"checks": {"eligible_for_30mm_stl_replay": False}},
        }
        eligible = {
            "alpha": 0.25,
            "summary": {
                "combined_part_failures": 0,
                "shape_correlation": 0.95,
                "gradient_correlation": 0.95,
                "normalized_rmse": 0.05,
            },
            "decision": {"checks": {"eligible_for_30mm_stl_replay": True}},
        }

        self.assertIs(_selected_candidate([ineligible, eligible]), eligible)


if __name__ == "__main__":
    unittest.main()
