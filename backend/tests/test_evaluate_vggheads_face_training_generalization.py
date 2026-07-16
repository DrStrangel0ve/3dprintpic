import unittest

from backend.benchmark.evaluate_vggheads_face_training_generalization import (
    CHALLENGE_CONTEXTS,
    _final_decision,
    select_generalization_suite,
    select_train_variant,
)
from backend.benchmark.makehuman_face_training_corpus import (
    EXPRESSIONS,
    SPLIT_BY_IDENTITY,
)


def _row(
    identity,
    expression,
    suffix,
    context,
    *,
    height,
    occluded=False,
):
    split = SPLIT_BY_IDENTITY[identity]
    dimension, yaw, background, lighting = context
    return {
        "row_id": f"{identity}__{expression}_{suffix}",
        "split": split,
        "identity_group": identity,
        "expression": expression,
        "render": {"face_bbox_height_pixels": height},
        "spec": {
            "target_dimension": dimension,
            "camera_yaw_deg": yaw,
            "background_profile": background,
            "lighting_profile": lighting,
            "occlusion": "eye_band" if occluded else None,
        },
    }


def _summary(rows):
    return {
        "combined_part_failures": sum(
            row["combined_part_failures"] for row in rows
        ),
        "median_shape_correlation": sum(
            row["shape_correlation"] for row in rows
        )
        / len(rows),
        "median_gradient_correlation": sum(
            row["gradient_correlation"] for row in rows
        )
        / len(rows),
        "median_normalized_rmse": sum(
            row["normalized_rmse"] for row in rows
        )
        / len(rows),
        "rows": rows,
    }


def _quality(row_id, failures, *, shape=0.9, gradient=0.7, rmse=0.1):
    return {
        "row_id": row_id,
        "combined_part_failures": failures,
        "shape_correlation": shape,
        "gradient_correlation": gradient,
        "normalized_rmse": rmse,
    }


class VGGHeadsFaceTrainingGeneralizationTests(unittest.TestCase):
    def test_suite_balances_contexts_and_preserves_identity_splits(self):
        rows = []
        identities = sorted(SPLIT_BY_IDENTITY)
        for identity in identities:
            for expression in EXPRESSIONS:
                for index, context in enumerate(CHALLENGE_CONTEXTS):
                    rows.append(
                        _row(
                            identity,
                            expression,
                            f"small_{index}",
                            context,
                            height=70 + index,
                        )
                    )
                rows.append(
                    _row(
                        identity,
                        expression,
                        "occluded",
                        CHALLENGE_CONTEXTS[0],
                        height=72,
                        occluded=True,
                    )
                )
                rows.append(
                    _row(
                        identity,
                        expression,
                        "large",
                        CHALLENGE_CONTEXTS[1],
                        height=180,
                    )
                )

        suite = select_generalization_suite(rows)

        challenge = suite["challenge"]
        self.assertEqual(len(challenge), 32)
        self.assertEqual(len(suite["occluded_controls"]), 8)
        self.assertEqual(len(suite["large_controls"]), 8)
        self.assertEqual(
            {row["identity_group"] for row in challenge},
            set(SPLIT_BY_IDENTITY),
        )
        self.assertEqual(
            {row["expression"] for row in challenge},
            set(EXPRESSIONS),
        )
        self.assertEqual(
            {
                (
                    row["spec"]["target_dimension"],
                    row["spec"]["camera_yaw_deg"],
                    row["spec"]["background_profile"],
                    row["spec"]["lighting_profile"],
                )
                for row in challenge
            },
            set(CHALLENGE_CONTEXTS),
        )
        for identity in SPLIT_BY_IDENTITY:
            self.assertEqual(
                {
                    (
                        row["spec"]["target_dimension"],
                        row["spec"]["camera_yaw_deg"],
                        row["spec"]["background_profile"],
                        row["spec"]["lighting_profile"],
                    )
                    for row in challenge
                    if row["identity_group"] == identity
                },
                set(CHALLENGE_CONTEXTS),
            )
        for expression in EXPRESSIONS:
            counts = {
                context: sum(
                    row["expression"] == expression
                    and (
                        row["spec"]["target_dimension"],
                        row["spec"]["camera_yaw_deg"],
                        row["spec"]["background_profile"],
                        row["spec"]["lighting_profile"],
                    )
                    == context
                    for row in challenge
                )
                for context in CHALLENGE_CONTEXTS
            }
            self.assertEqual(set(counts.values()), {2})
        self.assertEqual(
            {
                split: sum(row["split"] == split for row in challenge)
                for split in ("train", "validation", "sealed")
            },
            {"train": 24, "validation": 4, "sealed": 4},
        )

    def test_train_selector_prefers_eligible_variant(self):
        baseline = _summary(
            [_quality("a", 3), _quality("b", 3), _quality("c", 3)]
        )
        diagnostic = {
            "variant_id": "diagnostic",
            "provider_alpha": 0.25,
            "sigma_ratio": 0.04,
            "maximum_provider_yaw_deg": 30.0,
            "summary": _summary(
                [_quality("a", 2), _quality("b", 2), _quality("c", 5)]
            ),
        }
        eligible = {
            "variant_id": "eligible",
            "provider_alpha": 0.5,
            "sigma_ratio": 0.06,
            "maximum_provider_yaw_deg": 30.0,
            "summary": _summary(
                [_quality("a", 2), _quality("b", 3), _quality("c", 3)]
            ),
        }
        for variant in (diagnostic, eligible):
            for row in variant["summary"]["rows"]:
                row["background_value_exact"] = True
                row["boundary_value_exact"] = True

        selection = select_train_variant(
            [diagnostic, eligible],
            baseline,
        )

        self.assertEqual(selection["selected_variant_id"], "eligible")
        self.assertTrue(selection["selected_is_train_eligible"])

    def test_final_decision_rejects_held_out_row_regression(self):
        baseline_rows = [_quality("a", 3), _quality("b", 3)]
        candidate_rows = [_quality("a", 2), _quality("b", 3)]
        baseline = _summary(baseline_rows)
        candidate = _summary(candidate_rows)
        baseline_by_split = {
            "train": baseline,
            "validation": baseline,
            "sealed": baseline,
            "all": baseline,
        }
        candidate_by_split = {
            "train": candidate,
            "validation": candidate,
            "sealed": _summary(
                [_quality("a", 2), _quality("b", 4)]
            ),
            "all": candidate,
        }
        candidate_contract_rows = [
            {
                "background_value_exact": True,
                "boundary_value_exact": True,
                "provider": {
                    "confidence": 0.95,
                    "absolute_yaw_magnitude_error_deg": 2.0,
                },
                "fusion": {"provider_crop_coverage": 0.75},
                "policy": {"applied": True},
            }
        ]
        anchor_baseline = {
            **_quality("hard", 10),
            "policy_applied": False,
        }
        anchor_candidate = {
            **_quality(
                "hard",
                9,
                shape=0.91,
                gradient=0.7,
                rmse=0.09,
            ),
            "policy_applied": True,
        }

        decision = _final_decision(
            baseline_by_split,
            candidate_by_split,
            {"selected_is_train_eligible": True},
            candidate_contract_rows,
            [
                {
                    "candidate_maximum_absolute_difference": 0.0,
                }
            ],
            anchor_baseline=anchor_baseline,
            anchor_candidate=anchor_candidate,
        )

        self.assertFalse(decision["checks"]["sealed_split_passes"])
        self.assertFalse(
            decision["checks"]["eligible_for_30mm_stl_replay"]
        )


if __name__ == "__main__":
    unittest.main()
