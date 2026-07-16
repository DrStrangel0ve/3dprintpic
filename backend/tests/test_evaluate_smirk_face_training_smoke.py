import unittest
from unittest.mock import patch

import numpy as np

from backend.benchmark.evaluate_smirk_face_training_smoke import (
    SMOKE_ROW_IDS,
    _bbox_iou,
    _similarity_matrix,
    apply_xy_registration,
    detect_landmark_regions,
    effective_provider_alpha,
    final_decision,
    fit_landmark_registration,
    gate_provider_alpha_by_registration,
    select_evaluation_rows,
    select_landmark_region,
    select_smoke_rows,
    select_train_variant,
)


def _quality(row_id, split, failures, *, shape=0.9, gradient=0.7, rmse=0.1):
    return {
        "row_id": row_id,
        "split": split,
        "combined_part_failures": failures,
        "shape_correlation": shape,
        "gradient_correlation": gradient,
        "normalized_rmse": rmse,
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


class SMIRKFaceTrainingSmokeTests(unittest.TestCase):
    def test_similarity_matrix_recovers_known_transform(self):
        source = np.array(
            [
                [10.0, 20.0],
                [40.0, 20.0],
                [10.0, 70.0],
                [55.0, 65.0],
            ],
            dtype=np.float64,
        )
        angle = np.deg2rad(12.0)
        scale = 1.15
        linear = scale * np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        translation = np.array([7.5, -3.0])
        target = source @ linear.T + translation

        matrix = _similarity_matrix(source, target)

        np.testing.assert_allclose(matrix[:, :2], linear, atol=1e-10)
        np.testing.assert_allclose(matrix[:, 2], translation, atol=1e-10)

    def test_xy_registration_preserves_provider_depth(self):
        points = np.array(
            [[1.0, 2.0, -0.4], [3.0, 4.0, 0.8]],
            dtype=np.float32,
        )
        matrix = np.array(
            [[2.0, 0.0, 5.0], [0.0, 2.0, -1.0]],
            dtype=np.float32,
        )

        registered = apply_xy_registration(points, matrix)

        np.testing.assert_allclose(
            registered[:, :2],
            np.array([[7.0, 3.0], [11.0, 7.0]]),
        )
        np.testing.assert_array_equal(registered[:, 2], points[:, 2])

    def test_robust_landmark_registration_rejects_outliers(self):
        indices = np.arange(105, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(
            np.linspace(30.0, 170.0, 15),
            np.linspace(20.0, 180.0, 7),
        )
        source = np.column_stack(
            (
                grid_x.reshape(-1),
                grid_y.reshape(-1),
                np.linspace(-0.5, 0.5, 105),
            )
        ).astype(np.float32)
        angle = np.deg2rad(-8.0)
        scale = 1.08
        linear = scale * np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        translation = np.array([9.0, 6.0])
        target_pixels = source[:, :2] @ linear.T + translation
        target_pixels[:15] += np.array([45.0, -30.0])
        detected = np.zeros((478, 3), dtype=np.float32)
        detected[indices, 0] = target_pixels[:, 0] / 255.0
        detected[indices, 1] = target_pixels[:, 1] / 255.0

        matrix, evidence = fit_landmark_registration(
            source,
            detected,
            indices,
            image_width=256,
            image_height=256,
        )

        self.assertTrue(evidence["checks"]["passed"])
        self.assertGreaterEqual(evidence["inlier_ratio"], 0.80)
        np.testing.assert_allclose(matrix[:, :2], linear, atol=1e-5)
        np.testing.assert_allclose(matrix[:, 2], translation, atol=1e-4)

    def test_failed_registration_forces_provider_bypass(self):
        alpha, policy = gate_provider_alpha_by_registration(
            0.25,
            {"branch": "high", "effective_provider_alpha": 0.25},
            "similarity",
            {"checks": {"passed": False}},
        )

        self.assertEqual(alpha, 0.0)
        self.assertEqual(policy["branch"], "registration-bypass")
        self.assertEqual(policy["branch_before_registration"], "high")

    def test_smoke_rows_lock_split_coverage(self):
        rows = []
        for index, row_id in enumerate(SMOKE_ROW_IDS):
            split = (
                "train"
                if index < 4
                else "validation"
                if index < 6
                else "sealed"
            )
            rows.append({"row_id": row_id, "split": split})

        selected = select_smoke_rows(rows)

        self.assertEqual(
            [row["row_id"] for row in selected],
            list(SMOKE_ROW_IDS),
        )

    def test_evaluation_suite_rejects_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown SMIRK"):
            select_evaluation_rows([], "unknown")

    def test_landmark_region_prefers_overlap_and_minimum_landmarks(self):
        regions = [
            {
                "bbox": [0, 0, 10, 10],
                "landmark_count": 478,
            },
            {
                "bbox": [22, 22, 48, 48],
                "landmark_count": 478,
            },
            {
                "bbox": [20, 20, 50, 50],
                "landmark_count": 5,
            },
        ]

        selected, overlap = select_landmark_region(
            regions,
            [20, 20, 50, 50],
        )

        self.assertEqual(selected["bbox"], [22, 22, 48, 48])
        self.assertAlmostEqual(
            overlap,
            _bbox_iou([22, 22, 48, 48], [20, 20, 50, 50]),
        )

    def test_landmark_detection_uses_guided_upscale_when_direct_misses(self):
        guided = {
            "bbox": [22, 22, 48, 48],
            "landmark_count": 478,
        }
        with (
            patch(
                "backend.benchmark.evaluate_smirk_face_training_smoke."
                "_detect_faces_mediapipe",
                return_value=[],
            ),
            patch(
                "backend.benchmark.evaluate_smirk_face_training_smoke."
                "_upgrade_yunet_regions_with_mediapipe",
                return_value=[guided],
            ) as upgrade,
        ):
            regions, evidence = detect_landmark_regions(
                np.zeros((64, 64, 3), dtype=np.uint8),
                [20, 20, 50, 50],
                np.ones((64, 64), dtype=bool),
            )

        self.assertEqual(regions, [guided])
        self.assertEqual(
            evidence["method"],
            "cached-box-guided-upscaled-mediapipe",
        )
        upgrade.assert_called_once()

    def test_jaw_policy_uses_gentle_blend_for_closed_mouth(self):
        variant = {
            "provider_alpha": 0.25,
            "low_provider_alpha": 0.10,
            "minimum_jaw_open": 0.075,
        }

        low_alpha, low = effective_provider_alpha(
            variant,
            {"jaw_params": [0.05, 0.0, 0.0]},
        )
        high_alpha, high = effective_provider_alpha(
            variant,
            {"jaw_params": [0.12, 0.0, 0.0]},
        )

        self.assertEqual(low_alpha, 0.10)
        self.assertEqual(low["branch"], "low")
        self.assertEqual(high_alpha, 0.25)
        self.assertEqual(high["branch"], "high")

    def test_pose_policy_bypasses_turned_face_before_jaw_gate(self):
        alpha, policy = effective_provider_alpha(
            {
                "provider_alpha": 0.25,
                "low_provider_alpha": 0.10,
                "outside_pose_provider_alpha": 0.0,
                "minimum_jaw_open": 0.075,
                "maximum_absolute_pose_y_radians": 0.45,
            },
            {
                "pose_params": [0.0, 0.61, 0.0],
                "jaw_params": [0.12, 0.0, 0.0],
            },
        )

        self.assertEqual(alpha, 0.0)
        self.assertEqual(policy["branch"], "pose-bypass")

    def test_train_selector_uses_only_eligible_train_variant(self):
        baseline = _summary(
            [
                _quality("a", "train", 3),
                _quality("b", "train", 3),
            ]
        )
        regressing = {
            "variant_id": "regressing",
            "provider_alpha": 0.5,
            "sigma_ratio": 0.02,
            "train": _summary(
                [
                    _quality("a", "train", 1),
                    _quality("b", "train", 4),
                ]
            ),
        }
        eligible = {
            "variant_id": "eligible",
            "provider_alpha": 0.25,
            "sigma_ratio": 0.02,
            "train": _summary(
                [
                    _quality("a", "train", 2),
                    _quality("b", "train", 3),
                ]
            ),
        }

        selection = select_train_variant(
            baseline,
            [regressing, eligible],
        )

        self.assertEqual(selection["selected_variant_id"], "eligible")
        self.assertTrue(selection["selected_is_train_eligible"])

    def test_final_decision_rejects_sealed_regression(self):
        train_baseline = [_quality("a", "train", 3)]
        validation_baseline = [_quality("b", "validation", 3)]
        sealed_baseline = [_quality("c", "sealed", 3)]
        train_candidate = [_quality("a", "train", 2)]
        validation_candidate = [_quality("b", "validation", 3)]
        sealed_candidate = [_quality("c", "sealed", 4)]
        baseline = {
            "train": _summary(train_baseline),
            "validation": _summary(validation_baseline),
            "sealed": _summary(sealed_baseline),
            "all": _summary(
                train_baseline + validation_baseline + sealed_baseline
            ),
        }
        candidate = {
            "train": _summary(train_candidate),
            "validation": _summary(validation_candidate),
            "sealed": _summary(sealed_candidate),
            "all": _summary(
                train_candidate + validation_candidate + sealed_candidate
            ),
        }
        selected_rows = [
            {
                "background_value_exact": True,
                "boundary_value_exact": True,
                "provider": {
                    "landmark_bbox_iou": 0.5,
                    "refinement_crop_coverage": 0.7,
                    "registration": {"checks": {"passed": True}},
                    "raster": {
                        "finite_pixels": 100,
                        "degenerate_faces": 0,
                    },
                },
            }
        ]

        decision = final_decision(
            baseline,
            candidate,
            selected_rows,
            {"selected_is_train_eligible": True},
        )

        self.assertFalse(decision["checks"]["sealed_non_regression"])
        self.assertFalse(
            decision["checks"]["eligible_for_30mm_replay"]
        )


if __name__ == "__main__":
    unittest.main()
