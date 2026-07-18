import unittest

import numpy as np

from backend.benchmark.train_face_depth_head import (
    CachedFace,
    _epoch_rank,
    _near_high_target,
    _padded_box,
    _select_blend_candidate,
    _strictly_improves,
    _target_depth_record,
    _training_loss,
    _validate_training_corpus_summary,
)


class TrainFaceDepthHeadTests(unittest.TestCase):
    @staticmethod
    def _mhr_training_rows():
        rows = []
        for identity_index in range(40):
            split = (
                "train"
                if identity_index < 30
                else "validation"
                if identity_index < 35
                else "sealed"
            )
            for scene_index in range(8):
                rows.append(
                    {
                        "row_id": f"identity_{identity_index:03d}_{scene_index:02d}",
                        "identity_group": f"identity_{identity_index:03d}",
                        "split": split,
                    }
                )
        return rows

    def test_padded_box_matches_production_ratio_and_clamps(self):
        self.assertEqual(_padded_box((20, 30, 60, 80), 100, 100), (6, 12, 74, 98))
        self.assertEqual(_padded_box((0, 0, 20, 20), 30, 30), (0, 0, 27, 27))

    def test_near_high_target_reverses_exact_far_high_depth(self):
        exact = np.tile(np.linspace(0.2, 0.8, 16, dtype=np.float32), (16, 1))
        face = np.ones_like(exact, dtype=bool)
        target = _near_high_target(exact, face)

        self.assertGreater(float(target[:, 0].mean()), float(target[:, -1].mean()))
        self.assertTrue(np.all(np.isfinite(target)))
        self.assertGreater(float(np.ptp(target)), 0.9)

    def test_near_high_target_is_finite_outside_supervised_geometry(self):
        exact = np.full((16, 16), np.nan, dtype=np.float32)
        exact[4:12, 4:12] = np.linspace(
            0.2,
            0.4,
            64,
            dtype=np.float32,
        ).reshape(8, 8)
        face = np.zeros((16, 16), dtype=bool)
        face[4:12, 4:12] = True

        target = _near_high_target(exact, face)

        self.assertTrue(np.all(np.isfinite(target)))
        self.assertTrue(np.all(target[~face] == 0.0))

    def test_near_high_target_rejects_tiny_support(self):
        with self.assertRaisesRegex(ValueError, "at least 64"):
            _near_high_target(
                np.zeros((8, 8), dtype=np.float32),
                np.eye(8, dtype=bool),
            )

    def test_mhr_rows_prefer_floating_camera_z_targets(self):
        camera = {"path": "camera.npy", "sha256": "a" * 64}
        scene = {"path": "scene.npy", "sha256": "b" * 64}
        record, representation = _target_depth_record(
            {"exact_camera_depth": camera, "exact_depth": scene}
        )
        self.assertIs(record, camera)
        self.assertEqual(representation, "floating-normalized-camera-z")
        legacy, legacy_representation = _target_depth_record(
            {"exact_depth": scene}
        )
        self.assertIs(legacy, scene)
        self.assertEqual(legacy_representation, "normalized-scene-depth")

    def test_mhr_training_contract_requires_complete_deterministic_matrix(self):
        summary = {
            "provider": "meta-mhr-v1.0.1-camera-depth",
            "source_revision": "4998cec385b1aaa07abdefba71bfba2f83c7db32",
            "matrix_kind": "training",
            "corpus_complete": True,
            "training_eligible": True,
            "identity_disjoint_splits": True,
            "promotion_eligible": False,
            "deterministic_generation": {"device_requirement_met": True},
            "rows": self._mhr_training_rows(),
        }
        contract = _validate_training_corpus_summary(summary)
        self.assertTrue(contract["camera_aligned"])
        self.assertEqual(
            contract["target_depth"],
            "floating-normalized-camera-z",
        )
        with self.assertRaisesRegex(ValueError, "deterministic_cpu"):
            _validate_training_corpus_summary(
                {
                    **summary,
                    "deterministic_generation": {"device_requirement_met": False},
                }
            )

    def test_mhr_training_contract_rejects_duplicate_and_leaked_identities(self):
        summary = {
            "provider": "meta-mhr-v1.0.1-camera-depth",
            "source_revision": "4998cec385b1aaa07abdefba71bfba2f83c7db32",
            "matrix_kind": "training",
            "corpus_complete": True,
            "training_eligible": True,
            "identity_disjoint_splits": True,
            "promotion_eligible": False,
            "deterministic_generation": {"device_requirement_met": True},
            "rows": self._mhr_training_rows(),
        }
        duplicate = {**summary, "rows": [dict(row) for row in summary["rows"]]}
        duplicate["rows"][1]["row_id"] = duplicate["rows"][0]["row_id"]
        with self.assertRaisesRegex(ValueError, "identity_disjoint"):
            _validate_training_corpus_summary(duplicate)

        missing_id = {**summary, "rows": [dict(row) for row in summary["rows"]]}
        missing_id["rows"][0]["row_id"] = None
        with self.assertRaisesRegex(ValueError, "identity_disjoint"):
            _validate_training_corpus_summary(missing_id)

        leaked = {**summary, "rows": [dict(row) for row in summary["rows"]]}
        leaked["rows"][-1]["identity_group"] = leaked["rows"][0][
            "identity_group"
        ]
        with self.assertRaisesRegex(ValueError, "identity_disjoint"):
            _validate_training_corpus_summary(leaked)

    def test_small_face_objective_selects_epoch_using_small_face_metrics(self):
        def summary(global_failures, small_gradient):
            return {
                "combined_part_failures": global_failures,
                "median_shape_correlation": 0.9,
                "median_gradient_correlation": 0.8,
                "median_normalized_rmse": 0.1,
                "rows": [
                    {
                        "face_height_pixels": 75,
                        "shape_correlation": 0.9,
                        "gradient_correlation": small_gradient,
                        "normalized_rmse": 0.1,
                        "shape_failed_parts": [],
                        "affine_failed_parts": [],
                    }
                ],
            }

        globally_better = summary(8, 0.7)
        small_face_better = summary(9, 0.8)
        self.assertLess(
            _epoch_rank(globally_better, "global-failures"),
            _epoch_rank(small_face_better, "global-failures"),
        )
        self.assertLess(
            _epoch_rank(small_face_better, "small-face-gradient"),
            _epoch_rank(globally_better, "small-face-gradient"),
        )

    def test_strict_selector_requires_part_and_all_global_improvements(self):
        baseline = {
            "combined_part_failures": 10,
            "median_shape_correlation": 0.80,
            "median_gradient_correlation": 0.60,
            "median_normalized_rmse": 0.20,
        }
        passing = {
            "combined_part_failures": 9,
            "median_shape_correlation": 0.81,
            "median_gradient_correlation": 0.61,
            "median_normalized_rmse": 0.19,
        }
        self.assertTrue(_strictly_improves(passing, baseline))
        for field, value in (
            ("combined_part_failures", 10),
            ("median_shape_correlation", 0.79),
            ("median_gradient_correlation", 0.59),
            ("median_normalized_rmse", 0.21),
        ):
            candidate = dict(passing)
            candidate[field] = value
            self.assertFalse(_strictly_improves(candidate, baseline))

    def test_training_loss_uses_explicit_gradient_weight(self):
        import torch

        target = torch.from_numpy(
            np.tile(np.linspace(0.0, 1.0, 8, dtype=np.float32), (8, 1))
        )
        item = CachedFace(
            row={},
            feature=torch.zeros(1),
            baseline=torch.zeros((8, 8)),
            target=target,
            face_mask=torch.ones((8, 8), dtype=torch.bool),
            part_weight=torch.zeros((8, 8)),
            patch_height=1,
            patch_width=1,
            bbox=(0, 0, 8, 8),
        )
        prediction = torch.zeros((8, 8), requires_grad=True)
        low, low_components = _training_loss(
            prediction,
            item,
            gradient_loss_weight=0.0,
        )
        high, high_components = _training_loss(
            prediction,
            item,
            gradient_loss_weight=2.0,
        )
        self.assertGreater(low_components["gradient"], 0.0)
        self.assertEqual(low_components["gradient"], high_components["gradient"])
        self.assertGreater(float(high.detach()), float(low.detach()))

    def test_small_face_selector_maximizes_validated_small_gradient(self):
        candidates = [
            {
                "alpha": 0.0,
                "eligible": False,
                "small_face_eligible": False,
                "combined_part_failures": 10,
                "median_normalized_rmse": 0.2,
                "median_shape_correlation": 0.8,
                "small_face": {
                    "combined_part_failures": 6,
                    "median_gradient_correlation": 0.6,
                    "median_normalized_rmse": 0.2,
                    "median_shape_correlation": 0.8,
                },
            },
            {
                "alpha": 0.5,
                "eligible": True,
                "small_face_eligible": True,
                "combined_part_failures": 9,
                "median_normalized_rmse": 0.19,
                "median_shape_correlation": 0.81,
                "small_face": {
                    "combined_part_failures": 5,
                    "median_gradient_correlation": 0.679,
                    "median_normalized_rmse": 0.19,
                    "median_shape_correlation": 0.81,
                },
            },
            {
                "alpha": 0.75,
                "eligible": True,
                "small_face_eligible": True,
                "combined_part_failures": 8,
                "median_normalized_rmse": 0.18,
                "median_shape_correlation": 0.82,
                "small_face": {
                    "combined_part_failures": 5,
                    "median_gradient_correlation": 0.68,
                    "median_normalized_rmse": 0.18,
                    "median_shape_correlation": 0.82,
                },
            },
            {
                "alpha": 1.0,
                "eligible": True,
                "small_face_eligible": True,
                "combined_part_failures": 7,
                "median_normalized_rmse": 0.17,
                "median_shape_correlation": 0.83,
                "small_face": {
                    "combined_part_failures": 4,
                    "median_gradient_correlation": 0.66,
                    "median_normalized_rmse": 0.17,
                    "median_shape_correlation": 0.83,
                },
            },
        ]
        self.assertEqual(
            _select_blend_candidate(candidates, "global-failures")["alpha"],
            1.0,
        )
        self.assertEqual(
            _select_blend_candidate(candidates, "small-face-gradient")["alpha"],
            0.75,
        )
        self.assertEqual(
            _select_blend_candidate(
                candidates,
                "small-face-gradient-conservative",
            )["alpha"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
