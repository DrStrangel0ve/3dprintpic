import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.benchmark.evaluate_sheap_small_face_exact_gate import (
    EXPECTED_INPUT_HASH_KEYS,
    EXPECTED_PART_MASK_SHA256,
    _compare_parts,
    _expected_input_hashes,
    _measurement_source_manifest,
    _registration_checks,
    _select_variant_and_decision,
    _verify_input_hashes,
)
from backend.benchmark.sheap_depth_provider import (
    SHEAP_REQUIRED_SOURCE_FILES,
    SHEAP_SOURCE_REVISION,
    SHEAP_VERTICAL_FOV_DEGREES,
    apply_sheap_image_similarity,
    fit_sheap_landmark_similarity,
    project_sheap_vertices,
    rasterize_sheap_camera_depth,
    sheap_crop_transform,
    sheap_head_crop_bbox,
    sheap_head_crop_bbox_from_landmarks,
    sheap_preflight,
)


class SHeaPDepthProviderTests(unittest.TestCase):
    @staticmethod
    def _part_quality() -> dict:
        names = tuple(sorted(EXPECTED_PART_MASK_SHA256))
        return {
            "parts": {
                name: {
                    "shape_correlation": 0.80,
                    "face_normalized_shape_rmse": 0.20,
                    "minimum_raw_gradient_correlation": 0.50,
                    "shape_passed": False,
                }
                for name in names
            },
            "affine_parts": {
                name: {
                    "rmse_mm": 1.0,
                    "p95_absolute_error_mm": 2.0,
                    "bias_mm": 0.5,
                    "span_retention": 0.8,
                    "affine_passed": False,
                }
                for name in names
            },
        }

    def test_preflight_rejects_unknown_model_variant(self):
        with self.assertRaisesRegex(ValueError, "Unsupported SHeaP model type"):
            sheap_preflight(".", "a", "b", "c", "d", model_type="unknown")

    def test_preflight_rejects_every_unpinned_model_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider_root = root / "provider"
            for relative in SHEAP_REQUIRED_SOURCE_FILES:
                path = provider_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            checkpoint = root / "model_expressive.pt"
            flame_tensor = root / "generic_model.pt"
            flame_source = root / "generic_model.pkl"
            eyelids = root / "eyelids.pt"
            for path in (checkpoint, flame_tensor, flame_source, eyelids):
                path.write_bytes(b"not-the-pinned-asset")
            with patch(
                "backend.benchmark.sheap_depth_provider._git_output",
                side_effect=(SHEAP_SOURCE_REVISION, ""),
            ):
                evidence = sheap_preflight(
                    provider_root,
                    checkpoint,
                    flame_tensor,
                    flame_source,
                    eyelids,
                )

        self.assertTrue(evidence["checks"]["source_revision_pinned"])
        self.assertTrue(evidence["checks"]["source_clean"])
        self.assertTrue(evidence["checks"]["source_files_complete"])
        self.assertFalse(evidence["checks"]["checkpoint_size_pinned"])
        self.assertFalse(evidence["checks"]["checkpoint_hash_pinned"])
        self.assertFalse(evidence["checks"]["flame_source_hash_pinned"])
        self.assertFalse(evidence["checks"]["flame_tensor_hash_pinned"])
        self.assertFalse(evidence["checks"]["eyelids_hash_pinned"])
        self.assertFalse(evidence["runnable"])
        self.assertTrue(evidence["license"]["research_only"])
        self.assertFalse(evidence["license"]["production_eligible"])

    def test_crop_preserves_rectangle_and_half_pixel_resize_mapping(self):
        crop = sheap_crop_transform(
            (40.0, 30.0, 80.0, 90.0),
            image_width=200,
            image_height=120,
        )

        self.assertEqual(crop["integer_crop_box_xyxy"], [40, 30, 80, 90])
        self.assertEqual(crop["crop_width_pixels"], 40)
        self.assertEqual(crop["crop_height_pixels"], 60)
        inverse = crop["network_to_source_matrix"]
        network = np.array(
            [[0.0, 0.0, 1.0], [223.0, 223.0, 1.0]],
            dtype=np.float64,
        )
        source = network @ inverse.T
        source = source[:, :2] / source[:, 2:3]
        np.testing.assert_allclose(
            source,
            np.array(
                [
                    [39.589285714285715, 29.633928571428573],
                    [79.41071428571429, 89.36607142857143],
                ]
            ),
            atol=1e-10,
        )

    def test_crop_rejects_bbox_outside_image(self):
        with self.assertRaisesRegex(ValueError, "does not intersect"):
            sheap_crop_transform(
                (220.0, 130.0, 260.0, 170.0),
                image_width=200,
                image_height=120,
            )

    def test_head_crop_matches_official_margin_and_upward_shift(self):
        bbox = sheap_head_crop_bbox(
            (162.0, 102.0, 193.0, 139.0),
            image_width=256,
            image_height=256,
        )

        np.testing.assert_allclose(
            bbox,
            np.array([142, 76, 212, 146]),
        )

    def test_landmark_crop_uses_extrema_and_official_live_parameters(self):
        bbox = sheap_head_crop_bbox_from_landmarks(
            np.array(
                [[162.0, 102.0], [193.0, 102.0], [193.0, 139.0]],
                dtype=np.float32,
            ),
            image_width=256,
            image_height=256,
        )

        self.assertEqual(bbox, [142, 76, 212, 146])

    def test_landmark_crop_matches_upstream_landscape_aspect_and_clamp(self):
        bbox = sheap_head_crop_bbox_from_landmarks(
            np.array(
                [[150.0, 50.0], [250.0, 50.0], [250.0, 150.0]],
                dtype=np.float32,
            ),
            image_width=400,
            image_height=200,
        )

        self.assertEqual(bbox, [105, 0, 295, 170])

    def test_projection_uses_official_perspective_and_camera_depth(self):
        crop = sheap_crop_transform(
            (40.0, 30.0, 80.0, 90.0),
            image_width=200,
            image_height=120,
        )
        tangent = math.tan(math.radians(SHEAP_VERTICAL_FOV_DEGREES) * 0.5)
        projected = project_sheap_vertices(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [tangent, tangent, 0.0],
                    [0.0, 0.0, 0.25],
                ],
                dtype=np.float32,
            ),
            crop,
        )

        np.testing.assert_allclose(
            projected[0],
            np.array([59.5, 59.5, 1.0]),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            projected[1],
            np.array([79.5, 29.5, 1.0]),
            atol=1e-5,
        )
        self.assertAlmostEqual(float(projected[2, 2]), 0.75)
        self.assertLess(float(projected[2, 2]), float(projected[0, 2]))

    def test_camera_depth_raster_uses_reciprocal_interpolation(self):
        vertices = np.array(
            [[1.0, 1.0, 1.0], [5.0, 1.0, 2.0], [1.0, 5.0, 4.0]],
            dtype=np.float32,
        )
        depth, stats = rasterize_sheap_camera_depth(
            vertices,
            np.array([[0, 1, 2]], dtype=np.int32),
            height=7,
            width=7,
        )

        self.assertAlmostEqual(float(depth[2, 2]), 1.0 / 0.6875, places=6)
        self.assertEqual(
            stats["interpolation"],
            "perspective-correct-reciprocal-depth",
        )

    def test_projection_fails_when_mesh_crosses_camera_plane(self):
        crop = sheap_crop_transform(
            (40.0, 30.0, 80.0, 90.0),
            image_width=200,
            image_height=120,
        )
        with self.assertRaisesRegex(ValueError, "camera plane"):
            project_sheap_vertices(
                np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                crop,
            )

    def test_similarity_recovers_registration_and_preserves_depth(self):
        source = np.array(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]],
            dtype=np.float32,
        )
        angle = math.radians(12.0)
        expected = np.array(
            [
                [1.2 * math.cos(angle), -1.2 * math.sin(angle), 4.0],
                [1.2 * math.sin(angle), 1.2 * math.cos(angle), -3.0],
            ],
            dtype=np.float32,
        )
        target = np.column_stack(
            (source, np.ones(len(source), dtype=np.float32))
        ) @ expected.T

        matrix, stats = fit_sheap_landmark_similarity(source, target)
        vertices = np.column_stack(
            (source, np.arange(1, len(source) + 1, dtype=np.float32))
        )
        transformed = apply_sheap_image_similarity(vertices, matrix)

        np.testing.assert_allclose(matrix, expected, atol=1e-6)
        np.testing.assert_allclose(transformed[:, :2], target, atol=1e-6)
        np.testing.assert_array_equal(transformed[:, 2], vertices[:, 2])
        self.assertAlmostEqual(stats["scale"], 1.2, places=6)
        self.assertAlmostEqual(stats["rotation_degrees"], 12.0, places=5)
        self.assertLess(stats["residual_maximum_pixels"], 1e-6)

    def test_exact_gate_pins_every_measurement_input(self):
        expected = _expected_input_hashes()

        self.assertEqual(set(expected), set(EXPECTED_INPUT_HASH_KEYS))
        self.assertEqual(
            {
                name.removeprefix("part_mask:")
                for name in expected
                if name.startswith("part_mask:")
            },
            set(EXPECTED_PART_MASK_SHA256),
        )
        for required in (
            "metadata",
            "local_face_depth",
            "face_mask",
            "mediapipe_embedding",
            "face_landmarker_model",
        ):
            self.assertIn(required, expected)
        with self.assertRaisesRegex(ValueError, "path schema changed"):
            _verify_input_hashes(
                {
                    name: Path(name)
                    for name in expected
                    if name != "metadata"
                }
            )

    def test_measurement_manifest_covers_transitive_local_code(self):
        manifest = _measurement_source_manifest()
        file_names = set(manifest["files"])
        function_names = set(manifest["functions"])

        for expected_file in (
            "backend/benchmark/evaluate_sheap_small_face_exact_gate.py",
            "backend/benchmark/evaluate_vggheads_small_face_exact_gate.py",
            "backend/benchmark/face_part_metrics.py",
            "backend/benchmark/run_cc0_live_face_variation_matrix.py",
            "backend/benchmark/run_makehuman_face_depth_smoke.py",
            "backend/benchmark/sheap_depth_provider.py",
            "backend/benchmark/vggheads_depth_provider.py",
        ):
            self.assertIn(expected_file, file_names)
        for expected_function in (
            "backend.face_depth_refinement.fuse_face_surface_residual",
            "backend.face_depth_refinement._fit_face_depth",
            "backend.face_depth_refinement._resize_float",
            "backend.face_depth_refinement._resize_mask",
            "backend.face_depth_refinement._robust_span",
            "backend.pic_to_3d._resize_nan_aware",
        ):
            self.assertIn(expected_function, function_names)
        for sha256 in (
            *manifest["files"].values(),
            *manifest["functions"].values(),
        ):
            self.assertRegex(sha256, r"^[0-9a-f]{64}$")

    def test_registration_acceptance_fails_closed_on_each_guard(self):
        fit = {
            "correspondences": 105,
            "scale": 1.0,
            "rotation_degrees": 0.0,
            "residual_median_pixels": 0.5,
            "residual_p95_pixels": 1.0,
        }
        self.assertTrue(all(_registration_checks(fit, 0.9).values()))

        invalid = copy.deepcopy(fit)
        invalid["scale"] = 1.11
        checks = _registration_checks(invalid, 0.9)
        self.assertFalse(checks["scale_bounded"])
        self.assertFalse(all(checks.values()))
        checks = _registration_checks(fit, 0.79)
        self.assertFalse(checks["detection_matches_baseline"])
        self.assertFalse(all(checks.values()))

    def test_part_comparison_requires_all_42_metrics(self):
        baseline = self._part_quality()
        candidate = copy.deepcopy(baseline)
        for part in candidate["parts"].values():
            part["shape_correlation"] += 0.01
            part["face_normalized_shape_rmse"] -= 0.01
            part["minimum_raw_gradient_correlation"] += 0.01
        for part in candidate["affine_parts"].values():
            part["rmse_mm"] -= 0.1
            part["p95_absolute_error_mm"] -= 0.1
            part["bias_mm"] -= 0.1
            part["span_retention"] += 0.1

        comparison = _compare_parts(baseline, candidate)
        self.assertEqual(comparison["passed_checks"], 42)
        self.assertEqual(comparison["total_checks"], 42)
        self.assertTrue(comparison["all_part_metrics_non_regressing"])
        self.assertEqual(
            comparison["parts_with_strict_shape_gradient_affine_improvement"],
            6,
        )

        candidate["parts"]["right_eye"]["shape_correlation"] = 0.79
        comparison = _compare_parts(baseline, candidate)
        self.assertEqual(comparison["passed_checks"], 41)
        self.assertFalse(comparison["all_part_metrics_non_regressing"])

    def test_selector_cannot_promote_aggregate_gain_past_failed_check(self):
        variant = {
            "variant_id": "aggregate-only",
            "eligible": False,
            "comparison": {"passed_checks": 41, "total_checks": 42},
            "quality": {
                "combined_part_failures": 0,
                "shape_correlation": 0.99,
                "gradient_correlation": 0.99,
                "normalized_rmse": 0.01,
            },
            "checks": {
                "all_six_part_metrics_non_regressing": False,
                "overall_shape_non_regressing": True,
            },
        }

        ordered, decision = _select_variant_and_decision([variant])

        self.assertEqual(ordered[0]["variant_id"], "aggregate-only")
        self.assertEqual(decision["status"], "hold")
        self.assertFalse(decision["eligible_for_30mm_stl_replay"])
        self.assertEqual(
            decision["failed_checks"],
            ["all_six_part_metrics_non_regressing"],
        )
        self.assertFalse(decision["production_changed"])


if __name__ == "__main__":
    unittest.main()
