import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from backend import gnm_face_foundation as gnm
from backend.face_depth_refinement import face_masks_from_box, refine_depth_for_faces


class GNMFaceFoundationTests(unittest.TestCase):
    def test_active_detection_kwargs_are_opt_in_only(self):
        original = gnm._DEFAULT_PROVIDER
        try:
            gnm._DEFAULT_PROVIDER = None
            self.assertEqual(gnm.active_gnm_detection_kwargs(), {})

            provider = Mock()
            provider.requires_face_blendshapes = True
            gnm._DEFAULT_PROVIDER = provider
            self.assertEqual(
                gnm.active_gnm_detection_kwargs(),
                {"output_face_blendshapes": True},
            )

            provider.requires_facial_transformation_matrix = True
            self.assertEqual(
                gnm.active_gnm_detection_kwargs(),
                {
                    "output_face_blendshapes": True,
                    "output_facial_transformation_matrixes": True,
                },
            )
        finally:
            gnm._DEFAULT_PROVIDER = original

    def test_mediapipe_mapping_uses_pinned_correspondence_and_pair_averages(self):
        points = np.column_stack(
            (
                np.arange(478, dtype=np.float64),
                np.arange(478, dtype=np.float64) * -2.0,
            )
        )

        mapped = gnm.mediapipe_to_dlib68(points)

        self.assertEqual(mapped.shape, (68, 2))
        np.testing.assert_allclose(mapped[0], points[127])
        np.testing.assert_allclose(mapped[3], np.mean(points[[132, 58]], axis=0))
        np.testing.assert_allclose(mapped[27], np.mean(points[[168, 6]], axis=0))

    def test_mediapipe_mapping_rejects_incomplete_input(self):
        with self.assertRaisesRegex(ValueError, "at least 468"):
            gnm.mediapipe_to_dlib68(np.zeros((467, 2), dtype=np.float32))

    def test_configured_asset_fails_closed_on_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "wrong.npz"
            path.write_bytes(b"not the pinned model")
            with patch.dict(
                "os.environ",
                {"GNM_HEAD_MODEL_PATH": str(path)},
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    gnm.resolve_gnm_model()

    def test_rasterizer_selects_frontmost_triangle(self):
        vertices = np.asarray(
            [
                [1.0, 1.0, 1.0],
                [6.0, 1.0, 1.0],
                [1.0, 6.0, 1.0],
                [1.0, 1.0, 2.0],
                [6.0, 1.0, 2.0],
                [1.0, 6.0, 2.0],
            ],
            dtype=np.float64,
        )
        triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mask = np.ones((8, 8), dtype=np.uint8)

        surface = gnm._rasterize_front_surface(vertices, triangles, mask)

        self.assertEqual(float(surface[2, 2]), 2.0)
        self.assertTrue(np.isnan(surface[7, 7]))

    def test_geometry_coefficients_apply_the_pinned_linear_bases(self):
        template = np.zeros((2, 3), dtype=np.float64)
        identity_basis = np.zeros(
            (gnm.GNM_IDENTITY_DIMENSION, 2, 3),
            dtype=np.float64,
        )
        expression_basis = np.zeros(
            (gnm.GNM_EXPRESSION_DIMENSION, 2, 3),
            dtype=np.float64,
        )
        identity_basis[0, 0, 2] = 2.0
        expression_basis[0, 1, 1] = -3.0
        identity = np.zeros(gnm.GNM_IDENTITY_DIMENSION)
        expression = np.zeros(gnm.GNM_EXPRESSION_DIMENSION)
        identity[0] = 0.5
        expression[0] = 2.0

        vertices = gnm._apply_geometry_coefficients(
            template,
            identity_basis,
            expression_basis,
            identity,
            expression,
        )

        np.testing.assert_array_equal(
            vertices,
            np.asarray([[0.0, 0.0, 1.0], [0.0, -6.0, 0.0]]),
        )

    def test_geometry_coefficients_fail_closed_on_partial_or_nonfinite_input(self):
        template = np.zeros((2, 3), dtype=np.float64)
        identity_basis = np.zeros((gnm.GNM_IDENTITY_DIMENSION, 2, 3))
        expression_basis = np.zeros((gnm.GNM_EXPRESSION_DIMENSION, 2, 3))
        with self.assertRaisesRegex(ValueError, "identity coefficients"):
            gnm._apply_geometry_coefficients(
                template,
                identity_basis,
                expression_basis,
                np.zeros(gnm.GNM_IDENTITY_DIMENSION - 1),
                np.zeros(gnm.GNM_EXPRESSION_DIMENSION),
            )
        identity = np.zeros(gnm.GNM_IDENTITY_DIMENSION)
        identity[0] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            gnm._apply_geometry_coefficients(
                template,
                identity_basis,
                expression_basis,
                identity,
                np.zeros(gnm.GNM_EXPRESSION_DIMENSION),
            )

    def test_surface_selector_requires_pareto_live_depth_alignment(self):
        rows, columns = np.indices((32, 32), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:28, 4:28] = 255
        depth = 0.2 + rows * 0.01 + columns * 0.02
        fallback = 0.5 + rows * 0.02 + columns * 0.04
        candidate = fallback.copy()
        candidate += 0.002 * np.sin(rows * 0.4)
        fallback[mask == 0] = np.nan
        candidate[mask == 0] = np.nan
        candidates = gnm.GNMSurfaceCandidateSet(
            candidate_surface=candidate,
            candidate_stats={"method": "candidate"},
            fallback_surface=fallback,
            fallback_stats={"method": "fallback"},
        )

        selected, stats, selection = gnm.select_gnm_surface_candidate(
            depth,
            candidates,
            mask,
        )

        np.testing.assert_array_equal(selected, fallback)
        self.assertEqual(stats["method"], "fallback")
        self.assertEqual(selection["selected"], "mean_fallback")

    def test_surface_selector_accepts_strict_pareto_improvement(self):
        rows, columns = np.indices((32, 32), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:28, 4:28] = 255
        depth = 0.2 + rows * 0.01 + columns * 0.02
        fallback = depth + 0.04 * np.sin(rows * 0.45)
        candidate = depth + 0.01 * np.sin(rows * 0.45)
        fallback[mask == 0] = np.nan
        candidate[mask == 0] = np.nan
        candidates = gnm.GNMSurfaceCandidateSet(
            candidate_surface=candidate,
            candidate_stats={"method": "candidate"},
            fallback_surface=fallback,
            fallback_stats={"method": "fallback"},
        )

        selected, stats, selection = gnm.select_gnm_surface_candidate(
            depth,
            candidates,
            mask,
        )

        np.testing.assert_array_equal(selected, candidate)
        self.assertEqual(stats["method"], "candidate")
        self.assertEqual(selection["selected"], "conditioned")

    def test_bounded_fusion_preserves_boundary_and_caps_correction(self):
        rows, columns = np.indices((40, 40), dtype=np.float32)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[8:32, 8:32] = 255
        depth = 0.2 + columns * 0.01 + rows * 0.004
        depth += 0.08 * np.exp(
            -((rows - 20.0) ** 2 + (columns - 20.0) ** 2) / 18.0
        )
        surface = 0.5 + columns * 0.02 + rows * 0.008
        surface[mask == 0] = np.nan

        refined, weight, stats = gnm.fuse_gnm_face_foundation(
            depth,
            surface,
            mask,
        )

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["reliability_tier"], "high")
        self.assertEqual(stats["correction_strength"], 1.0)
        self.assertGreater(stats["max_abs_correction"], 0.0)
        self.assertLessEqual(
            stats["max_abs_correction"],
            stats["correction_limit"] + 1e-7,
        )
        self.assertLessEqual(stats["boundary_max_abs_correction"], 1e-7)
        np.testing.assert_array_equal(refined[mask == 0], depth[mask == 0])
        self.assertEqual(float(np.max(weight[8, 8:32])), 0.0)

    def test_bounded_fusion_preserves_detected_expression_parts(self):
        rows, columns = np.indices((40, 40), dtype=np.float32)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[8:32, 8:32] = 255
        preserve = np.zeros_like(mask)
        preserve[18:24, 15:25] = 255
        depth = 0.2 + columns * 0.01 + rows * 0.004
        depth += 0.08 * np.exp(
            -((rows - 20.0) ** 2 + (columns - 20.0) ** 2) / 18.0
        )
        surface = 0.5 + columns * 0.02 + rows * 0.008
        surface[mask == 0] = np.nan

        refined, _weight, stats = gnm.fuse_gnm_face_foundation(
            depth,
            surface,
            mask,
            preserve_detail_mask=preserve,
        )

        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["detail_preservation"]["enabled"])
        self.assertEqual(
            stats["detail_preservation"]["maximum_core_abs_correction"],
            0.0,
        )
        np.testing.assert_array_equal(refined[preserve > 0], depth[preserve > 0])
        self.assertGreater(float(np.max(np.abs(refined - depth))), 0.0)

    def test_bounded_fusion_limits_correction_to_central_face_parts(self):
        rows, columns = np.indices((40, 40), dtype=np.float32)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[8:32, 8:32] = 255
        correction_region = np.zeros_like(mask)
        correction_region[17:23, 17:23] = 255
        depth = 0.2 + columns * 0.01 + rows * 0.004
        depth += 0.08 * np.exp(
            -((rows - 20.0) ** 2 + (columns - 20.0) ** 2) / 18.0
        )
        surface = 0.5 + columns * 0.02 + rows * 0.008
        surface[mask == 0] = np.nan

        refined, weight, stats = gnm.fuse_gnm_face_foundation(
            depth,
            surface,
            mask,
            correction_region_mask=correction_region,
        )

        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["correction_region"]["enabled"])
        self.assertEqual(stats["correction_region"]["core_pixels"], 36)
        self.assertEqual(
            stats["correction_region"]["dilation_radius_pixels"],
            gnm.GNM_CENTRAL_CORRECTION_DILATION_PIXELS,
        )
        self.assertGreater(float(np.max(weight)), 0.0)
        self.assertEqual(float(np.max(weight[:10, :])), 0.0)
        self.assertEqual(
            float(np.max(np.abs(refined[:10, :] - depth[:10, :]))),
            0.0,
        )

    def test_bounded_fusion_fails_closed_on_empty_correction_region(self):
        rows, columns = np.indices((40, 40), dtype=np.float32)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[8:32, 8:32] = 255
        depth = 0.2 + columns * 0.01 + rows * 0.004
        surface = 0.5 + columns * 0.02 + rows * 0.008
        surface[mask == 0] = np.nan

        refined, weight, stats = gnm.fuse_gnm_face_foundation(
            depth,
            surface,
            mask,
            correction_region_mask=np.zeros_like(mask),
        )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "empty_correction_region")
        self.assertFalse(stats["correction_region"]["enabled"])
        np.testing.assert_array_equal(refined, depth)
        self.assertEqual(float(np.max(weight)), 0.0)

    def test_bounded_fusion_rejects_misaligned_correction_region(self):
        rows, columns = np.indices((40, 40), dtype=np.float32)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[8:32, 8:32] = 255
        depth = 0.2 + columns * 0.01 + rows * 0.004
        surface = 0.5 + columns * 0.02 + rows * 0.008
        surface[mask == 0] = np.nan

        with self.assertRaisesRegex(ValueError, "must match"):
            gnm.fuse_gnm_face_foundation(
                depth,
                surface,
                mask,
                correction_region_mask=np.ones((20, 20), dtype=np.uint8),
            )

    def test_bounded_fusion_skips_faces_above_small_face_gate(self):
        depth = np.arange(80 * 80, dtype=np.float32).reshape(80, 80)
        mask = np.zeros_like(depth, dtype=np.uint8)
        mask[8:72, 8:72] = 255

        refined, weight, stats = gnm.fuse_gnm_face_foundation(
            depth,
            depth.copy(),
            mask,
        )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "face_support_above_small_face_gate")
        np.testing.assert_array_equal(refined, depth)
        self.assertEqual(float(np.max(weight)), 0.0)

    def test_fusion_uses_guarded_strength_for_moderate_alignment(self):
        rows, columns = np.indices((40, 40), dtype=np.float32)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[8:32, 8:32] = 255
        surface = 0.5 + columns * 0.02 + rows * 0.008
        surface[mask == 0] = np.nan
        depth = 0.2 + columns * 0.01 + rows * 0.004
        depth += 0.15 * np.sin(columns * 0.6) * np.cos(rows * 0.4)

        _refined, _weight, stats = gnm.fuse_gnm_face_foundation(
            depth,
            surface,
            mask,
        )

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["reliability_tier"], "guarded")
        self.assertEqual(
            stats["correction_strength"],
            gnm.GNM_GUARDED_CORRECTION_STRENGTH,
        )
        self.assertLessEqual(
            stats["max_abs_correction"],
            stats["effective_correction_limit"] + 1e-7,
        )

    def test_fusion_skips_low_correlation_alignment(self):
        rows, columns = np.indices((40, 40), dtype=np.float32)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[8:32, 8:32] = 255
        surface = 0.5 + columns * 0.02 + rows * 0.008
        surface[mask == 0] = np.nan
        depth = 0.2 + columns * 0.01 + rows * 0.004
        depth += 0.3 * np.sin(columns * 0.6) * np.cos(rows * 0.4)

        refined, weight, stats = gnm.fuse_gnm_face_foundation(
            depth,
            surface,
            mask,
        )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "alignment_reliability_gate")
        self.assertEqual(stats["reliability_tier"], "skip")
        np.testing.assert_array_equal(refined, depth)
        self.assertEqual(float(np.max(weight)), 0.0)

    def test_face_pipeline_applies_provider_and_records_auditable_metadata(self):
        class FakeProvider:
            calls = 0

            def fit_and_render(self, landmarks, face_mask):
                self.calls += 1
                self.assert_landmarks = np.asarray(landmarks)
                rows, columns = np.indices(face_mask.shape, dtype=np.float32)
                surface = 0.4 + columns * 0.02 + rows * 0.006
                surface += 0.12 * np.exp(
                    -((rows - rows.mean()) ** 2 + (columns - columns.mean()) ** 2)
                    / 24.0
                )
                surface[face_mask == 0] = np.nan
                return surface, {
                    "enabled": True,
                    "method": "fake-gnm-provider",
                    "coverage_ratio": 1.0,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "portrait.png"
            depth_path = root / "depth.npy"
            image = np.full((96, 96, 3), 180, dtype=np.uint8)
            Image.fromarray(image).save(image_path)
            rows, columns = np.indices((48, 48), dtype=np.float32)
            depth = 0.2 + columns * 0.008 + rows * 0.003
            np.save(depth_path, depth)
            face_mask, feature_mask = face_masks_from_box(
                image.shape,
                (28, 22, 68, 72),
            )
            angles = np.linspace(0.0, 8.0 * np.pi, 478, endpoint=False)
            landmarks = np.column_stack(
                (
                    0.5 + 0.16 * np.cos(angles),
                    0.5 + 0.22 * np.sin(angles),
                    np.zeros_like(angles),
                )
            ).astype(np.float32)

            def detector(_image):
                return [
                    {
                        "bbox": [28, 22, 68, 72],
                        "face_mask": face_mask,
                        "feature_mask": feature_mask,
                        "part_masks": {},
                        "detector": "mediapipe-face-landmarker",
                        "landmark_count": 478,
                        "landmarks_xyz": landmarks,
                    }
                ]

            def infer_depth(crop_path, output_dir):
                crop = Image.open(crop_path)
                crop_rows, crop_columns = np.indices(
                    (crop.height, crop.width),
                    dtype=np.float32,
                )
                values = 0.5 + crop_columns * 0.01 + crop_rows * 0.003
                output_path = Path(output_dir) / "output_depth_data.npy"
                np.save(output_path, values)
                return output_path

            provider = FakeProvider()
            with patch.object(
                gnm,
                "get_gnm_mean_face_foundation",
                return_value=provider,
            ):
                _output_path, metadata = refine_depth_for_faces(
                    image_path,
                    depth_path,
                    root,
                    infer_depth=infer_depth,
                    detector=detector,
                    mode="on",
                )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.assert_landmarks.shape, (478, 2))
        self.assertEqual(metadata["parametric_foundation_faces"], 1)
        foundation = metadata["faces"][0]["parametric_face_foundation"]
        self.assertTrue(foundation["enabled"])
        self.assertEqual(
            foundation["method"],
            "bounded-camera-aligned-gnm-mean-face",
        )
        self.assertLessEqual(foundation["boundary_max_abs_correction"], 1e-7)
        self.assertTrue(foundation["detail_preservation"]["enabled"])
        self.assertTrue(foundation["correction_region"]["enabled"])
        self.assertGreater(foundation["correction_region"]["core_pixels"], 0)
        self.assertGreater(
            foundation["detail_preservation"]["core_pixels"],
            0,
        )
        self.assertEqual(
            foundation["detail_preservation"][
                "maximum_core_abs_correction"
            ],
            0.0,
        )

    def test_conditioned_provider_without_transform_capability_keeps_interface(self):
        class LegacyConditionedProvider:
            requires_facial_transformation_matrix = False

            def __init__(self):
                self.calls = 0

            def fit_and_render_conditioned(
                self,
                landmarks,
                face_mask,
                *,
                media_pipe_landmarks_xyz,
                face_image_rgb,
                media_pipe_blendshape_names,
                media_pipe_blendshape_scores,
            ):
                self.calls += 1
                self.landmarks_xyz = np.asarray(media_pipe_landmarks_xyz)
                rows, columns = np.indices(face_mask.shape, dtype=np.float32)
                surface = 0.4 + columns * 0.02 + rows * 0.006
                surface[face_mask == 0] = np.nan
                return surface, {"enabled": True, "method": "legacy-conditioned"}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "portrait.png"
            depth_path = root / "depth.npy"
            Image.fromarray(np.full((96, 96, 3), 180, dtype=np.uint8)).save(
                image_path
            )
            rows, columns = np.indices((48, 48), dtype=np.float32)
            np.save(depth_path, 0.2 + columns * 0.008 + rows * 0.003)
            face_mask, feature_mask = face_masks_from_box(
                (96, 96, 3),
                (28, 22, 68, 72),
            )
            angles = np.linspace(0.0, 8.0 * np.pi, 478, endpoint=False)
            landmarks = np.column_stack(
                (
                    0.5 + 0.16 * np.cos(angles),
                    0.5 + 0.22 * np.sin(angles),
                    np.zeros_like(angles),
                )
            ).astype(np.float32)

            def detector(_image):
                return [
                    {
                        "bbox": [28, 22, 68, 72],
                        "face_mask": face_mask,
                        "feature_mask": feature_mask,
                        "part_masks": {},
                        "detector": "mediapipe-face-landmarker",
                        "landmark_count": 478,
                        "landmarks_xyz": landmarks,
                    }
                ]

            def infer_depth(crop_path, output_dir):
                crop = Image.open(crop_path)
                crop_rows, crop_columns = np.indices(
                    (crop.height, crop.width),
                    dtype=np.float32,
                )
                output_path = Path(output_dir) / "output_depth_data.npy"
                np.save(
                    output_path,
                    0.5 + crop_columns * 0.01 + crop_rows * 0.003,
                )
                return output_path

            provider = LegacyConditionedProvider()
            with patch.object(
                gnm,
                "get_gnm_mean_face_foundation",
                return_value=provider,
            ):
                _output, metadata = refine_depth_for_faces(
                    image_path,
                    depth_path,
                    root,
                    infer_depth=infer_depth,
                    detector=detector,
                    mode="on",
                )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.landmarks_xyz.shape, (478, 3))
        self.assertNotEqual(
            metadata["faces"][0]["parametric_face_foundation"].get(
                "reason"
            ),
            "provider_error",
        )


if __name__ == "__main__":
    unittest.main()
