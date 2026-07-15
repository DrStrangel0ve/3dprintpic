import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from scipy.ndimage import gaussian_filter

import backend.main as main_module
import backend.face_depth_refinement as face_module
from backend.face_depth_refinement import (
    EYEWEAR_LANDMARK_INDICES,
    FACE_PART_NAMES,
    _detect_eyewear_occlusion_weight,
    detect_face_regions_in_roi,
    _effective_min_face_pixels,
    _reconstruct_eyewear_occlusion,
    _resolve_verified_model,
    _yunet_keypoints_are_face_like,
    detect_face_regions,
    face_blend_weight,
    face_masks_from_box,
    fuse_face_depth,
    fuse_face_landmark_shape_prior,
    refine_depth_for_faces,
)


def gaussian_peak(shape, center, sigma, amplitude):
    yy, xx = np.indices(shape, dtype=np.float32)
    cy, cx = center
    return amplitude * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))


class FaceDepthRefinementTest(unittest.TestCase):
    def test_yunet_keypoint_gate_rejects_non_face_geometry(self):
        box = (20, 10, 120, 150)
        face_like = np.asarray(
            [[45, 55], [95, 54], [70, 82], [50, 115], [91, 114]],
            dtype=np.float32,
        )
        collapsed = face_like.copy()
        collapsed[1] = collapsed[0]
        inverted = face_like.copy()
        inverted[3:, 1] = 45

        self.assertTrue(_yunet_keypoints_are_face_like(face_like, box))
        self.assertFalse(_yunet_keypoints_are_face_like(collapsed, box))
        self.assertFalse(_yunet_keypoints_are_face_like(inverted, box))

    def test_verified_model_download_is_atomic_and_size_bounded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            with (
                patch.object(face_module.Path, "home", return_value=home),
                patch.object(
                    face_module.urllib.request,
                    "urlopen",
                    return_value=io.BytesIO(b"oversized"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "exceeded 4 bytes"):
                    _resolve_verified_model(
                        environment_name="UNSET_TEST_FACE_MODEL",
                        cache_name="test-model.bin",
                        url="https://example.invalid/test-model.bin",
                        expected_sha256="0" * 64,
                        maximum_bytes=4,
                    )

            cache_dir = home / ".cache" / "3dprintpic"
            self.assertFalse((cache_dir / "test-model.bin").exists())
            self.assertEqual(list(cache_dir.glob("*.part")), [])

    def test_verified_model_reuses_checksum_valid_cache(self):
        payload = b"small verified model"
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            with (
                patch.object(face_module.Path, "home", return_value=home),
                patch.object(
                    face_module.urllib.request,
                    "urlopen",
                    return_value=io.BytesIO(payload),
                ) as download,
            ):
                first = _resolve_verified_model(
                    environment_name="UNSET_TEST_FACE_MODEL",
                    cache_name="test-model.bin",
                    url="https://example.invalid/test-model.bin",
                    expected_sha256=expected_sha256,
                    maximum_bytes=1024,
                )
                second = _resolve_verified_model(
                    environment_name="UNSET_TEST_FACE_MODEL",
                    cache_name="test-model.bin",
                    url="https://example.invalid/test-model.bin",
                    expected_sha256=expected_sha256,
                    maximum_bytes=1024,
                )

            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), payload)
            self.assertEqual(download.call_count, 1)

    def test_verified_model_rejects_unverified_configured_override_without_path_leak(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            configured = Path(temp_dir) / "private-server-path" / "model.onnx"
            configured.parent.mkdir()
            configured.write_bytes(b"unverified")
            with patch.dict(
                face_module.os.environ,
                {"TEST_FACE_MODEL_PATH": str(configured)},
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Configured face model checksum mismatch: TEST_FACE_MODEL_PATH",
                ) as raised:
                    _resolve_verified_model(
                        environment_name="TEST_FACE_MODEL_PATH",
                        cache_name="test-model.bin",
                        url="https://example.invalid/test-model.bin",
                        expected_sha256="0" * 64,
                        maximum_bytes=1024,
                    )

            self.assertNotIn(str(configured), str(raised.exception))

    def test_face_minimum_adapts_to_standard_256_pixel_workflow(self):
        self.assertEqual(_effective_min_face_pixels((256, 256, 3), 96), 64)
        self.assertEqual(_effective_min_face_pixels((512, 512, 3), 96), 96)
        self.assertEqual(_effective_min_face_pixels((256, 256, 3), 32), 32)

    def test_detect_face_regions_passes_adaptive_minimum_to_mediapipe(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        expected_region = {"bbox": [90, 60, 165, 161]}
        with (
            patch.object(
                face_module,
                "_detect_faces_mediapipe",
                return_value=[expected_region],
            ) as mediapipe_detector,
            patch.object(face_module, "_detect_faces_yunet") as yunet_detector,
        ):
            regions, errors = detect_face_regions(image)

        self.assertEqual(regions, [expected_region])
        self.assertEqual(errors, [])
        mediapipe_detector.assert_called_once_with(image, 3, 64)
        yunet_detector.assert_not_called()

    def test_detect_face_regions_uses_yunet_before_haar(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        expected_region = {"bbox": [90, 60, 165, 161], "detector": "opencv-yunet-2023mar"}
        with (
            patch.object(face_module, "_detect_faces_mediapipe", return_value=[]),
            patch.object(face_module, "_detect_faces_yunet", return_value=[expected_region]),
            patch.object(face_module, "_detect_faces_opencv") as haar_detector,
        ):
            regions, errors = detect_face_regions(image)

        self.assertEqual(regions, [expected_region])
        self.assertEqual(errors, [])
        haar_detector.assert_not_called()

    def test_yunet_guide_recovers_mediapipe_landmarks_from_upscaled_crop(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        guide_face, guide_features = face_masks_from_box(
            image.shape, (72, 52, 184, 210)
        )
        guide = {
            "bbox": [72, 52, 184, 210],
            "face_mask": guide_face,
            "feature_mask": guide_features,
            "detector": "opencv-yunet-2023mar",
            "confidence": 0.92,
        }

        def landmarker(crop, _max_faces, minimum):
            self.assertEqual(max(crop.shape[:2]), 384)
            self.assertEqual(minimum, face_module.MIN_FACE_PIXELS_FLOOR)
            height, width = crop.shape[:2]
            face_mask, feature_mask = face_masks_from_box(
                crop.shape, (width * 0.22, height * 0.16, width * 0.78, height * 0.86)
            )
            landmarks = np.zeros((478, 3), dtype=np.float32)
            landmarks[:, 0] = 0.50
            landmarks[:, 1] = 0.51
            return [
                {
                    "bbox": [
                        int(width * 0.22),
                        int(height * 0.16),
                        int(width * 0.78),
                        int(height * 0.86),
                    ],
                    "face_mask": face_mask,
                    "feature_mask": feature_mask,
                    "part_masks": {},
                    "detector": "mediapipe-face-landmarker",
                    "landmark_count": 478,
                    "landmarks_xyz": landmarks,
                }
            ]

        with patch.object(
            face_module, "_detect_faces_mediapipe", side_effect=landmarker
        ):
            upgraded = face_module._upgrade_yunet_regions_with_mediapipe(
                image, [guide], max_faces=3
            )

        self.assertEqual(len(upgraded), 1)
        self.assertEqual(
            upgraded[0]["detector"],
            "yunet-guided:mediapipe-face-landmarker",
        )
        self.assertEqual(upgraded[0]["detection_scope"], "yunet-face-upscaled")
        self.assertEqual(upgraded[0]["landmark_count"], 478)
        self.assertGreaterEqual(upgraded[0]["selection_overlap_ratio"], 0.5)

    def test_detect_face_regions_falls_back_to_haar_when_yunet_is_unavailable(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        expected_region = {"bbox": [90, 60, 165, 161], "detector": "opencv-haar"}
        with (
            patch.object(face_module, "_detect_faces_mediapipe", return_value=[]),
            patch.object(
                face_module,
                "_detect_faces_yunet",
                side_effect=RuntimeError("model unavailable"),
            ),
            patch.object(face_module, "_detect_faces_opencv", return_value=[expected_region]),
        ):
            regions, errors = detect_face_regions(image)

        self.assertEqual(regions, [expected_region])
        self.assertEqual(errors, ["yunet:RuntimeError:model unavailable"])

    def test_detect_face_regions_redacts_configured_model_paths_from_errors(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            mediapipe_path = Path(temp_dir) / "private-mediapipe" / "model.task"
            yunet_path = Path(temp_dir) / "private-yunet" / "model.onnx"
            with (
                patch.dict(
                    face_module.os.environ,
                    {
                        "FACE_LANDMARKER_MODEL_PATH": str(mediapipe_path),
                        "YUNET_FACE_DETECTOR_MODEL_PATH": str(yunet_path),
                    },
                ),
                patch.object(
                    face_module,
                    "_detect_faces_mediapipe",
                    side_effect=RuntimeError(
                        f"could not load {str(mediapipe_path).upper()}"
                    ),
                ),
                patch.object(
                    face_module,
                    "_detect_faces_yunet",
                    side_effect=RuntimeError(f"could not load {yunet_path}"),
                ),
                patch.object(face_module, "_detect_faces_opencv", return_value=[]),
            ):
                regions, errors = detect_face_regions(image)

        self.assertEqual(regions, [])
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("<configured-model-path>" in error for error in errors))
        self.assertTrue(all(str(mediapipe_path) not in error for error in errors))
        self.assertTrue(all(str(mediapipe_path).upper() not in error for error in errors))
        self.assertTrue(all(str(yunet_path) not in error for error in errors))

    def test_detect_face_regions_falls_back_to_haar_when_yunet_finds_no_face(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        expected_region = {"bbox": [90, 60, 165, 161], "detector": "opencv-haar"}
        with (
            patch.object(face_module, "_detect_faces_mediapipe", return_value=[]),
            patch.object(face_module, "_detect_faces_yunet", return_value=[]),
            patch.object(face_module, "_detect_faces_opencv", return_value=[expected_region]),
        ):
            regions, errors = detect_face_regions(image)

        self.assertEqual(regions, [expected_region])
        self.assertEqual(errors, [])

    def test_selection_roi_detection_upscales_and_maps_landmarks(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        roi_mask = np.zeros((256, 256), dtype=np.uint8)
        roi_mask[90:166, 36:118] = 255
        observed = {}

        def detector(crop, max_faces, min_face_pixels):
            observed["shape"] = crop.shape
            observed["minimum"] = min_face_pixels
            height, width = crop.shape[:2]
            face_mask, feature_mask = face_masks_from_box(
                crop.shape,
                (width * 0.20, height * 0.18, width * 0.80, height * 0.84),
            )
            landmarks = np.zeros((478, 3), dtype=np.float32)
            landmarks[:, 0] = 0.50
            landmarks[:, 1] = 0.51
            return [
                {
                    "bbox": [
                        int(width * 0.20),
                        int(height * 0.18),
                        int(width * 0.80),
                        int(height * 0.84),
                    ],
                    "face_mask": face_mask,
                    "feature_mask": feature_mask,
                    "part_masks": {},
                    "detector": "fixture",
                    "landmark_count": 478,
                    "landmarks_xyz": landmarks,
                }
            ], []

        regions, errors, stats = detect_face_regions_in_roi(
            image,
            roi_mask,
            detector=detector,
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(regions), 1)
        self.assertGreaterEqual(max(observed["shape"][:2]), 384)
        self.assertGreaterEqual(observed["minimum"], 48)
        self.assertLess(observed["minimum"], 96)
        self.assertEqual(regions[0]["face_mask"].shape, image.shape[:2])
        self.assertEqual(regions[0]["landmark_count"], 478)
        self.assertTrue(regions[0]["detector"].startswith("selection-roi:"))
        self.assertGreater(float(regions[0]["landmarks_xyz"][0, 0]), 0.0)
        self.assertLess(float(regions[0]["landmarks_xyz"][0, 0]), 1.0)
        self.assertEqual(stats["detected_faces"], 1)
        self.assertEqual(stats["selection_detail_fallback_regions"], 0)
        self.assertGreaterEqual(regions[0]["selection_overlap_ratio"], 0.5)
        self.assertGreaterEqual(
            min(
                regions[0]["bbox"][2] - regions[0]["bbox"][0],
                regions[0]["bbox"][3] - regions[0]["bbox"][1],
            ),
            24,
        )

    def test_selection_roi_can_emit_labeled_generic_detail_fallback(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        roi_mask = np.zeros((256, 256), dtype=np.uint8)
        roi_mask[90:166, 36:118] = 255

        regions, errors, stats = detect_face_regions_in_roi(
            image,
            roi_mask,
            detector=lambda *_args: ([], []),
            allow_selection_detail_fallback=True,
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["detector"], "selection-detail-fallback")
        self.assertEqual(regions[0]["semantic_scope"], "selected-component-detail")
        self.assertEqual(regions[0]["landmark_count"], 0)
        self.assertEqual(regions[0]["face_mask"].shape, image.shape[:2])
        self.assertEqual(stats["detected_faces"], 0)
        self.assertEqual(stats["validated_face_regions"], 0)
        self.assertEqual(stats["selection_detail_fallback_regions"], 1)

    def test_selection_roi_mapping_accepts_valid_landmarks_below_global_floor(self):
        component = np.zeros((256, 256), dtype=bool)
        component[92:165, 126:205] = True
        face_mask = np.zeros((384, 384), dtype=np.uint8)
        face_mask[160:240, 160:240] = 255
        region = {
            "bbox": [160, 160, 240, 240],
            "face_mask": face_mask,
            "feature_mask": face_mask.copy(),
            "part_masks": {},
            "detector": "mediapipe-face-landmarker",
            "landmark_count": 478,
        }

        mapped = face_module._map_roi_face_region(
            region,
            roi_box=(90, 53, 241, 204),
            roi_shape=(151, 151),
            image_shape=(256, 256, 3),
            scale_x=384 / 151,
            scale_y=384 / 151,
            component_mask=component,
        )

        self.assertIsNotNone(mapped)
        mapped_width = mapped["bbox"][2] - mapped["bbox"][0]
        self.assertGreaterEqual(mapped_width, 24)
        self.assertLess(mapped_width, face_module.MIN_FACE_PIXELS_FLOOR)
        self.assertGreaterEqual(mapped["selection_overlap_ratio"], 0.5)

    def test_selection_detail_fallback_is_suppressed_on_detector_error(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        roi_mask = np.zeros((256, 256), dtype=np.uint8)
        roi_mask[90:166, 36:118] = 255

        regions, errors, stats = detect_face_regions_in_roi(
            image,
            roi_mask,
            detector=lambda *_args: ([], ["mediapipe:RuntimeError:model failed"]),
            allow_selection_detail_fallback=True,
        )

        self.assertEqual(regions, [])
        self.assertTrue(errors)
        self.assertFalse(stats["fallback_errors_clean"])
        self.assertEqual(stats["selection_detail_fallback_regions"], 0)

    @staticmethod
    def _synthetic_eyewear_landmarks():
        indices = np.arange(478, dtype=np.float64)
        angles = indices * (np.pi * (3.0 - np.sqrt(5.0)))
        radii = 0.90 * np.sqrt((indices + 0.5) / 478.0)
        points = np.column_stack(
            (
                64.0 + 39.0 * radii * np.cos(angles),
                64.0 + 51.0 * radii * np.sin(angles),
            )
        )
        eyewear_indices = np.asarray(EYEWEAR_LANDMARK_INDICES, dtype=np.int64)
        eyewear_angles = np.linspace(0.0, 2.0 * np.pi, len(eyewear_indices), endpoint=False)
        points[eyewear_indices] = np.column_stack(
            (
                64.0 + 34.0 * np.cos(eyewear_angles),
                50.0 + 9.0 * np.sin(eyewear_angles),
            )
        )
        points[1] = (64.0, 62.0)
        return points

    def test_broad_dark_eyewear_component_is_detected(self):
        image = np.full((128, 128, 3), 210, dtype=np.uint8)
        face_mask, _ = face_masks_from_box(image.shape, (20, 8, 108, 120))
        points = self._synthetic_eyewear_landmarks()
        image[38:63, 25:103] = 24

        weight, stats = _detect_eyewear_occlusion_weight(image, face_mask, points)

        self.assertTrue(stats["enabled"])
        self.assertGreaterEqual(stats["dark_coverage_ratio"], 0.50)
        self.assertGreaterEqual(stats["component_width_ratio"], 0.60)
        self.assertGreater(stats["core_pixels"], 32)
        self.assertGreater(float(np.max(weight)), 0.99)

    def test_low_luminance_connected_shadow_does_not_trigger_eyewear(self):
        image = np.full((128, 128, 3), 44, dtype=np.uint8)
        face_mask, _ = face_masks_from_box(image.shape, (20, 8, 108, 120))
        points = self._synthetic_eyewear_landmarks()
        image[46:55, 30:98] = 24
        image[46:55, 34:98:7] = 44

        weight, stats = _detect_eyewear_occlusion_weight(image, face_mask, points)

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "eyewear_detection_gate")
        self.assertIn("dark_coverage", stats["failures"])
        self.assertIn("component_coverage", stats["failures"])
        self.assertLess(stats["dark_coverage_ratio"], 0.40)
        self.assertLess(stats["component_coverage_ratio"], 0.45)
        self.assertGreaterEqual(stats["component_width_ratio"], 0.60)
        self.assertEqual(float(np.max(weight)), 0.0)

    def test_separate_eye_shadows_do_not_trigger_eyewear_reconstruction(self):
        image = np.full((128, 128, 3), 210, dtype=np.uint8)
        face_mask, _ = face_masks_from_box(image.shape, (20, 8, 108, 120))
        points = self._synthetic_eyewear_landmarks()
        cv2.ellipse(image, (46, 52), (8, 4), 0, 0, 360, (24, 24, 24), -1)
        cv2.ellipse(image, (82, 52), (8, 4), 0, 0, 360, (24, 24, 24), -1)

        weight, stats = _detect_eyewear_occlusion_weight(image, face_mask, points)

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "eyewear_detection_gate")
        self.assertEqual(float(np.max(weight)), 0.0)

    def test_eyewear_sheet_is_replaced_by_bounded_landmark_surface(self):
        shape = (96, 96)
        yy, xx = np.indices(shape, dtype=np.float32)
        prior = 0.35 + xx * 0.001 + yy * 0.0004
        source = prior.copy()
        occlusion = np.zeros(shape, dtype=np.float32)
        occlusion[30:52, 20:76] = 1.0
        source[occlusion > 0] += 0.08

        candidate, stats = _reconstruct_eyewear_occlusion(
            source,
            source,
            prior,
            occlusion,
            reference_span=0.50,
        )

        self.assertTrue(stats["enabled"])
        self.assertGreaterEqual(stats["residual_reduction_ratio"], 0.35)
        self.assertLessEqual(stats["output_residual_p95_ratio"], 0.05 + 1e-6)
        self.assertLessEqual(stats["maximum_correction_observed_ratio"], 0.15 + 1e-6)
        self.assertLessEqual(stats["saturated_core_ratio"], 0.05)
        np.testing.assert_array_equal(candidate[occlusion == 0], source[occlusion == 0])

    def test_eyewear_reconstruction_fails_closed_when_correction_cap_saturates(self):
        prior = np.full((64, 64), 0.35, dtype=np.float32)
        source = prior.copy()
        occlusion = np.zeros_like(source)
        occlusion[18:46, 10:54] = 1.0
        source[occlusion > 0] += 0.50

        candidate, stats = _reconstruct_eyewear_occlusion(
            source,
            source,
            prior,
            occlusion,
            reference_span=0.50,
        )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "quality_gate_failed")
        self.assertIn("correction_saturation", stats["quality_gates"]["failures"])
        np.testing.assert_array_equal(candidate, source)

    def test_landmark_relative_z_adds_bounded_coarse_shape_without_a_seam(self):
        shape = (128, 128)
        yy, xx = np.indices(shape, dtype=np.float32)
        face_mask, feature_mask = face_masks_from_box(shape, (20, 8, 108, 120))
        rng = np.random.default_rng(4040)
        angles = rng.uniform(0.0, 2.0 * np.pi, 468)
        radii = np.sqrt(rng.uniform(0.0, 0.92**2, 468))
        points = np.column_stack(
            (
                64.0 + 40.0 * radii * np.cos(angles),
                64.0 + 51.0 * radii * np.sin(angles),
            )
        )
        points[1] = (64.0, 62.0)
        px = (points[:, 0] - 64.0) / 40.0
        py = (points[:, 1] - 64.0) / 51.0
        landmark_shape = 0.62 * np.exp(-(px**2 + py**2) / 0.62)
        landmark_shape += 0.38 * np.exp(-(px / 0.18) ** 2 - ((py + 0.02) / 0.28) ** 2)
        relative_z = -landmark_shape

        nose = np.exp(-((xx - 64.0) / 8.0) ** 2 - ((yy - 62.0) / 14.0) ** 2)
        global_depth = 0.34 + xx * 0.00015 + nose * 0.012
        refined, weight, stats = fuse_face_landmark_shape_prior(
            global_depth,
            face_mask,
            feature_mask,
            points,
            relative_z,
            max_correction_ratio=0.40,
            minimum_abs_correlation=0.08,
        )
        correction = refined - global_depth

        self.assertTrue(stats["enabled"])
        self.assertGreater(abs(stats["correlation"]), stats["minimum_abs_correlation"])
        self.assertGreater(float(np.max(np.abs(correction))), 1e-4)
        self.assertEqual(float(np.max(np.abs(correction[face_mask == 0]))), 0.0)
        self.assertLessEqual(stats["boundary_max_abs_correction"], 1e-7)
        self.assertLessEqual(float(np.max(np.abs(correction))), stats["correction_limit"] + 1e-6)
        self.assertGreater(float(weight[64, 64]), float(weight[10, 64]))

    def test_landmark_shape_prior_rejects_large_yaw_proxy(self):
        shape = (96, 96)
        face_mask, feature_mask = face_masks_from_box(shape, (16, 6, 80, 90))
        angles = np.linspace(0.0, 2.0 * np.pi, 468, endpoint=False)
        points = np.column_stack((48.0 + 28.0 * np.cos(angles), 48.0 + 38.0 * np.sin(angles)))
        points[1, 0] = 74.0
        relative_z = -np.cos(angles)
        global_depth = np.linspace(0.2, 0.8, shape[1], dtype=np.float32)[None, :]
        global_depth = np.repeat(global_depth, shape[0], axis=0)

        refined, weight, stats = fuse_face_landmark_shape_prior(
            global_depth,
            face_mask,
            feature_mask,
            points,
            relative_z,
        )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "yaw_gate")
        np.testing.assert_array_equal(refined, global_depth)
        self.assertEqual(float(np.max(weight)), 0.0)

    def test_box_fallback_builds_separate_face_and_feature_masks(self):
        face_mask, feature_mask = face_masks_from_box((120, 100), (20, 10, 80, 110))

        self.assertEqual(face_mask.shape, (120, 100))
        self.assertEqual(feature_mask.shape, face_mask.shape)
        self.assertEqual(int(face_mask[60, 50]), 255)
        self.assertEqual(int(face_mask[0, 0]), 0)
        self.assertGreater(np.count_nonzero(face_mask), np.count_nonzero(feature_mask))
        self.assertEqual(np.count_nonzero((feature_mask > 0) & (face_mask == 0)), 0)
        self.assertEqual(int(face_mask[65, 15]), 255)
        self.assertEqual(int(feature_mask[65, 15]), 255)

    def test_face_detail_fusion_preserves_features_without_a_boundary_ridge(self):
        shape = (128, 128)
        yy, xx = np.indices(shape, dtype=np.float32)
        global_depth = 0.35 + xx * 0.0012 + yy * 0.0004
        local_depth = global_depth.copy()
        local_depth += gaussian_peak(shape, (72, 64), sigma=3.0, amplitude=0.10)
        local_depth -= gaussian_peak(shape, (50, 48), sigma=2.5, amplitude=0.04)
        local_depth -= gaussian_peak(shape, (50, 80), sigma=2.5, amplitude=0.04)
        face_mask, feature_mask = face_masks_from_box(shape, (20, 8, 108, 120))

        refined, weight, stats = fuse_face_depth(
            global_depth,
            local_depth,
            face_mask,
            feature_mask,
            detail_strength=1.25,
            feather_ratio=0.20,
            max_correction_ratio=0.50,
        )
        correction = refined - global_depth

        self.assertGreater(float(abs(correction[72, 64])), 0.01)
        self.assertEqual(float(np.max(np.abs(correction[face_mask == 0]))), 0.0)
        self.assertLessEqual(stats["boundary_max_abs_correction"], 1e-7)
        self.assertLessEqual(float(np.max(np.abs(correction))), stats["correction_limit"] + 1e-6)
        self.assertGreater(float(weight[72, 64]), float(weight[16, 64]))

    def test_face_oval_is_a_safety_envelope_not_a_crop_blend_region(self):
        shape = (160, 160)
        face_mask, feature_mask = face_masks_from_box(shape, (28, 12, 132, 150))
        weight, _distance, feather_px = face_blend_weight(
            face_mask,
            feature_mask,
            feather_ratio=0.20,
        )

        distance_from_feature = cv2.distanceTransform(
            (feature_mask == 0).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        oval_only = (face_mask > 0) & (distance_from_feature > feather_px)

        self.assertTrue(np.any(oval_only))
        self.assertEqual(float(np.max(weight[oval_only])), 0.0)
        self.assertGreater(float(np.max(weight[feature_mask > 0])), 0.5)

    def test_crop_detail_cannot_change_head_shape_away_from_features(self):
        shape = (160, 160)
        yy, xx = np.indices(shape, dtype=np.float32)
        global_depth = 0.3 + xx * 0.001 + yy * 0.0003
        local_depth = global_depth + 0.04 * np.sin(xx * 0.7) * np.cos(yy * 0.6)
        face_mask, feature_mask = face_masks_from_box(shape, (28, 12, 132, 150))

        refined, weight, stats = fuse_face_depth(
            global_depth,
            local_depth,
            face_mask,
            feature_mask,
            detail_strength=1.0,
            max_correction_ratio=0.5,
        )
        correction = refined - global_depth

        self.assertEqual(stats["blend_strategy"], "feature-supported-shape-preserving")
        self.assertEqual(float(np.max(np.abs(correction[weight == 0]))), 0.0)
        self.assertGreater(float(np.max(np.abs(correction[weight > 0.5]))), 0.0)

    def test_smoother_face_crop_cannot_erase_existing_global_detail(self):
        shape = (96, 96)
        global_depth = np.full(shape, 0.4, dtype=np.float32)
        global_depth += gaussian_peak(shape, (54, 48), sigma=2.0, amplitude=0.12)
        local_depth = gaussian_filter(global_depth, sigma=4.0)
        face_mask, feature_mask = face_masks_from_box(shape, (16, 6, 80, 90))

        refined, _weight, stats = fuse_face_depth(
            global_depth,
            local_depth,
            face_mask,
            feature_mask,
            max_correction_ratio=0.50,
        )

        self.assertEqual(stats["detail_fusion"], "monotonic-excess")
        self.assertGreaterEqual(float(refined[54, 48]), float(global_depth[54, 48]) - 1e-7)
        self.assertLess(float(np.max(np.abs(refined - global_depth))), 1e-3)

    def test_file_pipeline_writes_refined_depth_and_auditable_masks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "portrait.png"
            depth_path = root / "output_depth_data.npy"
            image = np.full((128, 128, 3), 210, dtype=np.uint8)
            Image.fromarray(image).save(image_path)
            yy, xx = np.indices((64, 64), dtype=np.float32)
            global_depth = 0.25 + xx * 0.003 + yy * 0.001
            np.save(depth_path, global_depth)
            face_mask, feature_mask = face_masks_from_box(image.shape, (32, 18, 96, 114))
            landmark_indices = np.arange(478, dtype=np.float64)
            landmark_angles = landmark_indices * (np.pi * (3.0 - np.sqrt(5.0)))
            landmark_radii = 0.90 * np.sqrt((landmark_indices + 0.5) / 478.0)
            landmark_x = 64.0 + 28.0 * landmark_radii * np.cos(landmark_angles)
            landmark_y = 66.0 + 40.0 * landmark_radii * np.sin(landmark_angles)
            landmarks_xyz = np.column_stack(
                (
                    landmark_x / 127.0,
                    landmark_y / 127.0,
                    -(0.75 * landmark_x / 127.0 + 0.25 * landmark_y / 127.0),
                )
            ).astype(np.float32)
            landmarks_xyz[1, 0] = 64.0 / 127.0
            detector_part_masks = {
                name: feature_mask.copy() for name in FACE_PART_NAMES
            }
            detector_part_masks["nose"] = np.zeros_like(feature_mask)

            def detector(_image):
                return [
                    {
                        "bbox": [32, 18, 96, 114],
                        "face_mask": face_mask,
                        "feature_mask": feature_mask,
                        "detector": "test-landmarks",
                        "landmark_count": 478,
                        "landmarks_xyz": landmarks_xyz,
                        "part_masks": detector_part_masks,
                    }
                ]

            def infer_depth(crop_path, output_dir):
                crop = Image.open(crop_path)
                height, width = crop.height, crop.width
                crop_y, crop_x = np.indices((height, width), dtype=np.float32)
                values = 1.8 + crop_x * 0.004 + crop_y * 0.001
                values += gaussian_peak((height, width), (height * 0.58, width * 0.50), 4.0, 0.12)
                output_path = Path(output_dir) / "output_depth_data.npy"
                np.save(output_path, values.astype(np.float32))
                return output_path

            refined_path, metadata = refine_depth_for_faces(
                image_path,
                depth_path,
                root,
                infer_depth=infer_depth,
                mode="on",
                detector=detector,
                detail_strength=1.2,
                max_correction_ratio=0.30,
            )

            refined = np.load(refined_path)
            self.assertTrue(metadata["applied"])
            self.assertEqual(metadata["detected_faces"], 1)
            self.assertEqual(metadata["refined_faces"], 1)
            self.assertEqual(metadata["eyewear_deoccluded_faces"], 0)
            self.assertEqual(metadata["faces"][0]["landmark_count"], 478)
            self.assertEqual(metadata["part_mask_faces"], 1)
            self.assertEqual(tuple(metadata["part_names"]), FACE_PART_NAMES)
            self.assertTrue(metadata["faces"][0]["part_masks"]["complete"])
            face_part_mask = np.asarray(
                Image.open(root / metadata["faces"][0]["part_masks"]["face_file"])
            )
            self.assertEqual(face_part_mask.shape, global_depth.shape)
            self.assertGreater(np.count_nonzero(face_part_mask), 0)
            for name, relative_path in metadata["faces"][0]["part_masks"]["files"].items():
                self.assertIn(name, FACE_PART_NAMES)
                part_mask = np.asarray(Image.open(root / relative_path))
                self.assertEqual(part_mask.shape, global_depth.shape)
                self.assertGreater(np.count_nonzero(part_mask), 0)
            self.assertTrue(metadata["faces"][0]["landmark_shape_prior"]["enabled"])
            self.assertFalse(metadata["faces"][0]["eyewear_deocclusion"]["enabled"])
            self.assertGreater(float(np.max(np.abs(refined - global_depth))), 0.0)
            self.assertTrue((root / "output_depth_face_refined_preview.png").is_file())
            self.assertTrue((root / "output_face_refinement_weight.png").is_file())
            self.assertTrue((root / "output_face_refinement_region.png").is_file())
            self.assertTrue((root / "output_face_refinement_occlusion.png").is_file())
            self.assertEqual(
                int(
                    np.max(
                        np.asarray(
                            Image.open(root / "output_face_refinement_occlusion.png")
                        )
                    )
                ),
                0,
            )
            self.assertTrue((root / "output_face_refinement_metadata.json").is_file())
            self.assertEqual(metadata["region_file"], "output_face_refinement_region.png")

    def test_auto_mode_keeps_original_depth_when_no_face_is_found(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "image.png"
            depth_path = root / "depth.npy"
            Image.new("RGB", (32, 32), "white").save(image_path)
            np.save(depth_path, np.ones((16, 16), dtype=np.float32))

            output_path, metadata = refine_depth_for_faces(
                image_path,
                depth_path,
                root,
                infer_depth=lambda *_args: self.fail("Depth inference must not run without a detected face"),
                mode="auto",
                detector=lambda _image: [],
            )

            self.assertEqual(Path(output_path), depth_path)
            self.assertFalse(metadata["applied"])
            self.assertEqual(metadata["reason"], "no_face_detected")

    def test_selected_component_detail_fallback_refines_without_landmark_prior(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "selected.png"
            depth_path = root / "depth.npy"
            image = np.full((128, 128, 3), 245, dtype=np.uint8)
            image[28:108, 30:98] = (165, 118, 96)
            Image.fromarray(image).save(image_path)
            yy, xx = np.indices((64, 64), dtype=np.float32)
            global_depth = 0.3 + 0.001 * xx + 0.0005 * yy
            np.save(depth_path, global_depth)
            roi_mask = np.zeros((128, 128), dtype=np.uint8)
            roi_mask[28:108, 30:98] = 255

            def infer_depth(crop_path, output_dir):
                crop = Image.open(crop_path)
                rows, cols = np.indices((crop.height, crop.width), dtype=np.float32)
                local = 0.6 + 0.002 * cols + 0.001 * rows
                local += 0.06 * np.sin(cols * 0.35) * np.cos(rows * 0.31)
                output = Path(output_dir) / "output_depth_data.npy"
                np.save(output, local.astype(np.float32))
                return output

            refined_path, metadata = refine_depth_for_faces(
                image_path,
                depth_path,
                root,
                infer_depth=infer_depth,
                mode="auto",
                detector=lambda _image: [],
                detection_roi_mask=roi_mask,
            )

            refined = np.load(refined_path)
            self.assertTrue(metadata["applied"])
            self.assertEqual(metadata["selection_detail_fallback_regions"], 1)
            self.assertEqual(metadata["refined_faces"], 0)
            self.assertEqual(metadata["detected_faces"], 0)
            self.assertEqual(metadata["refined_selection_detail_regions"], 1)
            self.assertEqual(metadata["refined_regions_total"], 1)
            self.assertEqual(
                metadata["faces"][0]["detector"],
                "selection-detail-fallback",
            )
            self.assertEqual(
                metadata["faces"][0]["semantic_scope"],
                "selected-component-detail",
            )
            self.assertFalse(metadata["faces"][0]["landmark_shape_prior"]["enabled"])
            self.assertGreater(float(np.max(np.abs(refined - global_depth))), 0.0)
            region_mask = np.asarray(Image.open(root / metadata["region_file"]))
            weight_mask = np.asarray(Image.open(root / metadata["weight_file"]))
            self.assertEqual(int(np.max(region_mask)), 0)
            self.assertGreater(int(np.max(weight_mask)), 0)

    def test_required_mode_raises_when_all_detector_backends_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "image.png"
            depth_path = root / "depth.npy"
            Image.new("RGB", (32, 32), "white").save(image_path)
            np.save(depth_path, np.ones((16, 16), dtype=np.float32))

            with self.assertRaisesRegex(RuntimeError, "Face detection failed"):
                refine_depth_for_faces(
                    image_path,
                    depth_path,
                    root,
                    infer_depth=lambda *_args: self.fail("Depth inference must not run"),
                    mode="on",
                    detector=lambda _image: (
                        [],
                        ["mediapipe:RuntimeError:failed", "opencv:RuntimeError:failed"],
                    ),
                )

    def test_process_image_passes_face_controls_and_returns_refinement_audit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "output"

            def fake_complete(input_path, **_kwargs):
                return input_path, None

            def fake_depth(_image_path, output_dir, **_kwargs):
                depth_path = Path(output_dir) / "output_depth_data.npy"
                rows, cols = np.indices((8, 8), dtype=np.float32)
                np.save(depth_path, 0.1 + 0.02 * rows + 0.03 * cols)
                return str(depth_path)

            refinement_audit = {
                "mode": "on",
                "applied": True,
                "detected_faces": 1,
                "refined_faces": 1,
                "faces": [{"boundary_max_abs_correction": 0.0}],
            }
            with (
                patch.object(main_module, "OUTPUT_DIR", output_root),
                patch.object(main_module, "complete_image", side_effect=fake_complete),
                patch.object(main_module, "process_image_get_depth_data", side_effect=fake_depth),
                patch.object(
                    main_module,
                    "refine_depth_for_faces",
                    side_effect=lambda _image, depth, _output, **_kwargs: (depth, refinement_audit),
                ) as refine_mock,
            ):
                portrait_bytes = cv2.imencode(
                    ".png",
                    np.full((32, 32, 3), 255, dtype=np.uint8),
                )[1].tobytes()
                response = TestClient(main_module.app).post(
                    "/process_image",
                    files={"file": ("portrait.png", portrait_bytes, "image/png")},
                    data={
                        "target_dimension": "8",
                        "z_scale": "2",
                        "sigma": "0",
                        "base_border_px": "0",
                        "trim_top_background": "false",
                        "face_refinement_mode": "on",
                        "face_detail_strength": "1.4",
                        "face_feather_ratio": "0.25",
                        "face_max_correction_ratio": "0.05",
                    },
                )

            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            call = refine_mock.call_args
            self.assertEqual(call.kwargs["mode"], "on")
            self.assertAlmostEqual(call.kwargs["detail_strength"], 1.4)
            self.assertAlmostEqual(call.kwargs["feather_ratio"], 0.25)
            self.assertAlmostEqual(call.kwargs["max_correction_ratio"], 0.05)
            self.assertIsNone(call.kwargs["detection_roi_mask"])
            self.assertEqual(payload["face_refinement"], refinement_audit)
            self.assertIn("face_refinement_seconds", payload["timings"])


if __name__ == "__main__":
    unittest.main()
