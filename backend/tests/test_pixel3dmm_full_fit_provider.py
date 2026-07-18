import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.benchmark.pixel3dmm_full_fit_provider import (
    PIXEL3DMM_REQUIRED_SOURCE_FILES,
    PIXEL3DMM_SOURCE_REVISION,
    pixel3dmm_crop_transform,
    pixel3dmm_transfer_preflight,
    project_pixel3dmm_camera_vertices,
    rasterize_pixel3dmm_camera_depth,
    transform_pixel3dmm_camera_vertices,
)


class Pixel3DMMFullFitProviderTests(unittest.TestCase):
    SOURCE_SHA256 = "a" * 64

    def _crop(self, bounds, **overrides):
        arguments = {
            "source_height": 300,
            "source_width": 400,
            "source_sha256": self.SOURCE_SHA256,
            "crop_source_sha256": self.SOURCE_SHA256,
            "frame_id": 0,
            "crop_frame_id": 0,
            "checkpoint_image_size": np.array([256, 256]),
        }
        arguments.update(overrides)
        return pixel3dmm_crop_transform(np.asarray(bounds), **arguments)

    def test_transfer_preflight_rejects_unpinned_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider_root = root / "provider"
            for relative in PIXEL3DMM_REQUIRED_SOURCE_FILES:
                path = provider_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            uv_checkpoint = root / "uv.ckpt"
            normal_checkpoint = root / "normals.ckpt"
            uv_checkpoint.write_bytes(b"not-the-pinned-uv-checkpoint")
            normal_checkpoint.write_bytes(b"not-the-pinned-normal-checkpoint")
            with patch(
                "backend.benchmark.pixel3dmm_full_fit_provider._git_output",
                side_effect=(PIXEL3DMM_SOURCE_REVISION, ""),
            ):
                evidence = pixel3dmm_transfer_preflight(
                    provider_root,
                    uv_checkpoint,
                    normal_checkpoint,
                )

        self.assertTrue(evidence["checks"]["source_revision_pinned"])
        self.assertTrue(evidence["checks"]["source_clean"])
        self.assertTrue(evidence["checks"]["source_files_complete"])
        self.assertFalse(evidence["checks"]["uv_checkpoint_size_pinned"])
        self.assertFalse(evidence["checks"]["uv_checkpoint_hash_pinned"])
        self.assertFalse(evidence["checks"]["normal_checkpoint_size_pinned"])
        self.assertFalse(evidence["checks"]["normal_checkpoint_hash_pinned"])
        self.assertFalse(evidence["transfer_ready"])
        self.assertFalse(evidence["runtime_ready"])
        self.assertTrue(evidence["license"]["research_only"])
        self.assertFalse(evidence["license"]["production_eligible"])

    def test_camera_transform_applies_head_then_world_to_camera(self):
        posed = np.array(
            [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]],
            dtype=np.float32,
        )
        quarter_turn = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        camera, metadata = transform_pixel3dmm_camera_vertices(
            posed,
            head_rotation=quarter_turn,
            head_translation=np.array([2.0, 3.0, -1.0]),
            camera_rotation=np.eye(3),
            camera_translation=np.array([-1.0, -1.0, -2.0]),
        )

        np.testing.assert_allclose(
            camera,
            np.array(
                [[1.0, 2.0, -4.0], [1.0, 3.0, -4.0], [0.0, 2.0, -4.0]],
                dtype=np.float32,
            ),
        )
        self.assertEqual(metadata["depth_semantics"], "positive -z_cam")
        self.assertEqual(metadata["depth_min"], 4.0)
        self.assertEqual(metadata["depth_max"], 4.0)

    def test_camera_transform_rejects_reflection_and_near_plane_crossing(self):
        vertices = np.array(
            [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]],
            dtype=np.float32,
        )
        reflection = np.diag([-1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            transform_pixel3dmm_camera_vertices(
                vertices,
                head_rotation=reflection,
                head_translation=np.zeros(3),
                camera_rotation=np.eye(3),
                camera_translation=np.zeros(3),
            )
        with self.assertRaisesRegex(ValueError, "near plane"):
            transform_pixel3dmm_camera_vertices(
                -vertices,
                head_rotation=np.eye(3),
                head_translation=np.zeros(3),
                camera_rotation=np.eye(3),
                camera_translation=np.zeros(3),
            )

    def test_crop_inversion_uses_exclusive_bounds_and_half_pixel_centers(self):
        crop = self._crop([10, 110, 20, 220])
        tracking = np.array(
            [[0.0, 0.0, 1.0], [255.0, 255.0, 1.0]],
            dtype=np.float64,
        )
        source = tracking @ crop["tracking_to_source_matrix"].T
        source = source[:, :2] / source[:, 2:3]

        np.testing.assert_allclose(
            source,
            np.array(
                [[19.890625, 9.6953125], [219.109375, 109.3046875]],
                dtype=np.float64,
            ),
            atol=1e-12,
        )
        self.assertEqual(crop["crop_width_pixels"], 200)
        self.assertEqual(crop["crop_height_pixels"], 100)
        self.assertEqual(crop["prediction_size"], 512)
        self.assertEqual(crop["tracking_size"], 256)
        identity = (
            crop["tracking_to_source_matrix"]
            @ crop["source_to_tracking_matrix"]
        )
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-12)

    def test_crop_rejects_out_of_bounds_metadata(self):
        for bounds, message in (
            ([0, 100, 0, 201], "horizontal"),
            ([-1, 100, 0, 200], "vertical"),
            ([0, 100, 0.5, 200], "integer"),
        ):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ValueError, message):
                    self._crop(
                        bounds,
                        source_height=100,
                        source_width=200,
                    )

    def test_crop_rejects_stale_source_frame_and_checkpoint_metadata(self):
        cases = (
            (
                {"crop_source_sha256": "b" * 64},
                "different source image",
            ),
            ({"crop_frame_id": 1}, "frame ID"),
            (
                {"checkpoint_image_size": np.array([512, 512])},
                "tracking grid",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    self._crop([10, 110, 20, 220], **overrides)

    def test_projection_uses_clean_opengl_intrinsics_then_inverse_crop(self):
        crop = self._crop(
            [0, 256, 0, 256],
            source_height=256,
            source_width=256,
        )
        projected, metadata = project_pixel3dmm_camera_vertices(
            np.array(
                [[0.0, 0.0, -2.0], [0.5, 0.5, -2.0]],
                dtype=np.float32,
            ),
            crop,
            focal_length=1.0,
            principal_point=np.array([0.0, 0.0]),
        )

        np.testing.assert_allclose(
            projected,
            np.array(
                [[128.0, 128.0, 2.0], [192.0, 64.0, 2.0]],
                dtype=np.float32,
            ),
        )
        self.assertEqual(metadata["focal_length_pixels"], 256.0)
        self.assertEqual(
            metadata["principal_point_tracking_pixels"],
            [128.5, 128.5],
        )
        self.assertEqual(metadata["viewport_to_array_index_offset"], -0.5)

    def test_depth_raster_uses_reciprocal_interpolation_without_antialiasing(self):
        projected = np.array(
            [[1.0, 1.0, 1.0], [5.0, 1.0, 2.0], [1.0, 5.0, 4.0]],
            dtype=np.float32,
        )
        depth, metadata = rasterize_pixel3dmm_camera_depth(
            projected,
            np.array([[0, 1, 2]], dtype=np.int32),
            height=7,
            width=7,
        )

        self.assertAlmostEqual(float(depth[2, 2]), 1.0 / 0.6875, places=6)
        self.assertEqual(
            metadata["interpolation"],
            "perspective-correct-reciprocal-depth",
        )
        self.assertFalse(metadata["silhouette_depth_antialiasing"])
        self.assertTrue(metadata["source_alignment_requires_landmark_replay"])


if __name__ == "__main__":
    unittest.main()
