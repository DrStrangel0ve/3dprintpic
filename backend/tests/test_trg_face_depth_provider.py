import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np

from backend.benchmark.trg_face_depth_provider import (
    TRG_REQUIRED_SOURCE_FILES,
    TRG_SOURCE_REVISION,
    _upstream_import_context,
    bbox_iou,
    crop_intrinsic_matrix,
    official_crop_bbox,
    project_camera_vertices,
    trg_preflight,
)


class TRGFaceDepthProviderTests(unittest.TestCase):
    def test_upstream_import_context_restores_shadowed_modules(self):
        previous_models = sys.modules.get("models")
        previous_child = sys.modules.get("models.existing")
        sentinel = types.ModuleType("models")
        sentinel_child = types.ModuleType("models.existing")
        sys.modules["models"] = sentinel
        sys.modules["models.existing"] = sentinel_child
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with _upstream_import_context(root):
                    self.assertNotIn("models", sys.modules)
                    sys.modules["models"] = types.ModuleType("models")
                    sys.modules["models.trg_temporary"] = types.ModuleType(
                        "models.trg_temporary"
                    )
                self.assertIs(sys.modules.get("models"), sentinel)
                self.assertIs(sys.modules.get("models.existing"), sentinel_child)
                self.assertNotIn("models.trg_temporary", sys.modules)
        finally:
            for name, module in (
                ("models", previous_models),
                ("models.existing", previous_child),
            ):
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_official_crop_is_square_and_one_point_five_times_support(self):
        crop = official_crop_bbox([160, 102, 193, 139])

        self.assertEqual(crop, (148, 92, 204, 148))
        self.assertEqual(crop[2] - crop[0], crop[3] - crop[1])

    def test_crop_intrinsic_matches_official_crop_then_resize_contract(self):
        intrinsic = crop_intrinsic_matrix(
            256,
            256,
            (148, 92, 204, 148),
        )

        scale = 192.0 / 56.0
        self.assertAlmostEqual(float(intrinsic[0, 0]), 5000.0 * scale, places=3)
        self.assertAlmostEqual(float(intrinsic[1, 1]), 5000.0 * scale, places=3)
        self.assertAlmostEqual(
            float(intrinsic[2, 0]),
            (128.0 - 148.0) * scale,
            places=4,
        )
        self.assertAlmostEqual(
            float(intrinsic[2, 1]),
            (128.0 - 92.0) * scale,
            places=4,
        )
        np.testing.assert_array_equal(intrinsic[3], np.zeros(3))

    def test_camera_projection_preserves_positive_camera_z(self):
        vertices = np.array(
            [[0.0, 0.0, 2.0], [0.1, -0.2, 2.0]],
            dtype=np.float32,
        )

        projected = project_camera_vertices(
            vertices,
            image_height=200,
            image_width=300,
            focal_length_px=100.0,
        )

        np.testing.assert_allclose(projected[0], [150.0, 100.0, 2.0])
        np.testing.assert_allclose(projected[1], [155.0, 90.0, 2.0])
        with self.assertRaisesRegex(ValueError, "in front"):
            project_camera_vertices(
                np.array([[0.0, 0.0, 0.0]]),
                image_height=200,
                image_width=300,
            )

    def test_bbox_iou_is_bounded_and_symmetric(self):
        first = [0.0, 0.0, 10.0, 10.0]
        second = [5.0, 0.0, 15.0, 10.0]

        self.assertAlmostEqual(bbox_iou(first, second), 1.0 / 3.0)
        self.assertEqual(bbox_iou(first, second), bbox_iou(second, first))

    def test_preflight_rejects_unpinned_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider_root = root / "provider"
            for relative in TRG_REQUIRED_SOURCE_FILES:
                path = provider_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            checkpoint = root / "state_dict.bin"
            checkpoint.write_bytes(b"not-the-pinned-checkpoint")
            dependencies = patch(
                "backend.benchmark.trg_face_depth_provider.importlib.util.find_spec",
                return_value=object(),
            )
            revision = patch(
                "backend.benchmark.trg_face_depth_provider._git_output",
                side_effect=(TRG_SOURCE_REVISION, ""),
            )
            with dependencies, revision:
                evidence = trg_preflight(provider_root, checkpoint)

        self.assertTrue(evidence["checks"]["source_revision_pinned"])
        self.assertFalse(evidence["checks"]["checkpoint_size_pinned"])
        self.assertFalse(evidence["checks"]["checkpoint_hash_pinned"])
        self.assertFalse(evidence["runnable"])
        self.assertFalse(evidence["license"]["production_eligible"])
        self.assertFalse(
            evidence["camera_contract"]["metric_depth_calibrated_for_input_camera"]
        )


if __name__ == "__main__":
    unittest.main()
