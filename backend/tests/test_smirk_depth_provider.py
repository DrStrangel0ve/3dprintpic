import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.benchmark.smirk_depth_provider import (
    SMIRK_REQUIRED_SOURCE_FILES,
    SMIRK_SOURCE_REVISION,
    _crop_transform,
    project_smirk_vertices,
    smirk_preflight,
)


class SMIRKDepthProviderTests(unittest.TestCase):
    def test_preflight_rejects_unpinned_checkpoint_and_flame_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider_root = root / "provider"
            for relative in SMIRK_REQUIRED_SOURCE_FILES:
                path = provider_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            checkpoint = root / "SMIRK_em1.pt"
            checkpoint.write_bytes(b"not-the-pinned-checkpoint")
            flame_model = root / "generic_model.pkl"
            flame_model.write_bytes(b"not-the-pinned-flame-model")
            with patch(
                "backend.benchmark.smirk_depth_provider._git_output",
                side_effect=(SMIRK_SOURCE_REVISION, ""),
            ):
                evidence = smirk_preflight(
                    provider_root,
                    checkpoint,
                    flame_model,
                )

        self.assertTrue(evidence["checks"]["source_revision_pinned"])
        self.assertTrue(evidence["checks"]["source_clean"])
        self.assertTrue(evidence["checks"]["source_files_complete"])
        self.assertFalse(evidence["checks"]["checkpoint_size_pinned"])
        self.assertFalse(evidence["checks"]["checkpoint_hash_pinned"])
        self.assertFalse(evidence["checks"]["flame_model_hash_pinned"])
        self.assertFalse(evidence["runnable"])
        self.assertTrue(evidence["license"]["research_only"])

    def test_crop_transform_matches_official_square_crop_scale(self):
        crop = _crop_transform(
            (40.0, 30.0, 80.0, 90.0),
            image_width=200,
            image_height=120,
        )

        self.assertAlmostEqual(crop["crop_side_pixels"], 70.0)
        self.assertEqual(crop["crop_box_xyxy"], [25.0, 25.0, 95.0, 95.0])

    def test_crop_transform_truncates_side_like_official_demo(self):
        crop = _crop_transform(
            (82.0, 103.0, 117.0, 144.0),
            image_width=256,
            image_height=256,
        )

        self.assertAlmostEqual(
            crop["crop_side_unrounded_pixels"],
            53.2,
        )
        self.assertEqual(crop["crop_side_pixels"], 53.0)
        self.assertEqual(
            crop["crop_box_xyxy"],
            [73.0, 97.0, 126.0, 150.0],
        )

    def test_projection_maps_camera_center_to_crop_center(self):
        crop = _crop_transform(
            (40.0, 30.0, 80.0, 90.0),
            image_width=200,
            image_height=120,
        )
        projected = project_smirk_vertices(
            np.array(
                [
                    [-0.1, 0.2, 0.3],
                    [0.0, 0.0, -0.5],
                ],
                dtype=np.float32,
            ),
            np.array([2.0, 0.1, -0.2], dtype=np.float32),
            crop,
        )

        np.testing.assert_allclose(
            projected[0],
            np.array([60.0, 60.0, -0.6]),
            atol=1e-6,
        )
        self.assertAlmostEqual(float(projected[1, 2]), 1.0)


if __name__ == "__main__":
    unittest.main()
