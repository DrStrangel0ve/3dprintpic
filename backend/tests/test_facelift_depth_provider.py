import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.benchmark.facelift_depth_provider import (
    FACELIFT_FRONT_CAMERA_INDEX,
    FACELIFT_REQUIRED_MODEL_FILES,
    FACELIFT_REQUIRED_SOURCE_FILES,
    facelift_preflight,
    load_facelift_camera,
    load_facelift_gaussians,
    normalize_depth,
    rasterize_gaussian_center_depth,
)


class FaceLiftDepthProviderTests(unittest.TestCase):
    def test_preflight_rejects_unpinned_weight_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider_root = root / "provider"
            model_root = root / "model"
            for relative in FACELIFT_REQUIRED_SOURCE_FILES:
                path = provider_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            for relative in FACELIFT_REQUIRED_MODEL_FILES:
                path = model_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")

            evidence = facelift_preflight(provider_root, model_root)

        self.assertTrue(evidence["checks"]["model_files_complete"])
        self.assertFalse(evidence["checks"]["model_weight_hashes_pinned"])
        self.assertFalse(evidence["runnable"])
        self.assertTrue(evidence["license"]["research_only"])

    def test_ply_dependency_is_lazy_and_reports_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gaussians.ply"
            path.write_text("ply\n", encoding="utf-8")

            with patch.dict(sys.modules, {"plyfile": None}):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "optional 'plyfile' dependency",
                ):
                    load_facelift_gaussians(path)

    def test_front_camera_uses_pinned_face_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera_dir = root / "utils_folder"
            camera_dir.mkdir()
            frames = []
            for index in range(6):
                frames.append(
                    {
                        "fx": 100.0 + index,
                        "fy": 110.0 + index,
                        "cx": 16.0,
                        "cy": 16.0,
                        "w2c": np.eye(4).tolist(),
                    }
                )
            (camera_dir / "opencv_cameras.json").write_text(
                json.dumps({"frames": frames}),
                encoding="utf-8",
            )

            camera = load_facelift_camera(root)

        self.assertEqual(camera["index"], FACELIFT_FRONT_CAMERA_INDEX)
        self.assertEqual(camera["fx"], 102.0)
        self.assertEqual(camera["fy"], 112.0)

    def test_front_to_back_composition_prefers_near_gaussian(self):
        xyz = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
        opacity = np.array([0.99, 0.99], dtype=np.float64)
        scales = np.full((2, 3), 0.03, dtype=np.float64)
        camera = {
            "w2c": np.eye(4),
            "fx": 100.0,
            "fy": 100.0,
            "cx": 16.0,
            "cy": 16.0,
        }

        depth, alpha, stats = rasterize_gaussian_center_depth(
            xyz,
            opacity,
            scales,
            camera,
            height=32,
            width=32,
        )

        self.assertTrue(np.isfinite(depth[16, 16]))
        self.assertLess(float(depth[16, 16]), 1.05)
        self.assertGreater(float(alpha[16, 16]), 0.99)
        self.assertEqual(stats["rendered_gaussians"], 2)

    def test_depth_normalization_preserves_near_is_smaller(self):
        depth = np.linspace(1.0, 3.0, 100, dtype=np.float32).reshape(10, 10)

        normalized, metadata = normalize_depth(depth)

        self.assertLess(float(normalized[0, 0]), float(normalized[-1, -1]))
        self.assertEqual(float(np.nanmin(normalized)), 0.0)
        self.assertEqual(float(np.nanmax(normalized)), 1.0)
        self.assertTrue(metadata["near_is_smaller"])


if __name__ == "__main__":
    unittest.main()
