import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.benchmark.run_makehuman_face_depth_smoke import run


class MakeHumanFaceDepthSmokeTests(unittest.TestCase):
    def test_oracle_validates_depth_part_and_background_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = run(
                temporary,
                providers=("oracle",),
                render_size=160,
                allow_failures=True,
            )

        row = summary["rows"][0]
        self.assertTrue(summary["checks"]["oracle_passed"])
        self.assertFalse(summary["checks"]["at_least_one_challenger_passed"])
        self.assertTrue(row["metrics"]["named_parts"]["passed"])
        self.assertTrue(row["metrics"]["affine_mm"]["passed"])
        self.assertTrue(row["metrics"]["background"]["passed"])

    def test_exact_mock_challenger_passes_expansion_gate(self):
        def exact_prediction(image_path, **_kwargs):
            exact = np.load(Path(image_path).parent / "exact_depth.npy")
            return exact, {
                "provider": "mock-moge2",
                "model": "exact-test-double",
                "relief_transform": "one-minus-depth",
                "inference_seconds": 0.0,
                "peak_vram_gb": 0.0,
            }

        with tempfile.TemporaryDirectory() as temporary, patch(
            "backend.benchmark.run_makehuman_face_depth_smoke._infer_moge2",
            side_effect=exact_prediction,
        ), patch(
            "backend.benchmark.run_makehuman_face_depth_smoke._git_provenance",
            return_value={"available": True, "clean": True, "revision": "test"},
        ):
            summary = run(
                temporary,
                providers=("oracle", "moge2-vitb-normal"),
                render_size=160,
            )

        self.assertTrue(summary["checks"]["passed"])
        self.assertTrue(summary["checks"]["implementation_provenance_clean"])
        self.assertTrue(summary["rows"][1]["metrics"]["checks"]["passed"])

    def test_unknown_provider_fails_before_rendering(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Unknown providers"):
                run(temporary, providers=("future-depth",), render_size=96)


if __name__ == "__main__":
    unittest.main()
