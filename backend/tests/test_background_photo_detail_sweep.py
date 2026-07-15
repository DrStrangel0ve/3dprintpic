import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark import run_background_photo_detail_sweep as detail_sweep


class BackgroundPhotoDetailSweepTests(unittest.TestCase):
    def test_detail_signal_is_deterministic_and_sensitive_to_texture(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            image = np.zeros((48, 48, 3), dtype=np.uint8)
            image[:, ::4] = 255
            Image.fromarray(image).save(path)

            first = detail_sweep._photo_detail_signal(path, (48, 48))
            second = detail_sweep._photo_detail_signal(path, (48, 48))

        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(np.std(first)), 0.05)

    def test_detail_metrics_accept_correlated_background_and_exact_face(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, ::4] = 255
            Image.fromarray(image).save(path)
            signal = detail_sweep._photo_detail_signal(path, (64, 64))
            scale = float(np.percentile(np.abs(signal), 98.0))
            baseline = np.full((64, 64), 4.0, dtype=np.float32)
            face = np.zeros_like(baseline, dtype=bool)
            face[16:48, 24:40] = True
            candidate = baseline + 0.30 * np.clip(signal / scale, -1.0, 1.0)
            candidate[face] = baseline[face]

            metrics = detail_sweep._detail_metrics(
                baseline,
                candidate,
                face,
                path,
                0.30,
            )

        self.assertTrue(metrics["checks"]["passed"])
        self.assertGreater(metrics["source_detail_correlation"], 0.9)
        self.assertEqual(metrics["max_face_interior_change_mm"], 0.0)
        self.assertEqual(metrics["max_attachment_boundary_change_mm"], 0.0)

    def test_detail_metrics_reject_face_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, ::4] = 255
            Image.fromarray(image).save(path)
            baseline = np.zeros((64, 64), dtype=np.float32)
            face = np.zeros_like(baseline, dtype=bool)
            face[16:48, 24:40] = True
            candidate = baseline.copy()
            candidate[face] = 0.02

            metrics = detail_sweep._detail_metrics(
                baseline,
                candidate,
                face,
                path,
                0.30,
            )

        self.assertFalse(metrics["checks"]["face_interior"])
        self.assertFalse(metrics["checks"]["passed"])

    def test_run_rejects_invalid_level_matrix_before_loading_fixture(self):
        for levels in ((0.12, 0.30), (0.0, -0.1), (0.0, 0.3, 0.3)):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                detail_sweep.run("unused", detail_levels_mm=levels)


if __name__ == "__main__":
    unittest.main()
