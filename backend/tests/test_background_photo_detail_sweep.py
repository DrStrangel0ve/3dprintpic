import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark import run_background_photo_detail_sweep as detail_sweep
from backend.pic_to_3d import (
    _effective_background_photo_detail_mm,
    _photo_detail_sampling,
)


class BackgroundPhotoDetailSweepTests(unittest.TestCase):
    def test_unprotected_photo_detail_retains_legacy_cap(self):
        self.assertEqual(_effective_background_photo_detail_mm(0.60, None), 0.12)
        self.assertEqual(
            _effective_background_photo_detail_mm(0.60, np.ones((2, 2), dtype=bool)),
            0.60,
        )
        self.assertEqual(
            _effective_background_photo_detail_mm(0.60, np.zeros((2, 2), dtype=bool)),
            0.12,
        )
        self.assertEqual(
            _effective_background_photo_detail_mm(2.0, np.ones((2, 2), dtype=bool)),
            0.60,
        )
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _effective_background_photo_detail_mm(
                    invalid, np.ones((2, 2), dtype=bool)
                )

    def test_photo_detail_halo_is_constant_in_physical_units(self):
        coarse_pitch, coarse_halo = _photo_detail_sampling(96.0, (256, 256))
        fine_pitch, fine_halo = _photo_detail_sampling(96.0, (512, 512))

        self.assertAlmostEqual(coarse_pitch * coarse_halo, 5.0)
        self.assertAlmostEqual(fine_pitch * fine_halo, 5.0)
        self.assertGreater(fine_halo, coarse_halo)

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

    def test_detail_signal_replays_surface_grid_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[::3, :, :] = 255
            Image.fromarray(image).save(path)
            transform = {
                "target_depth_shape": [36, 48],
                "mesh_shape_before_crop": [24, 32],
                "crop_bbox_rc": [2, 3, 22, 29],
                "emitted_shape": [20, 26],
            }
            signal = detail_sweep._photo_detail_signal(
                path,
                (20, 26),
                surface_grid_transform=transform,
            )

        self.assertEqual(signal.shape, (20, 26))
        self.assertGreater(float(np.std(signal)), 0.01)

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
        self.assertGreater(
            metrics["source_detail_correlation_intended_background"], 0.9
        )
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

    def test_detail_metrics_reject_large_attachment_boundary_movement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, ::4] = 255
            Image.fromarray(image).save(path)
            baseline = np.zeros((64, 64), dtype=np.float32)
            face = np.zeros_like(baseline, dtype=bool)
            face[16:48, 16:48] = True
            candidate = baseline.copy()
            candidate[16, 16:48] = 0.15

            metrics = detail_sweep._detail_metrics(
                baseline, candidate, face, path, 0.60
            )

        self.assertFalse(metrics["checks"]["attachment_boundary"])
        self.assertFalse(metrics["checks"]["passed"])

    def test_partial_candidate_cannot_hide_full_background_correlation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, ::4] = 255
            Image.fromarray(image).save(path)
            signal = detail_sweep._photo_detail_signal(path, (64, 64))
            scale = float(np.percentile(np.abs(signal), 98.0))
            baseline = np.zeros((64, 64), dtype=np.float32)
            face = np.zeros_like(baseline, dtype=bool)
            face[16:48, 24:40] = True
            candidate = baseline.copy()
            candidate[:4] = 0.60 * np.clip(signal[:4] / scale, -1.0, 1.0)

            metrics = detail_sweep._detail_metrics(
                baseline, candidate, face, path, 0.60
            )

        self.assertGreater(metrics["active_source_detail_correlation"], 0.9)
        self.assertFalse(metrics["checks"]["source_aligned_capture"])
        self.assertFalse(metrics["checks"]["passed"])

    def test_intended_background_metric_excludes_declared_quiet_halo(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, ::4] = 255
            Image.fromarray(image).save(path)
            signal = detail_sweep._photo_detail_signal(path, (64, 64))
            scale = float(np.percentile(np.abs(signal), 98.0))
            baseline = np.zeros((64, 64), dtype=np.float32)
            face = np.zeros_like(baseline, dtype=bool)
            face[20:44, 24:40] = True
            feather = detail_sweep.gaussian_filter(
                detail_sweep.maximum_filter(face.astype(np.float32), size=17),
                sigma=5.0,
            )
            intended = (~face) & (feather < 0.5)
            candidate = baseline.copy()
            candidate[intended] = 0.60 * np.clip(
                signal[intended] / scale, -1.0, 1.0
            )

            metrics = detail_sweep._detail_metrics(
                baseline, candidate, face, path, 0.60, protection_halo_px=8.0
            )

        self.assertGreater(
            metrics["source_detail_correlation_intended_background"], 0.9
        )
        self.assertTrue(metrics["checks"]["intended_background_source_detail_correlation"])

    def test_run_rejects_invalid_level_matrix_before_loading_fixture(self):
        for levels in ((0.12, 0.30), (0.0, -0.1), (0.0, 0.3, 0.3)):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                detail_sweep.run("unused", detail_levels_mm=levels)


if __name__ == "__main__":
    unittest.main()
