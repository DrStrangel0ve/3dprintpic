import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.face_part_metrics import (
    face_part_affine_surface_error_metrics,
    face_part_cross_height_metrics,
    load_persisted_face_part_masks,
)
from backend.face_depth_refinement import FACE_PART_NAMES


class FacePartMetricsTest(unittest.TestCase):
    @staticmethod
    def _surface_and_masks():
        rows, cols = np.indices((81, 101), dtype=np.float32)
        face = (rows - 40.0) ** 2 / 30.0**2 + (cols - 50.0) ** 2 / 26.0**2 <= 1.0
        parts = {
            "left_eye": (rows - 34.0) ** 2 / 5.0**2 + (cols - 40.0) ** 2 / 8.0**2 <= 1.0,
            "right_eye": (rows - 34.0) ** 2 / 5.0**2 + (cols - 60.0) ** 2 / 8.0**2 <= 1.0,
            "nose": (rows - 43.0) ** 2 / 10.0**2 + (cols - 50.0) ** 2 / 6.0**2 <= 1.0,
            "mouth": (rows - 55.0) ** 2 / 5.0**2 + (cols - 50.0) ** 2 / 12.0**2 <= 1.0,
        }
        surface = (
            2.0
            + 0.02 * rows
            + 0.03 * cols
            + 3.5 * np.exp(-((rows - 42.0) ** 2 + (cols - 50.0) ** 2) / 210.0)
            + 0.35 * np.exp(-((rows - 55.0) ** 2 + (cols - 50.0) ** 2) / 20.0)
        ).astype(np.float32)
        return surface, face, parts

    def test_shared_face_normalization_accepts_global_scale_and_offset(self):
        reference, face, parts = self._surface_and_masks()
        candidate = 1.8 * reference + 5.0

        metrics = face_part_cross_height_metrics(
            reference,
            candidate,
            face,
            parts,
            sample_pitch_mm=0.4,
        )

        self.assertTrue(metrics["passed"])
        self.assertEqual(metrics["measured_part_count"], len(parts))
        self.assertTrue(all(record["passed"] for record in metrics["parts"]))
        self.assertLess(
            max(record["face_normalized_shape_rmse"] for record in metrics["parts"]),
            1e-5,
        )

    def test_local_flattening_and_oversharpening_fail_independent_part_gates(self):
        reference, face, parts = self._surface_and_masks()
        flattened = reference.copy()
        flattened[parts["mouth"]] = float(np.mean(reference[parts["mouth"]]))
        sharpened = reference.copy()
        sharpened[parts["nose"]] = (
            np.mean(reference[parts["nose"]])
            + 4.0
            * (reference[parts["nose"]] - np.mean(reference[parts["nose"]]))
        )

        flattened_metrics = face_part_cross_height_metrics(
            reference,
            flattened,
            face,
            parts,
            sample_pitch_mm=0.4,
        )
        sharpened_metrics = face_part_cross_height_metrics(
            reference,
            sharpened,
            face,
            parts,
            sample_pitch_mm=0.4,
        )

        self.assertFalse(flattened_metrics["passed"])
        self.assertIn("mouth", flattened_metrics["failed_parts"])
        self.assertFalse(sharpened_metrics["passed"])
        self.assertIn("nose", sharpened_metrics["failed_parts"])

    def test_nan_background_does_not_contaminate_part_derivatives(self):
        reference, face, parts = self._surface_and_masks()
        reference = reference.astype(np.float64)
        candidate = 1.4 * reference + 2.0
        reference[~face] = np.nan
        candidate[~face] = np.nan

        metrics = face_part_cross_height_metrics(
            reference,
            candidate,
            face,
            parts,
            sample_pitch_mm=0.4,
        )

        self.assertTrue(metrics["passed"])
        self.assertTrue(all(record["passed"] for record in metrics["parts"]))

    def test_physical_gradient_gate_tolerates_raw_raster_noise_but_keeps_floor(self):
        reference, face, parts = self._surface_and_masks()
        noise = np.random.default_rng(7).standard_normal(reference.shape)
        distribution_overrides = {
            "minimum_slope_q95_retention": 0.0,
            "maximum_slope_q95_retention": 1000.0,
            "minimum_curvature_q95_retention": 0.0,
            "maximum_curvature_q95_retention": 1000.0,
            "maximum_slope_wasserstein_ratio": 1000.0,
            "maximum_curvature_wasserstein_ratio": 1000.0,
        }
        mild = face_part_cross_height_metrics(
            reference,
            reference + 0.03 * noise,
            face,
            parts,
            sample_pitch_mm=0.4,
            gates=distribution_overrides,
        )
        severe = face_part_cross_height_metrics(
            reference,
            reference + 0.20 * noise,
            face,
            parts,
            sample_pitch_mm=0.4,
            gates=distribution_overrides,
        )

        self.assertTrue(mild["passed"])
        self.assertLess(
            min(part["minimum_raw_gradient_correlation"] for part in mild["parts"]),
            0.75,
        )
        self.assertGreater(
            min(part["minimum_gradient_correlation"] for part in mild["parts"]),
            0.99,
        )
        self.assertFalse(severe["passed"])
        self.assertTrue(
            any(
                not part["checks"]["raw_gradient_correlation"]
                for part in severe["parts"]
            )
        )

    def test_thin_visible_part_keeps_full_support_instead_of_biased_skeleton(self):
        reference, face, _parts = self._surface_and_masks()
        thin_nose = np.zeros_like(face)
        thin_nose[30:60, 49:52] = True

        metrics = face_part_cross_height_metrics(
            reference,
            reference.copy(),
            face,
            {"nose": thin_nose},
            sample_pitch_mm=0.4,
        )

        self.assertTrue(metrics["passed"])
        nose = metrics["parts"][0]
        self.assertFalse(nose["boundary_exclusion_applied"])
        self.assertEqual(nose["samples"], 90)
        self.assertEqual(nose["visible_samples_before_boundary_exclusion"], 90)
        self.assertEqual(
            nose["boundary_exclusion_reason"],
            "insufficient_retained_fraction",
        )
        self.assertEqual(nose["boundary_exclusion_eroded_samples"], 28)
        self.assertAlmostEqual(
            nose["boundary_exclusion_attempted_retained_fraction"],
            28.0 / 90.0,
        )
        self.assertEqual(nose["boundary_exclusion_retained_fraction"], 1.0)

        flattened = reference.copy()
        flattened[thin_nose] = float(np.mean(reference[thin_nose]))
        damaged = face_part_cross_height_metrics(
            reference,
            flattened,
            face,
            {"nose": thin_nose},
            sample_pitch_mm=0.4,
        )
        self.assertFalse(damaged["passed"])
        self.assertIn("nose", damaged["failed_parts"])
        damaged_affine = face_part_affine_surface_error_metrics(
            reference,
            flattened,
            face,
            {"nose": thin_nose},
        )
        self.assertFalse(damaged_affine["passed"])
        self.assertIn("nose", damaged_affine["failed_parts"])

        boundary_damaged = reference.copy()
        boundary_damaged[30:60, 49] += 2.0
        boundary_damaged[30:60, 51] += 2.0
        boundary_affine = face_part_affine_surface_error_metrics(
            reference,
            boundary_damaged,
            face,
            {"nose": thin_nose},
        )
        self.assertFalse(boundary_affine["passed"])
        self.assertIn("nose", boundary_affine["failed_parts"])

    def test_missing_part_pixel_and_flat_part_fail_closed(self):
        reference, face, parts = self._surface_and_masks()
        missing = reference.astype(np.float64)
        missing[40, 50] = np.nan
        missing_metrics = face_part_cross_height_metrics(
            reference,
            missing,
            face,
            parts,
            sample_pitch_mm=0.4,
        )

        flat_reference = np.zeros_like(reference)
        flat_reference[face & ~parts["mouth"]] = reference[face & ~parts["mouth"]]
        flat_metrics = face_part_cross_height_metrics(
            flat_reference,
            flat_reference.copy(),
            face,
            {"mouth": parts["mouth"]},
            sample_pitch_mm=0.4,
        )

        self.assertFalse(missing_metrics["passed"])
        self.assertIn("nose", missing_metrics["failed_parts"])
        self.assertFalse(flat_metrics["passed"])
        mouth = flat_metrics["parts"][0]
        self.assertIsNone(mouth["minimum_slope_q95_retention"])
        self.assertFalse(mouth["checks"]["slope_retention"])

    def test_empty_smoothing_configuration_fails_closed(self):
        reference, face, parts = self._surface_and_masks()

        metrics = face_part_cross_height_metrics(
            reference,
            reference.copy(),
            face,
            parts,
            sample_pitch_mm=0.4,
            smoothing_radii_mm=(),
        )

        self.assertFalse(metrics["available"])
        self.assertFalse(metrics["passed"])
        self.assertEqual(metrics["reason"], "invalid_smoothing_radii")

    def test_shared_face_affine_metric_reports_local_mm_error(self):
        reference, face, parts = self._surface_and_masks()
        candidate = 0.55 * reference + 4.0
        passing = face_part_affine_surface_error_metrics(
            reference,
            candidate,
            face,
            parts,
        )
        damaged = candidate.copy()
        damaged[parts["nose"]] += 2.0
        failing = face_part_affine_surface_error_metrics(
            reference,
            damaged,
            face,
            parts,
        )

        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["face_affine_fit"]["scale"], 0.55, places=6)
        self.assertLess(max(part["rmse_mm"] for part in passing["parts"]), 1e-5)
        self.assertFalse(failing["passed"])
        self.assertIn("nose", failing["failed_parts"])

    def test_affine_metric_shape_mismatch_keeps_failed_part_provenance(self):
        reference = np.zeros((16, 16), dtype=np.float64)
        candidate = np.zeros((15, 16), dtype=np.float64)
        face = np.ones((16, 16), dtype=bool)
        parts = {
            "nose": np.ones((16, 16), dtype=bool),
            "mouth": np.ones((16, 16), dtype=bool),
        }

        metrics = face_part_affine_surface_error_metrics(
            reference,
            candidate,
            face,
            parts,
        )

        self.assertFalse(metrics["available"])
        self.assertFalse(metrics["passed"])
        self.assertEqual(metrics["reason"], "shape_mismatch")
        self.assertEqual(metrics["failed_parts"], ["mouth", "nose"])
        self.assertIsNone(metrics["face_affine_fit"])

    def test_loader_resizes_and_flips_the_persisted_masks_once(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            part_dir = root / "face_refinement" / "face_00_parts"
            part_dir.mkdir(parents=True)
            face = np.zeros((20, 30), dtype=np.uint8)
            face[3:17, 4:26] = 255
            left_eye = np.zeros_like(face)
            left_eye[7:10, 7:12] = 255
            Image.fromarray(face).save(part_dir / "face.png")
            for name in FACE_PART_NAMES:
                Image.fromarray(left_eye).save(part_dir / f"{name}.png")
            metadata = {
                "faces": [
                    {
                        "index": 0,
                        "detector": "test",
                        "landmark_count": 478,
                        "part_masks": {
                            "complete": True,
                            "face_file": "face_refinement/face_00_parts/face.png",
                            "files": {
                                name: f"face_refinement/face_00_parts/{name}.png"
                                for name in FACE_PART_NAMES
                            },
                        },
                    }
                ]
            }
            metadata_path = root / "output_face_refinement_metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            records = load_persisted_face_part_masks(
                metadata_path,
                (40, 60),
                flip_horizontal=True,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["face_mask"].shape, (40, 60))
        self.assertTrue(records[0]["part_masks"]["left_eye"][16, 43])

    def test_loader_replays_the_exporter_resize_flip_and_crop_transform(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            part_dir = root / "parts"
            part_dir.mkdir()
            mask = np.zeros((12, 16), dtype=np.uint8)
            mask[3:8, 2:6] = 255
            Image.fromarray(mask).save(part_dir / "face.png")
            for name in FACE_PART_NAMES:
                Image.fromarray(mask).save(part_dir / f"{name}.png")
            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "faces": [
                            {
                                "index": 0,
                                "part_masks": {
                                    "complete": True,
                                    "face_file": "parts/face.png",
                                    "files": {
                                        name: f"parts/{name}.png"
                                        for name in FACE_PART_NAMES
                                    },
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            transform = {
                "input_depth_shape": [12, 16],
                "target_depth_shape": [6, 8],
                "flip_x": True,
                "mesh_shape_before_crop": [12, 16],
                "crop_bbox_rc": [2, 3, 11, 15],
                "emitted_shape": [9, 12],
            }

            records = load_persisted_face_part_masks(
                metadata_path,
                surface_grid_transform=transform,
            )

        self.assertEqual(records[0]["face_mask"].shape, (9, 12))
        self.assertTrue(np.any(records[0]["part_masks"]["nose"]))

    def test_loader_rejects_incomplete_parts_and_stale_input_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            mask = np.ones((12, 16), dtype=np.uint8) * 255
            Image.fromarray(mask).save(root / "face.png")
            Image.fromarray(mask).save(root / "nose.png")
            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "faces": [
                            {
                                "part_masks": {
                                    "complete": True,
                                    "face_file": "face.png",
                                    "files": {"nose": "nose.png"},
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly the required parts"):
                load_persisted_face_part_masks(metadata_path, (12, 16))

            for name in FACE_PART_NAMES:
                Image.fromarray(mask).save(root / f"{name}.png")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["faces"][0]["part_masks"]["files"] = {
                name: f"{name}.png" for name in FACE_PART_NAMES
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            stale_transform = {
                "input_depth_shape": [10, 14],
                "target_depth_shape": [6, 8],
                "flip_x": True,
                "mesh_shape_before_crop": [12, 16],
                "crop_bbox_rc": [0, 0, 12, 16],
                "emitted_shape": [12, 16],
            }
            with self.assertRaisesRegex(ValueError, "recorded input depth shape"):
                load_persisted_face_part_masks(
                    metadata_path,
                    surface_grid_transform=stale_transform,
                )


if __name__ == "__main__":
    unittest.main()
