import unittest

import numpy as np

from backend.benchmark.face_relief_local_metrics import (
    DERIVED_REGION_NAMES,
    derived_face_region_masks,
    globally_align_face_surface,
    localized_face_surface_metrics,
    transform_mask_to_emitted_grid,
)


class FaceReliefLocalMetricsTests(unittest.TestCase):
    @staticmethod
    def _surface_and_masks(size: int = 96):
        yy, xx = np.mgrid[:size, :size]
        face = ((xx - 48.0) / 34.0) ** 2 + ((yy - 48.0) / 42.0) ** 2 <= 1.0
        surface = (
            4.0
            + 0.025 * xx
            + 0.018 * yy
            + 4.0 * np.exp(-((xx - 48.0) ** 2 + (yy - 48.0) ** 2) / 150.0)
            + 0.7 * np.sin(xx / 5.0) * np.cos(yy / 7.0)
        ).astype(np.float64)

        def region(x0, y0, x1, y1):
            mask = np.zeros((size, size), dtype=bool)
            mask[y0:y1, x0:x1] = True
            return mask & face

        parts = {
            "left_eye": region(29, 35, 43, 44),
            "right_eye": region(53, 35, 67, 44),
            "left_eyebrow": region(28, 28, 44, 34),
            "right_eyebrow": region(52, 28, 68, 34),
            "nose": region(42, 40, 55, 62),
            "mouth": region(36, 65, 61, 75),
        }
        return surface, face, parts

    def test_global_alignment_uses_one_face_fit_and_enforces_orientation(self):
        reference, face, _parts = self._surface_and_masks()
        candidate = -2.0 * reference + 7.0

        aligned, record = globally_align_face_surface(
            reference,
            candidate,
            face,
            expected_scale_sign=-1.0,
        )

        np.testing.assert_allclose(aligned[face], reference[face], atol=1e-10)
        self.assertLess(record["scale"], 0.0)
        with self.assertRaisesRegex(ValueError, "orientation"):
            globally_align_face_surface(
                reference,
                candidate,
                face,
                expected_scale_sign=1.0,
            )

    def test_derived_regions_are_nonempty_and_remain_inside_face(self):
        _surface, face, parts = self._surface_and_masks()

        regions = derived_face_region_masks(face, parts, sample_pitch_mm=0.4)

        self.assertEqual(set(parts) | set(DERIVED_REGION_NAMES), set(regions))
        for name, mask in regions.items():
            self.assertGreater(np.count_nonzero(mask), 12, name)
            self.assertTrue(np.all(mask <= face), name)

    def test_mask_transform_matches_flip_resize_and_crop_contract(self):
        mask = np.zeros((2, 4), dtype=bool)
        mask[:, :2] = True
        transform = {
            "input_depth_shape": [2, 4],
            "target_depth_shape": [2, 4],
            "mesh_shape_before_crop": [4, 8],
            "crop_bbox_rc": [1, 2, 3, 6],
            "emitted_shape": [2, 4],
            "flip_x": True,
        }

        transformed = transform_mask_to_emitted_grid(mask, transform, (2, 4))

        self.assertFalse(np.any(transformed[:, :2]))
        self.assertTrue(np.all(transformed[:, 2:]))

    def test_perfect_surface_passes_and_local_nose_damage_is_named(self):
        reference, face, parts = self._surface_and_masks()
        perfect = localized_face_surface_metrics(
            reference,
            reference.copy(),
            face,
            parts,
            sample_pitch_mm=0.4,
        )
        damaged = reference.copy()
        damaged[parts["nose"]] = float(np.median(reference[face]))
        damaged_metrics = localized_face_surface_metrics(
            reference,
            damaged,
            face,
            parts,
            sample_pitch_mm=0.4,
        )

        self.assertTrue(perfect["checks"]["passed"])
        self.assertEqual(
            {record["name"] for record in perfect["cross_height"]["parts"]},
            set(parts),
        )
        self.assertFalse(
            perfect["derived_region_diagnostics"]["hard_gate"]
        )
        self.assertEqual(
            {record["name"] for record in perfect["derived_region_diagnostics"]["parts"]},
            set(DERIVED_REGION_NAMES),
        )
        failed_cross = set(perfect["cross_height"]["failed_parts"])
        self.assertFalse(failed_cross)
        self.assertFalse(damaged_metrics["checks"]["passed"])
        self.assertIn("nose", damaged_metrics["cross_height"]["failed_parts"])
        self.assertIn("nose", damaged_metrics["multi_light_appearance"]["failed_parts"])


if __name__ == "__main__":
    unittest.main()
