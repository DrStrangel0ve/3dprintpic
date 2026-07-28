import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from backend import face_relief_geometry
from backend.face_relief_geometry import align_face_to_scene_gradient_domain
from backend.pic_to_3d import (
    _expand_face_region_to_depth_connected_head,
    _face_translation_core_mask,
)


def _analytic_face(shape=(81, 81)):
    rows, cols = np.indices(shape, dtype=np.float64)
    x = (cols - 40.0) / 22.0
    y = (rows - 34.0) / 28.0
    head = np.square(x) + np.square(y) <= 1.0
    face = 0.48 * np.exp(-(np.square(x / 0.72) + np.square(y / 0.86)))
    face += 0.92 * np.exp(-(np.square(x / 0.17) + np.square((y - 0.02) / 0.27)))
    face -= 0.18 * np.exp(-(np.square((x - 0.30) / 0.14) + np.square((y + 0.20) / 0.09)))
    face -= 0.18 * np.exp(-(np.square((x + 0.30) / 0.14) + np.square((y + 0.20) / 0.09)))
    face += 0.16 * np.exp(-(np.square(x / 0.30) + np.square((y - 0.43) / 0.07)))
    return head, face


class FaceReliefGeometryTest(unittest.TestCase):
    def test_screened_attachment_matches_boundary_without_polynomial_face_warp(self):
        head, face = _analytic_face()
        rows, cols = np.indices(head.shape, dtype=np.float64)
        stabilized = 8.0 + face
        reference = 2.0 + 0.25 * face
        processed = 3.0 + 0.045 * cols + 0.0009 * np.square(rows - 38.0)

        aligned, stats = align_face_to_scene_gradient_domain(
            stabilized,
            reference,
            processed,
            head,
            screening_length_px=8.0,
        )

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["method"], "screened_poisson_shift")
        self.assertEqual(stats["solver_fallback_components"], 0)
        self.assertGreater(stats["hard_translation_core_pixels"], 0)
        self.assertLess(stats["boundary_residual_max_mm"], 1e-5)
        self.assertLess(stats["translation_aligned_shape_distortion_rmse_mm"], 0.55)
        self.assertLess(stats["gradient_distortion_p95_mm_per_px"], 0.20)
        np.testing.assert_array_equal(aligned[~head], processed[~head].astype(aligned.dtype))

    def test_screened_attachment_is_stable_across_large_scene_offsets(self):
        head, face = _analytic_face()
        reference = 4.0 + face
        face_core = head & (distance_transform_edt(head) >= 12.0)
        baseline = None
        baseline_span = None
        for scene_height in (20.0, 30.0, 40.0, 50.0):
            stabilized = scene_height * 0.47 + face
            processed = np.full(head.shape, scene_height * 0.16, dtype=np.float64)
            processed += np.linspace(0.0, scene_height * 0.12, head.shape[1])[None, :]
            aligned, stats = align_face_to_scene_gradient_domain(
                stabilized,
                reference,
                processed,
                head,
                screening_length_px=10.0,
                protected_core_mask=face_core,
                protected_face_mask=head,
                max_neighbor_step_mm=0.4,
            )
            self.assertEqual(stats["protected_core_pixels"], int(np.count_nonzero(face_core)))
            self.assertLess(stats["protected_core_translation_aligned_rmse_mm"], 1e-10)
            self.assertLess(stats["protected_core_gradient_distortion_p95_mm_per_px"], 1e-10)
            centered = aligned[face_core] - np.median(aligned[face_core])
            feature_span = float(np.ptp(centered))
            if baseline is None:
                baseline = centered
                baseline_span = feature_span
            else:
                np.testing.assert_allclose(centered, baseline, atol=1e-5)
                self.assertAlmostEqual(feature_span / baseline_span, 1.0, places=5)
            self.assertLess(stats["gradient_distortion_p95_mm_per_px"], 0.30)
            self.assertGreater(stats["soft_translation_support_pixels"], 0)
            self.assertLessEqual(
                stats["core_boundary_slope_ratio_max"],
                stats["max_core_boundary_slope_ratio"],
            )

    def test_protected_core_uses_component_boundary_shift_median(self):
        head, face = _analytic_face()
        rows, cols = np.indices(head.shape, dtype=np.float64)
        core = head & (distance_transform_edt(head) >= 12.0)
        stabilized = 8.0 + face
        reference = 3.0 + 0.25 * face
        processed = 1.5 + 0.02 * cols + 0.01 * rows
        boundary = head & ~binary_erosion(
            head,
            structure=np.ones((3, 3), dtype=bool),
            border_value=0,
        )
        expected_shift = float(np.median((processed - stabilized)[boundary]))

        aligned, stats = align_face_to_scene_gradient_domain(
            stabilized,
            reference,
            processed,
            head,
            protected_core_mask=core,
            protected_face_mask=head,
        )

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["core_anchor_source"], "boundary_shift_median")
        self.assertAlmostEqual(
            stats["protected_core_shift_median_mm"], expected_shift, places=6
        )
        np.testing.assert_allclose(
            aligned[core] - stabilized[core], expected_shift, atol=1e-6
        )

    def test_nonconverged_solver_fails_closed_to_processed_scene(self):
        head, face = _analytic_face()
        rows, cols = np.indices(head.shape, dtype=np.float64)
        stabilized = 9.0 + face
        reference = 3.0 + face
        processed = 2.0 + cols * 0.03 + rows * 0.01

        aligned, stats = align_face_to_scene_gradient_domain(
            stabilized,
            reference,
            processed,
            head,
            protected_core_mask=head & (distance_transform_edt(head) >= 10.0),
            solver_tolerance=1e-14,
            max_iterations=1,
        )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "solver_nonconvergence")
        self.assertGreater(stats["solver_fallback_components"], 0)
        np.testing.assert_array_equal(aligned, processed)

    def test_attachment_slope_gate_rejects_an_unsafe_transition(self):
        head, face = _analytic_face()
        stabilized = 8.0 + face
        reference = 2.0 + face
        processed = np.zeros(head.shape, dtype=np.float64)
        processed[:, head.shape[1] // 2 :] = 14.0

        aligned, stats = align_face_to_scene_gradient_domain(
            stabilized,
            reference,
            processed,
            head,
            protected_core_mask=head & (distance_transform_edt(head) >= 10.0),
            max_neighbor_step_mm=0.25,
            max_attachment_slope_ratio=1.1,
            max_attachment_slope_p99_ratio=1.05,
        )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "attachment_slope_gate")
        self.assertGreater(stats["slope_rejected_components"], 0)
        self.assertIn("attempted_attachment_slope_ratio_max", stats)
        np.testing.assert_array_equal(aligned, processed)

    def test_any_failed_face_rejects_the_whole_multi_face_alignment(self):
        region = np.zeros((54, 84), dtype=bool)
        region[8:32, 8:32] = True
        region[16:40, 52:76] = True
        core = np.zeros_like(region)
        core[14:26, 14:26] = True
        core[22:34, 58:70] = True
        stabilized = np.full(region.shape, 8.0, dtype=np.float64)
        reference = np.full(region.shape, 3.0, dtype=np.float64)
        processed = np.zeros(region.shape, dtype=np.float64)
        original_solver = face_relief_geometry._solve_screened_shift
        calls = 0

        def fail_second_component(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                return np.zeros(region.shape), 1, True, 0
            return original_solver(*args, **kwargs)

        with patch(
            "backend.face_relief_geometry._solve_screened_shift",
            side_effect=fail_second_component,
        ):
            aligned, stats = align_face_to_scene_gradient_domain(
                stabilized,
                reference,
                processed,
                region,
                protected_core_mask=core,
                protected_face_mask=region,
            )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["attempted_components"], 2)
        self.assertEqual(stats["aligned_components"], 1)
        self.assertEqual(stats["reason"], "solver_nonconvergence")
        np.testing.assert_array_equal(aligned, processed)

    def test_small_secondary_face_is_counted_and_fails_closed(self):
        region = np.zeros((48, 72), dtype=bool)
        region[8:32, 8:32] = True
        region[10:13, 58:61] = True
        stabilized = np.full(region.shape, 8.0, dtype=np.float64)
        reference = np.full(region.shape, 3.0, dtype=np.float64)
        processed = np.zeros(region.shape, dtype=np.float64)

        aligned, stats = align_face_to_scene_gradient_domain(
            stabilized,
            reference,
            processed,
            region,
        )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "partial_component_alignment")
        self.assertEqual(stats["attempted_components"], 2)
        self.assertEqual(stats["aligned_components"], 1)
        self.assertEqual(stats["skipped_small_components"], 1)
        np.testing.assert_array_equal(aligned, processed)

    def test_core_boundary_has_an_independent_slope_gate(self):
        head, face = _analytic_face()
        core = head & (distance_transform_edt(head) >= 10.0)
        stabilized = 8.0 + face
        reference = 2.0 + face
        processed = np.zeros(head.shape, dtype=np.float64)

        aligned, stats = align_face_to_scene_gradient_domain(
            stabilized,
            reference,
            processed,
            head,
            protected_core_mask=core,
            protected_face_mask=head,
            max_neighbor_step_mm=0.25,
            max_attachment_slope_ratio=100.0,
            max_attachment_slope_p99_ratio=100.0,
            max_core_boundary_slope_ratio=0.1,
            max_core_boundary_slope_p99_ratio=0.1,
        )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "attachment_slope_gate")
        self.assertEqual(stats["core_slope_rejected_components"], 1)
        np.testing.assert_array_equal(aligned, processed)

    def test_head_expansion_protects_a_connected_neck_without_taking_background(self):
        face = np.zeros((72, 72), dtype=bool)
        face[18:43, 22:50] = True
        reference = np.zeros(face.shape, dtype=np.float32)
        reference[face] = 6.0
        reference[10:18, 28:44] = 6.2
        reference[43:59, 29:43] = 5.4
        reference[43:59, :20] = 5.4

        head, stats = _expand_face_region_to_depth_connected_head(
            face,
            reference,
            depth_tolerance_mm=2.0,
        )

        self.assertTrue(stats["enabled"])
        self.assertGreater(stats["neck_added_pixels"], 0)
        self.assertTrue(head[48, 36])
        self.assertFalse(head[52, 12])
        self.assertFalse(np.any(head[63:]))

    def test_translation_core_keeps_inner_face_and_releases_oval_boundary(self):
        head, _face = _analytic_face()

        core, stats = _face_translation_core_mask(head, head.shape, margin_ratio=0.06)

        self.assertTrue(stats["enabled"])
        self.assertGreater(stats["core_pixels"], 0)
        self.assertLess(stats["core_pixels"], stats["face_pixels"])
        self.assertTrue(core[40, 40])
        self.assertFalse(np.any(core & ~head))
        self.assertFalse(np.any(core & ~binary_erosion(head)))

    def test_translation_core_is_built_for_every_face_component(self):
        faces = np.zeros((80, 100), dtype=bool)
        faces[8:54, 8:54] = True
        faces[58:74, 76:94] = True

        core, stats = _face_translation_core_mask(faces, faces.shape, margin_ratio=0.08)

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["component_count"], 2)
        self.assertEqual(stats["components_with_core"], 2)
        self.assertTrue(np.any(core[8:54, 8:54]))
        self.assertTrue(np.any(core[58:74, 76:94]))


if __name__ == "__main__":
    unittest.main()
