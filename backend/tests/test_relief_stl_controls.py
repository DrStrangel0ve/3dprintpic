import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.ndimage import laplace
from stl import mesh
from PIL import Image

from backend import pic_to_3d
from backend.pic_to_3d import (
    RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
    _load_depth_input_image,
    _flatten_border,
    _enhance_weighted_relief_features,
    _guard_weighted_feature_updates,
    _align_stabilized_head_to_reference_boundary,
    _attach_face_boundary_to_local_surface,
    _background_relief_preservation_metrics,
    _cap_selection_background_relief,
    _bridge_weighted_face_features,
    _compress_relief_gradients,
    _expand_face_region_to_depth_connected_head,
    _face_detail_preservation_metrics,
    _guard_face_detail_updates,
    _inject_photo_relief_detail,
    _limit_positive_relief_slope,
    _prepare_relief_for_printing,
    _restore_background_from_reference,
    _restore_stabilized_face_surface,
    _resize_nan_aware,
    _shape_relief_values,
    _stabilize_face_relief_height,
    _surface_lighting_agreement_metrics,
    _top_silhouette_mask,
    compose_selection_depth_with_context,
    depth_data_to_3d_model,
    relief_value_transform_for_model,
)


class ReliefStlControlsTest(unittest.TestCase):
    def test_depth_input_applies_exif_orientation_before_mask_alignment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "phone-photo.jpg"
            source = Image.new("RGB", (8, 4), color=(30, 60, 90))
            exif = source.getexif()
            exif[274] = 6
            source.save(source_path, exif=exif)

            oriented = _load_depth_input_image(source_path)

        self.assertEqual(oriented.size, (4, 8))

    def test_context_selection_depth_preserves_subject_support_and_background(self):
        rows, cols = np.indices((61, 81), dtype=np.float32)
        depth = 0.2 + 0.004 * cols + 0.08 * np.exp(
            -((rows - 34.0) ** 2 + (cols - 42.0) ** 2) / 90.0
        )
        selected = ((rows - 34.0) ** 2 / 180.0 + (cols - 42.0) ** 2 / 260.0) <= 1.0

        composed, stats = compose_selection_depth_with_context(
            depth,
            selected,
            relief_height_mm=30.0,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
        )

        np.testing.assert_array_equal(composed[selected], depth[selected])
        self.assertTrue(stats["enabled"])
        self.assertEqual(
            stats["method"],
            "full_scene_depth_with_bounded_background_context_v3",
        )
        self.assertGreater(stats["support_halo_pixels"], 0)
        self.assertLess(stats["base_canonical_value"], stats["selected_canonical_p01"])
        self.assertAlmostEqual(float(composed[0, 0]), stats["base_canonical_value"], places=6)
        self.assertGreater(float(np.max(composed[~selected])), float(composed[0, 0]))
        self.assertTrue(stats["background_context_enabled"])
        self.assertGreater(stats["background_context_pixels"], 0)
        self.assertGreater(stats["background_context_normalized_correlation"], 0.99)
        self.assertGreater(stats["background_context_normalized_rms_retention"], 0.99)
        self.assertGreater(stats["background_context_recoverable_coverage_ratio"], 0.5)
        self.assertLessEqual(
            stats["background_context_slope_guard"]["final_audit"][
                "accepted_surface_ratio_max"
            ],
            2.500001,
        )
        self.assertGreater(stats["background_output_span_ratio"], 0.2)
        self.assertGreater(float(composed[5, 75]), float(composed[5, 5]))

    def test_context_selection_depth_can_replay_legacy_flat_background(self):
        rows, cols = np.indices((61, 81), dtype=np.float32)
        depth = 0.2 + 0.004 * cols
        selected = ((rows - 30.0) ** 2 + (cols - 40.0) ** 2) <= 10.0**2

        composed, stats = compose_selection_depth_with_context(
            depth,
            selected,
            relief_height_mm=30.0,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            background_depth_ratio=0.0,
        )

        self.assertFalse(stats["background_context_enabled"])
        self.assertEqual(stats["background_context_pixels"], 0)
        self.assertAlmostEqual(float(composed[0, 0]), stats["base_canonical_value"], places=6)
        self.assertAlmostEqual(float(composed[0, -1]), stats["base_canonical_value"], places=6)

    def test_context_selection_depth_rejects_invalid_background_ratio(self):
        depth = np.ones((20, 20), dtype=np.float32)
        selected = np.zeros(depth.shape, dtype=bool)
        selected[5:15, 5:15] = True

        with self.assertRaisesRegex(ValueError, "supported range"):
            compose_selection_depth_with_context(
                depth,
                selected,
                background_depth_ratio=1.1,
            )

    def test_background_preservation_metric_rejects_flattened_scene_context(self):
        rows, cols = np.indices((80, 100), dtype=np.float32)
        foreground = ((rows - 47.0) ** 2 / 210.0 + (cols - 52.0) ** 2 / 330.0) <= 1.0
        reference = (
            2.0
            + 0.035 * cols
            + 0.9 * np.sin(cols / 9.0)
            + 0.5 * np.cos(rows / 7.0)
            + 4.0 * foreground
        ).astype(np.float32)

        preserved = _background_relief_preservation_metrics(
            reference,
            reference.copy(),
            foreground,
            sample_pitch_mm=0.4,
        )
        flattened = reference.copy()
        flattened[~foreground] = float(np.median(reference[~foreground]))
        rejected = _background_relief_preservation_metrics(
            reference,
            flattened,
            foreground,
            sample_pitch_mm=0.4,
        )

        self.assertTrue(preserved["passed"])
        self.assertAlmostEqual(preserved["correlation"], 1.0, places=6)
        self.assertAlmostEqual(preserved["span_retention"], 1.0, places=6)
        self.assertFalse(rejected["passed"])
        self.assertIn("correlation", rejected["quality_failures"])
        self.assertIn("span_retention", rejected["quality_failures"])
        self.assertIn("gradient_rms_retention", rejected["quality_failures"])

        flat_reference = np.full(reference.shape, 2.0, dtype=np.float32)
        flat_reference[foreground] = 6.0
        flat = _background_relief_preservation_metrics(
            flat_reference,
            flat_reference.copy(),
            foreground,
            sample_pitch_mm=0.4,
        )
        self.assertTrue(flat["passed"])
        self.assertEqual(flat["rms_retention"], 1.0)
        self.assertEqual(flat["span_retention"], 1.0)

        background = ~foreground
        amplified = reference.copy()
        background_mean = float(np.mean(reference[background]))
        amplified[background] = background_mean + 10.0 * (
            reference[background] - background_mean
        )
        amplification_metrics = _background_relief_preservation_metrics(
            reference,
            amplified,
            foreground,
            sample_pitch_mm=0.4,
        )
        self.assertFalse(amplification_metrics["passed"])
        self.assertIn("rms_retention", amplification_metrics["quality_failures"])
        self.assertIn("span_retention", amplification_metrics["quality_failures"])

        shifted = reference.copy()
        shifted[background] += 20.0
        shift_metrics = _background_relief_preservation_metrics(
            reference,
            shifted,
            foreground,
            sample_pitch_mm=0.4,
        )
        self.assertFalse(shift_metrics["passed"])
        self.assertIn("mean_shift", shift_metrics["quality_failures"])
        self.assertIn("boundary_jump", shift_metrics["quality_failures"])

        missing = reference.copy()
        missing[5:25, 5:25] = np.nan
        missing_metrics = _background_relief_preservation_metrics(
            reference,
            missing,
            foreground,
            sample_pitch_mm=0.4,
        )
        self.assertTrue(missing_metrics["available"])
        self.assertFalse(missing_metrics["passed"])
        self.assertIn("coverage", missing_metrics["quality_failures"])
        self.assertLess(missing_metrics["candidate_coverage_ratio"], 1.0)

        localized_reference = reference.copy()
        localized_bump = 2.5 * np.exp(
            -((rows - 24.0) ** 2 + (cols - 82.0) ** 2) / 28.0
        )
        localized_reference[background] += localized_bump[background]
        localized_loss = localized_reference.copy()
        localized_loss[background] -= localized_bump[background]
        localized_metrics = _background_relief_preservation_metrics(
            localized_reference,
            localized_loss,
            foreground,
            sample_pitch_mm=0.4,
        )
        self.assertGreater(localized_metrics["correlation"], 0.8)
        self.assertGreater(localized_metrics["gradient_correlation"], 0.58)
        self.assertFalse(localized_metrics["passed"])
        self.assertIn("localized_structure", localized_metrics["quality_failures"])
        self.assertGreater(
            localized_metrics["localized_structure"]["failed_window_count"],
            0,
        )

    def test_background_preservation_metric_reports_insufficient_context_unavailable(self):
        rows, cols = np.indices((40, 40), dtype=np.float32)
        foreground = np.ones((40, 40), dtype=bool)
        foreground[:3, :] = False
        reference = (2.0 + 0.02 * rows + 0.01 * cols).astype(np.float32)

        metrics = _background_relief_preservation_metrics(
            reference,
            reference.copy(),
            foreground,
            sample_pitch_mm=0.4,
        )

        self.assertFalse(metrics["available"])
        self.assertFalse(metrics["passed"])
        self.assertEqual(metrics["reason"], "insufficient_reference_background_samples")

    def test_surface_lighting_agreement_is_offset_invariant_and_detects_flattening(self):
        rows, cols = np.indices((61, 81), dtype=np.float32)
        region = np.ones((61, 81), dtype=bool)
        reference = (
            2.0
            + 0.03 * cols
            + 0.02 * rows
            + 3.5 * np.exp(-((rows - 29.0) ** 2 + (cols - 39.0) ** 2) / 95.0)
            + 0.8 * np.sin(cols / 6.0)
        ).astype(np.float32)

        shifted = _surface_lighting_agreement_metrics(
            reference,
            reference + 7.0,
            region,
            sample_pitch_mm=0.4,
        )
        flattened = _surface_lighting_agreement_metrics(
            reference,
            np.full(reference.shape, float(np.mean(reference)), dtype=np.float32),
            region,
            sample_pitch_mm=0.4,
        )

        self.assertTrue(shifted["available"])
        self.assertAlmostEqual(shifted["normal_mean_cosine"], 1.0, places=6)
        self.assertLess(shifted["normal_angle_p95_deg"], 1e-3)
        self.assertAlmostEqual(shifted["minimum_lighting_correlation"], 1.0, places=6)
        self.assertAlmostEqual(shifted["maximum_lighting_mae"], 0.0, places=6)
        self.assertTrue(flattened["available"])
        self.assertLess(flattened["normal_mean_cosine"], 0.95)
        self.assertGreater(flattened["normal_angle_p95_deg"], 20.0)
        self.assertLess(flattened["minimum_lighting_correlation"], 0.2)
        self.assertGreater(flattened["maximum_lighting_mae"], 0.1)

    def test_surface_lighting_agreement_reports_missing_pixels_per_component(self):
        rows, cols = np.indices((81, 101), dtype=np.float32)
        first = (rows - 40.0) ** 2 + (cols - 29.0) ** 2 <= 15.0**2
        second = (rows - 40.0) ** 2 + (cols - 73.0) ** 2 <= 12.0**2
        region = first | second
        reference = (
            2.0
            + 2.5 * np.exp(-((rows - 40.0) ** 2 + (cols - 29.0) ** 2) / 70.0)
            + 2.0 * np.exp(-((rows - 40.0) ** 2 + (cols - 73.0) ** 2) / 55.0)
        ).astype(np.float32)
        candidate = reference.copy()
        candidate[second] = np.nan

        metrics = _surface_lighting_agreement_metrics(
            reference,
            candidate,
            region,
            sample_pitch_mm=0.4,
            component_metrics=True,
        )

        self.assertTrue(metrics["available"])
        self.assertLess(metrics["candidate_coverage_ratio"], 1.0)
        self.assertGreater(metrics["missing_candidate_samples"], 0)
        self.assertEqual(metrics["component_count"], 2)
        self.assertEqual(metrics["measured_component_count"], 1)
        self.assertEqual(metrics["unavailable_component_count"], 1)
        self.assertTrue(metrics["components"][0]["available"])
        self.assertFalse(metrics["components"][1]["available"])

    def test_background_preservation_unavailable_blocks_stl_emission(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            rows, cols = np.indices((40, 40), dtype=np.float32)
            selected = np.ones((40, 40), dtype=bool)
            selected[:3, :] = False
            depth = 0.2 + 0.006 * cols + 0.08 * np.exp(
                -((rows - 21.0) ** 2 + (cols - 20.0) ** 2) / 70.0
            )
            depth_path = root / "depth.npy"
            stl_path = root / "unavailable.stl"
            np.save(depth_path, depth.astype(np.float32))

            with self.assertRaisesRegex(ValueError, "telemetry is unavailable"):
                depth_data_to_3d_model(
                    depth_path,
                    output_stl_path=str(stl_path),
                    target_dimension=-1,
                    z_scale=30.0,
                    max_xy_size=15.6,
                    sigma=0.0,
                    relief_gamma=1.0,
                    detail_boost=0.0,
                    low_percentile=0.0,
                    high_percentile=100.0,
                    base_border_px=0,
                    value_transform="linear",
                    minimum_feature_mm=0.8,
                    max_relief_slope=2.0,
                    selection_region_mask=selected,
                    selection_background_depth_ratio=0.45,
                )

            self.assertFalse(stl_path.exists())

    def test_selection_background_physical_cap_preserves_support_and_limits_far_context(self):
        values = np.full((61, 81), 20.0, dtype=np.float32)
        values[0, :] = 0.0
        values[-1, :] = 0.0
        values[:, 0] = 0.0
        values[:, -1] = 0.0
        selected = np.zeros(values.shape, dtype=bool)
        selected[25:36, 35:46] = True
        values[selected] = 27.0

        capped, stats = _cap_selection_background_relief(
            values,
            selected,
            relief_height_mm=30.0,
            sample_pitch_mm=0.5,
            max_slope_mm_per_mm=2.0,
            background_depth_ratio=0.45,
        )

        np.testing.assert_array_equal(capped[selected], values[selected])
        self.assertTrue(stats["passed"])
        self.assertGreater(stats["affected_pixels"], 0)
        self.assertLessEqual(stats["far_background_max_mm"], 13.5 + 1e-5)
        self.assertGreater(float(capped[24, 40]), stats["far_background_ceiling_mm"])
        self.assertAlmostEqual(float(capped[5, 5]), 13.5, places=5)
        self.assertLessEqual(
            stats["attachment_jump_max_mm"],
            stats["attachment_step_limit_mm"] + 1e-5,
        )

        low_subject = values.copy()
        low_subject[selected] = 5.0
        low_capped, low_stats = _cap_selection_background_relief(
            low_subject,
            selected,
            relief_height_mm=30.0,
            sample_pitch_mm=0.5,
            max_slope_mm_per_mm=2.0,
            background_depth_ratio=0.45,
        )
        self.assertTrue(low_stats["passed"])
        self.assertAlmostEqual(float(low_capped[24, 40]), 6.0, places=5)
        self.assertAlmostEqual(float(low_capped[5, 5]), 13.5, places=5)
        self.assertEqual(low_stats["attachment_slope_violation_mm"], 0.0)

    def test_background_reference_fallback_cannot_steepen_accepted_surface(self):
        candidate = np.full((61, 81), 5.0, dtype=np.float32)
        reference = candidate.copy()
        foreground = np.zeros(candidate.shape, dtype=bool)
        foreground[25:36, 35:46] = True
        reference[~foreground] = 20.0

        restored, stats = _restore_background_from_reference(
            candidate,
            reference,
            foreground,
            sample_pitch_mm=1.0,
            feather_mm=1.5,
            max_neighbor_step_mm=1.0,
        )

        horizontal = np.abs(restored[:, 1:] - restored[:, :-1])
        vertical = np.abs(restored[1:, :] - restored[:-1, :])
        self.assertTrue(stats["slope_guard"]["enabled"])
        self.assertLessEqual(float(np.max(horizontal)), 1.0001)
        self.assertLessEqual(float(np.max(vertical)), 1.0001)
        self.assertLessEqual(
            stats["slope_guard"]["final_audit"]["accepted_surface_ratio_max"],
            1.0001,
        )

    def test_background_cap_reports_incompatible_subject_boundary_constraints(self):
        values = np.full((21, 21), 20.0, dtype=np.float32)
        values[[0, -1], :] = 0.0
        values[:, [0, -1]] = 0.0
        selected = np.zeros(values.shape, dtype=bool)
        selected[10, 9] = True
        selected[9, 10] = True
        selected[10, 11] = True
        selected[11, 10] = True
        values[10, 9] = 5.0
        values[9, 10] = 10.0
        values[10, 11] = 5.0
        values[11, 10] = 10.0

        _, stats = _cap_selection_background_relief(
            values,
            selected,
            relief_height_mm=30.0,
            sample_pitch_mm=0.5,
            max_slope_mm_per_mm=2.0,
            background_depth_ratio=0.45,
        )

        self.assertTrue(stats["far_background_cap_passed"])
        self.assertTrue(stats["feasible_attachment_constraints_passed"])
        self.assertTrue(stats["emission_passed"])
        self.assertFalse(stats["attachment_constraints_passed"])
        self.assertFalse(stats["passed"])
        self.assertGreater(stats["attachment_constraint_conflicts"], 0)

    def test_context_selection_depth_supports_metric_far_high_values(self):
        depth = np.full((31, 31), 8.0, dtype=np.float32)
        selected = np.zeros(depth.shape, dtype=bool)
        selected[9:22, 9:22] = True
        depth[selected] = np.linspace(2.0, 4.0, np.count_nonzero(selected), dtype=np.float32)

        composed, stats = compose_selection_depth_with_context(
            depth,
            selected,
            value_transform=RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
            relief_height_mm=20.0,
            sample_pitch_mm=0.5,
            max_slope_mm_per_mm=2.0,
        )

        np.testing.assert_allclose(composed[selected], depth[selected], rtol=1e-6)
        self.assertTrue(np.all(np.isfinite(composed)))
        self.assertGreater(float(composed[0, 0]), float(depth[9, 9]))
        self.assertEqual(stats["value_transform"], RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH)

    def test_context_selection_depth_ignores_nonpositive_inverse_depth_samples(self):
        depth = np.full((31, 31), 8.0, dtype=np.float32)
        selected = np.zeros(depth.shape, dtype=bool)
        selected[9:22, 9:22] = True
        depth[selected] = 3.0
        depth[15, 15] = 0.0
        depth[:13, :13] = 0.0

        composed, stats = compose_selection_depth_with_context(
            depth,
            selected,
            value_transform=RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
            relief_height_mm=20.0,
            sample_pitch_mm=0.5,
            max_slope_mm_per_mm=2.0,
        )

        self.assertTrue(np.all(np.isfinite(composed)))
        self.assertGreater(float(composed[15, 15]), 0.0)
        self.assertEqual(stats["mask_pixels"], int(np.count_nonzero(selected & (depth > 0))))

    def test_context_selection_depth_rejects_misaligned_aspect_ratio(self):
        depth = np.ones((40, 80), dtype=np.float32)
        mismatched_mask = np.ones((80, 40), dtype=bool)

        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            compose_selection_depth_with_context(depth, mismatched_mask)

    def test_face_detail_metrics_catch_one_flattened_face(self):
        rows, cols = np.indices((64, 96), dtype=np.float32)
        reference = 0.01 * rows + 0.02 * cols
        first = ((rows - 30.0) ** 2 + (cols - 25.0) ** 2) <= 13.0**2
        second = ((rows - 30.0) ** 2 + (cols - 70.0) ** 2) <= 13.0**2
        reference += first * (0.8 * np.sin(cols * 0.55))
        reference += second * (0.8 * np.sin(cols * 0.55))
        candidate = reference.copy()
        candidate[first] = 0.01 * rows[first] + 0.02 * cols[first]

        metrics = _face_detail_preservation_metrics(reference, candidate, first | second)

        self.assertTrue(metrics["available"])
        self.assertEqual(len(metrics["components"]), 2)
        self.assertLess(metrics["minimum_component_rms_retention"], 0.1)
        self.assertGreater(metrics["components"][1]["rms_retention"], 0.99)

    def test_face_detail_guard_attenuates_a_component_regression(self):
        rows, cols = np.indices((48, 80), dtype=np.float32)
        face = np.zeros((48, 80), dtype=bool)
        face[10:38, 10:34] = True
        face[10:38, 46:70] = True
        reference = 0.03 * rows + 0.02 * cols + face * (0.4 * np.sin(cols * 0.7))
        baseline = reference.copy()
        candidate = baseline.copy()
        candidate[10:38, 10:34] += 1.2 * np.cos(cols[10:38, 10:34] * 1.9)

        guarded, stats = _guard_face_detail_updates(
            baseline,
            candidate,
            reference,
            face,
            minimum_correlation=0.8,
            minimum_rms_retention=0.6,
        )

        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["attenuated"])
        self.assertLess(stats["applied_scale"], 1.0)
        self.assertGreaterEqual(stats["final"]["minimum_component_correlation"], 0.8)
        np.testing.assert_array_equal(guarded[~face], baseline[~face])

    def test_flat_face_detail_violation_is_json_safe_and_fails_closed(self):
        reference = np.zeros((40, 40), dtype=np.float32)
        face = np.zeros(reference.shape, dtype=bool)
        face[8:32, 8:32] = True
        candidate = reference.copy()
        candidate[14:26, 14:26] = 0.4

        metrics = _face_detail_preservation_metrics(reference, candidate, face)
        guarded, guard_stats = _guard_face_detail_updates(
            reference,
            candidate,
            reference,
            face,
        )

        self.assertTrue(metrics["flat_reference_violation"])
        self.assertIsNone(metrics["rms_retention"])
        self.assertIsNone(metrics["components"][0]["rms_retention"])
        json.dumps(metrics, allow_nan=False)
        np.testing.assert_array_equal(guarded, reference)
        self.assertEqual(guard_stats["applied_scale"], 0.0)

    def test_face_detail_metrics_fail_closed_for_tiny_detected_component(self):
        rows, cols = np.indices((48, 64), dtype=np.float32)
        reference = 0.02 * rows + 0.03 * cols + 0.2 * np.sin(cols * 0.4)
        face = np.zeros(reference.shape, dtype=bool)
        face[8:36, 8:34] = True
        face[42:44, 58:60] = True

        metrics = _face_detail_preservation_metrics(reference, reference, face)

        self.assertFalse(metrics["available"])
        self.assertEqual(metrics["component_count"], 2)
        self.assertEqual(metrics["unavailable_component_count"], 1)

    def test_save_depth_outputs_writes_normalized_depth_preview_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            depth_path = pic_to_3d._save_depth_outputs(
                np.array([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32),
                output_dir,
                metadata={
                    "provider": "transformers",
                    "requested_model": "apple/DepthPro-hf",
                    "effective_model": "depth-anything/Depth-Anything-V2-Large-hf",
                    "fallback_reason": "fallback",
                },
            )

            depth = np.load(depth_path)
            metadata = json.loads((output_dir / "output_depth_metadata.json").read_text(encoding="utf-8"))
            preview_exists = (output_dir / "output_depth_preview.png").exists()

        self.assertAlmostEqual(float(depth.min()), 0.0)
        self.assertAlmostEqual(float(depth.max()), 1.0)
        self.assertTrue(preview_exists)
        self.assertEqual(metadata["requested_model"], "apple/DepthPro-hf")
        self.assertEqual(metadata["effective_model"], "depth-anything/Depth-Anything-V2-Large-hf")
        self.assertTrue(metadata["stored_depth_normalized"])

    def test_save_depth_outputs_preserves_metric_depth_for_inverse_relief(self):
        source = np.array([[1.0, 2.0], [10.0, 100.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            depth_path = pic_to_3d._save_depth_outputs(
                source,
                output_dir,
                metadata={"effective_model": "apple/DepthPro-hf"},
                normalize_depth=False,
                preview_value_transform=RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
            )

            saved = np.load(depth_path)
            metadata = json.loads((output_dir / "output_depth_metadata.json").read_text(encoding="utf-8"))
            relief_preview_exists = (output_dir / "output_relief_preview.png").exists()

        np.testing.assert_array_equal(saved, source)
        self.assertFalse(metadata["stored_depth_normalized"])
        self.assertEqual(metadata["relief_value_transform"], RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH)
        self.assertEqual(metadata["relief_preview"], "output_relief_preview.png")
        self.assertTrue(relief_preview_exists)

    def test_depth_fallback_preserves_requested_model_and_reason(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(pic_to_3d, "process_image_get_depth_data_transformers", return_value="depth.npy") as fallback:
                result = pic_to_3d._run_depth_fallback(
                    "input.png",
                    tmp_dir,
                    "apple/DepthPro-hf",
                    "cpu",
                    "Depth Pro unavailable",
                )

        self.assertEqual(result, "depth.npy")
        fallback.assert_called_once_with(
            "input.png",
            output_dir=tmp_dir,
            model_name=pic_to_3d.DEFAULT_DEPTH_FALLBACK_MODEL,
            device="cpu",
            requested_model_name="apple/DepthPro-hf",
            fallback_reason="Depth Pro unavailable",
        )

    def test_depth_fallback_rejects_recursive_fallback_model(self):
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.dict(os.environ, {"DEPTH_FALLBACK_MODEL": "apple/DepthPro-hf"}),
            patch.object(pic_to_3d, "process_image_get_depth_data_transformers") as fallback,
        ):
            with self.assertRaisesRegex(RuntimeError, "Depth Pro unavailable"):
                pic_to_3d._run_depth_fallback(
                    "input.png",
                    tmp_dir,
                    "apple/DepthPro-hf",
                    "cpu",
                    "Depth Pro unavailable",
                )

        fallback.assert_not_called()

    def test_detail_boost_lifts_local_features(self):
        base = np.tile(np.linspace(0.2, 0.8, 41, dtype=np.float32), (41, 1))
        base[20, 20] += 0.08

        plain = _shape_relief_values(base, detail_boost=0, detail_radius=2, low_percentile=0, high_percentile=100)
        boosted = _shape_relief_values(base, detail_boost=2.0, detail_radius=2, low_percentile=0, high_percentile=100)

        self.assertGreater(boosted[20, 20] - boosted[20, 19], plain[20, 20] - plain[20, 19])

    def test_detail_boost_does_not_create_a_halo_at_depth_steps(self):
        step = np.full((41, 41), 0.2, dtype=np.float32)
        step[:, 21:] = 0.8

        plain = _shape_relief_values(
            step,
            gamma=1.0,
            detail_boost=0,
            low_percentile=0,
            high_percentile=100,
        )
        boosted = _shape_relief_values(
            step,
            gamma=1.0,
            detail_boost=2.0,
            detail_radius=2.0,
            detail_edge_threshold=0.12,
            low_percentile=0,
            high_percentile=100,
        )

        edge_change = np.max(np.abs(boosted[:, 19:23] - plain[:, 19:23]))
        self.assertLess(edge_change, 0.01)

    def test_detail_boost_does_not_raise_soft_depth_boundary_shoulders(self):
        soft_step = np.full((41, 41), 0.2, dtype=np.float32)
        soft_step[:, 18] = 0.23
        soft_step[:, 19] = 0.30
        soft_step[:, 20] = 0.45
        soft_step[:, 21] = 0.65
        soft_step[:, 22:] = 0.8

        plain = _shape_relief_values(
            soft_step,
            gamma=1.0,
            detail_boost=0,
            low_percentile=0,
            high_percentile=100,
        )
        boosted = _shape_relief_values(
            soft_step,
            gamma=1.0,
            detail_boost=3.0,
            detail_radius=2.0,
            detail_edge_threshold=0.12,
            low_percentile=0,
            high_percentile=100,
        )

        edge_change = np.max(np.abs(boosted[:, 14:28] - plain[:, 14:28]))
        self.assertLess(edge_change, 0.01)

    def test_background_detail_boost_respects_face_protection_mask(self):
        yy, xx = np.indices((64, 64), dtype=np.float32)
        values = 0.25 + xx * 0.006
        values += 0.025 * np.sin(xx * 0.9) * np.cos(yy * 0.7)
        face_region = np.zeros_like(values, dtype=bool)
        face_region[:, 32:] = True

        baseline = _shape_relief_values(
            values,
            gamma=1.0,
            detail_boost=0.8,
            detail_radius=1.5,
            detail_edge_threshold=1.0,
            low_percentile=0,
            high_percentile=100,
            detail_protection_mask=face_region,
            background_detail_boost=1.0,
        )
        boosted = _shape_relief_values(
            values,
            gamma=1.0,
            detail_boost=0.8,
            detail_radius=1.5,
            detail_edge_threshold=1.0,
            low_percentile=0,
            high_percentile=100,
            detail_protection_mask=face_region,
            background_detail_boost=2.0,
        )

        change = np.abs(boosted - baseline)
        background_change = float(np.mean(change[:, 4:24]))
        protected_change = float(np.mean(change[:, 40:60]))
        self.assertGreater(background_change, protected_change * 8)

    def test_photo_detail_fusion_restores_background_without_touching_faces(self):
        relief = np.tile(np.linspace(0.15, 0.85, 64, dtype=np.float32), (64, 1))
        photo = np.zeros((64, 64, 3), dtype=np.uint8)
        photo[:, ::4] = 255
        face_region = np.zeros_like(relief, dtype=bool)
        face_region[:, 40:] = True

        fused, stats = _inject_photo_relief_detail(
            relief,
            photo,
            max_detail_ratio=0.02,
            protection_mask=face_region,
        )

        self.assertTrue(stats["enabled"])
        self.assertGreater(float(np.mean(np.abs(fused[:, :24] - relief[:, :24]))), 0.002)
        np.testing.assert_array_equal(fused[face_region], relief[face_region])

    def test_top_silhouette_mask_removes_only_pixels_above_content(self):
        source = np.full((20, 24, 3), 255, dtype=np.uint8)
        source[8:, :12] = 30
        source[4:, 12:] = 30

        silhouette, stats = _top_silhouette_mask(source, source.shape[:2], padding_px=0)

        self.assertTrue(stats["enabled"])
        # The helper flips horizontally to match the STL coordinate system.
        skyline_rows = np.argmax(silhouette, axis=0)
        np.testing.assert_array_equal(skyline_rows[:13], np.full(13, 3))
        np.testing.assert_array_equal(skyline_rows[13:], np.full(11, 7))

    def test_weighted_feature_depth_is_signed_bounded_and_masked(self):
        yy, xx = np.indices((48, 48), dtype=np.float32)
        values = 5.0 + 0.3 * np.sin(xx * 0.8) * np.cos(yy * 0.7)
        weights = np.zeros_like(values, dtype=np.float32)
        weights[12:36, 12:36] = 1.0

        enhanced, stats = _enhance_weighted_relief_features(
            values,
            weights,
            max_feature_depth_mm=0.4,
        )

        change = enhanced - values
        self.assertTrue(stats["enabled"])
        self.assertGreater(float(np.max(change)), 0.1)
        self.assertLess(float(np.min(change)), -0.1)
        self.assertLessEqual(float(np.max(np.abs(change))), 0.40001)
        np.testing.assert_array_equal(change[weights == 0], np.zeros(np.count_nonzero(weights == 0)))

    def test_feature_bridge_fills_peripheral_gaps_but_preserves_central_holes(self):
        values = np.full((40, 40), 5.0, dtype=np.float32)
        region = np.zeros_like(values, dtype=bool)
        region[6:34, 6:34] = True
        weights = np.zeros_like(values, dtype=np.float32)
        weights[15:25, 6:12] = 1.0
        weights[15:25, 28:34] = 1.0
        values[18:22, 7:11] = 3.0
        values[18:22, 29:33] = 3.0
        values[18:22, 18:22] = 3.0

        bridged, stats = _bridge_weighted_face_features(
            values,
            region,
            weights,
            max_bridge_depth_mm=0.8,
        )

        self.assertTrue(stats["enabled"])
        self.assertGreater(stats["bridged_pixels"], 0)
        self.assertGreater(float(np.mean(bridged[18:22, 7:11])), 3.1)
        self.assertGreater(float(np.mean(bridged[18:22, 29:33])), 3.1)
        np.testing.assert_array_equal(bridged[18:22, 18:22], values[18:22, 18:22])
        np.testing.assert_array_equal(bridged[~region], values[~region])

    def test_accessory_exclusion_blocks_feature_enhancement_and_bridge_after_expansion(self):
        yy, xx = np.indices((40, 40), dtype=np.float32)
        values = 5.0 + 0.3 * np.sin(xx * 0.8) * np.cos(yy * 0.7)
        weights = np.ones_like(values, dtype=np.float32)
        exclusion = np.zeros_like(values, dtype=np.uint8)
        exclusion[13:27, 13:27] = 255
        flipped_exclusion = np.flip(exclusion > 0, axis=1)

        enhanced, enhancement_stats = _enhance_weighted_relief_features(
            values,
            weights,
            max_feature_depth_mm=0.4,
            feature_exclusion_mask=exclusion,
        )

        enhancement_change = enhanced - values
        self.assertTrue(enhancement_stats["enabled"])
        np.testing.assert_array_equal(
            enhancement_change[flipped_exclusion],
            np.zeros(np.count_nonzero(flipped_exclusion)),
        )
        self.assertGreater(float(np.max(np.abs(enhancement_change[~flipped_exclusion]))), 0.1)

        bridge_values = np.full((40, 40), 5.0, dtype=np.float32)
        region = np.zeros_like(bridge_values, dtype=bool)
        region[6:34, 6:34] = True
        bridge_weights = np.zeros_like(bridge_values, dtype=np.float32)
        bridge_weights[15:25, 6:12] = 1.0
        bridge_weights[15:25, 28:34] = 1.0
        bridge_values[18:22, 7:11] = 3.0
        bridge_values[18:22, 29:33] = 3.0
        bridge_exclusion = np.zeros_like(bridge_values, dtype=np.uint8)
        bridge_exclusion[18:22, 29:33] = 255

        bridged, bridge_stats = _bridge_weighted_face_features(
            bridge_values,
            region,
            bridge_weights,
            max_bridge_depth_mm=0.8,
            feature_exclusion_mask=bridge_exclusion,
        )

        self.assertTrue(bridge_stats["enabled"])
        np.testing.assert_array_equal(bridged[18:22, 7:11], bridge_values[18:22, 7:11])
        self.assertGreater(float(np.mean(bridged[18:22, 29:33])), 3.1)

    def test_unreadable_accessory_exclusion_fails_feature_enhancement_closed(self):
        values = np.linspace(4.0, 6.0, 24 * 24, dtype=np.float32).reshape(24, 24)
        weights = np.ones_like(values)

        enhanced, stats = _enhance_weighted_relief_features(
            values,
            weights,
            max_feature_depth_mm=0.4,
            feature_exclusion_mask="missing-accessory-exclusion.png",
        )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "feature_exclusion_unreadable")
        np.testing.assert_array_equal(enhanced, values)

    def test_feature_guard_attenuates_new_horizontal_vertical_and_diagonal_cliffs(self):
        baseline = np.zeros((24, 24), dtype=np.float32)
        candidate = baseline.copy()
        candidate[12, 12] = 1.0

        guarded, stats = _guard_weighted_feature_updates(
            baseline,
            candidate,
            max_neighbor_step_mm=0.2,
        )

        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["attenuated"])
        self.assertGreater(stats["applied_scale"], 0.0)
        self.assertLess(stats["applied_scale"], 0.25)
        self.assertGreater(
            stats["candidate_audit"]["accepted_surface_ratio_max"],
            stats["max_ratio"],
        )
        self.assertLessEqual(
            stats["final_audit"]["accepted_surface_ratio_max"],
            stats["max_ratio"] + 1e-6,
        )
        self.assertGreater(float(guarded[12, 12]), 0.0)
        self.assertLess(float(guarded[12, 12]), 0.21)

    def test_feature_guard_preserves_an_already_safe_update(self):
        rows, cols = np.indices((20, 20), dtype=np.float32)
        baseline = rows * 0.02
        candidate = baseline + cols * 0.03

        guarded, stats = _guard_weighted_feature_updates(
            baseline,
            candidate,
            max_neighbor_step_mm=0.2,
        )

        self.assertTrue(stats["enabled"])
        self.assertFalse(stats["attenuated"])
        self.assertEqual(stats["applied_scale"], 1.0)
        np.testing.assert_array_equal(guarded, candidate)

    def test_feature_guard_uses_safe_baseline_when_projection_does_not_converge(self):
        rng = np.random.default_rng(0)
        baseline = (rng.normal(size=(7, 7)) * 2.0).astype(np.float32)
        candidate = (baseline + rng.normal(size=(7, 7)) * 10.0).astype(np.float32)

        guarded, stats = _guard_weighted_feature_updates(
            baseline,
            candidate,
            max_neighbor_step_mm=1.0,
            max_ratio=1.0,
        )

        self.assertLessEqual(
            stats["final_audit"]["accepted_surface_ratio_max"],
            1.0 + 1e-6,
        )
        if stats["fell_back_to_baseline"]:
            np.testing.assert_array_equal(guarded, baseline)

    def test_face_boundary_attachment_preserves_detail_and_does_not_raise_background(self):
        values = np.full((81, 81), 2.0, dtype=np.float32)
        region = np.zeros_like(values, dtype=bool)
        region[20:61, 20:61] = True
        values[region] = 12.0
        values[39:42, 39:42] = 12.6

        attached, stats = _attach_face_boundary_to_local_surface(
            values,
            region,
            max_separation_mm=1.0,
            sample_pitch_mm=0.4,
        )

        self.assertTrue(stats["enabled"])
        self.assertGreater(stats["adjusted_boundary_pixels"], 0)
        self.assertLessEqual(float(np.max(attached[20, 20:61] - values[19, 20:61])), 1.00001)
        np.testing.assert_array_equal(attached[~region], values[~region])
        neighboring_surface = float((attached[40, 38] + attached[40, 42]) * 0.5)
        self.assertGreater(float(attached[40, 40]) - neighboring_surface, 0.5)

    def test_face_relief_shape_stays_in_reference_range_at_large_scene_heights(self):
        values = np.full((41, 41), 3.0, dtype=np.float32)
        region = np.zeros_like(values, dtype=bool)
        region[10:31, 10:31] = True
        values[region] = 20.0
        values[18:23, 18:23] = 30.0

        stabilized, stats = _stabilize_face_relief_height(
            values,
            region,
            relief_height_mm=50.0,
            reference_face_height_mm=10.0,
        )

        self.assertTrue(stats["enabled"])
        self.assertAlmostEqual(stats["shape_scale"], 0.2)
        np.testing.assert_array_equal(stabilized[~region], values[~region])
        self.assertAlmostEqual(float(np.median(stabilized[region])), 20.0, places=4)
        self.assertAlmostEqual(float(stabilized[20, 20]), 22.0, places=4)

    def test_head_stabilization_region_adds_connected_hair_without_background(self):
        face = np.zeros((48, 48), dtype=bool)
        face[14:34, 13:35] = True
        reference = np.zeros(face.shape, dtype=np.float32)
        reference[face] = 6.0
        reference[8:14, 18:30] = 6.0

        head, stats = _expand_face_region_to_depth_connected_head(
            face,
            reference,
            depth_tolerance_mm=2.0,
        )

        self.assertTrue(stats["enabled"])
        self.assertTrue(np.any(head[8:14, 18:30]))
        self.assertFalse(np.any(head[:5]))
        self.assertGreater(stats["head_pixels"], stats["face_pixels"])

    def test_reference_boundary_transfer_meets_the_processed_scene(self):
        head = np.zeros((45, 45), dtype=bool)
        head[10:35, 10:35] = True
        stable = np.full(head.shape, 3.0, dtype=np.float32)
        stable[head] = 10.0
        reference = np.full(head.shape, 2.0, dtype=np.float32)
        reference[head] = 4.0
        processed = np.full(head.shape, 3.0, dtype=np.float32)
        processed[head] = 1.0

        aligned, stats = _align_stabilized_head_to_reference_boundary(
            stable,
            reference,
            processed,
            head,
            transition_width_px=5.0,
        )

        self.assertTrue(stats["enabled"])
        self.assertLess(stats["boundary_residual_max_mm"], 1e-5)
        self.assertAlmostEqual(float(aligned[10, 22]), 1.0, places=5)
        np.testing.assert_array_equal(aligned[~head], processed[~head])

    def test_stabilized_face_is_restored_after_global_scene_processing(self):
        protected = np.full((21, 21), 4.0, dtype=np.float32)
        region = np.zeros_like(protected, dtype=bool)
        region[5:16, 5:16] = True
        protected[region] = 9.0
        processed = protected.copy()
        processed[region] = 5.0
        processed[:, :3] = 2.0

        restored, stats = _restore_stabilized_face_surface(
            processed,
            protected,
            region,
            enabled=True,
        )

        self.assertTrue(stats["enabled"])
        np.testing.assert_array_equal(restored[region], protected[region])
        np.testing.assert_array_equal(restored[~region], processed[~region])

    def test_skyline_trim_stl_remains_watertight(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            depth_path = tmp_path / "depth.npy"
            stl_path = tmp_path / "skyline.stl"
            source = np.full((24, 32, 3), 255, dtype=np.uint8)
            source[8:, :10] = 20
            source[4:, 10:22] = 20
            source[11:, 22:] = 20
            np.save(depth_path, np.full((24, 32), 0.5, dtype=np.float32))

            postprocess = depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(stl_path),
                target_dimension=-1,
                z_scale=4,
                sigma=0,
                detail_boost=0,
                relief_gamma=1,
                low_percentile=0,
                high_percentile=100,
                base_border_px=0,
                source_image=source,
                trim_top_background=True,
            )

            triangles = mesh.Mesh.from_file(str(stl_path)).vectors
            rounded = np.round(triangles, decimals=5)
            edges = np.concatenate((rounded[:, [0, 1]], rounded[:, [1, 2]], rounded[:, [2, 0]]), axis=0)
            edges = np.sort(edges, axis=1)
            edge_keys = edges.reshape(edges.shape[0], -1)
            _, counts = np.unique(edge_keys, axis=0, return_counts=True)

        self.assertTrue(postprocess["top_silhouette"]["enabled"])
        self.assertGreater(postprocess["top_silhouette"]["removed_area_ratio"], 0.1)
        self.assertTrue(np.all(counts == 2))

    def test_print_filter_resamples_to_two_samples_per_minimum_feature(self):
        values = np.tile(np.linspace(0.0, 1.0, 401, dtype=np.float32), (401, 1))
        values[200, 200] = 1.0

        filtered, stats = _prepare_relief_for_printing(
            values,
            max_xy_size=40.0,
            minimum_feature_mm=0.8,
        )

        self.assertEqual(filtered.shape, (101, 101))
        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["resampled"])
        self.assertAlmostEqual(stats["input_sample_pitch_mm"], 0.1)
        self.assertAlmostEqual(stats["mesh_sample_pitch_mm"], 0.4)
        self.assertLess(float(filtered.max()), 1.0)

    def test_slope_limiter_caps_narrow_positive_extrusions_in_mm(self):
        values = np.zeros((21, 21), dtype=np.float32)
        values[8:13, 8:13] = 10.0

        limited, stats = _limit_positive_relief_slope(
            values,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
        )

        max_horizontal_step = float(np.max(np.abs(np.diff(limited, axis=1))))
        max_vertical_step = float(np.max(np.abs(np.diff(limited, axis=0))))
        self.assertLessEqual(max_horizontal_step, 0.80001)
        self.assertLessEqual(max_vertical_step, 0.80001)
        self.assertLess(float(limited.max()), 10.0)
        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["preserved_structural_edge_pairs"], 0)

    def test_gradient_compression_preserves_face_detail_without_peak_only_shaving(self):
        rows, cols = np.indices((64, 64), dtype=np.float32)
        values = 0.025 * rows + 0.01 * cols
        values += 1.4 * np.exp(-((rows - 32.0) ** 2 + (cols - 32.0) ** 2) / 90.0)
        values += 0.3 * np.sin(cols * 0.8) * np.exp(-((rows - 31.0) ** 2) / 28.0)
        values[:, 48:] += 12.0
        detail_region = np.zeros(values.shape, dtype=bool)
        detail_region[22:43, 16:45] = True

        compressed, stats = _compress_relief_gradients(
            values,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            detail_region_mask=detail_region,
        )

        correction = compressed - values
        source_detail = laplace(values)[27:38, 19:44]
        output_detail = laplace(compressed)[27:38, 19:44]
        correlation = float(np.corrcoef(source_detail.ravel(), output_detail.ravel())[0, 1])
        detail_retention = float(np.std(output_detail) / np.std(source_detail))
        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["quality_gates"]["passed"])
        self.assertTrue(stats["diagonal_metrics_available"])
        self.assertLessEqual(
            stats["output_edge_ratio_p99"],
            stats["quality_gates"]["maximum_output_edge_p99_ratio"],
        )
        self.assertLessEqual(
            stats["diagonal_edge_ratio_max"],
            stats["quality_gates"]["maximum_output_edge_ratio"],
        )
        self.assertEqual(stats["solver_info"], 0)
        self.assertGreater(correlation, 0.98)
        self.assertGreater(detail_retention, 0.75)
        self.assertLess(stats["target_edge_max_mm"], stats["input_edge_max_mm"])
        self.assertLess(float(np.min(correction)), -0.05)
        self.assertGreater(float(np.max(correction)), 0.05)

    def test_gradient_compression_fails_closed_when_solver_does_not_converge(self):
        values = np.linspace(0.0, 8.0, 32 * 32, dtype=np.float32).reshape(32, 32)
        with patch.object(
            pic_to_3d,
            "cg",
            return_value=(np.zeros(values.size, dtype=np.float64), 7),
        ):
            compressed, stats = _compress_relief_gradients(
                values,
                sample_pitch_mm=0.4,
                max_slope_mm_per_mm=2.0,
            )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "solver_nonconvergence")
        self.assertEqual(stats["solver_info"], 7)
        np.testing.assert_array_equal(compressed, values)

    def test_gradient_compression_rejects_a_converged_but_unsafe_surface(self):
        rows, cols = np.indices((40, 40), dtype=np.float32)
        values = 0.02 * rows + 0.01 * cols
        values[:, 30:] += 10.0

        compressed, stats = _compress_relief_gradients(
            values,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            maximum_correction_span_ratio=0.001,
        )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "quality_gate")
        self.assertIn("correction_span_ratio", stats["quality_gates"]["failures"])
        np.testing.assert_array_equal(compressed, values)

    def test_gradient_compression_rejects_a_detail_destroying_solution(self):
        rows, cols = np.indices((32, 32), dtype=np.float32)
        values = 0.02 * rows + 0.01 * cols
        values += 0.8 * np.sin(cols * 0.9) * np.exp(-((rows - 16.0) ** 2) / 20.0)
        detail_region = np.zeros(values.shape, dtype=bool)
        detail_region[8:25, 5:27] = True
        with patch.object(
            pic_to_3d,
            "cg",
            return_value=(np.zeros(values.size, dtype=np.float64), 0),
        ):
            compressed, stats = _compress_relief_gradients(
                values,
                sample_pitch_mm=0.4,
                max_slope_mm_per_mm=2.0,
                detail_region_mask=detail_region,
                minimum_height_span_ratio=1e-6,
                maximum_correction_span_ratio=2.0,
            )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "quality_gate")
        self.assertIn("detail_correlation", stats["quality_gates"]["failures"])
        self.assertIn("detail_rms_retention", stats["quality_gates"]["failures"])
        np.testing.assert_array_equal(compressed, values)

    def test_gradient_compression_invalid_solver_controls_fail_closed(self):
        values = np.arange(64, dtype=np.float32).reshape(8, 8)
        for options in ({"max_iterations": 0}, {"solver_tolerance": "invalid"}):
            with self.subTest(options=options):
                compressed, stats = _compress_relief_gradients(
                    values,
                    sample_pitch_mm=0.4,
                    max_slope_mm_per_mm=2.0,
                    **options,
                )
                self.assertFalse(stats["enabled"])
                self.assertEqual(stats["reason"], "invalid_configuration")
                np.testing.assert_array_equal(compressed, values)

    def test_gradient_compression_handles_a_grid_without_diagonal_edges(self):
        values = np.array([[0.0, 0.4, 0.8, 1.2]], dtype=np.float32)

        compressed, stats = _compress_relief_gradients(
            values,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
        )

        self.assertTrue(stats["enabled"])
        self.assertFalse(stats["diagonal_metrics_available"])
        self.assertIsNone(stats["diagonal_edge_ratio_p99"])
        self.assertTrue(np.all(np.isfinite(compressed)))

    def test_high_face_relief_uses_gradient_domain_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            depth_path = root / "depth.npy"
            stl_path = root / "face.stl"
            rows, cols = np.indices((48, 48), dtype=np.float32)
            depth = 0.15 + 0.65 * np.exp(
                -((rows - 24.0) ** 2 / 210.0 + (cols - 24.0) ** 2 / 150.0)
            )
            depth += 0.05 * np.exp(
                -((rows - 26.0) ** 2 / 18.0 + (cols - 24.0) ** 2 / 8.0)
            )
            face = ((rows - 24.0) ** 2 / 260.0 + (cols - 24.0) ** 2 / 180.0) <= 1.0
            np.save(depth_path, depth.astype(np.float32))

            postprocess = depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(stl_path),
                target_dimension=-1,
                z_scale=30.0,
                sigma=0.0,
                relief_gamma=1.0,
                detail_boost=0.0,
                low_percentile=0.0,
                high_percentile=100.0,
                face_region_mask=face,
                max_relief_slope=2.0,
            )
            stl_exists = stl_path.is_file()

        compression = postprocess["face_height_stabilization"]["gradient_compression"]
        self.assertEqual(
            postprocess["face_boundary_alignment"]["method"],
            "screened_gradient_domain_compression",
        )
        self.assertTrue(compression["enabled"])
        self.assertTrue(compression["quality_gates"]["passed"])
        self.assertAlmostEqual(compression["screening_weight"], 0.05)
        self.assertGreaterEqual(
            compression["detail_preservation"]["correlation"],
            compression["quality_gates"]["minimum_detail_correlation"],
        )
        self.assertGreaterEqual(
            compression["detail_preservation"]["rms_retention"],
            compression["quality_gates"]["minimum_detail_rms_retention"],
        )
        self.assertGreaterEqual(
            compression["detail_preservation"]["minimum_component_rms_retention"],
            compression["quality_gates"]["minimum_detail_rms_retention"],
        )
        self.assertLessEqual(
            compression["detail_preservation"]["maximum_component_rms_retention"],
            compression["quality_gates"]["maximum_detail_rms_retention"],
        )
        restoration = compression["post_solve_detail_restoration"]
        self.assertTrue(restoration["accepted"])
        self.assertLessEqual(restoration["applied_correction_max_mm"], 0.6001)
        self.assertLessEqual(restoration["boundary_correction_max_mm"], 1e-5)
        self.assertFalse(compression["hard_slope_limit_enforced"])
        self.assertEqual(compression["solver_info"], 0)
        self.assertEqual(compression["sample_pitch_source"], "implicit_stl_grid_unit")
        self.assertAlmostEqual(compression["max_neighbor_step_mm"], 2.0)
        self.assertTrue(stl_exists)

    def test_high_face_relief_records_rejected_gradient_candidate_before_fallback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            depth_path = root / "depth.npy"
            rows, cols = np.indices((32, 32), dtype=np.float32)
            depth = 0.2 + 0.6 * np.exp(
                -((rows - 16.0) ** 2 + (cols - 16.0) ** 2) / 100.0
            )
            face = ((rows - 16.0) ** 2 + (cols - 16.0) ** 2) <= 100.0
            np.save(depth_path, depth)
            rejected = {
                "enabled": False,
                "method": "screened_gradient_domain_compression",
                "reason": "quality_gate",
                "quality_gates": {"passed": False, "failures": ["detail_correlation"]},
            }
            with patch.object(
                pic_to_3d,
                "_compress_relief_gradients",
                return_value=(depth, rejected),
            ):
                postprocess = depth_data_to_3d_model(
                    depth_path,
                    output_stl_path=str(root / "fallback.stl"),
                    target_dimension=-1,
                    z_scale=30.0,
                    sigma=0.0,
                    relief_gamma=1.0,
                    detail_boost=0.0,
                    low_percentile=0.0,
                    high_percentile=100.0,
                    face_region_mask=face,
                )

        attempt = postprocess["face_height_stabilization"][
            "gradient_compression_attempt"
        ]
        self.assertEqual(attempt["reason"], "quality_gate")
        self.assertNotEqual(
            postprocess["face_boundary_alignment"].get("method"),
            "screened_gradient_domain_compression",
        )

    def test_selected_object_relief_uses_gradient_domain_compression(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            depth_path = root / "depth.npy"
            stl_path = root / "selected.stl"
            rows, cols = np.indices((64, 80), dtype=np.float32)
            selected = ((rows - 36.0) ** 2 / 300.0 + (cols - 41.0) ** 2 / 520.0) <= 1.0
            depth = 0.12 + selected * (
                0.5
                + 0.12 * np.sin(cols * 0.32)
                + 0.08 * np.cos(rows * 0.27)
            )
            np.save(depth_path, depth.astype(np.float32))

            postprocess = depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(stl_path),
                target_dimension=-1,
                z_scale=30.0,
                max_xy_size=32.0,
                sigma=0.0,
                relief_gamma=1.0,
                detail_boost=0.0,
                low_percentile=0.0,
                high_percentile=100.0,
                base_border_px=1,
                minimum_feature_mm=0.8,
                selection_region_mask=selected,
                selection_background_depth_ratio=0.0,
            )
            stl_exists = stl_path.is_file()

        compression = postprocess["selection_gradient_compression"]
        self.assertTrue(compression["enabled"])
        self.assertTrue(compression["quality_gates"]["passed"])
        self.assertGreater(compression["retained_detail_gradient_pairs"], 0)
        self.assertTrue(stl_exists)

    def test_selected_object_context_uses_final_foreground_for_background_reference(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            depth_path = root / "depth.npy"
            rows, cols = np.indices((96, 128), dtype=np.float32)
            selected = (
                (rows - 55.0) ** 2 / 560.0 + (cols - 64.0) ** 2 / 980.0
            ) <= 1.0
            background = (
                0.08
                + 0.16 * cols / cols.max()
                + 0.05 * np.sin(cols / 8.0)
                + 0.04 * np.cos(rows / 9.0)
                + 0.12
                * np.exp(-((rows - 22.0) ** 2 + (cols - 102.0) ** 2) / 90.0)
            )
            depth = background + selected * (
                0.62
                + 0.16 * np.sin(cols * 0.29)
                + 0.09 * np.cos(rows * 0.37)
            )
            np.save(depth_path, depth.astype(np.float32))

            postprocess = depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(root / "selected-context.stl"),
                target_dimension=-1,
                z_scale=30.0,
                max_xy_size=51.2,
                sigma=0.0,
                relief_gamma=1.0,
                detail_boost=0.0,
                low_percentile=0.0,
                high_percentile=100.0,
                base_border_px=1,
                minimum_feature_mm=0.8,
                max_relief_slope=2.0,
                selection_region_mask=selected,
                selection_background_depth_ratio=0.45,
            )

        background_stats = postprocess["background_depth_preservation"]
        physical_cap = postprocess["selection_background_physical_cap"]
        self.assertTrue(background_stats["available"])
        self.assertTrue(background_stats["passed"])
        self.assertGreater(background_stats["correlation"], 0.98)
        self.assertTrue(background_stats["localized_structure"]["passed"])
        self.assertTrue(physical_cap["emission_passed"])
        self.assertEqual(
            physical_cap["reference_cap"]["foreground_geometry_source"],
            "final_processed_selection",
        )

    def test_high_face_relief_also_preserves_nonface_selection_depth(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            depth_path = root / "depth.npy"
            stl_path = root / "face-and-selection.stl"
            rows, cols = np.indices((72, 96), dtype=np.float32)
            face = ((rows - 31.0) ** 2 / 210.0 + (cols - 27.0) ** 2 / 150.0) <= 1.0
            animal = ((rows - 41.0) ** 2 / 300.0 + (cols - 70.0) ** 2 / 210.0) <= 1.0
            selected = face | animal
            depth = 0.1 + 0.015 * cols / cols.max()
            depth += face * (
                0.58
                + 0.05 * np.sin(cols * 0.38)
                + 0.04 * np.cos(rows * 0.31)
            )
            depth += animal * (
                0.44
                + 0.12 * np.sin(cols * 0.29)
                + 0.07 * np.cos(rows * 0.43)
            )
            np.save(depth_path, depth.astype(np.float32))

            postprocess = depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(stl_path),
                target_dimension=-1,
                z_scale=30.0,
                max_xy_size=48.0,
                sigma=0.0,
                relief_gamma=1.0,
                detail_boost=0.0,
                low_percentile=0.0,
                high_percentile=100.0,
                base_border_px=1,
                minimum_feature_mm=0.8,
                max_relief_slope=2.0,
                face_region_mask=face,
                selection_region_mask=selected,
                selection_background_depth_ratio=0.0,
            )
            stl_exists = stl_path.is_file()

        selection_compression = postprocess["selection_gradient_compression"]
        self.assertTrue(selection_compression["enabled"])
        self.assertTrue(selection_compression["face_protection_passed"])
        self.assertGreater(selection_compression["retained_detail_gradient_pairs"], 0)
        self.assertEqual(
            selection_compression["protected_region_blend"]["protected_correction_max_mm"],
            0.0,
        )
        self.assertTrue(stl_exists)

    def test_slope_limiter_preserves_large_structural_silhouettes(self):
        values = np.zeros((81, 81), dtype=np.float32)
        values[20:61, 20:61] = 5.0

        limited, stats = _limit_positive_relief_slope(
            values,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
        )

        self.assertGreater(stats["preserved_structural_edge_pairs"], 0)
        self.assertEqual(float(limited[40, 40]), 5.0)
        self.assertEqual(float(limited[19, 40]), 0.0)
        self.assertEqual(float(limited[20, 40]), 5.0)

    def test_face_region_limits_outer_cliffs_without_treating_glasses_as_a_cut(self):
        values = np.zeros((81, 81), dtype=np.float32)
        values[20:61, 20:61] = 10.0
        values[36:40, 25:56] = 0.0
        face_region = np.zeros_like(values, dtype=bool)
        face_region[20:61, 20:61] = True

        limited, stats = _limit_positive_relief_slope(
            values,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            structural_region_mask=face_region,
        )

        internal_vertical_step = float(np.max(np.abs(np.diff(limited[25:56, 25:56], axis=0))))
        outer_boundary_step = float(abs(limited[20, 40] - limited[19, 40]))
        self.assertGreater(stats["suppressed_excessive_structural_pairs"], 0)
        self.assertGreater(stats["suppressed_excessive_region_boundary_pairs"], 0)
        self.assertEqual(stats["preserved_region_boundary_pairs"], 0)
        self.assertLessEqual(internal_vertical_step, 0.80001)
        self.assertLessEqual(outer_boundary_step, 0.80001)
        self.assertEqual(float(limited[19, 40]), 0.0)
        self.assertGreater(float(np.max(limited)), 8.0)

    def test_inverse_depth_transform_expands_near_subject_relief(self):
        metric_depth = np.array([[1.0, 2.0, 5.0, 100.0]], dtype=np.float32)
        linear = _shape_relief_values(
            metric_depth,
            invert=True,
            gamma=1.0,
            detail_boost=0,
            low_percentile=0,
            high_percentile=100,
        )
        proximity = _shape_relief_values(
            metric_depth,
            invert=True,
            gamma=1.0,
            detail_boost=0,
            low_percentile=0,
            high_percentile=100,
            value_transform=RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
        )

        self.assertGreater(proximity[0, 0], proximity[0, 1])
        self.assertGreater(proximity[0, 1], proximity[0, 2])
        self.assertGreater(proximity[0, 2], proximity[0, 3])
        self.assertLess(proximity[0, 1], linear[0, 1] - 0.4)

    def test_inverse_depth_transform_preserves_mold_polarity(self):
        metric_depth = np.array([[1.0, 5.0, 100.0]], dtype=np.float32)
        raised = _shape_relief_values(
            metric_depth,
            invert=True,
            gamma=1.0,
            detail_boost=0,
            low_percentile=0,
            high_percentile=100,
            value_transform=RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
        )
        mold = _shape_relief_values(
            metric_depth,
            invert=False,
            gamma=1.0,
            detail_boost=0,
            low_percentile=0,
            high_percentile=100,
            value_transform=RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
        )

        self.assertGreater(raised[0, 0], raised[0, -1])
        self.assertLess(mold[0, 0], mold[0, -1])
        np.testing.assert_allclose(mold, 1.0 - raised)

    def test_metric_models_select_inverse_depth_relief(self):
        self.assertEqual(
            relief_value_transform_for_model("apple/DepthPro-hf"),
            RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
        )
        self.assertEqual(
            relief_value_transform_for_model("depth-anything/Depth-Anything-V2-Large-hf"),
            "linear",
        )

    def test_flatten_border_locks_outer_wall_height(self):
        values = np.ones((8, 10), dtype=np.float32)
        flattened = _flatten_border(values, 2)

        self.assertTrue(np.all(flattened[:2, :] == 0))
        self.assertTrue(np.all(flattened[-2:, :] == 0))
        self.assertTrue(np.all(flattened[:, :2] == 0))
        self.assertTrue(np.all(flattened[:, -2:] == 0))
        self.assertTrue(np.all(flattened[2:-2, 2:-2] == 1))

    def test_max_xy_size_decouples_detail_resolution_from_physical_size(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            depth_path = tmp_path / "depth.npy"
            stl_path = tmp_path / "relief.stl"
            data = np.linspace(0.0, 1.0, 80 * 120, dtype=np.float32).reshape(80, 120)
            np.save(depth_path, data)

            postprocess = depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(stl_path),
                target_dimension=120,
                z_scale=12,
                invert=False,
                sigma=0,
                max_xy_size=40,
            )

            stl_mesh = mesh.Mesh.from_file(str(stl_path))
            mins = stl_mesh.vectors.reshape(-1, 3).min(axis=0)
            maxs = stl_mesh.vectors.reshape(-1, 3).max(axis=0)
            extents = maxs - mins

            self.assertAlmostEqual(max(extents[0], extents[1]), 40.0, places=4)
            self.assertGreater(extents[2], 9.5)
            self.assertLessEqual(extents[2], 12.1)
            self.assertEqual(postprocess["mesh_grid_shape"], [67, 101])
            self.assertAlmostEqual(postprocess["mesh_sample_pitch_mm"], 0.4)

    def test_depth_resize_preserves_near_target_detail_instead_of_stride_halving(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            depth_path = tmp_path / "depth.npy"
            stl_path = tmp_path / "near_target.stl"
            data = np.linspace(0.0, 1.0, 65 * 65, dtype=np.float32).reshape(65, 65)
            data[30:35, 30:35] += 0.2
            np.save(depth_path, data)

            depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(stl_path),
                target_dimension=64,
                z_scale=8,
                invert=False,
                sigma=0,
                detail_boost=0,
                relief_gamma=1.0,
                low_percentile=0,
                high_percentile=100,
                base_border_px=0,
                max_xy_size=24,
            )

            stl_mesh = mesh.Mesh.from_file(str(stl_path))
            vertices = stl_mesh.vectors.reshape(-1, 3)
            extents = vertices.max(axis=0) - vertices.min(axis=0)

            self.assertGreater(len(stl_mesh.vectors), 10000)
            self.assertAlmostEqual(max(extents[0], extents[1]), 24.0, places=4)

    def test_nan_aware_resize_preserves_object_cutout_holes(self):
        values = np.ones((20, 20), dtype=np.float32)
        values[8:12, 8:12] = np.nan

        resized = _resize_nan_aware(values, (10, 10))

        self.assertTrue(np.isnan(resized[4:6, 4:6]).all())
        self.assertTrue(np.isfinite(resized[0, 0]))

    def test_base_border_survives_final_smoothing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            depth_path = tmp_path / "depth.npy"
            stl_path = tmp_path / "bordered.stl"
            data = np.ones((30, 30), dtype=np.float32) * 0.4
            data[10:20, 10:20] = 1.0
            np.save(depth_path, data)

            depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(stl_path),
                target_dimension=-1,
                z_scale=10,
                invert=False,
                sigma=2.0,
                detail_boost=0,
                relief_gamma=1.0,
                low_percentile=0,
                high_percentile=100,
                base_border_px=2,
            )

            vertices = mesh.Mesh.from_file(str(stl_path)).vectors.reshape(-1, 3)
            mins = vertices.min(axis=0)
            maxs = vertices.max(axis=0)
            outer = (
                np.isclose(vertices[:, 0], mins[0])
                | np.isclose(vertices[:, 0], maxs[0])
                | np.isclose(vertices[:, 1], mins[1])
                | np.isclose(vertices[:, 1], maxs[1])
            )

            self.assertLessEqual(vertices[outer, 2].max(), 0.011)
            self.assertGreater(vertices[:, 2].max(), 9.0)


if __name__ == "__main__":
    unittest.main()
