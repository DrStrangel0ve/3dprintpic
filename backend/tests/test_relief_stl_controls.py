import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh
from scipy.ndimage import binary_erosion, laplace, zoom
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
    _audit_bounded_compression_surface,
    _audit_emitted_relief_printability,
    _background_relief_preservation_metrics,
    _cap_selection_background_relief,
    _bridge_weighted_face_features,
    _compress_relief_gradients,
    _compress_selected_relief_surface,
    _expand_face_region_to_depth_connected_head,
    _face_detail_preservation_metrics,
    _guard_face_detail_updates,
    _inject_photo_relief_detail,
    _photo_detail_background_gate,
    _prepare_depth_downsample_input,
    _depth_processor_output_size,
    _limit_positive_relief_slope,
    _minimal_component_connector_mask,
    _selection_grounded_backing_mask,
    _prepare_relief_for_printing,
    _relax_selection_attachment_conflicts,
    _restore_background_from_reference,
    _restore_stabilized_face_surface,
    _resize_nan_aware,
    _shape_relief_values,
    _stabilize_face_relief_height,
    _surface_lighting_agreement_metrics,
    _top_silhouette_mask,
    complete_selection_context_with_modern_inpaint,
    compose_selection_depth_with_context,
    depth_data_to_3d_model,
    relief_value_transform_for_model,
)


class ReliefStlControlsTest(unittest.TestCase):
    def test_selection_backing_fills_only_below_lowest_selected_pixel(self):
        selected = np.zeros((24, 32), dtype=bool)
        selected[4:10, 2:9] = True
        selected[8:23, 13:19] = True
        selected[6:14, 24:30] = True

        foundation, stats = _selection_grounded_backing_mask(
            selected,
            sample_pitch_mm=0.2,
            minimum_width_mm=0.8,
        )
        expected = np.zeros_like(selected)
        expected[10:23, 2:9] = True
        expected[14:23, 24:30] = True
        expected[19:23, 2:30] = True
        expected &= ~selected

        np.testing.assert_array_equal(foundation, expected)
        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["accepted"])
        self.assertEqual(stats["bottom_row"], 22)
        self.assertEqual(stats["active_columns"], 19)
        self.assertEqual(stats["foundation_pixels"], int(expected.sum()))
        self.assertEqual(stats["base_rail_height_px"], 4)
        self.assertFalse(np.any(foundation & selected))
        self.assertFalse(np.any(foundation[:10, 2:9]))

    def test_backing_connector_handles_diagonal_pixel_connection(self):
        selected = np.zeros((24, 32), dtype=bool)
        selected[2:11, 2:11] = True
        selected[11:21, 11:23] = True

        bridge, stats = _minimal_component_connector_mask(
            selected,
            sample_pitch_mm=0.2,
            minimum_width_mm=0.8,
        )
        connected = selected | bridge
        valid_cells = (
            connected[:-1, :-1]
            & connected[1:, :-1]
            & connected[:-1, 1:]
            & connected[1:, 1:]
        )
        _, component_count = pic_to_3d.label(
            valid_cells,
            structure=np.array(
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                dtype=np.uint8,
            ),
        )

        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["accepted"])
        self.assertEqual(stats["pixel_component_count_before"], 1)
        self.assertEqual(stats["mesh_component_count_before"], 2)
        self.assertEqual(stats["mesh_component_count_after"], 1)
        self.assertGreater(stats["mesh_bridge_segments"], 0)
        self.assertEqual(component_count, 1)
        self.assertFalse(np.any(bridge & selected))

    def test_backing_connector_rejects_fragmented_selection(self):
        selected = np.zeros((96, 96), dtype=bool)
        for row in range(8):
            for column in range(8):
                top = 2 + 11 * row
                left = 2 + 11 * column
                selected[top : top + 3, left : left + 3] = True

        bridge, stats = _minimal_component_connector_mask(
            selected,
            sample_pitch_mm=0.2,
            minimum_width_mm=0.8,
        )

        self.assertFalse(stats["accepted"])
        self.assertEqual(stats["reason"], "component_budget_exceeded")
        self.assertEqual(stats["component_count_before"], 64)
        self.assertFalse(np.any(bridge))

    def test_backing_connector_gives_thin_selection_pixels_mesh_support(self):
        selected = np.zeros((72, 96), dtype=bool)
        selected[10:60, 10:55] = True
        selected[34, 55:88] = True

        bridge, stats = _minimal_component_connector_mask(
            selected,
            sample_pitch_mm=0.2,
            minimum_width_mm=0.8,
        )
        supported = pic_to_3d._mesh_vertex_support_mask(selected | bridge)

        self.assertTrue(stats["accepted"])
        self.assertGreater(stats["initially_unsupported_selected_pixels"], 0)
        self.assertEqual(stats["unsupported_selected_mesh_pixels"], 0)
        self.assertTrue(np.all(supported[selected]))
        self.assertLessEqual(
            stats["bridge_ratio"],
            stats["maximum_bridge_ratio"],
        )

    def test_backing_connector_rejects_oversized_bridge(self):
        selected = np.zeros((96, 160), dtype=bool)
        selected[10:20, 10:20] = True
        selected[70:80, 140:150] = True

        _bridge, stats = _minimal_component_connector_mask(
            selected,
            sample_pitch_mm=0.2,
            minimum_width_mm=0.8,
        )

        self.assertFalse(stats["accepted"])
        self.assertEqual(stats["reason"], "bridge_budget_exceeded")
        self.assertGreater(stats["bridge_pixels"], stats["bridge_budget_pixels"])

    def test_grounded_backing_fills_below_component_bridge(self):
        selected = np.zeros((32, 48), dtype=bool)
        selected[4:24, 3:18] = True
        selected[10:31, 21:45] = True
        connector, connector_stats = _minimal_component_connector_mask(
            selected,
            sample_pitch_mm=0.2,
            minimum_width_mm=0.8,
        )
        foundation, foundation_stats = _selection_grounded_backing_mask(
            selected | connector,
            sample_pitch_mm=0.2,
            minimum_width_mm=0.8,
        )

        connector_rows, connector_columns = np.where(connector)
        self.assertTrue(connector_stats["accepted"])
        self.assertTrue(foundation_stats["accepted"])
        self.assertGreater(connector_columns.size, 0)
        grounded = selected | connector | foundation
        bottom_row = int(foundation_stats["bottom_row"])
        for column in np.unique(connector_columns):
            lowest_connector_row = int(connector_rows[connector_columns == column].max())
            self.assertTrue(
                np.all(
                    grounded[
                        lowest_connector_row + 1 : bottom_row + 1,
                        column,
                    ]
                )
            )

    def test_depth_processor_output_size_matches_aspect_preserving_dpt_resize(self):
        source = Image.new("RGB", (3840, 2160), (80, 120, 160))
        processor = SimpleNamespace(
            do_resize=True,
            size={"width": 518, "height": 518},
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
        )

        self.assertEqual(
            _depth_processor_output_size(source, processor),
            (924, 518),
        )

    def test_depth_input_sharpening_preserves_edges_through_model_resize(self):
        width = 1024
        x = np.arange(width, dtype=np.float32)
        signal = 127.0 + 54.0 * np.sin(2.0 * np.pi * x / 18.0)
        values = np.repeat(signal[None, :, None], width, axis=0)
        values = np.repeat(values, 3, axis=2).clip(0, 255).astype(np.uint8)
        source = Image.fromarray(values, mode="RGB")
        processor = SimpleNamespace(
            do_resize=True,
            size={"width": 128, "height": 128},
            keep_aspect_ratio=True,
            ensure_multiple_of=1,
        )

        prepared, metadata = _prepare_depth_downsample_input(
            source,
            processor,
            sharpening=0.35,
        )
        baseline = np.asarray(
            source.resize((128, 128), Image.Resampling.BICUBIC).convert("L"),
            dtype=np.float32,
        )
        sharpened = np.asarray(
            prepared.resize((128, 128), Image.Resampling.BICUBIC).convert("L"),
            dtype=np.float32,
        )
        baseline_gradient = float(np.mean(np.abs(np.diff(baseline, axis=1))))
        sharpened_gradient = float(np.mean(np.abs(np.diff(sharpened, axis=1))))

        self.assertTrue(metadata["enabled"])
        self.assertEqual(
            metadata["method"],
            "scale_aware_luminance_unsharp_before_model_resize_v1",
        )
        self.assertAlmostEqual(metadata["downsample_scale"], 8.0)
        self.assertGreater(sharpened_gradient, baseline_gradient * 1.05)

    def test_depth_input_sharpening_is_bounded_and_skips_small_images(self):
        source = Image.new("RGB", (96, 64), (80, 120, 160))
        processor = SimpleNamespace(
            do_resize=True,
            size={"width": 518, "height": 518},
        )

        prepared, metadata = _prepare_depth_downsample_input(
            source,
            processor,
            sharpening=4.0,
        )

        np.testing.assert_array_equal(np.asarray(prepared), np.asarray(source))
        self.assertFalse(metadata["enabled"])
        self.assertEqual(metadata["reason"], "input_not_downsampled")
        self.assertEqual(metadata["strength"], 1.0)

    def test_modern_selection_inpaint_never_conditions_on_removed_pixels(self):
        rows, cols = np.indices((32, 48))
        source_values = np.zeros((32, 48, 3), dtype=np.uint8)
        source_values[..., 0] = np.where((rows + cols) % 2 == 0, 255, 0)
        source_values[..., 1] = np.where(cols % 2 == 0, 0, 255)
        source_values[..., 2] = 40
        keep_values = np.zeros((32, 48), dtype=np.uint8)
        keep_values[6:28, 12:36] = 255
        source = Image.fromarray(source_values, mode="RGB")
        keep_mask = Image.fromarray(keep_values, mode="L")
        call = {}

        def fake_pipe(**kwargs):
            call.update(kwargs)
            return SimpleNamespace(
                images=[Image.new("RGB", kwargs["image"].size, (20, 80, 160))]
            )

        with patch.object(
            pic_to_3d,
            "_load_inpaint_pipeline",
            return_value=fake_pipe,
        ):
            completed, metadata = complete_selection_context_with_modern_inpaint(
                source,
                keep_mask,
                device="cpu",
                inpaint_max_dimension=48,
            )

        completed_values = np.asarray(completed)
        condition_values = np.asarray(call["image"])
        generation_mask = np.asarray(call["mask_image"])
        keep = keep_values > 0
        np.testing.assert_array_equal(completed_values[keep], source_values[keep])
        far_context = np.zeros_like(keep)
        far_context[:3, :3] = True
        self.assertTrue(np.all(completed_values[far_context] == [20, 80, 160]))
        self.assertFalse(np.array_equal(condition_values[~keep], source_values[~keep]))
        self.assertTrue(np.all(generation_mask[~keep] == 255))
        self.assertTrue(np.any(generation_mask[keep] == 255))
        self.assertTrue(np.any(generation_mask[keep] == 0))
        self.assertTrue(metadata["source_free_removed_context"])
        self.assertFalse(metadata["removed_source_pixels_conditioned"])
        self.assertTrue(metadata["selected_pixels_exact"])

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

    def test_background_reference_fallback_can_restore_an_accepted_reference(self):
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
            reference_is_accepted=True,
        )

        self.assertEqual(stats["slope_guard_baseline"], "accepted_reference")
        self.assertGreater(stats["restored_pixels"], 0)
        self.assertAlmostEqual(float(restored[0, 0]), 20.0, places=5)
        np.testing.assert_array_equal(restored[foreground], candidate[foreground])
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

        capped, _ = _cap_selection_background_relief(
            values,
            selected,
            relief_height_mm=30.0,
            sample_pitch_mm=0.5,
            max_slope_mm_per_mm=2.0,
            background_depth_ratio=0.45,
        )
        relaxed, relaxation = _relax_selection_attachment_conflicts(
            capped,
            selected,
            sample_pitch_mm=0.5,
            max_slope_mm_per_mm=2.0,
            feather_pixels=1,
        )
        _, relaxed_stats = _cap_selection_background_relief(
            relaxed,
            selected,
            relief_height_mm=30.0,
            sample_pitch_mm=0.5,
            max_slope_mm_per_mm=2.0,
            background_depth_ratio=0.45,
        )

        self.assertTrue(relaxation["enabled"])
        self.assertGreater(relaxation["seed_pixels"], 0)
        self.assertLess(
            relaxed_stats["attachment_jump_max_mm"],
            stats["attachment_jump_max_mm"],
        )

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

    def test_post_blend_audit_fails_closed_for_flat_reference_detail(self):
        reference = np.zeros((40, 40), dtype=np.float32)
        face = np.zeros(reference.shape, dtype=bool)
        face[8:32, 8:32] = True
        candidate = reference.copy()
        candidate[14:26, 14:26] = 0.4
        gates = {
            "minimum_detail_correlation": 0.8,
            "minimum_detail_rms_retention": 0.6,
            "maximum_detail_rms_retention": 2.0,
            "maximum_output_edge_p99_ratio": 12.0,
            "maximum_output_edge_ratio": 24.0,
            "minimum_height_span_ratio": 0.5,
            "maximum_height_span_ratio": 1.15,
            "maximum_correction_span_ratio": 0.9,
        }

        audit = _audit_bounded_compression_surface(
            reference,
            candidate,
            face,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            quality_gates=gates,
        )

        self.assertFalse(audit["enabled"])
        self.assertFalse(audit["quality_gates"]["passed"])
        self.assertIn(
            "detail_flat_reference_violation",
            audit["quality_gates"]["failures"],
        )
        self.assertIn(
            "detail_rms_retention",
            audit["quality_gates"]["failures"],
        )
        json.dumps(audit, allow_nan=False)

    def test_post_blend_audit_rejects_large_changed_edge_excess(self):
        rows, cols = np.indices((40, 40), dtype=np.float32)
        source = 0.02 * rows + 0.01 * cols
        source[:, 30:] += 10.0
        candidate = source.copy()
        candidate[10:30, 10:30] += 45.0
        gates = {
            "maximum_output_edge_p99_ratio": 12.0,
            "maximum_output_edge_ratio": 24.0,
            "minimum_height_span_ratio": 0.5,
            "maximum_height_span_ratio": 6.0,
            "maximum_correction_span_ratio": 6.0,
        }

        audit = _audit_bounded_compression_surface(
            source,
            candidate,
            None,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            quality_gates=gates,
        )

        self.assertFalse(audit["enabled"])
        self.assertGreater(audit["output_edge_excess_ratio_max"], 24.0)
        self.assertIn(
            "cardinal_edge_excess_max",
            audit["quality_gates"]["failures"],
        )

    def test_post_blend_audit_requires_identical_finite_coverage(self):
        rows, cols = np.indices((40, 40), dtype=np.float32)
        source = 0.02 * rows + 0.01 * cols
        candidate = source.copy()
        candidate[20:, :] = np.nan

        audit = _audit_bounded_compression_surface(
            source,
            candidate,
            None,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            quality_gates={},
        )

        self.assertFalse(audit["enabled"])
        self.assertEqual(audit["reason"], "finite_coverage_mismatch")
        self.assertFalse(audit["finite_mask_match"])
        self.assertEqual(
            audit["quality_gates"]["failures"],
            ["finite_coverage_mismatch"],
        )

    def test_post_blend_audit_rejects_material_edge_direction_reversal(self):
        rows, cols = np.indices((40, 40), dtype=np.float32)
        source = 2.5 * rows + 0.01 * cols
        source[:, 20:] += 5.0
        candidate = source.copy()
        candidate[:, 20:] -= 10.0
        gates = {
            "maximum_output_edge_p99_ratio": 12.0,
            "maximum_output_edge_ratio": 24.0,
            "minimum_height_span_ratio": 0.5,
            "maximum_height_span_ratio": 1.15,
            "maximum_correction_span_ratio": 0.9,
        }

        audit = _audit_bounded_compression_surface(
            source,
            candidate,
            None,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            quality_gates=gates,
            reject_direction_reversals=True,
        )

        self.assertFalse(audit["enabled"])
        self.assertGreater(audit["cardinal_edge_direction_reversal_count"], 0)
        self.assertIn(
            "cardinal_edge_direction_reversal",
            audit["quality_gates"]["failures"],
        )

    def test_post_blend_audit_rejects_substep_source_edge_reversed_into_cliff(self):
        rows, cols = np.indices((40, 40), dtype=np.float32)
        source = 2.5 * rows + 0.01 * cols
        source[:, 20:] += 0.7
        candidate = source.copy()
        candidate[:, 20:] -= 8.0
        gates = {
            "maximum_output_edge_p99_ratio": 12.0,
            "maximum_output_edge_ratio": 24.0,
            "minimum_height_span_ratio": 0.5,
            "maximum_height_span_ratio": 1.15,
            "maximum_correction_span_ratio": 0.9,
        }

        audit = _audit_bounded_compression_surface(
            source,
            candidate,
            None,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            quality_gates=gates,
            reject_direction_reversals=True,
        )

        self.assertFalse(audit["enabled"])
        self.assertGreater(audit["cardinal_edge_direction_reversal_count"], 0)
        self.assertIn(
            "cardinal_edge_direction_reversal",
            audit["quality_gates"]["failures"],
        )

    def test_post_blend_audit_allows_sparse_bounded_direction_reversals(self):
        rows, cols = np.indices((2, 40), dtype=np.float32)
        source = 2.5 * rows + 0.01 * cols
        source[:, 20:] += 1.5
        candidate = source.copy()
        candidate[:, 20:] -= 3.0
        gates = {
            "maximum_output_edge_p99_ratio": 12.0,
            "maximum_output_edge_ratio": 24.0,
            "minimum_height_span_ratio": 0.5,
            "maximum_height_span_ratio": 1.15,
            "maximum_correction_span_ratio": 2.0,
        }

        audit = _audit_bounded_compression_surface(
            source,
            candidate,
            None,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            quality_gates=gates,
            reject_direction_reversals=True,
        )

        self.assertTrue(audit["enabled"])
        self.assertEqual(audit["cardinal_edge_direction_reversal_count"], 2)
        self.assertLess(
            audit["total_edge_direction_reversal_count"],
            audit["minimum_direction_reversal_support_edges"],
        )

    def test_post_blend_audit_rejects_sparse_severe_direction_reversal(self):
        rows, cols = np.indices((2, 40), dtype=np.float32)
        source = 2.5 * rows + 0.01 * cols
        source[:, 20:] += 7.4
        candidate = source.copy()
        candidate[:, 20:] -= 14.8
        gates = {
            "maximum_output_edge_p99_ratio": 12.0,
            "maximum_output_edge_ratio": 24.0,
            "minimum_height_span_ratio": 0.5,
            "maximum_height_span_ratio": 1.15,
            "maximum_correction_span_ratio": 3.0,
        }

        audit = _audit_bounded_compression_surface(
            source,
            candidate,
            None,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            quality_gates=gates,
            reject_direction_reversals=True,
        )

        self.assertFalse(audit["enabled"])
        self.assertEqual(audit["cardinal_edge_direction_reversal_count"], 2)
        self.assertGreater(
            audit["cardinal_edge_direction_reversal_max_physical_ratio"],
            audit["maximum_sparse_direction_reversal_physical_ratio"],
        )
        self.assertIn(
            "cardinal_edge_direction_reversal",
            audit["quality_gates"]["failures"],
        )

    def test_post_blend_audit_combines_corner_reversal_support(self):
        rows, cols = np.indices((80, 100), dtype=np.float32)
        source = (
            0.2 * rows
            + 0.01 * cols
            + 0.1 * np.sin(cols * 1.2)
            + 0.05 * np.cos(rows)
        )
        detail = np.ones(source.shape, dtype=bool)
        source[0:1, 0:2] += 1.25
        candidate = source.copy()
        candidate[0:1, 0:2] -= 2.5
        gates = {
            "minimum_detail_correlation": 0.65,
            "minimum_detail_rms_retention": 0.15,
            "maximum_detail_rms_retention": 2.0,
            "maximum_output_edge_p99_ratio": 12.0,
            "maximum_output_edge_ratio": 24.0,
            "minimum_height_span_ratio": 0.5,
            "maximum_height_span_ratio": 1.15,
            "maximum_correction_span_ratio": 0.9,
        }

        audit = _audit_bounded_compression_surface(
            source,
            candidate,
            detail,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            quality_gates=gates,
            reject_direction_reversals=True,
        )

        self.assertEqual(audit["cardinal_edge_direction_reversal_count"], 3)
        self.assertEqual(audit["diagonal_edge_direction_reversal_count"], 3)
        self.assertLessEqual(
            audit["cardinal_edge_direction_reversal_max_physical_ratio"],
            audit["maximum_sparse_direction_reversal_physical_ratio"],
        )
        self.assertLessEqual(
            audit["diagonal_edge_direction_reversal_max_physical_ratio"],
            audit["maximum_sparse_direction_reversal_physical_ratio"],
        )
        self.assertFalse(audit["enabled"])
        self.assertIn(
            "cardinal_edge_direction_reversal",
            audit["quality_gates"]["failures"],
        )
        self.assertIn(
            "diagonal_edge_direction_reversal",
            audit["quality_gates"]["failures"],
        )

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
                    downsample_sharpening=0.35,
                )

        self.assertEqual(result, "depth.npy")
        fallback.assert_called_once_with(
            "input.png",
            output_dir=tmp_dir,
            model_name=pic_to_3d.DEFAULT_DEPTH_FALLBACK_MODEL,
            device="cpu",
            requested_model_name="apple/DepthPro-hf",
            fallback_reason="Depth Pro unavailable",
            downsample_sharpening=0.35,
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

    def test_subject_normalization_is_invariant_to_background_height(self):
        subject = np.zeros((32, 32), dtype=bool)
        subject[7:25, 8:24] = True
        subject_values = np.tile(
            np.linspace(0.2, 0.8, 32, dtype=np.float32),
            (32, 1),
        )
        low_background = np.full(subject.shape, -0.4, dtype=np.float32)
        raised_background = np.full(subject.shape, 0.45, dtype=np.float32)
        low_background[subject] = subject_values[subject]
        raised_background[subject] = subject_values[subject]

        low = _shape_relief_values(
            low_background,
            gamma=1.0,
            detail_boost=0.0,
            low_percentile=0.0,
            high_percentile=100.0,
            normalization_mask=subject,
        )
        raised = _shape_relief_values(
            raised_background,
            gamma=1.0,
            detail_boost=0.0,
            low_percentile=0.0,
            high_percentile=100.0,
            normalization_mask=subject,
        )

        np.testing.assert_array_equal(low[subject], raised[subject])
        self.assertGreater(float(np.mean(raised[~subject])), float(np.mean(low[~subject])))

    def test_subject_normalization_rejects_a_mismatched_mask(self):
        with self.assertRaisesRegex(ValueError, "normalization mask"):
            _shape_relief_values(
                np.ones((12, 12), dtype=np.float32),
                detail_boost=0.0,
                normalization_mask=np.ones((8, 8), dtype=bool),
            )

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

    def test_photo_detail_fusion_keeps_raised_masked_background_active(self):
        relief = np.full((64, 64), 0.1, dtype=np.float32)
        relief[:, :12] = 0.9
        photo = np.zeros((64, 64, 3), dtype=np.uint8)
        photo[:, ::4] = 255
        face_region = np.zeros_like(relief, dtype=bool)
        face_region[20:44, 24:40] = True

        fused, stats = _inject_photo_relief_detail(
            relief,
            photo,
            max_detail_ratio=0.04,
            protection_mask=face_region,
            protection_halo_px=4.0,
        )

        raised_change = float(np.mean(np.abs(fused[:, :12] - relief[:, :12])))
        low_change = float(np.mean(np.abs(fused[:, 52:] - relief[:, 52:])))
        self.assertTrue(stats["enabled"])
        self.assertGreater(raised_change, 0.002)
        self.assertGreater(low_change, 0.002)
        np.testing.assert_array_equal(fused[face_region], relief[face_region])

    def test_photo_detail_protection_reaches_full_gain_isotropically(self):
        protected = np.zeros((41, 41), dtype=bool)
        protected[20, 20] = True

        gate = _photo_detail_background_gate(protected, 10.0, zero_guard_px=4.0)

        self.assertEqual(float(gate[20, 20]), 0.0)
        self.assertEqual(float(gate[20, 24]), 0.0)
        self.assertAlmostEqual(float(gate[20, 27]), 0.5, places=6)
        self.assertEqual(float(gate[20, 30]), 1.0)
        self.assertEqual(float(gate[26, 28]), 1.0)
        self.assertLess(float(gate[26, 27]), 1.0)

    def test_photo_detail_fusion_treats_empty_mask_as_unprotected(self):
        relief = np.full((24, 24), 0.3, dtype=np.float32)
        photo = np.zeros((24, 24, 3), dtype=np.uint8)
        photo[:, ::3] = 255

        _fused, stats = _inject_photo_relief_detail(
            relief,
            photo,
            max_detail_ratio=0.02,
            protection_mask=np.zeros_like(relief, dtype=bool),
        )

        self.assertTrue(stats["enabled"])
        self.assertFalse(stats["face_protected"])
        self.assertEqual(stats["protection_halo_px"], 0.0)

    def test_top_silhouette_mask_removes_only_pixels_above_content(self):
        source = np.full((20, 24, 3), 255, dtype=np.uint8)
        source[8:, :12] = 30
        source[4:, 12:] = 30

        silhouette, stats = _top_silhouette_mask(source, source.shape[:2], padding_px=0)

        self.assertTrue(stats["enabled"])
        # The helper flips horizontally to match the STL coordinate system.
        skyline_rows = np.argmax(silhouette, axis=0)
        np.testing.assert_array_equal(skyline_rows[:13], np.full(13, 3))
        np.testing.assert_array_equal(skyline_rows[-9:], np.full(9, 7))
        self.assertTrue(np.all(np.diff(skyline_rows) >= 0))
        self.assertEqual(stats["method"], "structural_boundary_skyline_v2")

    def test_top_silhouette_ignores_smooth_cloud_gradient_but_keeps_structures(self):
        height, width = 80, 120
        yy, xx = np.indices((height, width), dtype=np.float32)
        sky = np.empty((height, width, 3), dtype=np.float32)
        sky[..., 0] = 38.0 + 0.18 * yy
        sky[..., 1] = 46.0 + 0.13 * yy
        sky[..., 2] = 59.0 + 0.08 * yy
        cloud = 30.0 * np.exp(-(((xx - 28.0) / 24.0) ** 2 + ((yy - 17.0) / 12.0) ** 2))
        sky += cloud[..., None]
        source = np.clip(sky, 0, 255).astype(np.uint8)
        source[42:, :35] = (35, 28, 24)
        source[8:, 48:62] = (118, 74, 35)
        source[30:, 80:] = (12, 34, 86)
        source[60:, :] = (32, 38, 34)

        silhouette, stats = _top_silhouette_mask(source, source.shape[:2], padding_px=0)
        skyline_rows = np.argmax(np.flip(silhouette, axis=1), axis=0)

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["method"], "structural_boundary_skyline_v2")
        self.assertGreater(stats["removed_area_ratio"], 0.35)
        self.assertGreaterEqual(int(np.median(skyline_rows[5:30])), 39)
        self.assertLessEqual(int(np.median(skyline_rows[5:30])), 43)
        self.assertLessEqual(int(np.median(skyline_rows[50:60])), 10)
        self.assertGreaterEqual(int(np.median(skyline_rows[88:112])), 27)
        self.assertLessEqual(int(np.median(skyline_rows[88:112])), 31)

    def test_top_silhouette_requires_sustained_depth_below_cloud_edges(self):
        height, width = 80, 120
        source = np.full((height, width, 3), (32, 40, 54), dtype=np.uint8)
        source[12:15, :] = (112, 116, 120)
        source[44:, :38] = (42, 28, 20)
        source[8:, 52:66] = (126, 78, 34)
        source[31:, 84:] = (14, 36, 90)
        depth = np.full((height, width), 0.12, dtype=np.float32)
        depth[44:, :38] = 0.64
        depth[8:, 52:66] = 0.82
        depth[31:, 84:] = 0.71

        silhouette, stats = _top_silhouette_mask(
            source,
            source.shape[:2],
            padding_px=0,
            depth_values=depth,
        )
        skyline_rows = np.argmax(np.flip(silhouette, axis=1), axis=0)

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["method"], "depth_supported_structural_skyline_v3")
        self.assertTrue(stats["depth_evidence"]["supported"])
        self.assertGreaterEqual(int(np.median(skyline_rows[5:32])), 43)
        self.assertLessEqual(int(np.median(skyline_rows[5:32])), 45)
        self.assertLessEqual(int(np.median(skyline_rows[55:63])), 8)
        self.assertGreaterEqual(int(np.median(skyline_rows[90:114])), 30)

    def test_top_silhouette_never_removes_selected_spire_pixels(self):
        height, width = 72, 96
        source = np.full((height, width, 3), (28, 35, 48), dtype=np.uint8)
        source[40:, :] = (44, 31, 22)
        protected = np.zeros((height, width), dtype=bool)
        protected[3:, 47] = True
        depth = np.full((height, width), 0.15, dtype=np.float32)
        depth[40:, :] = 0.72

        silhouette, stats = _top_silhouette_mask(
            source,
            source.shape[:2],
            padding_px=0,
            depth_values=depth,
            protected_region_mask=protected,
        )
        source_silhouette = np.flip(silhouette, axis=1)

        self.assertTrue(np.all(source_silhouette[protected]))
        self.assertEqual(stats["protected_removed_pixels"], 0)
        self.assertEqual(stats["protected_column_count"], 1)
        self.assertTrue(stats["protected_region_used"])
        self.assertLessEqual(int(np.argmax(source_silhouette[:, 47])), 3)

    def test_top_silhouette_selection_changes_only_protected_columns(self):
        height, width = 72, 96
        source = np.full((height, width, 3), (28, 35, 48), dtype=np.uint8)
        source[40:, :] = (44, 31, 22)
        depth = np.full((height, width), 0.15, dtype=np.float32)
        depth[40:, :] = 0.72
        protected = np.zeros((height, width), dtype=bool)
        protected[3:, 47] = True

        baseline, _ = _top_silhouette_mask(
            source,
            source.shape[:2],
            padding_px=0,
            depth_values=depth,
        )
        selected, _ = _top_silhouette_mask(
            source,
            source.shape[:2],
            padding_px=0,
            depth_values=depth,
            protected_region_mask=protected,
        )
        baseline_source = np.flip(baseline, axis=1)
        selected_source = np.flip(selected, axis=1)

        np.testing.assert_array_equal(
            selected_source[:, np.arange(width) != 47],
            baseline_source[:, np.arange(width) != 47],
        )
        self.assertTrue(np.all(selected_source[protected]))

    def test_top_silhouette_changes_only_final_mesh_coverage(self):
        height, width = 36, 48
        yy, xx = np.indices((height, width), dtype=np.float32)
        depth = 0.12 + 0.008 * yy + 0.004 * xx
        source = np.full((height, width, 3), (30, 39, 52), dtype=np.uint8)
        source[15:, :18] = (65, 42, 24)
        source[5:, 22:29] = (132, 78, 34)
        source[20:, 34:] = (18, 40, 88)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            depth_path = root / "depth.npy"
            source_path = root / "source.png"
            trimmed_surface_path = root / "trimmed.npy"
            rectangular_surface_path = root / "rectangular.npy"
            np.save(depth_path, depth)
            Image.fromarray(source).save(source_path)
            trimmed_stats = depth_data_to_3d_model(
                depth_path,
                output_stl_path=root / "trimmed.stl",
                target_dimension=-1,
                z_scale=10.0,
                max_xy_size=48.0,
                source_image=source_path,
                trim_top_background=True,
                surface_output_path=trimmed_surface_path,
                base_thickness_mm=2.4,
            )
            depth_data_to_3d_model(
                depth_path,
                output_stl_path=root / "rectangular.stl",
                target_dimension=-1,
                z_scale=10.0,
                max_xy_size=48.0,
                source_image=source_path,
                trim_top_background=False,
                surface_output_path=rectangular_surface_path,
                base_thickness_mm=2.4,
            )

            trimmed = np.load(trimmed_surface_path)
            rectangular = np.load(rectangular_surface_path)
            top, left, bottom, right = trimmed_stats["surface_grid_transform"][
                "crop_bbox_rc"
            ]
            rectangular_crop = rectangular[top:bottom, left:right]
            common = np.isfinite(trimmed) & np.isfinite(rectangular_crop)

            self.assertGreater(np.count_nonzero(~np.isfinite(trimmed)), 0)
            np.testing.assert_array_equal(trimmed[common], rectangular_crop[common])
            self.assertEqual(
                trimmed_stats["top_silhouette"]["application_stage"],
                "final_mesh_emission",
            )

    def test_top_silhouette_uses_alpha_without_color_guessing(self):
        source = np.zeros((30, 40, 4), dtype=np.uint8)
        source[..., :3] = (220, 30, 180)
        source[12:, 8:32, 3] = 255

        silhouette, stats = _top_silhouette_mask(source, source.shape[:2], padding_px=0)
        skyline_rows = np.argmax(np.flip(silhouette, axis=1), axis=0)

        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["source_alpha_used"])
        self.assertEqual(stats["method"], "alpha_silhouette_v2")
        np.testing.assert_array_equal(skyline_rows[9:31], np.full(22, 12))
        self.assertFalse(np.any(silhouette[:, :7]))
        self.assertFalse(np.any(silhouette[:, 33:]))
        self.assertEqual(stats["interpolated_column_count"], 0)

    def test_top_silhouette_fails_closed_without_a_structural_boundary(self):
        source = np.full((30, 40, 3), 90, dtype=np.uint8)

        silhouette, stats = _top_silhouette_mask(source, source.shape[:2], padding_px=0)

        np.testing.assert_array_equal(silhouette, np.ones(source.shape[:2], dtype=bool))
        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "no_trustworthy_structural_boundary")

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

    def test_print_filter_uses_full_detail_basis_before_final_xy_scaling(self):
        values = np.tile(np.linspace(0.0, 1.0, 401, dtype=np.float32), (301, 1))

        filtered, stats = _prepare_relief_for_printing(
            values,
            max_xy_size=40.0,
            minimum_feature_mm=0.8,
            detail_basis_mm=256.0,
        )

        self.assertEqual(filtered.shape, values.shape)
        self.assertTrue(stats["enabled"])
        self.assertFalse(stats["resampled"])
        self.assertTrue(stats["processed_before_final_xy_scale"])
        self.assertAlmostEqual(stats["detail_basis_mm"], 256.0)
        self.assertAlmostEqual(stats["processing_mesh_sample_pitch_mm"], 256.0 / 400.0)
        self.assertAlmostEqual(stats["mesh_sample_pitch_mm"], 40.0 / 400.0)
        self.assertAlmostEqual(stats["emitted_mesh_sample_pitch_mm"], 40.0 / 400.0)
        self.assertAlmostEqual(stats["final_xy_scale_ratio"], 40.0 / 256.0)

    def test_emitted_printability_audits_diagonal_stl_edges(self):
        values = np.array(
            [
                [1.5, 3.0],
                [0.0, 1.5],
            ],
            dtype=np.float32,
        )

        audit = _audit_emitted_relief_printability(
            values,
            sample_pitch_mm=1.0,
            max_relief_slope=2.0,
            minimum_feature_mm=0.8,
            final_xy_scale_ratio=1.0,
        )

        self.assertFalse(audit["slope_limit_passed"])
        self.assertEqual(audit["edge_sample_counts"]["cell_diagonal"], 1)
        self.assertAlmostEqual(audit["slope_max_mm_per_mm"], 3.0 / np.sqrt(2.0))

    def test_emitted_printability_ignores_edges_outside_valid_cells(self):
        values = np.array(
            [
                [0.0, 0.0, 10.0],
                [0.0, 0.0, np.nan],
            ],
            dtype=np.float32,
        )

        audit = _audit_emitted_relief_printability(
            values,
            sample_pitch_mm=1.0,
            max_relief_slope=2.0,
            minimum_feature_mm=0.8,
            final_xy_scale_ratio=1.0,
        )

        self.assertTrue(audit["slope_limit_passed"])
        self.assertEqual(audit["slope_violation_edge_count"], 0)
        self.assertAlmostEqual(audit["slope_max_mm_per_mm"], 0.0)

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

    def test_gradient_compression_can_defer_only_edge_failures_to_post_blend_audit(self):
        rows, cols = np.indices((40, 40), dtype=np.float32)
        values = 0.02 * rows + 0.01 * cols
        values[:, 30:] += 10.0

        compressed, stats = _compress_relief_gradients(
            values,
            sample_pitch_mm=0.4,
            max_slope_mm_per_mm=2.0,
            maximum_output_edge_p99_ratio=0.1,
            maximum_output_edge_ratio=0.1,
            allow_edge_only_candidate=True,
        )

        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["provisional_edge_only_candidate"])
        self.assertFalse(stats["quality_gates"]["passed"])
        self.assertTrue(stats["quality_gates"]["failures"])
        self.assertTrue(
            set(stats["quality_gates"]["failures"])
            <= {
                "cardinal_edge_p99",
                "cardinal_edge_max",
                "diagonal_edge_p99",
                "diagonal_edge_max",
            }
        )
        self.assertFalse(np.array_equal(compressed, values))

    def test_selected_relief_accepts_edge_only_candidate_after_exact_surface_audit(self):
        rows, cols = np.indices((48, 56), dtype=np.float32)
        selected = ((rows - 25.0) ** 2 / 260.0 + (cols - 28.0) ** 2 / 360.0) <= 1.0
        values = 0.02 * rows + 0.01 * cols
        candidate = values.copy()
        candidate[selected] += 0.12 * np.sin(cols[selected] * 0.4)
        provisional = {
            "enabled": True,
            "reason": "pending_baseline_aware_post_blend_audit",
            "provisional_edge_only_candidate": True,
            "max_neighbor_step_mm": 0.8,
            "detail_preservation": {
                "available": True,
                "components": [],
            },
            "quality_gates": {
                "passed": False,
                "failures": ["cardinal_edge_max"],
                "minimum_detail_correlation": 0.65,
                "minimum_detail_rms_retention": 0.15,
                "maximum_detail_rms_retention": 2.0,
                "maximum_output_edge_p99_ratio": 12.0,
                "maximum_output_edge_ratio": 24.0,
                "minimum_height_span_ratio": 0.5,
                "maximum_height_span_ratio": 1.15,
                "maximum_correction_span_ratio": 0.9,
            },
        }
        accepted_audit = {
            "enabled": True,
            "quality_gates": {
                **provisional["quality_gates"],
                "passed": True,
                "failures": [],
            },
        }
        with (
            patch.object(
                pic_to_3d,
                "_compress_relief_gradients",
                return_value=(candidate, provisional),
            ) as compress_mock,
            patch.object(
                pic_to_3d,
                "_restore_face_laplacian_detail",
                return_value=(candidate, {"enabled": False}),
            ),
            patch.object(
                pic_to_3d,
                "_audit_bounded_compression_surface",
                return_value=accepted_audit,
            ) as audit_mock,
        ):
            output, stats, slope_stats = _compress_selected_relief_surface(
                values,
                selected,
                sample_pitch_mm=0.4,
                max_slope_mm_per_mm=2.0,
                sample_pitch_source="physical_size",
            )

        self.assertTrue(compress_mock.call_args.kwargs["allow_edge_only_candidate"])
        audit_mock.assert_called_once()
        self.assertTrue(audit_mock.call_args.kwargs["reject_direction_reversals"])
        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["quality_gates"]["passed"])
        self.assertTrue(stats["provisional_edge_only_candidate_accepted"])
        self.assertFalse(stats["pre_blend_quality_gates"]["passed"])
        self.assertIs(slope_stats, stats)
        self.assertFalse(np.array_equal(output, values))

    def test_selected_relief_rejects_edge_only_candidate_when_exact_surface_audit_fails(self):
        rows, cols = np.indices((40, 48), dtype=np.float32)
        selected = ((rows - 21.0) ** 2 / 210.0 + (cols - 24.0) ** 2 / 280.0) <= 1.0
        values = 0.02 * rows + 0.01 * cols
        candidate = values + selected * 0.2
        provisional = {
            "enabled": True,
            "reason": "pending_baseline_aware_post_blend_audit",
            "provisional_edge_only_candidate": True,
            "max_neighbor_step_mm": 0.8,
            "detail_preservation": {"available": True, "components": []},
            "quality_gates": {
                "passed": False,
                "failures": ["cardinal_edge_max"],
                "minimum_detail_correlation": 0.65,
                "minimum_detail_rms_retention": 0.15,
                "maximum_detail_rms_retention": 2.0,
                "maximum_output_edge_p99_ratio": 12.0,
                "maximum_output_edge_ratio": 24.0,
                "minimum_height_span_ratio": 0.5,
                "maximum_height_span_ratio": 1.15,
                "maximum_correction_span_ratio": 0.9,
            },
        }
        rejected_audit = {
            "enabled": False,
            "reason": "quality_gate",
            "quality_gates": {
                **provisional["quality_gates"],
                "passed": False,
                "failures": ["cardinal_edge_max"],
            },
        }
        fallback_stats = {"enabled": True, "method": "test_fallback"}
        with (
            patch.object(
                pic_to_3d,
                "_compress_relief_gradients",
                return_value=(candidate, provisional),
            ),
            patch.object(
                pic_to_3d,
                "_restore_face_laplacian_detail",
                return_value=(candidate, {"enabled": False}),
            ),
            patch.object(
                pic_to_3d,
                "_audit_bounded_compression_surface",
                return_value=rejected_audit,
            ),
            patch.object(
                pic_to_3d,
                "_limit_positive_relief_slope",
                return_value=(values, fallback_stats),
            ),
        ):
            output, stats, slope_stats = _compress_selected_relief_surface(
                values,
                selected,
                sample_pitch_mm=0.4,
                max_slope_mm_per_mm=2.0,
                sample_pitch_source="physical_size",
            )

        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["reason"], "quality_gate")
        self.assertEqual(
            stats["post_blend_rejection_reason"],
            "post_blend_quality_gate",
        )
        self.assertFalse(stats["quality_gates"]["passed"])
        self.assertIs(slope_stats, fallback_stats)
        np.testing.assert_array_equal(output, values)

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
            surface_path = root / "surface.npy"
            reference_surface_path = root / "reference-surface.npy"
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
                feature_weight_mask=np.ones(depth.shape, dtype=np.float32),
                printable_feature_depth_mm=0.8,
                surface_output_path=surface_path,
                reference_surface_output_path=reference_surface_path,
            )
            stl_exists = stl_path.is_file()
            surface = np.load(surface_path)
            reference_surface = np.load(reference_surface_path)

        compression = postprocess["face_height_stabilization"]["gradient_compression"]
        self.assertEqual(
            postprocess["face_boundary_alignment"]["method"],
            "screened_gradient_domain_compression",
        )
        self.assertTrue(compression["enabled"])
        self.assertTrue(compression["quality_gates"]["passed"])
        self.assertAlmostEqual(compression["screening_weight"], 0.25)
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
        self.assertTrue(postprocess["printable_feature_depth"]["enabled"])
        self.assertFalse(
            postprocess["printable_feature_depth"][
                "suppressed_after_screened_face_reconstruction"
            ]
        )
        self.assertTrue(
            postprocess["printable_feature_depth"]["screened_face_reconstruction"]
        )
        self.assertAlmostEqual(postprocess["printable_feature_depth_mm"], 0.8)
        self.assertAlmostEqual(postprocess["effective_printable_feature_depth_mm"], 0.8)
        self.assertTrue(postprocess["post_feature_slope_guard"]["enabled"])
        self.assertFalse(
            postprocess["post_feature_slope_guard"]["fell_back_to_baseline"]
        )
        self.assertTrue(postprocess["face_detail_guard"]["enabled"])
        self.assertTrue(postprocess["face_detail_guard"]["final"]["available"])
        self.assertGreater(postprocess["face_detail_guard"]["applied_scale"], 0.0)
        transform = postprocess["surface_grid_transform"]
        self.assertEqual(transform["input_depth_shape"], [48, 48])
        self.assertEqual(transform["target_depth_shape"], [48, 48])
        self.assertTrue(transform["flip_x"])
        self.assertEqual(transform["emitted_shape"], postprocess["mesh_grid_shape"])
        self.assertEqual(transform["mask_interpolation"], "nearest")
        self.assertEqual(surface.shape, reference_surface.shape)
        self.assertEqual(list(reference_surface.shape), postprocess["mesh_grid_shape"])
        self.assertTrue(np.all(np.isfinite(reference_surface)))
        self.assertTrue(postprocess["reference_surface"]["emitted"])
        self.assertEqual(
            postprocess["reference_surface"]["kind"],
            "pre_high_relief_post_shape_surface",
        )

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

    def test_high_face_relief_retries_only_cardinal_edge_p99_rejection(self):
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
                "screening_weight": 0.25,
                "output_edge_ratio_p99": 13.2,
                "quality_gates": {
                    "passed": False,
                    "failures": ["cardinal_edge_p99"],
                },
            }
            accepted = {
                "enabled": True,
                "method": "screened_gradient_domain_compression",
                "screening_weight": 0.1,
                "output_edge_ratio_p99": 10.9,
                "detail_preservation": {"available": False},
                "quality_gates": {"passed": True, "failures": []},
            }
            bounded_audit = {
                "enabled": True,
                "output_edge_ratio_p99": 1.1,
                "output_edge_ratio_max": 1.2,
                "quality_gates": {"passed": True, "failures": []},
            }
            with (
                patch.object(
                    pic_to_3d,
                    "_compress_relief_gradients",
                    side_effect=((depth, rejected), (depth, accepted)),
                ) as compress_mock,
                patch.object(
                    pic_to_3d,
                    "_audit_bounded_compression_surface",
                    return_value=bounded_audit,
                ),
            ):
                postprocess = depth_data_to_3d_model(
                    depth_path,
                    output_stl_path=str(root / "retry.stl"),
                    target_dimension=-1,
                    z_scale=30.0,
                    sigma=0.0,
                    relief_gamma=1.0,
                    detail_boost=0.0,
                    low_percentile=0.0,
                    high_percentile=100.0,
                    face_region_mask=face,
                )

        self.assertEqual(compress_mock.call_count, 2)
        retry_call = compress_mock.call_args_list[1]
        self.assertAlmostEqual(
            retry_call.kwargs["screening_weight"],
            pic_to_3d.HIGH_RELIEF_FACE_CARDINAL_EDGE_RETRY_WEIGHT,
        )
        self.assertEqual(
            postprocess["face_boundary_alignment"].get("method"),
            "screened_gradient_domain_compression",
        )
        retry = postprocess["face_height_stabilization"]["gradient_compression"][
            "adaptive_screening_retry"
        ]
        self.assertTrue(retry["attempted"])
        self.assertEqual(retry["reason"], "retry_candidate_accepted")
        self.assertEqual(retry["trigger_failures"], ["cardinal_edge_p99"])
        self.assertAlmostEqual(retry["primary_output_edge_ratio_p99"], 13.2)
        self.assertAlmostEqual(retry["retry_output_edge_ratio_p99"], 10.9)
        self.assertIs(
            postprocess["face_height_stabilization"]["gradient_compression_attempt"],
            rejected,
        )
        self.assertIs(
            postprocess["face_height_stabilization"]["gradient_compression_selected"],
            accepted,
        )

    def test_high_face_relief_records_rejected_cardinal_edge_retry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            depth_path = root / "depth.npy"
            rows, cols = np.indices((32, 32), dtype=np.float32)
            depth = 0.2 + 0.6 * np.exp(
                -((rows - 16.0) ** 2 + (cols - 16.0) ** 2) / 100.0
            )
            face = ((rows - 16.0) ** 2 + (cols - 16.0) ** 2) <= 100.0
            np.save(depth_path, depth)
            primary = {
                "enabled": False,
                "method": "screened_gradient_domain_compression",
                "screening_weight": 0.25,
                "output_edge_ratio_p99": 13.2,
                "quality_gates": {
                    "passed": False,
                    "failures": ["cardinal_edge_p99"],
                },
            }
            retry = {
                "enabled": False,
                "method": "screened_gradient_domain_compression",
                "screening_weight": 0.1,
                "output_edge_ratio_p99": 12.4,
                "quality_gates": {
                    "passed": False,
                    "failures": ["cardinal_edge_p99"],
                },
            }
            with patch.object(
                pic_to_3d,
                "_compress_relief_gradients",
                side_effect=((depth, primary), (depth, retry)),
            ) as compress_mock:
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

        self.assertEqual(compress_mock.call_count, 2)
        self.assertNotEqual(
            postprocess["face_boundary_alignment"].get("method"),
            "screened_gradient_domain_compression",
        )
        attempt = postprocess["face_height_stabilization"][
            "gradient_compression_attempt"
        ]
        adaptive = attempt["adaptive_screening_retry"]
        self.assertTrue(adaptive["attempted"])
        self.assertEqual(adaptive["reason"], "retry_candidate_rejected")
        self.assertEqual(adaptive["retry_quality_failures"], ["cardinal_edge_p99"])

    def test_high_face_relief_retries_detail_loss_then_requires_post_blend_audit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            depth_path = root / "depth.npy"
            rows, cols = np.indices((32, 32), dtype=np.float32)
            depth = 0.2 + 0.6 * np.exp(
                -((rows - 16.0) ** 2 + (cols - 16.0) ** 2) / 100.0
            )
            face = ((rows - 16.0) ** 2 + (cols - 16.0) ** 2) <= 100.0
            np.save(depth_path, depth)
            primary = {
                "enabled": False,
                "method": "screened_gradient_domain_compression",
                "quality_gates": {
                    "passed": False,
                    "failures": ["cardinal_edge_p99", "detail_correlation"],
                },
            }
            retry = {
                "enabled": True,
                "method": "screened_gradient_domain_compression",
                "screening_weight": pic_to_3d.HIGH_RELIEF_FACE_DETAIL_RETRY_WEIGHT,
                "provisional_edge_only_candidate": True,
                "max_neighbor_step_mm": 2.0,
                "output_edge_ratio_p99": 8.0,
                "detail_preservation": {
                    "available": True,
                    "correlation": 0.97,
                    "rms_retention": 0.82,
                },
                "quality_gates": {
                    "passed": False,
                    "failures": ["diagonal_edge_max"],
                },
            }
            bounded_audit = {
                "enabled": True,
                "output_edge_ratio_p99": 1.1,
                "output_edge_ratio_max": 1.2,
                "quality_gates": {"passed": True, "failures": []},
            }
            with (
                patch.object(
                    pic_to_3d,
                    "_compress_relief_gradients",
                    side_effect=((depth, primary), (depth, retry)),
                ) as compress_mock,
                patch.object(
                    pic_to_3d,
                    "_audit_bounded_compression_surface",
                    return_value=bounded_audit,
                ),
            ):
                postprocess = depth_data_to_3d_model(
                    depth_path,
                    output_stl_path=str(root / "retry.stl"),
                    target_dimension=-1,
                    z_scale=40.0,
                    sigma=0.0,
                    relief_gamma=1.0,
                    detail_boost=0.0,
                    low_percentile=0.0,
                    high_percentile=100.0,
                    face_region_mask=face,
                )

        self.assertEqual(compress_mock.call_count, 2)
        retry_call = compress_mock.call_args_list[1]
        self.assertAlmostEqual(
            retry_call.kwargs["screening_weight"],
            pic_to_3d.HIGH_RELIEF_FACE_DETAIL_RETRY_WEIGHT,
        )
        self.assertTrue(retry_call.kwargs["allow_edge_only_candidate"])
        adaptive = postprocess["face_height_stabilization"][
            "gradient_compression"
        ]["adaptive_screening_retry"]
        self.assertEqual(adaptive["kind"], "high_detail_edge_audited")
        self.assertEqual(
            adaptive["reason"],
            "retry_candidate_pending_post_blend_audit",
        )
        self.assertTrue(adaptive["retry_provisional_edge_only_candidate"])
        projection = postprocess["face_height_stabilization"][
            "gradient_compression"
        ]["direction_reversal_projection"]
        self.assertTrue(projection["passed"])
        self.assertEqual(
            postprocess["face_boundary_alignment"]["method"],
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
            "final_conflict_relaxed_selection",
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
        selection_appearance = postprocess["surface_appearance_agreement"][
            "selection_nonface"
        ]
        self.assertTrue(selection_compression["enabled"])
        self.assertTrue(selection_compression["face_protection_passed"])
        self.assertAlmostEqual(selection_compression["screening_weight"], 2.0)
        self.assertAlmostEqual(
            selection_compression["detail_gradient_retention"],
            0.9,
        )
        self.assertGreater(selection_compression["retained_detail_gradient_pairs"], 0)
        self.assertEqual(
            selection_compression["protected_region_blend"]["protected_correction_max_mm"],
            0.0,
        )
        self.assertTrue(selection_appearance["available"])
        self.assertEqual(selection_appearance["candidate_coverage_ratio"], 1.0)
        self.assertEqual(selection_appearance["component_count"], 1)
        self.assertEqual(len(selection_appearance["components"]), 1)
        self.assertTrue(selection_appearance["components"][0]["available"])
        self.assertTrue(stl_exists)

    def test_low_face_relief_uses_face_aware_selection_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            depth_path = root / "depth.npy"
            surface_path = root / "surface.npy"
            rows, cols = np.indices((72, 96), dtype=np.float32)
            face = (
                (rows - 31.0) ** 2 / 210.0
                + (cols - 36.0) ** 2 / 150.0
            ) <= 1.0
            torso = (
                (rows - 53.0) ** 2 / 300.0
                + (cols - 42.0) ** 2 / 330.0
            ) <= 1.0
            selected = face | torso
            depth = 0.08 + 0.0012 * rows + 0.0018 * cols
            depth += selected * (
                0.52
                + 0.05 * np.cos(rows / 4.0)
                + 0.04 * np.sin(cols / 3.0)
            )
            np.save(depth_path, depth.astype(np.float32))

            postprocess = depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(root / "selected-low-face.stl"),
                target_dimension=-1,
                z_scale=10.0,
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
                selection_background_depth_ratio=0.65,
                surface_output_path=surface_path,
            )

        selection_compression = postprocess["selection_gradient_compression"]
        self.assertNotEqual(
            postprocess["face_boundary_alignment"]["reason"],
            "no_face_region",
        )
        self.assertNotEqual(
            postprocess["face_surface_protection"]["reason"],
            "selection_gradient_domain",
        )
        self.assertFalse(selection_compression["face_protection_passed"])
        self.assertFalse(selection_compression["enabled"])
        self.assertEqual(
            selection_compression["reason"],
            "face_protection_gate",
        )

    def test_subject_locked_selection_preserves_subject_surface(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            depth_path = root / "depth.npy"
            whole_surface_path = root / "whole-surface.npy"
            selected_surface_path = root / "selected-surface.npy"
            rows, cols = np.indices((72, 96), dtype=np.float32)
            face = (
                (rows - 31.0) ** 2 / 210.0
                + (cols - 36.0) ** 2 / 150.0
            ) <= 1.0
            torso = (
                (rows - 53.0) ** 2 / 300.0
                + (cols - 42.0) ** 2 / 330.0
            ) <= 1.0
            selected = face | torso
            depth = 0.08 + 0.0012 * rows + 0.0018 * cols
            depth += selected * (
                0.52
                + 0.05 * np.cos(rows / 4.0)
                + 0.04 * np.sin(cols / 3.0)
            )
            np.save(depth_path, depth.astype(np.float32))
            common = {
                "target_dimension": -1,
                "z_scale": 10.0,
                "max_xy_size": 48.0,
                "sigma": 0.35,
                "relief_gamma": 1.0,
                "detail_boost": 0.0,
                "trim_top_background": True,
                "low_percentile": 0.0,
                "high_percentile": 100.0,
                "base_border_px": 1,
                "minimum_feature_mm": 0.8,
                "max_relief_slope": 2.0,
                "face_region_mask": face,
            }

            depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(root / "whole.stl"),
                surface_output_path=whole_surface_path,
                **common,
            )
            postprocess = depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(root / "selected.stl"),
                selection_region_mask=selected,
                selection_background_depth_ratio=0.65,
                selection_subject_lock=True,
                surface_output_path=selected_surface_path,
                **common,
            )
            whole_surface = np.load(whole_surface_path)
            selected_surface = np.load(selected_surface_path)
            selected_mesh = trimesh.load_mesh(root / "selected.stl", force="mesh")
            selected_mesh_is_watertight = bool(selected_mesh.is_watertight)
            selected_mesh_winding_is_consistent = bool(
                selected_mesh.is_winding_consistent
            )

        emitted_selection = np.flip(selected, axis=1)
        subject_interior = binary_erosion(emitted_selection, iterations=3)
        self.assertGreater(np.count_nonzero(subject_interior), 100)
        np.testing.assert_array_equal(
            selected_surface[subject_interior],
            whole_surface[subject_interior],
        )
        self.assertTrue(postprocess["selection_subject_lock"])
        self.assertEqual(selected_surface.shape, whole_surface.shape)
        np.testing.assert_array_equal(
            np.isfinite(selected_surface),
            np.isfinite(whole_surface),
        )
        self.assertEqual(
            postprocess["selection_gradient_compression"]["reason"],
            "subject_surface_locked",
        )
        self.assertTrue(
            postprocess["selection_background_physical_cap"]["emission_passed"]
        )
        self.assertTrue(selected_mesh_is_watertight)
        self.assertTrue(selected_mesh_winding_is_consistent)

    def test_selected_emission_removes_only_unselected_surface_pixels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            depth_path = root / "depth.npy"
            context_surface_path = root / "context-surface.npy"
            emitted_surface_path = root / "emitted-surface.npy"
            emitted_reference_path = root / "emitted-reference.npy"
            rows, cols = np.indices((48, 72), dtype=np.float32)
            selected = np.zeros((48, 72), dtype=bool)
            selected[9:24, 5:24] = True
            selected[12:, 29:44] = True
            selected[14:31, 47:68] = True
            selected[:22, 34:39] = True
            depth = 0.08 + 0.0017 * rows + 0.0021 * cols
            depth += selected * (
                0.42
                + 0.04 * np.cos(rows / 4.0)
                + 0.03 * np.sin(cols / 3.0)
            )
            np.save(depth_path, depth.astype(np.float32))
            common = {
                "target_dimension": -1,
                "z_scale": 10.0,
                "max_xy_size": 48.0,
                "sigma": 0.35,
                "relief_gamma": 1.0,
                "detail_boost": 0.0,
                "trim_top_background": False,
                "low_percentile": 0.0,
                "high_percentile": 100.0,
                "base_border_px": 1,
                "minimum_feature_mm": 0.8,
                "max_relief_slope": 2.0,
                "selection_region_mask": selected,
                "selection_background_depth_ratio": 0.65,
                "selection_subject_lock": True,
            }

            depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(root / "context.stl"),
                surface_output_path=context_surface_path,
                **common,
            )
            postprocess = depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(root / "selected-only.stl"),
                surface_output_path=emitted_surface_path,
                reference_surface_output_path=emitted_reference_path,
                selection_emission_only=True,
                **common,
            )
            context_surface = np.load(context_surface_path)
            emitted_surface = np.load(emitted_surface_path)
            emitted_reference = np.load(emitted_reference_path)
            crop_top, crop_left, crop_bottom, crop_right = postprocess[
                "surface_grid_transform"
            ]["crop_bbox_rc"]
            context_surface = context_surface[
                crop_top:crop_bottom,
                crop_left:crop_right,
            ]
            emitted_selection = np.flip(selected, axis=1)[
                crop_top:crop_bottom,
                crop_left:crop_right,
            ]
            emitted_mesh = trimesh.load_mesh(root / "selected-only.stl", force="mesh")

        self.assertEqual(emitted_surface.shape, context_surface.shape)
        emitted_coverage = np.isfinite(emitted_surface)
        backing_only = emitted_coverage & ~emitted_selection
        self.assertTrue(np.all(emitted_coverage[emitted_selection]))
        np.testing.assert_array_equal(
            emitted_surface[emitted_selection],
            context_surface[emitted_selection],
        )
        self.assertTrue(np.any(backing_only))
        np.testing.assert_array_equal(
            emitted_surface[backing_only],
            np.full(np.count_nonzero(backing_only), 0.01, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            emitted_reference[backing_only],
            emitted_surface[backing_only],
        )
        np.testing.assert_array_equal(
            np.isfinite(emitted_reference),
            emitted_coverage,
        )
        emission = postprocess["selection_emission"]
        self.assertTrue(emission["enabled"])
        self.assertEqual(
            emission["method"],
            "full_scene_depth_grounded_selection_emission_v2",
        )
        self.assertEqual(emission["retained_unselected_pixels"], 0)
        self.assertEqual(emission["removed_selected_pixels"], 0)
        self.assertEqual(emission["retained_selection_ratio"], 1.0)
        self.assertGreater(emission["backing_foundation_pixels"], 0)
        self.assertEqual(
            emission["backing_foundation_pixels"]
            + emission["backing_connector_pixels"],
            np.count_nonzero(backing_only),
        )
        self.assertEqual(emission["unsupported_selected_mesh_pixels"], 0)
        self.assertEqual(
            emission["backing_connector"]["component_count_after"],
            1,
        )
        self.assertTrue(emitted_mesh.is_watertight)
        self.assertTrue(emitted_mesh.is_winding_consistent)
        self.assertEqual(len(emitted_mesh.split(only_watertight=False)), 1)

    def test_selected_surface_appearance_holds_across_physical_sample_pitches(self):
        from backend.benchmark.run_relief_visual_sweep import (
            _sweep_specs,
            _topology_scene,
        )

        source, face, selection = _topology_scene(_sweep_specs()[2])
        observed_correlations = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for factor in (1, 2):
                with self.subTest(factor=factor):
                    if factor == 1:
                        sampled_source = source
                        sampled_face = face
                        sampled_selection = selection
                    else:
                        sampled_source = zoom(source, factor, order=1)[:-1, :-1]
                        sampled_face = zoom(
                            face.astype(np.uint8), factor, order=0
                        )[:-1, :-1].astype(bool)
                        sampled_selection = zoom(
                            selection.astype(np.uint8), factor, order=0
                        )[:-1, :-1].astype(bool)
                    sample_pitch_mm = 48.0 / float(sampled_source.shape[0] - 1)
                    composed, _ = compose_selection_depth_with_context(
                        sampled_source,
                        sampled_selection,
                        relief_height_mm=40.0,
                        sample_pitch_mm=sample_pitch_mm,
                        max_slope_mm_per_mm=2.0,
                        background_depth_ratio=0.45,
                        background_feather_mm=1.5,
                        background_smoothing_mm=0.6,
                    )
                    depth_path = root / f"depth-{factor}.npy"
                    stl_path = root / f"relief-{factor}.stl"
                    np.save(depth_path, composed.astype(np.float32))
                    postprocess = depth_data_to_3d_model(
                        depth_path,
                        output_stl_path=str(stl_path),
                        target_dimension=-1,
                        z_scale=40.0,
                        max_xy_size=48.0,
                        sigma=0.0,
                        relief_gamma=1.0,
                        detail_boost=0.0,
                        low_percentile=0.0,
                        high_percentile=100.0,
                        base_border_px=1,
                        value_transform="linear",
                        minimum_feature_mm=2.0 * sample_pitch_mm,
                        max_relief_slope=2.0,
                        face_region_mask=sampled_face,
                        selection_region_mask=sampled_selection,
                        selection_background_depth_ratio=0.45,
                    )

                    compression = postprocess["selection_gradient_compression"]
                    appearance = postprocess["surface_appearance_agreement"]
                    selected = appearance["selection_nonface"]
                    self.assertTrue(compression["face_protection_passed"])
                    if not compression["enabled"]:
                        self.assertEqual(compression["reason"], "quality_gate")
                        self.assertFalse(compression["quality_gates"]["passed"])
                        self.assertTrue(compression["quality_gates"]["failures"])
                    self.assertAlmostEqual(compression["screening_weight"], 2.0)
                    self.assertAlmostEqual(
                        compression["detail_gradient_retention"],
                        0.9,
                    )
                    self.assertAlmostEqual(
                        compression["max_neighbor_step_mm"],
                        2.0 * sample_pitch_mm,
                        places=5,
                    )
                    self.assertTrue(selected["available"])
                    self.assertGreaterEqual(selected["normal_mean_cosine"], 0.95)
                    self.assertGreaterEqual(selected["normal_p05_cosine"], 0.85)
                    self.assertLessEqual(selected["normal_angle_p95_deg"], 30.0)
                    self.assertGreaterEqual(
                        selected["minimum_lighting_correlation"],
                        0.8,
                    )
                    self.assertLessEqual(selected["maximum_lighting_mae"], 0.08)
                    self.assertGreaterEqual(
                        selected["minimum_lighting_rms_retention"],
                        0.75,
                    )
                    self.assertLessEqual(
                        selected["maximum_lighting_rms_retention"],
                        1.4,
                    )
                    self.assertGreaterEqual(
                        appearance["face"]["minimum_lighting_correlation"],
                        0.8,
                    )
                    self.assertGreaterEqual(
                        appearance["background"]["minimum_lighting_correlation"],
                        0.95,
                    )
                    self.assertTrue(
                        postprocess["selection_background_physical_cap"][
                            "emission_passed"
                        ]
                    )
                    self.assertTrue(stl_path.is_file())
                    observed_correlations.append(
                        selected["minimum_lighting_correlation"]
                    )

        self.assertLessEqual(max(observed_correlations) - min(observed_correlations), 0.1)

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

    def test_flatten_border_can_preserve_selected_edge_pixels(self):
        values = np.ones((8, 10), dtype=np.float32)
        selected = np.zeros(values.shape, dtype=bool)
        selected[:4, :3] = True

        flattened = _flatten_border(values, 1, preserve_mask=selected)

        self.assertTrue(np.all(flattened[0, :3] == 1))
        self.assertTrue(np.all(flattened[:4, 0] == 1))
        self.assertTrue(np.all(flattened[0, 3:] == 0))
        self.assertTrue(np.all(flattened[4:, 0] == 0))
        self.assertTrue(np.all(flattened[-1, :] == 0))
        self.assertTrue(np.all(flattened[:, -1] == 0))

    def test_explicit_backing_plate_is_flat_and_preserves_relief_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            depth_path = root / "depth.npy"
            rows, cols = np.indices((12, 16), dtype=np.float32)
            np.save(depth_path, 0.2 + 0.01 * rows + 0.02 * cols)
            for base_border_px in (0, 2):
                with self.subTest(base_border_px=base_border_px):
                    common = {
                        "target_dimension": -1,
                        "z_scale": 10.0,
                        "max_xy_size": 32.0,
                        "sigma": 0.0,
                        "relief_gamma": 1.0,
                        "detail_boost": 0.0,
                        "low_percentile": 0.0,
                        "high_percentile": 100.0,
                        "base_border_px": base_border_px,
                    }
                    legacy_surface_path = root / f"legacy-surface-{base_border_px}.npy"
                    legacy_stl_path = root / f"legacy-{base_border_px}.stl"
                    depth_data_to_3d_model(
                        depth_path,
                        output_stl_path=str(legacy_stl_path),
                        surface_output_path=legacy_surface_path,
                        **common,
                    )
                    backed_surface_path = root / f"backed-surface-{base_border_px}.npy"
                    backed_stl_path = root / f"backed-{base_border_px}.stl"
                    postprocess = depth_data_to_3d_model(
                        depth_path,
                        output_stl_path=str(backed_stl_path),
                        surface_output_path=backed_surface_path,
                        base_thickness_mm=2.4,
                        **common,
                    )
                    legacy_surface = np.load(legacy_surface_path)
                    backed_surface = np.load(backed_surface_path)
                    legacy_mesh = trimesh.load_mesh(legacy_stl_path, force="mesh")
                    backed_mesh = trimesh.load_mesh(backed_stl_path, force="mesh")

                    np.testing.assert_allclose(
                        backed_surface - 2.4,
                        legacy_surface - 0.01,
                        atol=2e-6,
                    )
                    np.testing.assert_allclose(
                        backed_mesh.bounds[:, :2],
                        legacy_mesh.bounds[:, :2],
                        atol=1e-6,
                    )
                    self.assertEqual(len(backed_mesh.faces), len(legacy_mesh.faces))
                    self.assertAlmostEqual(
                        float(np.nanmin(backed_surface)),
                        2.4,
                        places=5,
                    )
                    self.assertAlmostEqual(float(backed_mesh.bounds[0, 2]), 0.0, places=6)
                    self.assertGreaterEqual(float(backed_mesh.bounds[1, 2]), 2.4)
                    intermediate_backing = (
                        (backed_mesh.vertices[:, 2] > 1e-6)
                        & (backed_mesh.vertices[:, 2] < 2.4 - 1e-6)
                    )
                    self.assertFalse(np.any(intermediate_backing))
                    self.assertTrue(backed_mesh.is_watertight)
                    self.assertTrue(backed_mesh.is_winding_consistent)
                    self.assertTrue(backed_mesh.is_volume)
                    self.assertEqual(postprocess["backing_plate"]["thickness_mm"], 2.4)
                    self.assertEqual(postprocess["backing_plate"]["bottom_z_mm"], 0.0)
                    self.assertEqual(postprocess["backing_plate"]["top_z_mm"], 2.4)

    def test_positive_context_preserves_cropped_selection_border_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            rows, cols = np.indices((48, 64), dtype=np.float32)
            face = (
                np.square((rows - 20.0) / 14.0)
                + np.square((cols - 5.0) / 13.0)
                <= 1.0
            )
            torso = (rows >= 31.0) & (cols <= 20.0)
            selected = face | torso
            depth = 0.08 + 0.0012 * rows + 0.0018 * cols
            depth += selected * (
                0.48
                + 0.05 * np.cos(rows / 5.0)
                + 0.04 * np.sin(cols / 4.0)
            )
            depth_path = root / "depth.npy"
            np.save(depth_path, depth.astype(np.float32))

            positive_surface = root / "positive-surface.npy"
            depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(root / "positive.stl"),
                target_dimension=-1,
                z_scale=30.0,
                max_xy_size=25.6,
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
                selection_background_depth_ratio=0.65,
                surface_output_path=positive_surface,
            )
            legacy_surface = root / "legacy-surface.npy"
            depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(root / "legacy.stl"),
                target_dimension=-1,
                z_scale=30.0,
                max_xy_size=25.6,
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
                surface_output_path=legacy_surface,
            )

            positive = np.load(positive_surface)
            legacy = np.load(legacy_surface)
            positive_mesh = trimesh.load_mesh(root / "positive.stl", process=True)
            legacy_mesh = trimesh.load_mesh(root / "legacy.stl", process=True)

        positive_face_border_max = float(np.max(positive[6:35, -1]))
        legacy_face_border_max = float(np.max(legacy[6:35, -1]))
        self.assertGreater(positive_face_border_max, legacy_face_border_max + 1.0)
        self.assertTrue(np.allclose(positive[36:, -1], 0.01))
        self.assertAlmostEqual(float(positive[0, -1]), 0.01, places=5)
        self.assertTrue(np.allclose(legacy[36:, -1], 0.01))
        for emitted in (positive_mesh, legacy_mesh):
            self.assertTrue(emitted.is_watertight)
            self.assertTrue(emitted.is_volume)
            self.assertTrue(emitted.is_winding_consistent)
            self.assertEqual(len(emitted.split(only_watertight=False)), 1)
            self.assertGreater(float(emitted.volume), 0.0)

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

    def test_detail_basis_preserves_heightfield_when_output_xy_is_smaller(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            depth_path = root / "depth.npy"
            data = np.linspace(0.0, 1.0, 80 * 120, dtype=np.float32).reshape(80, 120)
            selection = np.zeros(data.shape, dtype=bool)
            selection[8:72, 12:108] = True
            data = np.where(selection, data, np.nan)
            np.save(depth_path, data)
            large_surface = root / "large.npy"
            small_surface = root / "small.npy"

            common = {
                "target_dimension": 120,
                "z_scale": 12,
                "invert": False,
                "sigma": 0,
                "detail_boost": 0,
                "base_border_px": 0,
                "detail_basis_mm": 256.0,
                "surface_output_path": large_surface,
            }
            large = depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(root / "large.stl"),
                max_xy_size=256.0,
                **common,
            )
            common["surface_output_path"] = small_surface
            small = depth_data_to_3d_model(
                depth_path,
                output_stl_path=str(root / "small.stl"),
                max_xy_size=40.0,
                **common,
            )

            np.testing.assert_array_equal(np.load(large_surface), np.load(small_surface))
            self.assertEqual(large["mesh_grid_shape"], small["mesh_grid_shape"])
            self.assertAlmostEqual(
                large["processing_mesh_sample_pitch_mm"],
                small["processing_mesh_sample_pitch_mm"],
            )
            self.assertNotAlmostEqual(large["mesh_sample_pitch_mm"], small["mesh_sample_pitch_mm"])
            self.assertEqual(small["mesh_grid_shape"], [64, 96])
            self.assertAlmostEqual(small["emitted_mesh_sample_pitch_mm"], 40.0 / 95.0)
            self.assertTrue(small["processed_before_final_xy_scale"])
            self.assertTrue(small["emitted_printability"]["recognition_first_oversampling"])
            expected_scale_ratio = (40.0 / 95.0) / (256.0 / 119.0)
            self.assertAlmostEqual(small["final_xy_scale_ratio"], expected_scale_ratio)
            self.assertAlmostEqual(
                small["emitted_printability"][
                    "processing_minimum_feature_scaled_to_output_mm"
                ],
                0.8 * expected_scale_ratio,
            )

            large_mesh = mesh.Mesh.from_file(str(root / "large.stl"))
            small_mesh = mesh.Mesh.from_file(str(root / "small.stl"))
            large_extents = np.ptp(large_mesh.vectors.reshape(-1, 3), axis=0)
            small_extents = np.ptp(small_mesh.vectors.reshape(-1, 3), axis=0)
            self.assertAlmostEqual(max(large_extents[:2]), 256.0, places=4)
            self.assertAlmostEqual(max(small_extents[:2]), 40.0, places=4)
            self.assertAlmostEqual(large_extents[2], small_extents[2], places=5)

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
