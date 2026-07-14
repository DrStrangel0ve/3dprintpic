import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.ndimage import laplace
from stl import mesh

from backend import pic_to_3d
from backend.pic_to_3d import (
    RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
    _flatten_border,
    _enhance_weighted_relief_features,
    _guard_weighted_feature_updates,
    _align_stabilized_head_to_reference_boundary,
    _attach_face_boundary_to_local_surface,
    _bridge_weighted_face_features,
    _compress_relief_gradients,
    _expand_face_region_to_depth_connected_head,
    _inject_photo_relief_detail,
    _limit_positive_relief_slope,
    _prepare_relief_for_printing,
    _restore_stabilized_face_surface,
    _resize_nan_aware,
    _shape_relief_values,
    _stabilize_face_relief_height,
    _top_silhouette_mask,
    depth_data_to_3d_model,
    relief_value_transform_for_model,
)


class ReliefStlControlsTest(unittest.TestCase):
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
        self.assertGreaterEqual(
            compression["detail_preservation"]["correlation"],
            compression["quality_gates"]["minimum_detail_correlation"],
        )
        self.assertGreaterEqual(
            compression["detail_preservation"]["rms_retention"],
            compression["quality_gates"]["minimum_detail_rms_retention"],
        )
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
