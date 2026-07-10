import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from stl import mesh

from backend import pic_to_3d
from backend.pic_to_3d import (
    RELIEF_VALUE_TRANSFORM_INVERSE_DEPTH,
    _flatten_border,
    _shape_relief_values,
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

            depth_data_to_3d_model(
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
            self.assertGreater(extents[2], 11.0)
            self.assertLessEqual(extents[2], 12.1)

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
