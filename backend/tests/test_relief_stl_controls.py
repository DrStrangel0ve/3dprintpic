import tempfile
import unittest
from pathlib import Path

import numpy as np
from stl import mesh

from backend.pic_to_3d import _flatten_border, _shape_relief_values, depth_data_to_3d_model


class ReliefStlControlsTest(unittest.TestCase):
    def test_detail_boost_lifts_local_features(self):
        base = np.tile(np.linspace(0.2, 0.8, 41, dtype=np.float32), (41, 1))
        base[20, 20] += 0.08

        plain = _shape_relief_values(base, detail_boost=0, detail_radius=2, low_percentile=0, high_percentile=100)
        boosted = _shape_relief_values(base, detail_boost=2.0, detail_radius=2, low_percentile=0, high_percentile=100)

        self.assertGreater(boosted[20, 20] - boosted[20, 19], plain[20, 20] - plain[20, 19])

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
