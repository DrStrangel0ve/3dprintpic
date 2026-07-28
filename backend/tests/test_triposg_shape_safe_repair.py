from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

from backend.benchmark.direct_mesh import (
    mesh_is_printable_volume,
    postprocess_mesh_for_stl,
    repair_mesh_for_printable_stl,
)
from backend.benchmark.mesh_rendering import load_mesh
from backend.benchmark.run_stl_first_smoke import build_experiments, parse_args


class TripoSGShapeSafeRepairTest(unittest.TestCase):
    def test_adaptive_voxel_close_fits_budget_without_hull(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = trimesh.creation.box(extents=(1.0, 0.65, 0.4))
            source.update_faces(np.arange(len(source.faces) - 1))
            source.remove_unreferenced_vertices()
            source_path = root / "open_box.glb"
            output_path = root / "closed_box.glb"
            source.export(source_path)
            metrics = {}

            repair_mesh_for_printable_stl(
                source_path,
                output_path,
                mode="printable",
                target_faces=6_000,
                preconditioner="adaptive-voxel-close",
                voxel_resolution=64,
                voxel_fill_method="orthographic",
                allow_convex_hull_fallback=False,
                metrics=metrics,
            )

            repaired = load_mesh(output_path)
            self.assertTrue(mesh_is_printable_volume(repaired))
            self.assertLessEqual(len(repaired.faces), 6_000)
            self.assertFalse(metrics["repair_convex_hull_used"])
            self.assertLessEqual(
                metrics["repair_adaptive_voxel_selected_resolution"],
                64,
            )
            self.assertGreater(metrics["repair_adaptive_voxel_probe_count"], 0)

    def test_printable_repair_can_fail_closed_instead_of_hulling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plane = trimesh.Trimesh(
                vertices=np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ]
                ),
                faces=np.array([[0, 1, 2], [0, 2, 3]]),
                process=False,
            )
            source_path = root / "plane.ply"
            output_path = root / "output.glb"
            plane.export(source_path)

            with self.assertRaisesRegex(RuntimeError, "fallback is disabled"):
                repair_mesh_for_printable_stl(
                    source_path,
                    output_path,
                    mode="printable",
                    allow_convex_hull_fallback=False,
                )
            self.assertFalse(output_path.exists())

    def test_bounded_voxel_smoothing_preserves_printability(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = trimesh.creation.box(extents=(1.0, 0.65, 0.4))
            source.update_faces(np.arange(len(source.faces) - 1))
            source.remove_unreferenced_vertices()
            source_path = root / "open_box.glb"
            output_path = root / "smoothed.glb"
            source.export(source_path)
            metrics = {}

            repair_mesh_for_printable_stl(
                source_path,
                output_path,
                mode="printable",
                target_faces=6_000,
                preconditioner="adaptive-voxel-close",
                voxel_resolution=64,
                smoothing_iterations=2,
                allow_convex_hull_fallback=False,
                metrics=metrics,
            )

            repaired = load_mesh(output_path)
            self.assertTrue(mesh_is_printable_volume(repaired))
            self.assertTrue(metrics["repair_smoothing_applied"])
            self.assertEqual(metrics["repair_smoothing_iterations"], 2)
            self.assertFalse(metrics["repair_convex_hull_used"])
            self.assertLess(metrics["repair_smoothing_surface_hausdorff95_normalized"], 0.03)

    def test_uniform_bbox_mode_preserves_mesh_proportions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = trimesh.creation.box(extents=(1.0, 2.0, 0.1))
            source_path = root / "source.glb"
            output_path = root / "uniform.glb"
            source.export(source_path)

            postprocess_mesh_for_stl(
                source_path,
                output_path,
                target_bbox_extents=(95.0, 95.0, 50.0),
                target_bbox_mode="uniform-max",
            )

            processed = load_mesh(output_path)
            np.testing.assert_allclose(np.max(processed.extents), 95.0, rtol=1e-6)
            np.testing.assert_allclose(
                processed.extents / np.max(processed.extents),
                source.extents / np.max(source.extents),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_triposg_smoke_defaults_to_shape_safe_mirror_input(self):
        argv = [
            "run_stl_first_smoke",
            "--include-triposg",
            "--mesh-min-bbox-dimension",
            "12",
            "--mesh-max-bbox-aspect-ratio",
            "2.25",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        experiments = {row["name"]: row for row in build_experiments(args)}
        candidate = experiments["triposg_mirror_prefill_repaired_direct_mesh"]
        command = candidate["direct_mesh_command"]

        self.assertEqual(candidate["direct_mesh_input"], "mirror")
        self.assertIn("--mesh-repair-preconditioner adaptive-voxel-close", command)
        self.assertIn("--mesh-repair-voxel-resolution 128", command)
        self.assertIn("--no-mesh-allow-convex-hull-fallback", command)
        self.assertIn("--mesh-target-bbox-mode uniform-max", command)
        self.assertIn("--mesh-target-faces 10000", command)
        self.assertNotIn("--mesh-min-bbox-dimension", command)
        self.assertNotIn("--mesh-max-bbox-aspect-ratio", command)


if __name__ == "__main__":
    unittest.main()
