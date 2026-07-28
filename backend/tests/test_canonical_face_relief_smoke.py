import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.benchmark.mesh_rendering import CameraSpec, RenderConfig, render_mesh
from backend.benchmark.run_canonical_face_relief_smoke import (
    CANONICAL_FACE_SHA256,
    _run_row,
    canonical_face_masks,
    canonical_vertex_part_weights,
    load_canonical_face_fixture,
    make_structured_face_scene,
)


class CanonicalFaceReliefSmokeTest(unittest.TestCase):
    def test_pinned_canonical_face_has_ordered_landmark_topology(self):
        mesh, metadata = load_canonical_face_fixture()

        self.assertEqual(metadata["asset_sha256"], CANONICAL_FACE_SHA256)
        self.assertEqual(metadata["license"], "Apache-2.0")
        self.assertEqual(metadata["coordinate_unit"], "centimeter")
        self.assertEqual(len(mesh.vertices), 468)
        self.assertEqual(len(mesh.faces), 898)
        self.assertEqual(int(np.min(mesh.faces)), 0)
        self.assertEqual(int(np.max(mesh.faces)), 467)
        self.assertFalse(mesh.is_watertight)

    def test_front_render_produces_all_parts_and_structured_background(self):
        mesh, _metadata = load_canonical_face_fixture()
        camera = CameraSpec(azimuth_deg=0.0, elevation_deg=0.0)
        config = RenderConfig(size=96, ortho_scale=1.8)
        rendered = render_mesh(
            mesh,
            camera=camera,
            config=config,
            base_color=(188, 146, 128),
            vertex_part_weights=canonical_vertex_part_weights(len(mesh.vertices)),
        )

        face, parts = canonical_face_masks(
            rendered.silhouette,
            rendered.part_masks,
        )
        scene, background_rgb = make_structured_face_scene(
            rendered.depth,
            face,
            0.0,
        )

        self.assertGreater(np.count_nonzero(face), 2000)
        self.assertTrue(all(np.count_nonzero(mask) >= 12 for mask in parts.values()))
        self.assertLess(float(np.max(scene[face])), float(np.min(scene[~face])))
        self.assertGreater(float(np.std(scene[~face])), 0.02)
        self.assertEqual(background_rgb.shape, (96, 96, 3))
        self.assertEqual(rendered.surface_z.shape, (96, 96))
        self.assertTrue(np.all(np.isfinite(rendered.surface_z[face])))

    def test_oblique_40mm_row_uses_audited_selection_bounded_solve(self):
        mesh, _metadata = load_canonical_face_fixture()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            row = _run_row(
                root,
                mesh,
                yaw_deg=-30.0,
                relief_height_mm=40.0,
                render_size=256,
                physical_size_mm=96.0,
                background_depth_ratio=0.50,
            )
            postprocess = json.loads(
                (root / row["row_id"] / "postprocess.json").read_text(
                    encoding="utf-8"
                )
            )

        compression = postprocess["face_height_stabilization"][
            "gradient_compression"
        ]
        retry = compression["adaptive_screening_retry"]
        blend = compression["face_region_blend"]
        post_blend_audit = compression["post_blend_quality_audit"]
        self.assertTrue(row["checks"]["passed"])
        self.assertTrue(row["checks"]["bounded_feature_emboss"])
        self.assertAlmostEqual(
            row["feature_handling"]["requested_feature_depth_mm"],
            0.4,
        )
        self.assertAlmostEqual(
            row["feature_handling"]["effective_feature_depth_mm"],
            0.4,
        )
        self.assertFalse(row["feature_handling"]["emboss_suppressed"])
        self.assertTrue(row["feature_handling"]["slope_guard_enabled"])
        self.assertTrue(row["feature_handling"]["detail_guard_enabled"])
        self.assertTrue(retry["attempted"])
        self.assertEqual(retry["reason"], "retry_candidate_accepted")
        self.assertEqual(retry["trigger_failures"], ["cardinal_edge_p99"])
        self.assertTrue(blend["selection_bounded"])
        self.assertGreater(
            blend["outside_selection_correction_before_clamp_max_mm"],
            0.0,
        )
        self.assertEqual(blend["outside_selection_correction_max_mm"], 0.0)
        self.assertTrue(post_blend_audit["quality_gates"]["passed"])
        self.assertTrue(post_blend_audit["enabled"])
        self.assertEqual(
            postprocess["face_height_stabilization"][
                "gradient_compression_attempt"
            ]["quality_gates"]["failures"],
            ["cardinal_edge_p99"],
        )
        self.assertTrue(
            postprocess["face_height_stabilization"][
                "gradient_compression_selected"
            ]["quality_gates"]["passed"]
        )


if __name__ == "__main__":
    unittest.main()
