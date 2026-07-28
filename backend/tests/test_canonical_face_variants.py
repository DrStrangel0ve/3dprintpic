import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.benchmark.canonical_face_variants import (
    CANONICAL_FACE_VARIANTS,
    deform_canonical_face,
)
from backend.benchmark.mesh_rendering import CameraSpec, RenderConfig, render_mesh
from backend.benchmark.run_canonical_face_relief_smoke import (
    _run_row,
    canonical_face_masks,
    canonical_vertex_part_weights,
    load_canonical_face_fixture,
    make_structured_face_scene,
)
from backend.benchmark.run_canonical_face_variant_smoke import (
    VARIANT_RENDER_SETTINGS,
    _diversity_metrics,
)


class CanonicalFaceVariantsTest(unittest.TestCase):
    def test_variants_preserve_topology_and_meet_diversity_gates(self):
        source, _metadata = load_canonical_face_fixture()
        audits = []
        meshes = []
        for variant in CANONICAL_FACE_VARIANTS:
            mesh, audit = deform_canonical_face(source, variant)
            repeated_mesh, repeated_audit = deform_canonical_face(source, variant)
            meshes.append(mesh)
            audits.append(audit)
            self.assertTrue(audit["passed"], variant.name)
            self.assertEqual(audit["self_intersection_count"], 0)
            self.assertGreaterEqual(
                audit["minimum_signed_triangle_jacobian"],
                audit["deformation_gates"]["minimum_signed_triangle_jacobian"],
            )
            self.assertEqual(len(mesh.vertices), 468)
            self.assertEqual(len(mesh.faces), 898)
            np.testing.assert_array_equal(mesh.faces, source.faces)
            np.testing.assert_array_equal(mesh.vertices, repeated_mesh.vertices)
            self.assertEqual(
                audit["geometry_sha256"], repeated_audit["geometry_sha256"]
            )

        diversity = _diversity_metrics(audits)
        self.assertTrue(diversity["passed"])
        self.assertEqual(len({audit["geometry_sha256"] for audit in audits}), 6)
        np.testing.assert_allclose(meshes[0].vertices, source.vertices, atol=0.0)

    def test_all_named_parts_remain_visible_at_profile_yaws(self):
        source, _metadata = load_canonical_face_fixture()
        weights = canonical_vertex_part_weights()
        for variant in CANONICAL_FACE_VARIANTS:
            mesh, _audit = deform_canonical_face(source, variant)
            config = RenderConfig(
                size=128,
                ortho_scale=VARIANT_RENDER_SETTINGS[variant.name]["ortho_scale"],
            )
            for yaw_deg in (-45.0, 45.0):
                with self.subTest(variant=variant.name, yaw_deg=yaw_deg):
                    rendered = render_mesh(
                        mesh,
                        camera=CameraSpec(azimuth_deg=yaw_deg, elevation_deg=0.0),
                        config=config,
                        base_color=(188, 146, 128),
                        vertex_part_weights=weights,
                    )
                    _face, parts = canonical_face_masks(
                        rendered.silhouette,
                        rendered.part_masks,
                    )
                    self.assertTrue(
                        all(np.count_nonzero(mask) >= 12 for mask in parts.values())
                    )

    def test_background_phase_varies_context_without_changing_face_depth(self):
        depth = np.linspace(0.0, 1.0, 96 * 96, dtype=np.float32).reshape(96, 96)
        rows, cols = np.indices(depth.shape)
        face = ((rows - 48) ** 2 + (cols - 48) ** 2) <= 28**2
        baseline, baseline_rgb = make_structured_face_scene(depth, face, 15.0)
        explicit_zero, explicit_zero_rgb = make_structured_face_scene(
            depth,
            face,
            15.0,
            scene_phase_rad=0.0,
        )
        varied, varied_rgb = make_structured_face_scene(
            depth,
            face,
            15.0,
            scene_phase_rad=1.3,
        )

        np.testing.assert_array_equal(baseline, explicit_zero)
        np.testing.assert_array_equal(baseline_rgb, explicit_zero_rgb)
        np.testing.assert_array_equal(varied[face], baseline[face])
        self.assertGreater(float(np.mean(np.abs(varied[~face] - baseline[~face]))), 0.01)
        self.assertGreater(float(np.mean(np.abs(varied_rgb - baseline_rgb))), 0.001)

    def test_hard_profile_projected_nose_row_passes_absolute_gates(self):
        source, _metadata = load_canonical_face_fixture()
        variant = next(
            item for item in CANONICAL_FACE_VARIANTS if item.name == "narrow_projected_nose"
        )
        mesh, audit = deform_canonical_face(source, variant)
        self.assertTrue(audit["passed"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            row = _run_row(
                Path(tmp_dir),
                mesh,
                yaw_deg=45.0,
                relief_height_mm=30.0,
                render_size=256,
                physical_size_mm=96.0,
                render_ortho_scale=1.68,
                background_phase_rad=1.46,
            )

        self.assertTrue(row["checks"]["passed"])
        self.assertTrue(row["absolute_named_parts"]["passed"])
        self.assertTrue(row["absolute_named_part_mm_error"]["passed"])
        self.assertTrue(row["background"]["passed"])
        self.assertTrue(row["topology"]["printable"])
        self.assertTrue(row["shell"]["passed"])


if __name__ == "__main__":
    unittest.main()
