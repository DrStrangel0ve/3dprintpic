import unittest

import numpy as np
import trimesh

from backend.benchmark.mesh_rendering import (
    CameraSpec,
    RenderConfig,
    render_mesh,
)


class PerspectiveMeshRenderingTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.72)
        self.camera = CameraSpec(azimuth_deg=14.0, elevation_deg=-6.0)

    def test_default_projection_remains_identical_to_explicit_orthographic(self):
        implicit = render_mesh(
            self.mesh,
            self.camera,
            RenderConfig(size=96),
            (182, 132, 110),
        )
        explicit = render_mesh(
            self.mesh,
            self.camera,
            RenderConfig(size=96, projection="orthographic"),
            (182, 132, 110),
        )

        np.testing.assert_array_equal(implicit.rgb, explicit.rgb)
        np.testing.assert_array_equal(implicit.depth, explicit.depth)
        np.testing.assert_array_equal(implicit.silhouette, explicit.silhouette)

    def test_perspective_render_interpolates_colors_depth_and_parts(self):
        vertices = np.asarray(self.mesh.vertices)
        minimum = vertices.min(axis=0)
        span = np.maximum(vertices.max(axis=0) - minimum, 1e-6)
        colors = np.clip((vertices - minimum) / span, 0.0, 1.0)
        nose_weight = (vertices[:, 2] > np.percentile(vertices[:, 2], 65)).astype(
            np.float32
        )
        result = render_mesh(
            self.mesh,
            self.camera,
            RenderConfig(
                size=128,
                projection="perspective",
                perspective_fov_y_deg=38.0,
                camera_distance=3.2,
                specular=0.08,
            ),
            (182, 132, 110),
            vertex_part_weights={"nose": nose_weight},
            vertex_colors=colors,
        )

        self.assertGreater(int(np.count_nonzero(result.silhouette)), 1000)
        visible_depth = result.depth[result.silhouette]
        self.assertGreater(float(np.ptp(visible_depth)), 0.95)
        self.assertTrue(np.all(np.isfinite(result.surface_z[result.silhouette])))
        visible_rgb = result.rgb[result.silhouette]
        self.assertGreater(float(np.max(np.std(visible_rgb, axis=0))), 0.05)
        self.assertIn("nose", result.part_masks)
        self.assertGreater(int(np.count_nonzero(result.part_masks["nose"])), 20)
        self.assertTrue(
            np.all(result.part_masks["nose"] <= result.silhouette)
        )

    def test_perspective_z_buffer_keeps_nearer_overlapping_triangle(self):
        vertices = np.asarray(
            [
                [-0.45, -0.40, 0.25],
                [0.45, -0.40, 0.25],
                [0.00, 0.45, 0.25],
                [-0.45, -0.40, -0.25],
                [0.45, -0.40, -0.25],
                [0.00, 0.45, -0.25],
            ],
            dtype=np.float64,
        )
        colors = np.asarray(
            [[1.0, 0.0, 0.0]] * 3 + [[0.0, 0.0, 1.0]] * 3,
            dtype=np.float32,
        )
        config = RenderConfig(
            size=96,
            projection="perspective",
            camera_distance=3.0,
            ambient=1.0,
            diffuse=0.0,
            specular=0.0,
        )

        center_colors = []
        for faces in (
            np.asarray([[0, 1, 2], [3, 4, 5]]),
            np.asarray([[3, 4, 5], [0, 1, 2]]),
        ):
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            result = render_mesh(
                mesh,
                CameraSpec(azimuth_deg=0.0, elevation_deg=0.0),
                config,
                (255, 255, 255),
                vertex_colors=colors,
            )
            center_colors.append(result.rgb[48, 48])

        np.testing.assert_allclose(center_colors[0], (1.0, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(center_colors[1], center_colors[0], atol=1e-6)

    def test_perspective_rejects_invalid_projection_inputs(self):
        with self.assertRaisesRegex(ValueError, "Unsupported projection"):
            render_mesh(
                self.mesh,
                self.camera,
                RenderConfig(projection="fisheye"),
                (182, 132, 110),
            )
        with self.assertRaisesRegex(ValueError, "vertex_colors has shape"):
            render_mesh(
                self.mesh,
                self.camera,
                RenderConfig(projection="perspective"),
                (182, 132, 110),
                vertex_colors=np.zeros((3, 3), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
