import unittest

import numpy as np

from backend.benchmark.mhr_parametric_geometry import (
    camera_space_vertices,
    expand_head_identity,
    project_camera_vertices,
    project_head_to_support,
    validate_expression,
)


class MHRParametricGeometryTests(unittest.TestCase):
    def test_head_identity_expands_only_the_official_head_slice(self):
        head = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
        identity = expand_head_identity(head)
        self.assertEqual(identity.shape, (45,))
        np.testing.assert_array_equal(identity[20:40], head)
        np.testing.assert_array_equal(identity[:20], 0.0)
        np.testing.assert_array_equal(identity[40:], 0.0)

    def test_coefficient_validation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "20 finite"):
            expand_head_identity(np.zeros(19, dtype=np.float32))
        expression = np.zeros(72, dtype=np.float32)
        expression[3] = np.nan
        with self.assertRaisesRegex(ValueError, "72 finite"):
            validate_expression(expression)

    def test_camera_projection_preserves_floating_z(self):
        vertices = np.asarray(
            ((-1.0, -1.0, -0.5), (1.0, -1.0, 0.0), (0.0, 1.0, 0.5)),
            dtype=np.float32,
        )
        camera, record = camera_space_vertices(
            vertices,
            yaw_degrees=20.0,
            elevation_degrees=-4.0,
            distance=5.0,
        )
        intrinsics = np.asarray(
            ((100.0, 0.0, 32.0), (0.0, 100.0, 32.0), (0.0, 0.0, 1.0))
        )
        projected = project_camera_vertices(camera, intrinsics)
        np.testing.assert_allclose(projected[:, 2], camera[:, 2], atol=0.0)
        self.assertTrue(np.all(projected[:, 2] > 0.0))
        self.assertEqual(np.asarray(record["object_to_camera"]).shape, (4, 4))

    def test_projection_aligns_mesh_to_support_bbox(self):
        vertices = np.asarray(
            (
                (-1.0, -1.0, -0.4),
                (1.0, -1.0, -0.4),
                (1.0, 1.0, 0.4),
                (-1.0, 1.0, 0.4),
            ),
            dtype=np.float32,
        )
        support = np.zeros((80, 60), dtype=bool)
        support[20:60, 18:42] = True
        projected, record = project_head_to_support(
            vertices,
            support,
            yaw_degrees=0.0,
            elevation_degrees=0.0,
        )
        center = 0.5 * (
            np.min(projected[:, :2], axis=0) + np.max(projected[:, :2], axis=0)
        )
        np.testing.assert_allclose(center, (29.5, 39.5), atol=0.05)
        self.assertAlmostEqual(float(np.ptp(projected[:, 1])), 40.0, delta=0.25)
        self.assertEqual(record["target_bbox_xyxy"], [18, 20, 42, 60])


if __name__ == "__main__":
    unittest.main()
