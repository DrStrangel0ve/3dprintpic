import unittest

import torch

from backend.benchmark.face_depth_spatial_pyramid import (
    CONDITIONING_CHANNELS,
    SpatialPyramidConfig,
    SpatialPyramidLossWeights,
    build_spatial_pyramid,
    camera_points_from_z,
    camera_surface_normals,
    matched_deepest_control_features,
    parameter_count,
    spatial_pyramid_loss,
)


class FaceDepthSpatialPyramidTests(unittest.TestCase):
    def _features(self, batch=2):
        return (
            torch.randn(batch, 64, 16, 16),
            torch.randn(batch, 64, 8, 8),
            torch.randn(batch, 64, 4, 4),
            torch.randn(batch, 64, 2, 2),
        )

    def test_configuration_is_bounded_and_records_conditioning(self):
        provenance = SpatialPyramidConfig().provenance()
        self.assertEqual(provenance["feature_levels"], 4)
        self.assertIn("rgb_laplacian", provenance["conditioning_layout"])
        with self.assertRaisesRegex(ValueError, "four feature levels"):
            SpatialPyramidConfig(feature_channels=(64, 64, 64)).validated()
        with self.assertRaisesRegex(ValueError, "residual cap"):
            SpatialPyramidConfig(maximum_residual=2.0).validated()

    def test_candidate_and_control_use_identical_parameter_counts(self):
        candidate = build_spatial_pyramid()
        control = build_spatial_pyramid()
        self.assertEqual(parameter_count(candidate), parameter_count(control))
        self.assertGreater(parameter_count(candidate), 50_000)

    def test_matched_control_repeats_only_the_deepest_spatial_tensor(self):
        features = self._features(batch=1)
        control = matched_deepest_control_features(features)
        self.assertEqual([tuple(value.shape[-2:]) for value in control], [(16, 16), (8, 8), (4, 4), (2, 2)])
        torch.testing.assert_close(control[-1], features[-1])
        expected = torch.nn.functional.interpolate(
            features[-1], size=(16, 16), mode="bilinear", align_corners=False
        )
        torch.testing.assert_close(control[0], expected)

    def test_output_is_bounded_and_bit_exact_outside_supported_interior(self):
        model = build_spatial_pyramid(SpatialPyramidConfig(maximum_residual=0.2))
        torch.nn.init.constant_(model.output.bias, 10.0)
        conditioning = torch.zeros(2, CONDITIONING_CHANNELS, 20, 20)
        support = torch.zeros(2, 1, 20, 20)
        support[..., 3:17, 3:17] = 1.0
        boundary = torch.zeros_like(support)
        boundary[..., 3, 3:17] = 1.0
        residual = model(self._features(), conditioning, support, boundary)
        self.assertLessEqual(float(residual.detach().abs().max()), 0.2 + 1e-6)
        self.assertTrue(torch.equal(residual[support == 0], torch.zeros_like(residual[support == 0])))
        self.assertTrue(torch.equal(residual[boundary == 1], torch.zeros_like(residual[boundary == 1])))

    def test_camera_back_projection_and_planar_normals_are_finite(self):
        z = torch.full((1, 1, 5, 5), 2.0)
        x = torch.linspace(-0.2, 0.2, 5)
        y = torch.linspace(-0.2, 0.2, 5)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        rays = torch.stack((xx, yy, torch.ones_like(xx)), dim=0)[None]
        rays = torch.nn.functional.normalize(rays, dim=1)
        points = camera_points_from_z(z, rays)
        self.assertEqual(tuple(points.shape), (1, 3, 5, 5))
        normals = camera_surface_normals(z, rays)
        self.assertEqual(tuple(normals.shape), (1, 3, 4, 4))
        self.assertTrue(torch.isfinite(normals).all())
        self.assertGreater(float(normals[:, 2].abs().mean()), 0.99)

    def test_perfect_prediction_has_zero_geometry_loss(self):
        coordinates = torch.linspace(-1.0, 1.0, 8)
        target_y, target_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        target = (0.5 + 0.1 * torch.sin(target_x * 2.0) * torch.cos(target_y * 1.5))[
            None, None
        ]
        support = torch.ones_like(target)
        part = torch.zeros_like(target)
        part[..., 2:6, 2:6] = 1.0
        ray_coordinates = torch.linspace(-0.2, 0.2, 8)
        yy, xx = torch.meshgrid(ray_coordinates, ray_coordinates, indexing="ij")
        rays = torch.stack((xx, yy, torch.ones_like(xx)), dim=0)[None]
        rays = torch.nn.functional.normalize(rays, dim=1)
        total, components, prediction = spatial_pyramid_loss(
            torch.zeros_like(target),
            target,
            target,
            support,
            [part] * 6,
            torch.zeros_like(target),
            rays,
        )
        torch.testing.assert_close(prediction, target)
        self.assertLess(float(total), 1e-6)
        self.assertTrue(all(float(value) < 1e-6 for value in components.values()))

    def test_small_named_part_error_is_not_diluted(self):
        target = torch.zeros((1, 1, 8, 8))
        coarse = target.clone()
        residual = torch.zeros_like(target)
        residual[..., 2:4, 2:4] = 0.5
        support = torch.ones_like(target)
        part = torch.zeros_like(target)
        part[..., 2:4, 2:4] = 1.0
        rays = torch.zeros((1, 3, 8, 8))
        rays[:, 2] = 1.0
        total, components, _ = spatial_pyramid_loss(
            residual,
            coarse,
            target,
            support,
            [part] * 6,
            torch.zeros_like(target),
            rays,
            weights=SpatialPyramidLossWeights(
                camera_z=1.0,
                gradient=0.0,
                normal=0.0,
                laplacian=0.0,
                attachment=0.0,
                part_non_regression=0.0,
                residual=0.0,
            ),
        )
        self.assertGreater(float(components["camera_z"]), 0.05)
        self.assertGreater(float(total), 0.05)


if __name__ == "__main__":
    unittest.main()
