import unittest

import torch

from backend.benchmark.face_depth_rectified_flow import (
    MODEL_INPUT_CHANNELS,
    STATIC_CONDITIONING_CHANNELS,
    GeometryLossWeights,
    RectifiedFlowConfig,
    apply_supported_residual,
    build_static_conditioning,
    build_unet,
    camera_rays_from_intrinsics,
    deterministic_geometry_loss,
    flow_endpoint,
    flow_training_state,
    geometric_flow_loss,
    median_absolute_deviation,
    normalized_camera_rays,
    parameter_count,
    posterior_medoid,
    project_flow_state_to_support,
    sample_rectified_flow,
)


class _ConstantVelocityModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, sample, _timestep):
        output = torch.full_like(sample[:, :1], self.value)
        return type("Output", (), {"sample": output})()


class FaceDepthRectifiedFlowTests(unittest.TestCase):
    def test_configuration_is_bounded_and_records_no_vae(self):
        config = RectifiedFlowConfig().validated()
        provenance = config.provenance()
        self.assertEqual(provenance["model_input_channels"], MODEL_INPUT_CHANNELS)
        self.assertFalse(provenance["latent_vae"])
        with self.assertRaisesRegex(ValueError, "U-Net scale"):
            RectifiedFlowConfig(sample_size=95).validated()
        with self.assertRaisesRegex(ValueError, "norm groups"):
            RectifiedFlowConfig(block_out_channels=(30, 64)).validated()
        with self.assertRaisesRegex(ValueError, "unique"):
            RectifiedFlowConfig(inference_seeds=(1, 1)).validated()

    def test_randomly_initialized_flow_and_control_have_equal_parameter_counts(self):
        config = RectifiedFlowConfig(
            sample_size=32,
            block_out_channels=(16, 32),
            norm_num_groups=8,
            attention_head_dim=8,
        )
        flow = build_unet(config)
        control = build_unet(config)
        self.assertEqual(parameter_count(flow), parameter_count(control))
        self.assertGreater(parameter_count(flow), 100_000)
        self.assertEqual(flow.config.in_channels, MODEL_INPUT_CHANNELS)
        self.assertEqual(flow.config.out_channels, 1)

    def test_camera_rays_are_unit_length_and_centered(self):
        rays = normalized_camera_rays(5, 5, 32.0)
        self.assertEqual(tuple(rays.shape), (1, 3, 5, 5))
        torch.testing.assert_close(
            torch.linalg.vector_norm(rays, dim=1),
            torch.ones((1, 5, 5)),
        )
        torch.testing.assert_close(rays[0, :, 2, 2], torch.tensor((0.0, 0.0, 1.0)))

    def test_crop_camera_rays_inverse_warp_through_source_intrinsics(self):
        rays = camera_rays_from_intrinsics(
            4,
            4,
            ((100.0, 0.0, 50.0), (0.0, 100.0, 40.0), (0.0, 0.0, 1.0)),
            (46, 36, 54, 44),
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(rays, dim=1),
            torch.ones((1, 4, 4)),
        )
        self.assertLess(float(rays[0, 0, 1, 1]), 0.0)
        self.assertGreater(float(rays[0, 0, 1, 2]), 0.0)

    def test_conditioning_packs_exactly_ten_static_channels(self):
        rgb = torch.zeros((2, 3, 8, 8))
        scalar = torch.ones((2, 1, 8, 8))
        rays = normalized_camera_rays(8, 8, 40.0).expand(2, -1, -1, -1)
        packed = build_static_conditioning(
            rgb,
            scalar,
            scalar,
            scalar,
            rays,
            torch.tensor((1.0, 0.0)),
        )
        self.assertEqual(tuple(packed.shape), (2, STATIC_CONDITIONING_CHANNELS, 8, 8))
        self.assertTrue(torch.equal(packed[0, -1], torch.ones((8, 8))))
        self.assertTrue(torch.equal(packed[1, -1], torch.zeros((8, 8))))

    def test_true_flow_velocity_reconstructs_target_endpoint(self):
        target = torch.tensor((0.4, -0.2), dtype=torch.float32).reshape(2, 1, 1, 1)
        noise = torch.tensor((1.0, -1.0), dtype=torch.float32).reshape(2, 1, 1, 1)
        time = torch.tensor((0.25, 0.75))
        state, velocity = flow_training_state(target, time, noise, 0.15)
        endpoint = flow_endpoint(state, velocity, time)
        torch.testing.assert_close(endpoint, target)

    def test_exact_velocity_has_zero_geometry_loss(self):
        target = torch.linspace(-0.2, 0.2, 64).reshape(1, 1, 8, 8)
        noise = torch.zeros_like(target)
        time = torch.tensor((0.4,))
        state, velocity = flow_training_state(target, time, noise, 0.15)
        face = torch.ones_like(target)
        parts = [torch.ones_like(target) for _ in range(6)]
        boundary = torch.zeros_like(target)
        total, components, endpoint = geometric_flow_loss(
            velocity,
            velocity,
            state,
            time,
            target,
            face,
            parts,
            boundary,
        )
        torch.testing.assert_close(endpoint, target)
        self.assertLess(float(total), 1e-6)
        self.assertTrue(all(float(value) < 1e-6 for value in components.values()))

    def test_small_part_error_is_not_diluted_by_whole_face(self):
        target = torch.zeros((1, 1, 8, 8))
        predicted = target.clone()
        predicted[..., 2:4, 2:4] = 1.0
        face = torch.ones_like(target)
        part = torch.zeros_like(target)
        part[..., 2:4, 2:4] = 1.0
        parts = [part] + [torch.zeros_like(target) for _ in range(5)]
        boundary = torch.zeros_like(target)
        total, components, _ = deterministic_geometry_loss(
            predicted,
            target,
            face,
            parts,
            boundary,
        )
        self.assertGreater(float(total), 0.1)
        self.assertGreater(float(components["camera_z"]), 0.5)

    def test_boundary_term_rejects_nonzero_attachment_correction(self):
        target = torch.zeros((1, 1, 8, 8))
        predicted = torch.zeros_like(target)
        predicted[..., 0, :] = 0.5
        face = torch.ones_like(target)
        boundary = torch.zeros_like(target)
        boundary[..., 0, :] = 1.0
        total, components, _ = deterministic_geometry_loss(
            predicted,
            target,
            face,
            [],
            boundary,
            weights=GeometryLossWeights(
                flow=0.0,
                camera_z=0.0,
                gradient=0.0,
                normal=0.0,
                laplacian=0.0,
                boundary=1.0,
                ordering=0.0,
            ),
        )
        self.assertAlmostEqual(float(total), 0.5)
        self.assertAlmostEqual(float(components["boundary"]), 0.5)

    def test_deterministic_endpoint_is_exactly_the_predicted_residual(self):
        target = torch.zeros((1, 1, 4, 4))
        predicted = torch.full_like(target, 0.25)
        support = torch.ones_like(target)
        _, _, endpoint = deterministic_geometry_loss(
            predicted,
            target,
            support,
            [],
            torch.zeros_like(target),
        )
        torch.testing.assert_close(endpoint, predicted)

    def test_geometry_endpoint_discards_unsupported_predictions(self):
        target = torch.zeros((1, 1, 4, 4))
        predicted = torch.full_like(target, 9.0)
        support = torch.zeros_like(target)
        support[..., 1:3, 1:3] = 1.0
        _, _, endpoint = deterministic_geometry_loss(
            predicted,
            target,
            support,
            [],
            torch.zeros_like(target),
        )
        self.assertTrue(torch.equal(endpoint[support == 0], torch.zeros(12)))
        self.assertTrue(torch.equal(endpoint[support == 1], torch.full((4,), 9.0)))

    def test_control_matches_combined_flow_and_endpoint_value_weight(self):
        target = torch.ones((1, 1, 4, 4))
        predicted = torch.zeros_like(target)
        support = torch.ones_like(target)
        total, components, _ = deterministic_geometry_loss(
            predicted,
            target,
            support,
            [],
            torch.zeros_like(target),
            weights=GeometryLossWeights(
                flow=1.0,
                camera_z=0.5,
                gradient=0.0,
                normal=0.0,
                laplacian=0.0,
                boundary=0.0,
                ordering=0.0,
            ),
        )
        self.assertAlmostEqual(float(components["camera_z"]), 1.0)
        self.assertAlmostEqual(float(components["flow"]), 1.0)
        self.assertAlmostEqual(float(total), 1.5)

    def test_supported_residual_is_bit_exact_outside_face(self):
        coarse = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        residual = torch.ones_like(coarse)
        support = torch.zeros_like(coarse)
        support[..., 1:3, 1:3] = 1.0
        result = apply_supported_residual(coarse, residual, support)
        self.assertTrue(torch.equal(result[support == 0], coarse[support == 0]))
        self.assertTrue(torch.equal(result[support == 1], coarse[support == 1] + 1.0))

    def test_posterior_medoid_and_mad_report_sample_spread(self):
        samples = torch.tensor((0.0, 1.0, 10.0)).reshape(3, 1, 1, 1, 1)
        support = torch.ones((1, 1, 1, 1))
        medoid, indices = posterior_medoid(samples, support)
        self.assertEqual(int(indices[0]), 1)
        self.assertEqual(float(medoid[0, 0, 0, 0]), 1.0)
        mad = median_absolute_deviation(samples)
        self.assertEqual(float(mad[0, 0, 0, 0]), 1.0)

    def test_fixed_seed_euler_sampler_has_expected_shape_and_endpoint(self):
        model = _ConstantVelocityModel(0.25)
        conditioning = torch.zeros((1, STATIC_CONDITIONING_CHANNELS, 4, 4))
        conditioning[:, 5] = 1.0
        samples = sample_rectified_flow(
            model,
            conditioning,
            seeds=(1, 2),
            steps=4,
            noise_sigma=1e-8,
        )
        self.assertEqual(tuple(samples.shape), (2, 1, 1, 4, 4))
        torch.testing.assert_close(
            samples,
            torch.full_like(samples, 0.25),
            atol=5e-8,
            rtol=0.0,
        )

    def test_euler_sampler_projects_every_dynamic_step_to_support(self):
        model = _ConstantVelocityModel(4.0)
        conditioning = torch.zeros((1, STATIC_CONDITIONING_CHANNELS, 6, 6))
        conditioning[:, 5, 2:4, 2:4] = 1.0
        samples = sample_rectified_flow(
            model,
            conditioning,
            seeds=(7,),
            steps=3,
            noise_sigma=0.15,
        )
        support = conditioning[:, 5:6].bool()
        self.assertTrue(torch.equal(samples[0][~support], torch.zeros(32)))
        projected = project_flow_state_to_support(
            torch.ones((1, 1, 6, 6)),
            conditioning,
        )
        self.assertTrue(torch.equal(projected[~support], torch.zeros(32)))


if __name__ == "__main__":
    unittest.main()
