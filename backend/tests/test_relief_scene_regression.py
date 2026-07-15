import unittest
from inspect import signature

import numpy as np

from backend.benchmark.run_relief_scene_regression import (
    _boundary_shape_metrics,
    _certification_checks,
    _extreme,
    _negative_controls,
    _scene_checks,
    _scene_geometry,
    _scene_specs,
    _synthetic_scene,
    run,
)
from backend.pic_to_3d import DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO


class ReliefSceneRegressionTest(unittest.TestCase):
    def test_runner_defaults_to_production_background_ratio(self):
        self.assertEqual(
            signature(run).parameters["background_depth_ratio"].default,
            DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
        )

    def test_scene_matrix_is_varied_and_deterministic(self):
        specs = _scene_specs()

        self.assertEqual(len(specs), 10)
        self.assertEqual(len({spec.scene_id for spec in specs}), 10)
        self.assertEqual(len({spec.background for spec in specs}), 10)

        signatures = []
        geometry = []
        for spec in specs:
            with self.subTest(scene=spec.scene_id):
                source, face, subject = _synthetic_scene(spec)
                replay_source, replay_face, replay_subject = _synthetic_scene(spec)

                self.assertEqual(source.shape, (121, 121))
                self.assertEqual(face.shape, source.shape)
                self.assertEqual(subject.shape, source.shape)
                self.assertTrue(np.all(subject[face]))
                self.assertGreater(np.count_nonzero(face), 700)
                self.assertGreater(np.count_nonzero(~subject), 4000)
                self.assertTrue(np.all(np.isfinite(source)))
                self.assertGreater(float(np.ptp(source[~subject])), 0.05)
                np.testing.assert_array_equal(source, replay_source)
                np.testing.assert_array_equal(face, replay_face)
                np.testing.assert_array_equal(subject, replay_subject)
                signatures.append(
                    (
                        float(np.mean(source[~subject])),
                        float(np.std(source[~subject])),
                        int(np.count_nonzero(subject)),
                    )
                )
                geometry.append(_scene_geometry(spec, face, subject))

        self.assertEqual(len(set(signatures)), len(specs))
        self.assertLessEqual(min(item["face_scale_ratio"] for item in geometry), 0.65)
        self.assertGreaterEqual(max(item["face_scale_ratio"] for item in geometry), 1.5)
        self.assertLessEqual(
            min(item["subject_coverage_ratio"] for item in geometry), 0.25
        )
        self.assertGreaterEqual(
            max(item["subject_coverage_ratio"] for item in geometry), 0.6
        )
        self.assertGreaterEqual(sum(item["face_touches_frame"] for item in geometry), 2)
        clipped = [item for item in geometry if item["framing"].endswith("clipped")]
        self.assertEqual(len(clipped), 2)
        self.assertTrue(all(item["visible_face_fraction"] < 0.8 for item in clipped))

    def test_boundary_shape_gate_is_source_relative_and_stricter_than_global_gate(self):
        background = {
            "reference_boundary_jump_p99_mm": 0.9,
            "reference_boundary_jump_max_mm": 1.7,
            "output_boundary_jump_p99_mm": 1.55,
            "output_boundary_jump_max_mm": 3.1,
        }
        cap = {"attachment_step_limit_mm": 0.8}

        accepted = _boundary_shape_metrics(background, cap)
        self.assertTrue(accepted["passed"])
        self.assertEqual(accepted["p99_limit_mm"], 1.6)
        self.assertEqual(accepted["max_limit_mm"], 3.2)

        background["output_boundary_jump_p99_mm"] = 1.61
        rejected = _boundary_shape_metrics(background, cap)
        self.assertFalse(rejected["passed"])

        high_reference = {
            "reference_boundary_jump_p99_mm": 10.0,
            "reference_boundary_jump_max_mm": 12.0,
            "output_boundary_jump_p99_mm": 3.21,
            "output_boundary_jump_max_mm": 4.01,
        }
        absolute_rejection = _boundary_shape_metrics(
            high_reference,
            {"attachment_step_limit_mm": 1.2},
        )
        self.assertEqual(absolute_rejection["p99_limit_mm"], 3.2)
        self.assertEqual(absolute_rejection["max_limit_mm"], 4.0)
        self.assertFalse(absolute_rejection["passed"])

    def test_source_context_signal_is_a_hard_scene_gate(self):
        compose = {
            "background_context_enabled": True,
            "background_context_pixels": 100,
            "background_context_normalized_correlation": 0.1,
            "background_context_normalized_rms_retention": 1.0,
            "background_context_recoverable_coverage_ratio": 0.8,
        }
        postprocess = {
            "background_depth_preservation": {
                "available": True,
                "passed": True,
                "candidate_coverage_ratio": 1.0,
                "localized_structure": {"passed": True},
            },
            "selection_background_physical_cap": {
                "far_background_max_mm": 13.5,
                "far_background_ceiling_mm": 13.5,
                "emission_passed": True,
                "feasible_attachment_constraints_passed": True,
            },
            "face_detail_guard": {
                "final": {
                    "available": True,
                    "correlation": 0.95,
                    "rms_retention": 1.0,
                }
            },
        }

        checks = _scene_checks(compose, postprocess, {"printable": True})

        self.assertFalse(checks["source_context_signal"])

    def test_near_full_frame_context_uses_measurable_capacity_denominator(self):
        compose = {
            "background_context_enabled": True,
            "background_context_pixels": 100,
            "background_context_normalized_correlation": 0.99,
            "background_context_normalized_rms_retention": 1.0,
            "background_context_recoverable_coverage_ratio": 0.55,
            "background_context_measured_coverage_ratio": 0.75,
            "background_context_recoverable_measured_ratio": 0.70,
        }
        postprocess = {
            "background_depth_preservation": {
                "available": True,
                "passed": True,
                "candidate_coverage_ratio": 1.0,
                "localized_structure": {"passed": True},
            },
            "selection_background_physical_cap": {
                "far_background_max_mm": 13.5,
                "far_background_ceiling_mm": 13.5,
                "emission_passed": True,
                "feasible_attachment_constraints_passed": True,
            },
            "face_detail_guard": {
                "final": {
                    "available": True,
                    "correlation": 0.95,
                    "rms_retention": 1.0,
                }
            },
        }

        checks = _scene_checks(compose, postprocess, {"printable": True})
        self.assertTrue(checks["source_context_signal"])

        compose["background_context_measured_coverage_ratio"] = 0.2
        rejected = _scene_checks(compose, postprocess, {"printable": True})
        self.assertFalse(rejected["source_context_signal"])

    def test_partial_or_empty_telemetry_cannot_certify(self):
        controls = {"checks": {"passed": True}}
        provenance = {"available": True, "clean": True}
        checks = _certification_checks(
            [],
            expected_scene_count=0,
            full_scene_count=10,
            expected_scene_ids=(),
            provenance=provenance,
            negative_controls=controls,
        )

        self.assertFalse(checks["full_scene_matrix_complete"])
        self.assertFalse(checks["required_telemetry_complete"])
        self.assertFalse(all(checks.values()))
        self.assertIsNone(_extreme([], "background", "correlation", min))

    def test_certification_rejects_duplicate_or_substituted_scene_ids(self):
        rows = [
            {
                "scene_id": "scene-a",
                "checks": {
                    "passed": True,
                    "background_preservation": True,
                    "selection_boundary_shape": True,
                    "face_detail": True,
                    "physical_emission": True,
                    "printable_mesh": True,
                },
                "compose": {
                    "normalized_context_correlation": 1.0,
                    "normalized_context_rms_retention": 1.0,
                    "recoverable_context_coverage_ratio": 1.0,
                },
                "background": {"correlation": 1.0, "gradient_correlation": 1.0},
                "boundary_shape": {"output_p99_mm": 0.8, "output_max_mm": 0.8},
                "face": {"correlation": 1.0},
                "physical_cap": {"far_background_max_mm": 10.0},
                "geometry": {
                    "face_scale_ratio": 0.5,
                    "subject_coverage_ratio": 0.2,
                    "face_touches_frame": True,
                },
            },
            {
                "scene_id": "scene-a",
                "checks": {
                    "passed": True,
                    "background_preservation": True,
                    "selection_boundary_shape": True,
                    "face_detail": True,
                    "physical_emission": True,
                    "printable_mesh": True,
                },
                "compose": {
                    "normalized_context_correlation": 1.0,
                    "normalized_context_rms_retention": 1.0,
                    "recoverable_context_coverage_ratio": 1.0,
                },
                "background": {"correlation": 1.0, "gradient_correlation": 1.0},
                "boundary_shape": {"output_p99_mm": 0.8, "output_max_mm": 0.8},
                "face": {"correlation": 1.0},
                "physical_cap": {"far_background_max_mm": 10.0},
                "geometry": {
                    "face_scale_ratio": 1.6,
                    "subject_coverage_ratio": 0.7,
                    "face_touches_frame": True,
                },
            },
        ]

        checks = _certification_checks(
            rows,
            expected_scene_count=2,
            full_scene_count=2,
            expected_scene_ids=("scene-a", "scene-b"),
            provenance={"available": True, "clean": True},
            negative_controls={"checks": {"passed": True}},
        )

        self.assertFalse(checks["expected_scene_ids"])
        self.assertFalse(all(checks.values()))

    def test_negative_controls_reject_global_local_and_unavailable_context(self):
        controls = _negative_controls()

        self.assertTrue(controls["checks"]["passed"])
        self.assertGreater(controls["localized_loss_global_correlation"], 0.8)
        self.assertGreater(controls["localized_loss_global_gradient_correlation"], 0.58)
        self.assertGreater(controls["localized_loss_failed_windows"], 0)


if __name__ == "__main__":
    unittest.main()
