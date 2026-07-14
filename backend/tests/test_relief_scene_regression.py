import unittest

import numpy as np

from backend.benchmark.run_relief_scene_regression import (
    _certification_checks,
    _extreme,
    _negative_controls,
    _scene_checks,
    _scene_specs,
    _synthetic_scene,
)


class ReliefSceneRegressionTest(unittest.TestCase):
    def test_scene_matrix_is_varied_and_deterministic(self):
        specs = _scene_specs()

        self.assertEqual(len(specs), 6)
        self.assertEqual(len({spec.scene_id for spec in specs}), 6)
        self.assertEqual(len({spec.background for spec in specs}), 6)

        signatures = []
        for spec in specs:
            with self.subTest(scene=spec.scene_id):
                source, face, subject = _synthetic_scene(spec)
                replay_source, replay_face, replay_subject = _synthetic_scene(spec)

                self.assertEqual(source.shape, (121, 121))
                self.assertEqual(face.shape, source.shape)
                self.assertEqual(subject.shape, source.shape)
                self.assertTrue(np.all(subject[face]))
                self.assertGreater(np.count_nonzero(face), 1000)
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

        self.assertEqual(len(set(signatures)), len(specs))

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
            full_scene_count=6,
            provenance=provenance,
            negative_controls=controls,
        )

        self.assertFalse(checks["full_scene_matrix_complete"])
        self.assertFalse(checks["required_telemetry_complete"])
        self.assertFalse(all(checks.values()))
        self.assertIsNone(_extreme([], "background", "correlation", min))

    def test_negative_controls_reject_global_local_and_unavailable_context(self):
        controls = _negative_controls()

        self.assertTrue(controls["checks"]["passed"])
        self.assertGreater(controls["localized_loss_global_correlation"], 0.8)
        self.assertGreater(controls["localized_loss_global_gradient_correlation"], 0.58)
        self.assertGreater(controls["localized_loss_failed_windows"], 0)


if __name__ == "__main__":
    unittest.main()
