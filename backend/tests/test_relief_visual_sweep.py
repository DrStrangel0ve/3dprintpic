import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.benchmark.run_relief_visual_sweep import (
    FACE_APPEARANCE_GATES,
    RELIEF_HEIGHTS_MM,
    _appearance_checks,
    _appearance_negative_controls,
    _cross_height_face_shape_metrics,
    _mask_topology,
    _matrix_coverage,
    _stl_heightfield_agreement,
    _sweep_specs,
    _topology_scene,
    run,
)


class ReliefVisualSweepTest(unittest.TestCase):
    def test_topology_matrix_is_varied_deterministic_and_keeps_context(self):
        specs = _sweep_specs()

        self.assertEqual(len(specs), 4)
        self.assertEqual(len(RELIEF_HEIGHTS_MM), 3)
        self.assertEqual(len({spec.topology_id for spec in specs}), 4)

        coverages = []
        observed_topologies = []
        for spec in specs:
            with self.subTest(topology=spec.topology_id):
                source, face, subject = _topology_scene(spec)
                replay = _topology_scene(spec)
                topology = _mask_topology(subject)
                self.assertEqual(source.shape, (121, 121))
                self.assertTrue(np.all(subject[face]))
                self.assertGreater(np.count_nonzero(face), 1000)
                self.assertGreater(np.count_nonzero(~subject), 4000)
                self.assertTrue(np.all(np.isfinite(source)))
                np.testing.assert_array_equal(source, replay[0])
                np.testing.assert_array_equal(face, replay[1])
                np.testing.assert_array_equal(subject, replay[2])
                coverages.append(float(np.mean(subject)))
                observed_topologies.append(
                    (topology["component_count"], topology["hole_count"])
                )
                self.assertEqual(
                    topology["component_count"],
                    spec.expected_components,
                )
                self.assertEqual(topology["hole_count"], spec.expected_holes)

        self.assertLess(min(coverages), 0.4)
        self.assertGreater(max(coverages), 0.6)
        self.assertIn((2, 1), observed_topologies)

    def test_appearance_gates_fail_closed_and_negative_controls_are_effective(self):
        unavailable = _appearance_checks(
            {"available": False, "reason": "insufficient_surface_samples"},
            FACE_APPEARANCE_GATES,
        )
        controls = _appearance_negative_controls()

        self.assertFalse(unavailable["passed"])
        self.assertFalse(unavailable["telemetry"])
        self.assertTrue(controls["checks"]["constant_height_offset_passes"])
        self.assertTrue(controls["checks"]["flattened_surface_rejected"])
        self.assertTrue(controls["checks"]["heavy_smoothing_rejected"])
        self.assertTrue(controls["checks"]["missing_candidate_pixels_rejected"])
        self.assertTrue(controls["checks"]["single_component_damage_rejected"])
        self.assertTrue(controls["checks"]["cross_height_face_damage_rejected"])
        self.assertTrue(controls["checks"]["passed"])

    def test_cross_height_shape_gate_accepts_scaling_and_rejects_local_damage(self):
        rows, cols = np.indices((81, 101), dtype=np.float32)
        face = (rows - 40.0) ** 2 / 28.0**2 + (cols - 50.0) ** 2 / 24.0**2 <= 1.0
        reference = (
            2.0
            + 0.02 * rows
            + 0.03 * cols
            + 4.0 * np.exp(-((rows - 40.0) ** 2 + (cols - 50.0) ** 2) / 180.0)
        )
        scaled = 1.8 * reference + 5.0
        damaged = scaled.copy()
        damaged[27:55, 38:65] = float(np.mean(damaged[27:55, 38:65]))

        matching = _cross_height_face_shape_metrics(reference, scaled, face)
        mismatching = _cross_height_face_shape_metrics(reference, damaged, face)

        self.assertTrue(matching["passed"])
        self.assertAlmostEqual(matching["shape_correlation"], 1.0, places=10)
        self.assertLess(matching["normalized_shape_rmse"], 1e-6)
        self.assertFalse(mismatching["passed"])

    def test_partial_sweep_cannot_certify_but_emits_complete_row_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = run(
                root / "outputs",
                limit=1,
                allow_dirty=True,
                smoke=True,
            )

            row = summary["rows"][0]
            row_dir = root / "outputs" / row["row_id"]
            surface_path = row_dir / "emitted_surface.npy"
            stl_path = row_dir / "relief.stl"
            self.assertTrue(row["emitted_surface"]["passed"])
            surface = np.load(surface_path)
            np.save(surface_path, surface + 0.25)
            tampered = _stl_heightfield_agreement(stl_path, surface_path)

        self.assertEqual(len(summary["rows"]), 1)
        self.assertFalse(summary["checks"]["full_matrix_complete"])
        self.assertFalse(summary["checks"]["topology_coverage_complete"])
        self.assertFalse(summary["checks"]["height_coverage_complete"])
        self.assertTrue(summary["rows"][0]["face_appearance"]["available"])
        self.assertTrue(summary["rows"][0]["background_appearance"]["available"])
        self.assertTrue(summary["rows"][0]["topology"]["printable"])
        self.assertFalse(tampered["passed"])
        self.assertGreater(tampered["max_abs_error_mm"], 0.2)

    def test_matrix_coverage_rejects_duplicate_substitution(self):
        expected = {("a", 20.0), ("a", 30.0), ("b", 20.0)}
        rows = [
            {"topology_id": "a", "relief_height_mm": 20.0},
            {"topology_id": "a", "relief_height_mm": 30.0},
            {"topology_id": "a", "relief_height_mm": 30.0},
        ]

        coverage = _matrix_coverage(rows, expected)

        self.assertFalse(coverage["exact"])
        self.assertFalse(coverage["unique"])
        self.assertEqual(coverage["missing"], [["b", 20.0]])


if __name__ == "__main__":
    unittest.main()
