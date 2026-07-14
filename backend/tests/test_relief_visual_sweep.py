import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from stl import mesh

from backend.benchmark.run_relief_visual_sweep import (
    BACKGROUND_APPEARANCE_GATES,
    FACE_APPEARANCE_GATES,
    RELIEF_HEIGHTS_MM,
    SELECTION_APPEARANCE_GATES,
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
    def _run_sweep(self, *args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return run(*args, **kwargs)

    def _assert_component_appearance(self, row, region, expected_count, gates):
        telemetry = row[f"{region}_appearance"]
        checks = row[f"{region}_appearance_checks"]

        self.assertTrue(telemetry["available"])
        self.assertEqual(telemetry["component_count"], expected_count)
        self.assertEqual(len(telemetry["components"]), expected_count)
        self.assertTrue(checks["telemetry"])
        self.assertTrue(checks["components"])
        self.assertTrue(checks["passed"])
        for component in telemetry["components"]:
            component_metrics = dict(component)
            component_metrics.pop("components", None)
            self.assertTrue(
                _appearance_checks(component_metrics, gates)["passed"],
                msg=f"{row['row_id']} {region} component {component['component']}",
            )

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
                background_topology = _mask_topology(~subject)
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
                self.assertEqual(
                    background_topology["component_count"],
                    spec.expected_background_components,
                )

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
            summary = self._run_sweep(
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
            tampered_surface = surface.copy()
            tampered_surface[surface.shape[0] // 2, surface.shape[1] // 2] += 0.25
            np.save(surface_path, tampered_surface)
            tampered = _stl_heightfield_agreement(stl_path, surface_path)

        self.assertEqual(len(summary["rows"]), 1)
        self.assertFalse(summary["checks"]["full_matrix_complete"])
        self.assertFalse(summary["checks"]["topology_coverage_complete"])
        self.assertFalse(summary["checks"]["height_coverage_complete"])
        self.assertTrue(summary["rows"][0]["face_appearance"]["available"])
        self._assert_component_appearance(
            summary["rows"][0],
            "selection",
            1,
            SELECTION_APPEARANCE_GATES,
        )
        self._assert_component_appearance(
            summary["rows"][0],
            "background",
            1,
            BACKGROUND_APPEARANCE_GATES,
        )
        self.assertTrue(summary["rows"][0]["topology"]["printable"])
        self.assertFalse(tampered["passed"])
        self.assertGreater(tampered["max_abs_error_mm"], 0.2)

    def test_stl_alternate_diagonal_preserves_samples_but_fails_facet_geometry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary = self._run_sweep(
                root / "outputs",
                limit=1,
                allow_dirty=True,
                smoke=True,
            )
            row_dir = root / "outputs" / summary["rows"][0]["row_id"]
            surface_path = row_dir / "emitted_surface.npy"
            original_path = row_dir / "relief.stl"
            alternate_path = row_dir / "relief-alternate-diagonal.stl"
            missing_walls_path = row_dir / "relief-missing-walls.stl"
            reversed_wall_path = row_dir / "relief-reversed-wall.stl"
            alternate_mesh = mesh.Mesh.from_file(str(original_path))
            first_cell = alternate_mesh.vectors[:2].copy()
            corners = {
                (float(vertex[0]), float(vertex[1])): vertex.copy()
                for vertex in first_cell.reshape(-1, 3)
            }
            self.assertEqual(len(corners), 4)
            x_values = sorted({key[0] for key in corners})
            y_values = sorted({key[1] for key in corners})
            self.assertEqual(len(x_values), 2)
            self.assertEqual(len(y_values), 2)
            v0 = corners[(x_values[0], y_values[0])]
            v1 = corners[(x_values[0], y_values[1])]
            v2 = corners[(x_values[1], y_values[0])]
            v3 = corners[(x_values[1], y_values[1])]
            alternate_mesh.vectors[0] = np.asarray((v0, v3, v1))
            alternate_mesh.vectors[1] = np.asarray((v0, v2, v3))
            alternate_mesh.update_normals()
            alternate_mesh.save(str(alternate_path))

            agreement = _stl_heightfield_agreement(
                alternate_path,
                surface_path,
            )
            original_mesh = mesh.Mesh.from_file(str(original_path))
            bottom_vertices = np.isclose(original_mesh.vectors[:, :, 2], 0.0)
            side_faces = np.any(bottom_vertices, axis=1) & ~np.all(
                bottom_vertices,
                axis=1,
            )
            self.assertGreater(np.count_nonzero(side_faces), 0)
            kept_vectors = original_mesh.vectors[~side_faces]
            missing_walls_mesh = mesh.Mesh(
                np.zeros(len(kept_vectors), dtype=mesh.Mesh.dtype)
            )
            missing_walls_mesh.vectors[:] = kept_vectors
            missing_walls_mesh.update_normals()
            missing_walls_mesh.save(str(missing_walls_path))
            missing_walls = _stl_heightfield_agreement(
                missing_walls_path,
                surface_path,
            )

            reversed_wall_mesh = mesh.Mesh.from_file(str(original_path))
            wall_index = int(np.flatnonzero(side_faces)[0])
            reversed_wall_mesh.vectors[wall_index] = reversed_wall_mesh.vectors[
                wall_index, [0, 2, 1]
            ]
            reversed_wall_mesh.update_normals()
            reversed_wall_mesh.save(str(reversed_wall_path))
            reversed_wall = _stl_heightfield_agreement(
                reversed_wall_path,
                surface_path,
            )

        self.assertTrue(agreement["available"])
        self.assertTrue(agreement["sample_grid_passed"])
        self.assertEqual(agreement["max_abs_error_mm"], 0.0)
        self.assertTrue(agreement["triangle_count_match"])
        self.assertFalse(agreement["triangle_set_match"])
        self.assertFalse(agreement["facet_geometry_passed"])
        self.assertFalse(agreement["passed"])
        self.assertTrue(missing_walls["sample_grid_passed"])
        self.assertTrue(missing_walls["top_facet_geometry_passed"])
        self.assertFalse(missing_walls["shell_triangle_count_match"])
        self.assertFalse(missing_walls["complete_shell_verified"])
        self.assertFalse(missing_walls["passed"])
        self.assertTrue(reversed_wall["sample_grid_passed"])
        self.assertTrue(reversed_wall["top_facet_geometry_passed"])
        self.assertTrue(reversed_wall["shell_connectivity_match"])
        self.assertLess(reversed_wall["minimum_shell_normal_cosine"], -0.99)
        self.assertFalse(reversed_wall["complete_shell_verified"])
        self.assertFalse(reversed_wall["passed"])

    def test_full_relief_height_matrix_passes_all_quality_gates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = self._run_sweep(
                Path(tmp_dir) / "outputs",
                allow_dirty=True,
            )

        specs = {spec.topology_id: spec for spec in _sweep_specs()}
        expected_pairs = {
            (topology_id, float(height_mm))
            for topology_id in specs
            for height_mm in RELIEF_HEIGHTS_MM
        }
        observed_pairs = {
            (row["topology_id"], row["relief_height_mm"])
            for row in summary["rows"]
        }
        self.assertEqual(summary["matrix"]["expected_rows"], 12)
        self.assertEqual(summary["matrix"]["completed_rows"], 12)
        self.assertEqual(observed_pairs, expected_pairs)
        self.assertTrue(summary["matrix"]["coverage"]["exact"])
        self.assertTrue(summary["matrix"]["coverage"]["unique"])
        self.assertEqual(summary["matrix"]["coverage"]["missing"], [])
        self.assertEqual(summary["matrix"]["coverage"]["unexpected"], [])

        required_summary_checks = (
            "full_matrix_complete",
            "matrix_rows_unique",
            "expected_row_count",
            "topology_coverage_complete",
            "height_coverage_complete",
            "negative_controls_passed",
            "cross_height_face_coverage_complete",
            "cross_height_face_quality_passed",
            "all_row_gates_passed",
            "all_face_appearance_gates_passed",
            "all_selection_appearance_gates_passed",
            "all_background_appearance_gates_passed",
            "all_emitted_surfaces_match",
            "all_meshes_printable",
        )
        for check in required_summary_checks:
            self.assertTrue(summary["checks"][check], msg=check)

        for row in summary["rows"]:
            spec = specs[row["topology_id"]]
            with self.subTest(row=row["row_id"]):
                self.assertTrue(row["checks"]["passed"])
                self.assertTrue(row["checks"]["mask_topology"])
                self.assertTrue(row["checks"]["appearance_component_coverage"])
                self.assertTrue(row["face_appearance_checks"]["passed"])
                self._assert_component_appearance(
                    row,
                    "selection",
                    spec.expected_components,
                    SELECTION_APPEARANCE_GATES,
                )
                self._assert_component_appearance(
                    row,
                    "background",
                    spec.expected_background_components,
                    BACKGROUND_APPEARANCE_GATES,
                )
                self.assertEqual(
                    row["mask_topology"]["component_count"],
                    spec.expected_components,
                )
                self.assertEqual(
                    row["mask_topology"]["hole_count"],
                    spec.expected_holes,
                )
                self.assertTrue(row["topology"]["printable"])
                self.assertTrue(row["emitted_surface"]["sample_grid_passed"])
                self.assertTrue(row["emitted_surface"]["facet_geometry_passed"])
                self.assertTrue(row["emitted_surface"]["passed"])

        cross_height = summary["cross_height_face_consistency"]
        self.assertTrue(cross_height["coverage_complete"])
        self.assertTrue(cross_height["quality_passed"])
        self.assertTrue(cross_height["passed"])
        self.assertEqual(cross_height["comparison_count"], 12)
        self.assertTrue(all(record["passed"] for record in cross_height["records"]))

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
