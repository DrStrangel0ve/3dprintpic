import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.benchmark.makehuman_face_fixture import load_makehuman_face_fixture
from backend.benchmark.run_makehuman_face_relief_smoke import (
    DEFAULT_ASSET_DIR,
    DEFAULT_SCENES,
    SceneSpec,
    _crop_bounds,
    _emit_row,
    _render_scene,
    run,
)
from backend.pic_to_3d import _attenuate_direction_reversals_to_source


class MakeHumanFaceReliefSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = load_makehuman_face_fixture(DEFAULT_ASSET_DIR)

    def test_default_crop_bounds_match_pinned_framing_stress(self):
        self.assertEqual(_crop_bounds(384, 256, "centered"), (64, 320, 64, 320))
        self.assertEqual(_crop_bounds(384, 256, "left_frame"), (64, 320, 110, 366))
        self.assertEqual(_crop_bounds(384, 256, "right_frame"), (64, 320, 0, 256))

    def test_default_scenes_retain_named_parts_and_expected_frame_contact(self):
        records = []
        for scene in DEFAULT_SCENES:
            _, record = _render_scene(
                self.fixture,
                scene,
                render_size=384,
                crop_size=256,
            )
            records.append(record)
        self.assertTrue(all(record["checks"]["passed"] for record in records))
        self.assertFalse(records[0]["frame_contact"])
        self.assertTrue(records[1]["frame_contact"])
        self.assertTrue(records[2]["frame_contact"])
        self.assertFalse(records[0]["head_frame_contact"])
        self.assertTrue(records[1]["head_frame_contact"])
        self.assertTrue(records[2]["head_frame_contact"])
        self.assertGreaterEqual(min(records[1]["named_part_retention"].values()), 0.995)
        self.assertGreaterEqual(min(records[2]["named_part_retention"].values()), 0.995)

    def test_one_bounded_row_passes_relief_background_and_shell_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            row, context = _emit_row(
                Path(temporary),
                self.fixture,
                DEFAULT_SCENES[0],
                relief_height_mm=30.0,
                render_size=128,
                crop_size=96,
                physical_size_mm=96.0,
                background_depth_ratio=0.65,
            )
        self.assertTrue(row["checks"]["passed"])
        self.assertTrue(row["checks"]["background_preservation"])
        self.assertTrue(row["checks"]["feasible_attachment"])
        self.assertTrue(row["checks"]["printable_mesh"])
        self.assertTrue(row["checks"]["complete_shell"])
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
        self.assertEqual(context["face_mask"].shape, (96, 96))
        self.assertTrue(all(mask.any() for mask in context["part_masks"].values()))

    def test_direction_reversal_projection_is_local_and_fail_closed(self):
        source = np.tile(np.arange(7, dtype=np.float32), (7, 1))
        candidate = source.copy()
        candidate[3, 3] = -4.0
        adjustable = np.zeros(source.shape, dtype=bool)
        adjustable[1:6, 1:6] = True

        projected, stats = _attenuate_direction_reversals_to_source(
            source,
            candidate,
            adjustable,
            max_neighbor_step_mm=1.0,
        )

        self.assertTrue(stats["passed"])
        self.assertGreater(stats["initial_cardinal_reversals"], 0)
        self.assertGreater(stats["initial_diagonal_reversals"], 0)
        self.assertEqual(stats["final_cardinal_reversals"], 0)
        self.assertEqual(stats["final_diagonal_reversals"], 0)
        np.testing.assert_array_equal(projected[~adjustable], candidate[~adjustable])

        unchanged, no_op = _attenuate_direction_reversals_to_source(
            source,
            source,
            adjustable,
            max_neighbor_step_mm=1.0,
        )
        self.assertFalse(no_op["enabled"])
        self.assertTrue(no_op["passed"])
        np.testing.assert_array_equal(unchanged, source)

        outside_adjustable = np.zeros(source.shape, dtype=bool)
        outside_adjustable[0, 0] = True
        rejected, outside_stats = _attenuate_direction_reversals_to_source(
            source,
            candidate,
            outside_adjustable,
            max_neighbor_step_mm=1.0,
        )
        self.assertFalse(outside_stats["passed"])
        self.assertEqual(outside_stats["reason"], "reversals_outside_adjustable_region")
        np.testing.assert_array_equal(rejected, candidate)

        _, limited_stats = _attenuate_direction_reversals_to_source(
            source,
            candidate,
            adjustable,
            max_neighbor_step_mm=1.0,
            iterations=1,
        )
        self.assertFalse(limited_stats["passed"])
        self.assertEqual(limited_stats["reason"], "iteration_limit")

        _, invalid_stats = _attenuate_direction_reversals_to_source(
            source,
            candidate,
            adjustable,
            max_neighbor_step_mm=0.0,
        )
        self.assertEqual(invalid_stats["reason"], "invalid_max_neighbor_step")

        source_with_gap = source.copy()
        source_with_gap[0, 0] = np.nan
        _, coverage_stats = _attenuate_direction_reversals_to_source(
            source_with_gap,
            candidate,
            adjustable,
            max_neighbor_step_mm=1.0,
        )
        self.assertEqual(coverage_stats["reason"], "finite_coverage_mismatch")

    def test_right_frame_asymmetric_face_survives_40mm_relief(self):
        with tempfile.TemporaryDirectory() as temporary:
            row, _ = _emit_row(
                Path(temporary),
                self.fixture,
                DEFAULT_SCENES[2],
                relief_height_mm=40.0,
                render_size=384,
                crop_size=256,
                physical_size_mm=96.0,
                background_depth_ratio=0.65,
            )
        self.assertTrue(row["checks"]["passed"])
        # Keep this difficult crop tighter than the shared appearance gates.
        self.assertLess(row["absolute_face"]["normal_angle_p95_deg"], 15.0)
        self.assertGreater(
            row["absolute_face"]["minimum_lighting_correlation"],
            0.95,
        )
        self.assertEqual(row["absolute_named_parts"]["failed_parts"], [])
        self.assertTrue(row["background"]["passed"])
        projection = row["direction_reversal_projection"]
        self.assertTrue(projection["enabled"])
        self.assertTrue(projection["passed"])
        self.assertGreater(projection["corrected_pixels"], 0)
        self.assertEqual(projection["final_cardinal_reversals"], 0)
        self.assertEqual(projection["final_diagonal_reversals"], 0)

    def test_failed_reversal_projection_cannot_promote_screened_face(self):
        def reject_projection(_source, candidate, _region, **_kwargs):
            return candidate, {
                "enabled": True,
                "passed": False,
                "reason": "forced_test_failure",
                "corrected_pixels": 0,
                "final_cardinal_reversals": 1,
                "final_diagonal_reversals": 0,
            }

        with tempfile.TemporaryDirectory() as temporary, patch(
            "backend.pic_to_3d._attenuate_direction_reversals_to_source",
            side_effect=reject_projection,
        ):
            row, _ = _emit_row(
                Path(temporary),
                self.fixture,
                DEFAULT_SCENES[2],
                relief_height_mm=40.0,
                render_size=384,
                crop_size=256,
                physical_size_mm=96.0,
                background_depth_ratio=0.65,
            )

        self.assertFalse(row["direction_reversal_projection"]["passed"])
        self.assertFalse(row["checks"]["direction_reversal_projection"])
        self.assertFalse(row["checks"]["screened_face_reconstruction"])
        self.assertFalse(row["checks"]["passed"])

    def test_matrix_validation_fails_closed_before_rendering(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Exactly two distinct"):
                run(temporary, relief_heights_mm=(30.0, 30.0))
            with self.assertRaisesRegex(ValueError, "profile names must be unique"):
                run(
                    temporary,
                    scenes=(
                        SceneSpec("caucasian_female_smile", "centered", 0.0),
                        SceneSpec("caucasian_female_smile", "left_frame", 1.0),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
