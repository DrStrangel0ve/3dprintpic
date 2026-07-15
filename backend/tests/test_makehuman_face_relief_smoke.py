import tempfile
import unittest
from pathlib import Path

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
        self.assertEqual(context["face_mask"].shape, (96, 96))
        self.assertTrue(all(mask.any() for mask in context["part_masks"].values()))

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
