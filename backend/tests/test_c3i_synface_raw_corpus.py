import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend.benchmark import c3i_synface_raw_corpus as raw


VALID_POSE = """Camera Location: (-0.1000, -0.3296, 1.6065)
Head Point Location: (0.0000, -0.0296, 1.6065)
Camera Rotation: Yaw 0.00 Pitch 90.00 Roll -0.00
Head Rotation: Yaw 24.50 Pitch -2.00 Roll 1.00
"""


class C3ISynFaceRawCorpusTests(unittest.TestCase):
    def test_pose_parser_preserves_camera_and_head_geometry(self):
        pose = raw.parse_pose_text(VALID_POSE)
        self.assertEqual(pose["head_rotation_yaw_pitch_roll"], (24.5, -2.0, 1.0))
        self.assertLess(pose["camera_view_yaw_degrees"], 0.0)

    def test_pose_parser_rejects_unpublished_schema(self):
        with self.assertRaisesRegex(ValueError, "published schema"):
            raw.parse_pose_text(VALID_POSE + "unexpected: value\n")

    def test_raw_exr_requires_equal_finite_float_channels(self):
        depth = np.linspace(0.2, 0.4, 48, dtype=np.float32).reshape(4, 4, 3)
        depth[..., 1] = depth[..., 0]
        depth[..., 2] = depth[..., 0]
        with mock.patch.object(raw.cv2, "imread", return_value=depth):
            loaded = raw._load_raw_exr(Path("depth.exr"))
        self.assertEqual(loaded.shape, (4, 4))
        depth[..., 2] += 0.1
        with mock.patch.object(raw.cv2, "imread", return_value=depth):
            with self.assertRaisesRegex(ValueError, "channels disagree"):
                raw._load_raw_exr(Path("depth.exr"))

    def test_depth_resampling_masks_blender_no_hit_values(self):
        depth = np.linspace(0.2, 0.4, 16, dtype=np.float32).reshape(4, 4)
        depth[1, 2] = raw.BLENDER_NO_HIT_DEPTH

        resized, transform, validity = raw._resize_depth_and_place(
            depth,
            scale=1.0,
            dimension=4,
            center_xy=(2.0, 2.0),
            output_center_xy=(2.0, 2.0),
        )

        self.assertTrue(np.isnan(resized[1, 2]))
        self.assertEqual(validity["source_no_hit_pixels"], 1)
        self.assertEqual(validity["output_valid_pixels"], 15)
        self.assertEqual(transform["origin_xy"], [0, 0])

    def test_release_preflight_hashes_only_exact_sized_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / raw.ARCHIVE_FILENAME
            archive.write_bytes(b"verified archive")
            extracted = root / "extracted"
            extracted.mkdir()
            with (
                mock.patch.object(raw, "ARCHIVE_BYTES", archive.stat().st_size),
                mock.patch.object(raw, "ARCHIVE_SHA256", raw._sha256(archive)),
                mock.patch.object(raw, "EXTRACTED_FILE_COUNT", 0),
                mock.patch.object(raw, "EXTRACTED_BYTES", 0),
                mock.patch.object(raw, "PAIR_COUNT", 0),
            ):
                evidence = raw.raw_release_preflight(archive, extracted)
            self.assertTrue(evidence["ready"])
            with mock.patch.object(
                raw,
                "_sha256",
                side_effect=AssertionError("wrong-size archive must not be hashed"),
            ):
                evidence = raw.raw_release_preflight(root / "missing.7z", extracted)
            self.assertFalse(evidence["ready"])

    def test_balanced_selection_is_identity_disjoint(self):
        records = []
        for identity in raw.IDENTITIES:
            for background in raw.BACKGROUNDS:
                for expression in raw.EXPRESSIONS:
                    for motion in raw.MOTIONS:
                        for frame, yaw in (("0000", -26.0), ("0001", 26.0)):
                            records.append(
                                {
                                    "identity": identity,
                                    "split": raw.SPLIT_BY_IDENTITY[identity],
                                    "background": background,
                                    "expression": expression,
                                    "motion": motion,
                                    "frame": frame,
                                    "effective_yaw_degrees": yaw,
                                    "head_pitch_degrees": 0.0,
                                    "head_roll_degrees": 0.0,
                                    "pose": {"camera_view_yaw_degrees": 0.0},
                                }
                            )
        selected = raw.select_balanced_records(records)
        self.assertEqual(len(selected), 120)
        self.assertEqual(
            {row["identity"] for row in selected if row["split"] == "sealed"},
            {"0029"},
        )
        self.assertEqual(
            {row["identity"] for row in selected if row["split"] == "validation"},
            {"0025"},
        )
        self.assertEqual(
            {row["identity"] for row in selected if row["split"] == "train"},
            {"0020", "0024"},
        )

    def test_selection_penalizes_pitch_and_compound_camera_motion(self):
        base = {
            "motion": "HeadCameraRotTran",
            "frame": "0000",
            "effective_yaw_degrees": -26.0,
            "head_pitch_degrees": 20.0,
            "head_roll_degrees": 0.0,
            "pose": {"camera_view_yaw_degrees": 22.0},
        }
        clean = {
            **base,
            "frame": "0001",
            "effective_yaw_degrees": -22.0,
            "head_pitch_degrees": 0.5,
            "pose": {"camera_view_yaw_degrees": 4.0},
        }
        self.assertLess(
            raw._selection_score(clean, -26.0),
            raw._selection_score(base, -26.0),
        )

    def test_face_rejection_falls_back_within_the_same_balanced_cell(self):
        candidates = [
            {
                "identity": "0020",
                "background": "Barbershop",
                "expression": "Angry",
                "motion": "HeadRot",
                "frame": frame,
                "selection_rank": rank,
            }
            for rank, frame in enumerate(("0000", "0001"))
        ]

        def save(candidate):
            if candidate["selection_rank"] == 0:
                raise raw.FaceRowRejected("no complete face")
            return {"row_id": "fallback-row"}

        rows, rejections = raw._emit_balanced_rows([candidates], save)
        self.assertEqual(rows, [{"row_id": "fallback-row"}])
        self.assertEqual(len(rejections), 1)
        self.assertEqual(rejections[0]["frame"], "0000")

    def test_raw_path_parser_rejects_unknown_taxonomy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "9999" / "Complex" / "Classroom" / "Happy" / "HeadRot"
            path.mkdir(parents=True)
            pose = path / "data_0000.txt"
            pose.write_text(VALID_POSE, encoding="utf-8")
            (path / "rgb_0000.jpg").write_bytes(b"rgb")
            (path / "depthExr_0000.exr").write_bytes(b"depth")
            with self.assertRaisesRegex(ValueError, "taxonomy"):
                raw._record_from_pose_path(pose, root)


if __name__ == "__main__":
    unittest.main()
