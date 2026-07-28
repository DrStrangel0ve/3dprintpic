import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.benchmark import run_private_background_photo_detail_replay as replay


class PrivateBackgroundPhotoDetailReplayTests(unittest.TestCase):
    def test_transform_mask_replays_resize_flip_and_crop(self):
        mask = np.zeros((4, 6), dtype=bool)
        mask[1:3, 1:3] = True
        transform = {
            "input_depth_shape": [4, 6],
            "target_depth_shape": [4, 6],
            "flip_x": True,
            "mesh_shape_before_crop": [4, 6],
            "crop_bbox_rc": [1, 1, 4, 6],
            "emitted_shape": [3, 5],
        }
        transformed = replay._transform_mask_to_surface_grid(mask, transform)
        expected = np.flip(mask, axis=1)[1:4, 1:6]
        np.testing.assert_array_equal(transformed, expected)

    def test_transform_mask_requires_complete_contract(self):
        with self.assertRaisesRegex(ValueError, "Surface-grid transform is missing"):
            replay._transform_mask_to_surface_grid(
                np.ones((2, 2), dtype=bool), {"input_depth_shape": [2, 2]}
            )

    def test_aggregate_privacy_rejects_paths_and_hashes(self):
        replay._assert_aggregate_privacy({"scene_label": "portrait"})
        with self.assertRaisesRegex(ValueError, "private path"):
            replay._assert_aggregate_privacy({"value": r"C:\Users\private\source.png"})
        with self.assertRaisesRegex(ValueError, "content hashes"):
            replay._assert_aggregate_privacy({"value": "a" * 64})
        with self.assertRaisesRegex(ValueError, "private job identifiers"):
            replay._assert_aggregate_privacy({"value": "a" * 32})

    def test_output_must_be_gitignored(self):
        repository = Path(replay.__file__).resolve().parents[2]
        replay._require_ignored(repository / "backend/output/private-test", repository)
        with tempfile.TemporaryDirectory(dir=repository) as tmp_dir:
            with self.assertRaisesRegex(ValueError, "covered by .gitignore"):
                replay._require_ignored(Path(tmp_dir), repository)

    def test_scene_labels_are_non_semantic(self):
        repository = Path(replay.__file__).resolve().parents[2]
        with self.assertRaisesRegex(ValueError, "scene-NN"):
            replay._validate_scene({"label": "portrait"}, repository)

    def test_private_checksum_contract_rejects_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.bin"
            path.write_bytes(b"original")
            expected = {"depth_npy": replay._sha256(path)}
            replay._verify_private_checksums(
                "scene-01", {"depth_npy": path}, expected
            )
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                replay._verify_private_checksums(
                    "scene-01", {"depth_npy": path}, expected
                )

    def test_replay_rejects_unapproved_control_overrides(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            ValueError, "fixed to 30 mm"
        ):
            replay.run(
                Path(temporary) / "missing.json",
                Path(temporary) / "output",
                relief_height_mm=29.0,
            )


if __name__ == "__main__":
    unittest.main()
