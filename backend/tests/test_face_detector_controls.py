import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import backend.benchmark.run_face_detector_controls as controls


class FaceDetectorControlsTest(unittest.TestCase):
    def test_dirty_provenance_fails_closed_unless_explicitly_allowed(self):
        checks = {
            "implementation_provenance_clean": False,
            "detectors": True,
        }

        self.assertFalse(controls._checks_pass(checks, allow_dirty=False))
        self.assertTrue(controls._checks_pass(checks, allow_dirty=True))

    def test_checkerboard_is_deterministic_rgb(self):
        first = controls._checkerboard(64)
        second = controls._checkerboard(64)

        self.assertEqual(first.shape, (64, 64, 3))
        self.assertEqual(first.dtype, np.uint8)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(len(np.unique(first.reshape(-1, 3), axis=0)), 1)

    def test_subject_removed_background_replaces_only_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scene = Path(temp_dir)
            (scene / "face_parts").mkdir()
            source = np.full((8, 8, 3), 17, dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 3:5] = 255
            Image.fromarray(source).save(scene / "source.png")
            Image.fromarray(mask).save(scene / "face_parts" / "face.png")

            result = controls._subject_removed_background(scene, 8)

            self.assertTrue(np.all(result[mask > 0] == 245))
            self.assertTrue(np.all(result[mask == 0] == 17))

    def test_run_writes_bound_control_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = root / "fixture"
            procedural = root / "procedural"
            procedural.mkdir()
            for index, directory in enumerate(controls.FACE_CASES.values()):
                scene = fixture / directory
                (scene / "face_parts").mkdir(parents=True)
                image = np.full((32, 32, 3), 80 + index, dtype=np.uint8)
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[8:24, 8:24] = 255
                Image.fromarray(image).save(scene / "source.png")
                Image.fromarray(mask).save(scene / "face_parts" / "face.png")
            for index in range(3):
                Image.new("RGB", (32, 32), (index * 30, 20, 10)).save(
                    procedural / f"procedural_mesh_{index:04d}_v00_full.png"
                )
            yunet_model = root / "yunet.onnx"
            haar_cascade = root / "haarcascade_frontalface_default.xml"
            yunet_model.write_bytes(b"yunet")
            haar_cascade.write_bytes(b"haar")
            output = root / "summary.json"
            positive_region = {
                "detector": "opencv-yunet-2023mar",
                "confidence": 0.95,
            }

            with (
                patch.object(
                    controls,
                    "_sha256_file",
                    side_effect=lambda path: (
                        controls.YUNET_MODEL_SHA256
                        if Path(path) == yunet_model
                        else controls.HAAR_CASCADE_SHA256
                    ),
                ),
                patch.object(
                    controls,
                    "_detect_faces_yunet",
                    return_value=[positive_region],
                ),
                patch.object(
                    controls,
                    "detect_face_regions",
                    return_value=([positive_region], []),
                ),
                patch.object(
                    controls,
                    "_detector_record",
                    side_effect=lambda name, image, _minimum: {
                        "control": name,
                        "shape": list(image.shape),
                        "rgb_sha256": controls._rgb_sha256(image),
                        "mediapipe_count": 0,
                        "yunet_count": 0,
                        "yunet_confidences": [],
                        "haar_count": 0,
                        "final_chain_count": 0,
                        "final_chain_detectors": [],
                        "final_chain_errors": [],
                    },
                ),
                patch.object(
                    controls,
                    "_git_provenance",
                    return_value={"available": True, "clean": True, "revision": "abc"},
                ),
            ):
                summary = controls.run(
                    fixture_root=fixture,
                    procedural_dataset=procedural,
                    yunet_model=yunet_model,
                    haar_cascade=haar_cascade,
                    output=output,
                )

            self.assertTrue(summary["checks"]["passed"])
            self.assertEqual(len(summary["negative_controls"]), 8)
            self.assertTrue(summary["positive_control"]["rgb_sha256"])
            self.assertTrue(
                all(row["rgb_sha256"] for row in summary["negative_controls"])
            )
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
