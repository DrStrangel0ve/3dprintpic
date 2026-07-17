import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend import david_face_depth as david


class _FakeInput:
    name = "image"


class _FakeSession:
    def __init__(self):
        self.inputs = []

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, _outputs, inputs):
        self.inputs.append(inputs["image"].copy())
        ramp = np.linspace(0.2, 0.8, 512, dtype=np.float32)
        return [np.broadcast_to(ramp[None, None, None, :], (1, 1, 512, 512))]


class DAViDFaceDepthTests(unittest.TestCase):
    def test_prepare_uses_bgr_and_preserves_aspect_metadata(self):
        image = np.zeros((20, 10, 3), dtype=np.uint8)
        image[:, :, 0] = 10
        image[:, :, 1] = 20
        image[:, :, 2] = 30

        tensor, metadata = david._prepare_bgr_input(image)

        self.assertEqual(tensor.shape, (1, 3, 512, 512))
        self.assertAlmostEqual(float(tensor[0, 0, 256, 256]), 30.0 / 255.0)
        self.assertAlmostEqual(float(tensor[0, 2, 256, 256]), 10.0 / 255.0)
        self.assertEqual(metadata["original_shape"], [20, 10])
        restored = david._restore_dense_map(
            np.ones((1, 1, 512, 512), dtype=np.float32),
            metadata,
        )
        self.assertEqual(restored.shape, (20, 10))

    def test_normalize_relative_depth_rejects_flat_output(self):
        with self.assertRaisesRegex(ValueError, "no usable relative span"):
            david._normalize_relative_depth(np.ones((8, 8), dtype=np.float32))

    def test_inference_preserves_raw_z_orientation_and_normalizes_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.onnx"
            model_path.write_bytes(b"fake")
            session = _FakeSession()
            provider = david.DAViDFaceDepth.__new__(david.DAViDFaceDepth)
            provider.model_path = model_path
            provider.device = "cpu"
            provider.session = session
            provider.input_name = "image"
            provider.timings = []
            provider._free_before_load = None
            provider._free_after_first_inference = None
            provider.last_inference = {}

            depth = provider.infer_array(
                np.full((24, 16, 3), 128, dtype=np.uint8)
            )

        self.assertEqual(depth.shape, (24, 16))
        self.assertAlmostEqual(float(np.min(depth)), 0.0, places=5)
        self.assertAlmostEqual(float(np.max(depth)), 1.0, places=5)
        self.assertLess(float(depth[:, 0].mean()), float(depth[:, -1].mean()))
        self.assertEqual(session.inputs[0].shape, (1, 3, 512, 512))

    def test_configured_asset_checksum_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "wrong.onnx"
            path.write_bytes(b"wrong")
            with patch.dict(
                "os.environ",
                {"DAVID_DEPTH_MODEL_PATH": str(path)},
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                    david.resolve_david_depth_model()


if __name__ == "__main__":
    unittest.main()
