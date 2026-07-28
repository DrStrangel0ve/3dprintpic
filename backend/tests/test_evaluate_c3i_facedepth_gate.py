import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from backend.benchmark import evaluate_c3i_facedepth_gate as gate


class EvaluateC3IFaceDepthGateTests(unittest.TestCase):
    def test_checkpoint_preflight_requires_exact_bytes_and_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights_model.pth"
            path.write_bytes(b"pinned-checkpoint")
            with (
                mock.patch.object(
                    gate,
                    "EXPECTED_CHECKPOINT_BYTES",
                    path.stat().st_size,
                ),
                mock.patch.object(
                    gate,
                    "EXPECTED_CHECKPOINT_SHA256",
                    gate._sha256(path),
                ),
            ):
                evidence = gate.checkpoint_preflight(path)
            self.assertTrue(evidence["ready"])
            self.assertTrue(evidence["checks"]["checkpoint_sha256_exact"])

    def test_checkpoint_preflight_fails_closed_for_missing_checkpoint(self):
        evidence = gate.checkpoint_preflight("missing-checkpoint.pth")
        self.assertFalse(evidence["ready"])
        self.assertFalse(evidence["checks"]["checkpoint_present"])

    def test_infer_facedepth_resizes_and_normalizes_prediction(self):
        import torch

        class DummyModel:
            def __call__(self, tensor):
                self.input_shape = tuple(tensor.shape)
                rows = torch.linspace(
                    0.0,
                    1.0,
                    12,
                    device=tensor.device,
                ).reshape(1, 1, 3, 4)
                return rows

        model = DummyModel()
        image = Image.fromarray(np.full((24, 32, 3), 127, dtype=np.uint8))
        depth = gate.infer_facedepth(
            model,
            image,
            device="cpu",
            color_order="rgb",
            input_mode="official-stretch",
        )
        self.assertEqual(model.input_shape, (1, 3, 480, 640))
        self.assertEqual(depth.shape, (24, 32))
        self.assertAlmostEqual(float(depth.min()), 0.0, places=5)
        self.assertAlmostEqual(float(depth.max()), 1.0, places=5)

    def test_infer_facedepth_preserves_native_aspect(self):
        import torch

        class DummyModel:
            def __call__(self, tensor):
                self.input_shape = tuple(tensor.shape)
                return tensor[:, :1]

        model = DummyModel()
        image = Image.fromarray(np.full((120, 80, 3), 127, dtype=np.uint8))
        with self.assertRaisesRegex(RuntimeError, "no usable depth span"):
            gate.infer_facedepth(
                model,
                image,
                device="cpu",
                color_order="rgb",
                input_mode="native-aspect",
            )
        self.assertEqual(model.input_shape, (1, 3, 480, 320))

    def test_embed_crop_changes_only_requested_rectangle(self):
        reference = np.zeros((8, 10), dtype=np.float32)
        local = np.ones((3, 4), dtype=np.float32)
        embedded = gate._embed_crop(local, reference, (2, 1, 8, 6))
        self.assertTrue(np.allclose(embedded[1:6, 2:8], 1.0))
        outside = embedded.copy()
        outside[1:6, 2:8] = 0.0
        self.assertTrue(np.allclose(outside, 0.0))

    def test_candidate_decision_compares_provider_to_current(self):
        quality = {
            "available": True,
            "combined_part_failures": 4,
            "shape_correlation": 0.90,
            "gradient_correlation": 0.70,
            "normalized_rmse": 0.10,
            "checks": {
                "coverage": True,
                "depth_semantics_orientation": True,
            },
        }
        better = {
            **quality,
            "combined_part_failures": 3,
            "shape_correlation": 0.91,
            "gradient_correlation": 0.71,
            "normalized_rmse": 0.09,
        }
        rows = [
            {
                "row_id": "row-1",
                "current_refined": quality,
                "facedepth_rgb": better,
            }
        ]
        current = gate.summarize_rows(rows, "current_refined")
        candidate = gate._candidate_decision(rows, "facedepth_rgb", current)
        self.assertEqual(
            candidate["decision"]["status"],
            "advance-current-path",
        )

    def test_raw_diagnostic_is_not_promotion_eligible(self):
        self.assertFalse(
            gate.promotion_eligible_candidate(
                "facedepth_native_aspect_bgr_raw"
            )
        )
        self.assertTrue(
            gate.promotion_eligible_candidate(
                "facedepth_native_aspect_bgr_fused"
            )
        )


if __name__ == "__main__":
    unittest.main()
