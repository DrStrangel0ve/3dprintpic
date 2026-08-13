import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from backend import da3_depth_provider as provider


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def reset_peak_memory_stats(_device):
        return None

    @staticmethod
    def max_memory_allocated(_device):
        return 2 * 1024**3


class _FakeTorch(types.SimpleNamespace):
    cuda = _FakeCuda()

    @staticmethod
    def device(value):
        return value


class _FakeModel:
    def __init__(self):
        self.loaded_from = None
        self.device = None
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def inference(self, images, **kwargs):
        self.calls.append((images, kwargs))
        return types.SimpleNamespace(
            depth=np.ones((1, 8, 12), dtype=np.float32) * 3.0,
            conf=np.ones((1, 8, 12), dtype=np.float32) * 0.75,
            intrinsics=np.asarray(
                [[[500.0, 0.0, 6.0], [0.0, 510.0, 4.0], [0.0, 0.0, 1.0]]],
                dtype=np.float32,
            ),
        )


class DepthAnythingV3ProviderTests(unittest.TestCase):
    def setUp(self):
        provider.release_da3_models()

    def tearDown(self):
        provider.release_da3_models()

    def test_model_contract_is_pinned(self):
        self.assertTrue(provider.is_da3_model("depth-anything/DA3-LARGE-1.1"))
        self.assertTrue(provider.is_da3_model("depth-anything/DA3MONO-LARGE"))
        self.assertFalse(provider.is_da3_model("depth-anything/DA3-LARGE"))
        self.assertEqual(
            provider.DA3_LARGE_MODEL_REVISION,
            "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c",
        )
        self.assertEqual(
            provider.DA3_MODEL_SPECS[provider.DA3_LARGE_MODEL_ID]["license"],
            "CC BY-NC 4.0",
        )
        self.assertEqual(
            provider.DA3_MODEL_SPECS[provider.DA3_MONO_MODEL_ID]["license"],
            "Apache-2.0",
        )

    def test_cuda_request_fails_closed_without_cuda(self):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False)
        )
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            provider._resolved_device(fake_torch, "cuda")

    def test_inference_uses_pinned_snapshot_and_reports_far_high_depth(self):
        model = _FakeModel()
        snapshot_calls = []

        class FakeDepthAnything3:
            @staticmethod
            def from_pretrained(snapshot):
                model.loaded_from = snapshot
                return model

        def snapshot_download(model_id, **kwargs):
            snapshot_calls.append((model_id, kwargs))
            return "C:/models/da3-large-1.1"

        fake_api = types.SimpleNamespace(__file__="C:/wheel/depth_anything_3/api.py")
        fake_da3_package = types.ModuleType("depth_anything_3")
        fake_da3_package.__path__ = []
        fake_da3_api = types.ModuleType("depth_anything_3.api")
        fake_da3_api.__file__ = fake_api.__file__
        fake_da3_api.DepthAnything3 = FakeDepthAnything3
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.snapshot_download = snapshot_download

        modules = {
            "torch": _FakeTorch(),
            "depth_anything_3": fake_da3_package,
            "depth_anything_3.api": fake_da3_api,
            "huggingface_hub": fake_hub,
        }
        with patch.dict(sys.modules, modules), patch.object(
            provider,
            "_source_provenance",
            return_value={"revision_verified": False},
        ):
            depth, metadata = provider.infer_da3_depth(
                Image.new("RGB", (24, 16), "white"),
                device="cuda",
            )

        self.assertEqual(depth.shape, (8, 12))
        self.assertEqual(snapshot_calls[0][0], provider.DA3_LARGE_MODEL_ID)
        self.assertEqual(
            snapshot_calls[0][1]["revision"],
            provider.DA3_LARGE_MODEL_REVISION,
        )
        self.assertEqual(
            snapshot_calls[0][1]["allow_patterns"],
            ("config.json", "model.safetensors"),
        )
        self.assertEqual(model.device, "cuda")
        self.assertEqual(model.calls[0][1]["process_res"], 504)
        self.assertEqual(metadata["depth_value_semantics"], "relative_distance_far_high")
        self.assertEqual(metadata["license"], "CC BY-NC 4.0")
        self.assertEqual(metadata["peak_vram_gb"], 2.0)
        self.assertEqual(metadata["intrinsics"]["focal_px"], 505.0)


if __name__ == "__main__":
    unittest.main()
