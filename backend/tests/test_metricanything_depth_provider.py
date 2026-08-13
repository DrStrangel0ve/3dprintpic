import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend import metricanything_depth_provider as provider


class MetricAnythingDepthProviderTests(unittest.TestCase):
    def tearDown(self):
        provider.release_metricanything_models()

    def test_official_source_and_checkpoint_are_immutably_pinned(self):
        self.assertEqual(
            provider.METRICANYTHING_SOURCE_COMMIT,
            "616a5e6762f5fc40d1a4ef990fee04c800532f59",
        )
        self.assertEqual(
            provider.METRICANYTHING_MODEL_REVISION,
            "8f1f08c53c683d4e19864601dfaf78515f29a63b",
        )
        self.assertEqual(len(provider.METRICANYTHING_CHECKPOINT_SHA256), 64)
        self.assertEqual(
            provider.METRICANYTHING_DEPTHMAP_MODEL_REVISION,
            "cae9b4eb052e827048c9b385366c6d0dce83fb01",
        )
        self.assertEqual(len(provider.METRICANYTHING_DEPTHMAP_CHECKPOINT_SHA256), 64)
        self.assertEqual(provider.METRICANYTHING_LICENSE, "Apache-2.0")

    def test_cpu_and_missing_cuda_fail_closed(self):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False)
        )
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            provider._resolve_cuda_device(fake_torch, "cpu")
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            provider._resolve_cuda_device(fake_torch, "cuda")

    def test_checkpoint_hash_is_content_based(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            checkpoint.write_bytes(b"metric-anything")
            first = provider._checkpoint_sha256(checkpoint)
            second = provider._checkpoint_sha256(checkpoint)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_inference_rejects_invalid_resolution_before_source_access(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 9"):
            provider.infer_metricanything_depth(
                Image.new("RGB", (32, 32)),
                resolution_level=10,
            )

    def test_inference_rejects_unpinned_checkpoint_before_model_import(self):
        with patch.object(
            provider,
            "_resolve_source_dir",
            return_value=Path("C:/metric-anything"),
        ), patch.object(
            provider,
            "_source_provenance",
            return_value={"revision_verified": True},
        ), patch.object(
            provider,
            "_resolve_checkpoint_path",
            return_value=Path("C:/metric-anything/model.pt"),
        ), patch.object(
            provider,
            "_checkpoint_sha256",
            return_value="0" * 64,
        ):
            with self.assertRaisesRegex(RuntimeError, "pinned public artifact"):
                provider.infer_metricanything_depth(Image.new("RGB", (32, 32)))


if __name__ == "__main__":
    unittest.main()
