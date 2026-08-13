import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend import infinidepth_provider as provider


class InfiniDepthProviderTests(unittest.TestCase):
    def tearDown(self):
        provider.release_infinidepth_models()

    def test_official_source_and_checkpoint_are_immutably_pinned(self):
        self.assertEqual(
            provider.INFINIDEPTH_SOURCE_COMMIT,
            "36c6e0c31887fafc210184ee43ca475230704095",
        )
        self.assertEqual(
            provider.INFINIDEPTH_MODEL_REVISION,
            "b387ad877e922468fcd85190f24e5b78b28dcd66",
        )
        self.assertEqual(len(provider.INFINIDEPTH_CHECKPOINT_SHA256), 64)
        self.assertEqual(provider.INFINIDEPTH_LICENSE, "Apache-2.0")

    def test_scaled_shape_preserves_aspect_ratio_on_model_patch_grid(self):
        self.assertEqual(provider._scaled_shape(1920, 1080, 512), (512, 288))
        self.assertEqual(provider._scaled_shape(600, 900, 512), (336, 512))
        with self.assertRaisesRegex(ValueError, "at least 256"):
            provider._scaled_shape(640, 480, 128)

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
            checkpoint = Path(temporary) / "model.ckpt"
            checkpoint.write_bytes(b"pinned-checkpoint")
            first = provider._checkpoint_sha256(checkpoint)
            second = provider._checkpoint_sha256(checkpoint)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_inference_rejects_unpinned_checkpoint_before_model_import(self):
        with patch.object(
            provider,
            "_resolve_source_dir",
            return_value=Path("C:/infinidepth"),
        ), patch.object(
            provider,
            "_source_provenance",
            return_value={"revision_verified": True},
        ), patch.object(
            provider,
            "_resolve_checkpoint_path",
            return_value=Path("C:/infinidepth/model.ckpt"),
        ), patch.object(
            provider,
            "_checkpoint_sha256",
            return_value="0" * 64,
        ):
            with self.assertRaisesRegex(RuntimeError, "pinned public artifact"):
                provider.infer_infinidepth_disparity(Image.new("RGB", (32, 32)))


if __name__ == "__main__":
    unittest.main()
