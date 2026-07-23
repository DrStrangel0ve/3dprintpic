import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from backend.benchmark.train_dinov2_face_spatial_decoder import (
    CONDITIONING_CHANNELS,
    FEATURE_CHANNELS,
    FEATURE_SIZE,
    MAXIMUM_NORMALIZED_RESIDUAL,
    _paired_regression_audit,
    _select_validation_candidate,
    _select_overfit_items,
    _tagged_part_non_regression,
    _tokens_to_grid,
    aligned_processor_pixels,
    build_spatial_decoder,
    conditioning_from_tensors,
    overfit_gate,
    train_decoder,
    validate_cache_binding,
    validate_pinned_model_root,
)


class _Item:
    def __init__(self, row_id, identity, height, split="train"):
        self.row = {
            "row_id": row_id,
            "identity_group": identity,
            "split": split,
            "render": {"face_bbox_height_pixels": height},
        }


def _paired_face_summary_fixture():
    names = (
        "left_eye",
        "left_eyebrow",
        "mouth",
        "nose",
        "right_eye",
        "right_eyebrow",
    )

    def shape_part(name):
        return {
            "name": name,
            "available": True,
            "shape_correlation": 0.5,
            "minimum_raw_gradient_correlation": 0.2,
            "face_normalized_shape_rmse": 0.3,
        }

    def affine_part(name):
        return {
            "name": name,
            "available": True,
            "rmse_mm": 1.0,
            "p95_absolute_error_mm": 2.0,
            "bias_mm": 0.1,
            "span_retention": 1.0,
        }

    row = {
        "row_id": "one",
        "shape_correlation": 0.8,
        "gradient_correlation": 0.7,
        "normalized_rmse": 0.2,
        "shape_failed_parts": ["nose"],
        "affine_failed_parts": ["nose"],
        "combined_part_failures": 2,
        "named_part_shape": {"parts": [shape_part(name) for name in names]},
        "named_part_affine_mm": {
            "parts": [affine_part(name) for name in names]
        },
    }
    baseline = {
        "row_count": 1,
        "combined_part_failures": 2,
        "median_shape_correlation": 0.8,
        "median_gradient_correlation": 0.7,
        "median_normalized_rmse": 0.2,
        "shape_check_failure_counts": {},
        "global_check_failure_counts": {},
        "global_failure_row_count": 0,
        "unavailable_row_count": 0,
        "rows": [row],
    }
    candidate = json.loads(json.dumps(baseline))
    candidate.update(
        {
            "alpha": 0.5,
            "combined_part_failures": 1,
            "median_shape_correlation": 0.81,
            "median_gradient_correlation": 0.71,
            "median_normalized_rmse": 0.19,
        }
    )
    return candidate, baseline


def _cache_binding_fixture(root: Path) -> dict:
    corpus = root / "corpus"
    cache = root / "cache"
    row_root = corpus / "rows" / "one"
    part_root = row_root / "exact_face_parts"
    part_root.mkdir(parents=True)
    (cache / "rows").mkdir(parents=True)

    def artifact(relative_path: str, payload: bytes) -> dict:
        path = corpus / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return {
            "path": relative_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    parts = {}
    for index, name in enumerate(
        (
            "left_eye",
            "left_eyebrow",
            "mouth",
            "nose",
            "right_eye",
            "right_eyebrow",
        )
    ):
        parts[name] = artifact(
            f"rows/one/exact_face_parts/{name}.png",
            f"part-{index}".encode("ascii"),
        )
    row = {
        "row_id": "one",
        "source": artifact("rows/one/source.png", b"source"),
        "exact_depth": artifact("rows/one/exact_depth.npy", b"depth"),
        "selection_mask": artifact("rows/one/selection_mask.png", b"mask"),
        "exact_face_parts": parts,
    }
    summary_path = corpus / "summary.json"
    summary_path.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()

    row_path = cache / "rows" / "one.npz"
    np.savez_compressed(row_path, target=np.ones((2, 2)))
    manifest_path = cache / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "method": "cc0_production_native_face_surface_residual",
                "corpus_summary_sha256": summary_hash,
                "row_count": 1,
                "row_ids": ["one"],
            }
        ),
        encoding="utf-8",
    )
    row_hash = hashlib.sha256(row_path.read_bytes()).hexdigest()
    content_hash = hashlib.sha256(
        f"one\0{row_hash}\0{row_path.stat().st_size}\n".encode("ascii")
    ).hexdigest()
    return {
        "corpus": corpus,
        "cache": cache,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "summary_hash": summary_hash,
        "row_path": row_path,
        "content_hash": content_hash,
        "depth_path": corpus / row["exact_depth"]["path"],
    }


class TrainDinov2FaceSpatialDecoderTests(unittest.TestCase):
    def test_pinned_model_root_requires_exact_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {"config.json": b"config", "model.safetensors": b"weights"}
            for name, payload in payloads.items():
                (root / name).write_bytes(payload)
            expected = {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in payloads.items()
            }
            result = validate_pinned_model_root(root, expected_hashes=expected)
            self.assertTrue(result["local_files_only"])
            (root / "config.json").write_bytes(b"substituted")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_pinned_model_root(root, expected_hashes=expected)

    def test_tokens_reassemble_to_exact_spatial_grid(self):
        import torch

        tokens = torch.arange(
            2 * 257 * FEATURE_CHANNELS, dtype=torch.float32
        ).reshape(2, 257, FEATURE_CHANNELS)
        grid = _tokens_to_grid(tokens)
        self.assertEqual(
            tuple(grid.shape),
            (2, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        )
        self.assertTrue(torch.equal(grid[:, :, 0, 0], tokens[:, 1]))
        with self.assertRaisesRegex(ValueError, "16x16"):
            _tokens_to_grid(tokens[:, :-1])

    def test_aligned_processor_preserves_full_crop_without_center_crop(self):
        import torch

        class Processor:
            def __call__(self, *, images, **kwargs):
                import torch

                self.images = images
                self.kwargs = kwargs
                arrays = [
                    np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
                    / 255.0
                    for image in images
                ]
                return {"pixel_values": torch.from_numpy(np.stack(arrays))}

        pixels = np.zeros((160, 160, 3), dtype=np.uint8)
        pixels[0, 0] = (255, 0, 0)
        pixels[0, -1] = (0, 255, 0)
        pixels[-1, 0] = (0, 0, 255)
        pixels[-1, -1] = (255, 255, 255)
        processor = Processor()
        result = aligned_processor_pixels(processor, [Image.fromarray(pixels)])
        self.assertEqual(tuple(result.shape), (1, 3, 224, 224))
        self.assertFalse(processor.kwargs["do_resize"])
        self.assertFalse(processor.kwargs["do_center_crop"])
        self.assertEqual(processor.images[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(processor.images[0].getpixel((223, 0)), (0, 255, 0))
        self.assertEqual(processor.images[0].getpixel((0, 223)), (0, 0, 255))
        self.assertEqual(
            processor.images[0].getpixel((223, 223)), (255, 255, 255)
        )
        self.assertEqual(processor.images[0].getpixel((112, 112)), (0, 0, 0))
        self.assertAlmostEqual(float(result[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(result[0, 1, 0, -1]), 1.0)
        self.assertAlmostEqual(float(result[0, 2, -1, 0]), 1.0)

    def test_pinned_bit_processor_preserves_asymmetric_boundaries(self):
        from transformers import BitImageProcessor

        processor = BitImageProcessor(
            crop_size={"height": 224, "width": 224},
            do_center_crop=True,
            do_convert_rgb=True,
            do_normalize=True,
            do_rescale=True,
            do_resize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            resample=3,
            rescale_factor=1.0 / 255.0,
            size={"shortest_edge": 256},
        )
        pixels = np.zeros((160, 160, 3), dtype=np.uint8)
        pixels[:24, :24] = (255, 0, 0)
        pixels[:24, -24:] = (0, 255, 0)
        pixels[-24:, :24] = (0, 0, 255)
        pixels[-24:, -24:] = (255, 255, 255)

        result = aligned_processor_pixels(processor, [Image.fromarray(pixels)])

        self.assertEqual(tuple(result.shape), (1, 3, 224, 224))
        self.assertEqual(int(result[0, :, 0, 0].argmax()), 0)
        self.assertEqual(int(result[0, :, 0, -1].argmax()), 1)
        self.assertEqual(int(result[0, :, -1, 0].argmax()), 2)
        self.assertGreater(float(result[0, :, -1, -1].min()), 2.2)
        expected_black = np.asarray(
            [
                -0.485 / 0.229,
                -0.456 / 0.224,
                -0.406 / 0.225,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            result[0, :, 112, 112].numpy(),
            expected_black,
            rtol=0.0,
            atol=1e-6,
        )

    def test_conditioning_includes_incumbent_baseline_and_fusion_weight(self):
        import torch

        inputs = torch.zeros((1, 7, 8, 8), dtype=torch.float32)
        inputs[:, 3] = 0.2
        inputs[:, -2] = -0.5
        inputs[:, -1] = 0.5
        tensors = {
            "inputs": inputs,
            "baseline": torch.full((1, 1, 8, 8), 0.7),
            "support_face": torch.ones((1, 1, 8, 8)),
            "fusion_weight": torch.full((1, 1, 8, 8), 0.4),
        }
        packed = conditioning_from_tensors(tensors)
        self.assertEqual(tuple(packed.shape), (1, CONDITIONING_CHANNELS, 8, 8))
        self.assertAlmostEqual(float(packed[0, 4, 0, 0]), 0.7)
        self.assertAlmostEqual(float(packed[0, 6, 0, 0]), 0.4)

    def test_decoder_is_zero_initialized_bounded_and_support_limited(self):
        import torch

        model = build_spatial_decoder().eval()
        features = torch.randn((2, FEATURE_CHANNELS, FEATURE_SIZE, FEATURE_SIZE))
        conditioning = torch.randn((2, CONDITIONING_CHANNELS, 160, 160))
        support = torch.ones((2, 1, 160, 160))
        support[:, :, :, :80] = 0.0
        with torch.inference_mode():
            initial = model(features, conditioning, support)
        self.assertTrue(torch.equal(initial, torch.zeros_like(initial)))
        with torch.no_grad():
            model.output.bias.fill_(20.0)
            bounded = model(features, conditioning, support)
        self.assertLessEqual(
            float(bounded.abs().max()), MAXIMUM_NORMALIZED_RESIDUAL
        )
        self.assertEqual(int(torch.count_nonzero(bounded[:, :, :, :80])), 0)

    def test_overfit_selection_prioritizes_small_faces_and_identities(self):
        items = [
            _Item("large", "id-large", 220),
            _Item("small-b", "id-b", 75),
            _Item("small-a", "id-a", 74),
            _Item("validation", "id-validation", 70, "validation"),
        ]
        selected = _select_overfit_items(items, 2)
        self.assertEqual(
            [item.row["row_id"] for item in selected],
            ["small-a", "small-b"],
        )
        duplicate = [_Item("a", "same", 74), _Item("b", "same", 75)]
        with self.assertRaisesRegex(ValueError, "distinct identities"):
            _select_overfit_items(duplicate, 2)

    def test_overfit_gate_fails_closed_without_a_trained_epoch(self):
        passed = overfit_gate(
            {
                "initial_validation_loss": 1.0,
                "best_validation_loss": 0.5,
                "best_epoch": 2,
            },
            maximum_ratio=0.9,
        )
        self.assertTrue(passed["passed"])
        held = overfit_gate(
            {
                "initial_validation_loss": 1.0,
                "best_validation_loss": 0.5,
                "best_epoch": 0,
            },
            maximum_ratio=0.9,
        )
        self.assertFalse(held["passed"])

    def test_overfit_gate_keeps_the_declared_ratio_strict(self):
        held = overfit_gate(
            {
                "initial_validation_loss": 1.0,
                "best_validation_loss": 0.971,
                "best_epoch": 4,
            },
            maximum_ratio=0.97,
        )
        self.assertFalse(held["passed"])

    def test_post_encoding_horizontal_flip_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positional self-attention"):
            train_decoder(
                [object()],
                [object()],
                {},
                device="cpu",
                epochs=1,
                horizontal_flip_augmentation=True,
            )

    def test_part_selector_rejects_numeric_regression_on_failed_part(self):
        candidate, baseline = _paired_face_summary_fixture()
        self.assertTrue(_tagged_part_non_regression(candidate, baseline))
        candidate["rows"][0]["named_part_shape"]["parts"][3][
            "minimum_raw_gradient_correlation"
        ] = 0.19
        self.assertFalse(_tagged_part_non_regression(candidate, baseline))
        selected, candidates = _select_validation_candidate(
            [candidate], baseline
        )
        self.assertEqual(selected["alpha"], 0.0)
        self.assertFalse(candidates[0]["eligible"])
        self.assertGreater(
            candidates[0]["paired_regression_audit"][
                "named_part_regression_count"
            ],
            0,
        )

    def test_selector_rejects_whole_face_row_regression(self):
        candidate, baseline = _paired_face_summary_fixture()
        candidate["rows"][0]["shape_correlation"] = 0.79
        audit = _paired_regression_audit(candidate, baseline)
        self.assertEqual(audit["whole_face_regression_count"], 1)
        selected, candidates = _select_validation_candidate(
            [candidate], baseline
        )
        self.assertEqual(selected["alpha"], 0.0)
        self.assertFalse(candidates[0]["whole_face_non_regression"])
        self.assertFalse(candidates[0]["eligible"])

    def test_cache_binding_rejects_modified_row_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _cache_binding_fixture(Path(temporary))
            binding = validate_cache_binding(
                fixture["corpus"],
                fixture["cache"],
                expected_corpus_summary_sha256=fixture["summary_hash"],
                expected_ordered_row_content_sha256=fixture["content_hash"],
            )
            self.assertEqual(binding["artifact_count"], 9)

            fixture["depth_path"].write_bytes(b"substituted-depth")
            with self.assertRaisesRegex(ValueError, "artifact SHA256 mismatch"):
                validate_cache_binding(
                    fixture["corpus"],
                    fixture["cache"],
                    expected_corpus_summary_sha256=fixture["summary_hash"],
                    expected_ordered_row_content_sha256=fixture["content_hash"],
                )
            fixture["depth_path"].write_bytes(b"depth")
            np.savez_compressed(fixture["row_path"], target=np.zeros((2, 2)))
            with self.assertRaisesRegex(ValueError, "row-content SHA256 mismatch"):
                validate_cache_binding(
                    fixture["corpus"],
                    fixture["cache"],
                    expected_corpus_summary_sha256=fixture["summary_hash"],
                    expected_ordered_row_content_sha256=fixture["content_hash"],
                )

    def test_cache_binding_rejects_summary_substitution_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _cache_binding_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "summary SHA256 mismatch"):
                validate_cache_binding(
                    fixture["corpus"],
                    fixture["cache"],
                    expected_corpus_summary_sha256="0" * 64,
                    expected_ordered_row_content_sha256=fixture["content_hash"],
                )

            summary = json.loads(fixture["summary_path"].read_text())
            summary["rows"][0]["source"]["path"] = "../outside.png"
            fixture["summary_path"].write_text(
                json.dumps(summary), encoding="utf-8"
            )
            replacement_hash = hashlib.sha256(
                fixture["summary_path"].read_bytes()
            ).hexdigest()
            manifest = json.loads(fixture["manifest_path"].read_text())
            manifest["corpus_summary_sha256"] = replacement_hash
            fixture["manifest_path"].write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "escapes the corpus root"):
                validate_cache_binding(
                    fixture["corpus"],
                    fixture["cache"],
                    expected_corpus_summary_sha256=replacement_hash,
                    expected_ordered_row_content_sha256=fixture["content_hash"],
                )


if __name__ == "__main__":
    unittest.main()
