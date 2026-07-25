import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from backend.benchmark import train_dinov2_face_correction_gate as gate
from backend.benchmark import train_dinov2_face_spatial_decoder as base


class TrainDinov2FaceCorrectionGateTests(unittest.TestCase):
    def test_unique_items_deduplicates_unhashable_rows(self):
        class Item:
            __hash__ = None

            def __init__(self, row_id):
                self.row = {"row_id": row_id}

        first = Item("one")
        second = Item("two")
        self.assertEqual(
            gate._unique_items([first, first, second]),
            [first, second],
        )

    def test_oracle_accepts_helpful_direction_and_rejects_harmful_direction(self):
        import torch

        tensors = {
            "baseline": torch.zeros((1, 1, 5, 5)),
            "target": torch.ones((1, 1, 5, 5)),
            "support_face": torch.ones((1, 1, 5, 5)),
            "correction_limit": torch.ones((1, 1, 1, 1)),
        }
        helpful = gate.oracle_correction_gate(
            torch.ones((1, 1, 5, 5)),
            tensors,
            window=1,
            margin_ratio=0.0,
        )
        harmful = gate.oracle_correction_gate(
            -torch.ones((1, 1, 5, 5)),
            tensors,
            window=1,
            margin_ratio=0.0,
        )
        self.assertGreater(float(helpful.min()), 0.9)
        self.assertEqual(int(torch.count_nonzero(harmful)), 0)

        tensors["support_face"][:, :, :, :2] = 0.0
        supported = gate.oracle_correction_gate(
            torch.ones((1, 1, 5, 5)),
            tensors,
            window=1,
            margin_ratio=0.0,
        )
        self.assertEqual(int(torch.count_nonzero(supported[:, :, :, :2])), 0)

    def test_oracle_rejects_invalid_policy(self):
        import torch

        values = torch.zeros((1, 1, 3, 3))
        tensors = {
            "baseline": values,
            "target": values,
            "support_face": torch.ones_like(values),
            "correction_limit": torch.ones((1, 1, 1, 1)),
        }
        with self.assertRaisesRegex(ValueError, "positive odd"):
            gate.oracle_correction_gate(values, tensors, window=2)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            gate.oracle_correction_gate(values, tensors, margin_ratio=-0.1)

    def test_selective_gate_has_exact_abstention_and_support(self):
        import torch

        probability = torch.full((1, 1, 7, 7), 0.4)
        probability[:, :, 2:5, 2:5] = 0.9
        support = torch.ones_like(probability)
        support[:, :, :, 0] = 0.0
        selected = gate.selective_gate(
            probability,
            support,
            threshold=0.7,
            window=1,
        )
        self.assertEqual(int(torch.count_nonzero(selected[:, :, :2])), 0)
        self.assertGreater(float(selected[:, :, 3, 3]), 0.0)
        self.assertEqual(int(torch.count_nonzero(selected[:, :, :, 0])), 0)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            gate.selective_gate(probability, support, threshold=1.0)

    def test_gate_uses_only_declared_production_inputs_and_limits_support(self):
        import torch

        self.assertNotIn("target", gate.PRODUCTION_INPUTS)
        self.assertNotIn("parts", gate.PRODUCTION_INPUTS)
        profile = base.resolve_encoder_profile("pyramid-448")
        model = gate.build_correction_gate().eval()
        features = torch.randn(
            (
                1,
                profile["feature_channels"],
                profile["feature_size"],
                profile["feature_size"],
            )
        )
        conditioning = torch.randn((1, base.CONDITIONING_CHANNELS, 16, 16))
        residual = torch.randn((1, 1, 16, 16))
        support = torch.ones_like(residual)
        support[:, :, :, :8] = 0.0
        with torch.inference_mode():
            logits, probability = model(
                features,
                conditioning,
                residual,
                support,
            )
        self.assertEqual(tuple(logits.shape), tuple(residual.shape))
        self.assertGreaterEqual(float(probability.min()), 0.0)
        self.assertLessEqual(float(probability.max()), 1.0)
        self.assertEqual(
            int(torch.count_nonzero(probability[:, :, :, :8])),
            0,
        )

    def test_base_checkpoint_binding_fails_closed(self):
        import torch

        profile = base.resolve_encoder_profile("pyramid-448")
        model = gate.build_correction_gate()
        payload = {
            "corpus_summary_sha256": "a" * 64,
            "cache_manifest_sha256": "b" * 64,
            "ordered_cache_row_content_sha256": "c" * 64,
            "model_file_sha256": base.MODEL_FILE_HASHES,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "base.pt"
            path.write_bytes(b"fixture")
            with (
                patch.object(base, "_file_sha256", return_value="d" * 64),
                patch.object(
                    base,
                    "load_decoder_checkpoint",
                    return_value=(model, payload, profile),
                ),
            ):
                _model, _payload, _profile, record = (
                    gate.validate_base_checkpoint(
                        path,
                        expected_sha256="d" * 64,
                        expected_corpus_summary_sha256="a" * 64,
                        expected_cache_manifest_sha256="b" * 64,
                        expected_cache_row_sha256="c" * 64,
                        device="cpu",
                    )
                )
                self.assertTrue(all(record["checks"].values()))
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    gate.validate_base_checkpoint(
                        path,
                        expected_sha256="e" * 64,
                        expected_corpus_summary_sha256="a" * 64,
                        expected_cache_manifest_sha256="b" * 64,
                        expected_cache_row_sha256="c" * 64,
                        device="cpu",
                    )

    def test_capacity_gate_requires_real_improvement(self):
        passed = gate._capacity_gate(
            {
                "initial_validation_loss": 1.0,
                "best_validation_loss": 0.95,
                "best_epoch": 2,
            },
            maximum_ratio=0.97,
        )
        self.assertTrue(passed["passed"])
        held = gate._capacity_gate(
            {
                "initial_validation_loss": 1.0,
                "best_validation_loss": 0.95,
                "best_epoch": 0,
            },
            maximum_ratio=0.97,
        )
        self.assertFalse(held["passed"])


if __name__ == "__main__":
    unittest.main()
