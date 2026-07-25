import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backend.benchmark import train_coarse_to_fine_face_geometry as trainer
from backend.benchmark import train_dinov2_face_spatial_decoder as base


class TrainCoarseToFineFaceGeometryTests(unittest.TestCase):
    def test_surface_normal_exemplars_are_finite_unit_vectors(self):
        import torch

        depth = torch.linspace(0.0, 1.0, 16).reshape(1, 1, 4, 4)
        normals = trainer.surface_normal_exemplars(depth)
        self.assertEqual(tuple(normals.shape), (1, 3, 4, 4))
        self.assertTrue(torch.isfinite(normals).all())
        torch.testing.assert_close(
            torch.linalg.vector_norm(normals, dim=1),
            torch.ones((1, 4, 4)),
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            trainer.surface_normal_exemplars(torch.zeros((1, 2, 4, 4)))

    def test_decoder_starts_as_exact_abstention_and_limits_support(self):
        import torch

        profile = base.resolve_encoder_profile(trainer.ENCODER_PROFILE)
        model = trainer.build_geometry_decoder().eval()
        features = torch.randn(
            (
                1,
                profile["feature_channels"],
                profile["feature_size"],
                profile["feature_size"],
            )
        )
        conditioning = torch.randn((1, base.CONDITIONING_CHANNELS, 160, 160))
        support = torch.ones((1, 1, 160, 160))
        support[:, :, :, :80] = 0.0
        with torch.inference_mode():
            residual, coarse = model(features, conditioning, support)
        self.assertEqual(tuple(residual.shape), (1, 1, 160, 160))
        self.assertEqual(tuple(coarse.shape), (1, 1, 160, 160))
        self.assertEqual(int(torch.count_nonzero(residual)), 0)
        self.assertEqual(int(torch.count_nonzero(coarse)), 0)
        self.assertLessEqual(
            float(residual.abs().max()), trainer.TOTAL_RESIDUAL_LIMIT
        )

    def test_decoder_rejects_misaligned_inputs(self):
        import torch

        profile = base.resolve_encoder_profile(trainer.ENCODER_PROFILE)
        model = trainer.build_geometry_decoder()
        features = torch.zeros(
            (
                1,
                profile["feature_channels"],
                profile["feature_size"],
                profile["feature_size"],
            )
        )
        conditioning = torch.zeros(
            (1, base.CONDITIONING_CHANNELS, 159, 159)
        )
        support = torch.ones((1, 1, 159, 159))
        with self.assertRaisesRegex(ValueError, "160x160"):
            model(features, conditioning, support)

    def test_production_inputs_exclude_training_geometry(self):
        self.assertNotIn(
            "exact_camera_aligned_face_depth", trainer.PRODUCTION_INPUTS
        )
        self.assertNotIn("exact_six_part_face_masks", trainer.PRODUCTION_INPUTS)
        self.assertIn(
            "exact_camera_aligned_face_depth", trainer.TRAINING_ONLY_INPUTS
        )
        self.assertEqual(
            trainer.RESEARCH_SOURCES["coarse_to_fine_normal_refinement"][
                "license_status"
            ],
            "no repository license found; no code or weights imported",
        )

    def test_production_conditioning_ignores_exact_geometry_and_parts(self):
        import torch

        tensors = {
            "inputs": torch.randn((1, 7, 8, 8)),
            "baseline": torch.randn((1, 1, 8, 8)),
            "support_face": torch.ones((1, 1, 8, 8)),
            "fusion_weight": torch.ones((1, 1, 8, 8)),
            "target": torch.zeros((1, 1, 8, 8)),
            "part_masks": torch.zeros((1, 6, 8, 8)),
        }
        first = trainer.production_conditioning(tensors)
        tensors["target"].fill_(1000.0)
        tensors["part_masks"].fill_(1.0)
        second = trainer.production_conditioning(tensors)
        torch.testing.assert_close(first, second)

    def test_geometry_loss_penalizes_coarse_and_fine_frequency_leakage(self):
        import torch

        shape = (1, 1, 16, 16)
        residual = torch.zeros(shape)
        coarse = torch.zeros(shape)
        tensors = {
            "support_face": torch.ones(shape),
            "sample_weight": torch.ones((1, 1, 1, 1)),
            "target": torch.zeros(shape),
            "baseline": torch.zeros(shape),
        }
        with patch.object(
            base,
            "_loss",
            return_value=(torch.tensor(1.0), {"base": 1.0}),
        ):
            total, details = trainer.geometry_loss(
                residual, coarse, tensors
            )
            self.assertEqual(float(total), 1.0)
            residual[:, :, 4:12, 4:12] = 0.1
            total, details = trainer.geometry_loss(
                residual, coarse, tensors
            )
        self.assertGreater(float(total), 1.0)
        self.assertGreater(details["fine_low_frequency_leakage"], 0.0)

    def test_capacity_gate_requires_a_trained_epoch_and_loss_drop(self):
        passed = trainer.capacity_gate(
            {
                "initial_validation_loss": 1.0,
                "best_validation_loss": 0.95,
                "best_epoch": 2,
            },
            maximum_ratio=0.97,
        )
        self.assertTrue(passed["passed"])
        held = trainer.capacity_gate(
            {
                "initial_validation_loss": 1.0,
                "best_validation_loss": 0.98,
                "best_epoch": 2,
            },
            maximum_ratio=0.97,
        )
        self.assertFalse(held["passed"])

    def test_checkpoint_loader_fails_closed_on_schema_or_input_change(self):
        import torch

        model = trainer.build_geometry_decoder()
        payload = {
            "schema_version": trainer.CHECKPOINT_SCHEMA_VERSION,
            "method": trainer.METHOD,
            "model_id": base.MODEL_ID,
            "model_revision": base.MODEL_REVISION,
            "encoder_profile": base.resolve_encoder_profile(
                trainer.ENCODER_PROFILE
            ),
            "model_file_sha256": base.MODEL_FILE_HASHES,
            "production_inputs": list(trainer.PRODUCTION_INPUTS),
            "training_only_inputs": list(trainer.TRAINING_ONLY_INPUTS),
            "source_geometry_training_and_evaluation_only": True,
            "state_dict": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save(payload, path)
            loaded, loaded_payload = trainer.load_geometry_checkpoint(path)
            self.assertEqual(loaded_payload["method"], trainer.METHOD)
            self.assertFalse(loaded.training)
            payload["production_inputs"] = ["target"]
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "production_inputs"):
                trainer.load_geometry_checkpoint(path)


    def test_validation_is_invariant_to_requested_batch_size(self):
        import torch

        class Model:
            @staticmethod
            def eval():
                return None

            @staticmethod
            def __call__(_features, _conditioning, support):
                return support, support

        items = [
            SimpleNamespace(row={"row_id": str(index)})
            for index in range(3)
        ]
        tensors = {
            "support_face": torch.tensor([1.0, 2.0, 6.0]).reshape(
                3, 1, 1, 1
            ),
        }

        def fake_loss(_residual, _coarse, values, **_):
            value = float(values["support_face"].item())
            return torch.tensor(value), {"metric": value * 2.0}

        with (
            patch.object(trainer, "_prepare_tensors", return_value=(tensors, {})),
            patch.object(
                trainer.base,
                "_features_for",
                return_value=torch.zeros((1, 1, 1, 1)),
            ),
            patch.object(
                trainer,
                "production_conditioning",
                return_value=torch.zeros((1, 1, 1, 1)),
            ),
            patch.object(trainer, "geometry_loss", side_effect=fake_loss),
        ):
            one = trainer._validation_loss(
                Model(),
                items,
                {},
                device="cpu",
                batch_size=1,
                loss_options={},
                curriculum_options={},
            )
            many = trainer._validation_loss(
                Model(),
                items,
                {},
                device="cpu",
                batch_size=8,
                loss_options={},
                curriculum_options={},
            )
        self.assertEqual(one, many)
        self.assertAlmostEqual(one[0], 3.0)
        self.assertAlmostEqual(one[1]["metric"], 6.0)


if __name__ == "__main__":
    unittest.main()
