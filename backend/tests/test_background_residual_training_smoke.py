import unittest

import numpy as np

from backend.benchmark import run_background_residual_training_smoke as residual_smoke


class BackgroundResidualTrainingSmokeTests(unittest.TestCase):
    def test_corpus_split_is_fixed_unique_and_separate_from_final_holdout(self):
        train = [spec for spec in residual_smoke.CORPUS_SPECS if spec.split == "train"]
        validation = [
            spec for spec in residual_smoke.CORPUS_SPECS if spec.split == "validation"
        ]

        self.assertEqual(len(train), 16)
        self.assertEqual(len(validation), 4)
        self.assertEqual(
            len({spec.scene_id for spec in residual_smoke.CORPUS_SPECS}),
            20,
        )
        self.assertEqual(
            tuple((spec.profile_name, spec.framing) for spec in residual_smoke.FINAL_HELD_OUT_SCENES),
            (
                ("caucasian_female_smile", "centered"),
                ("asian_female_asymmetric", "right_frame"),
            ),
        )

    def test_procedural_background_is_deterministic_and_seed_sensitive(self):
        spec = residual_smoke.CORPUS_SPECS[0]
        first_depth, first_rgb, first_metadata = residual_smoke._procedural_background(
            (64, 64), spec
        )
        second_depth, second_rgb, second_metadata = residual_smoke._procedural_background(
            (64, 64), spec
        )
        changed = residual_smoke.CorpusSceneSpec(
            **{**spec.__dict__, "geometry_seed": spec.geometry_seed + 1}
        )
        changed_depth, changed_rgb, _ = residual_smoke._procedural_background(
            (64, 64), changed
        )

        np.testing.assert_array_equal(first_depth, second_depth)
        np.testing.assert_array_equal(first_rgb, second_rgb)
        self.assertEqual(first_metadata, second_metadata)
        self.assertFalse(np.array_equal(first_depth, changed_depth))
        self.assertFalse(np.array_equal(first_rgb, changed_rgb))
        self.assertGreaterEqual(float(first_depth.min()), 0.56)
        self.assertLessEqual(float(first_depth.max()), 0.94)

    def test_training_geometry_pairs_change_appearance_without_changing_depth(self):
        for index in range(0, 16, 2):
            first = residual_smoke.CORPUS_SPECS[index]
            second = residual_smoke.CORPUS_SPECS[index + 1]
            first_depth, first_rgb, _ = residual_smoke._procedural_background(
                (64, 64), first
            )
            second_depth, second_rgb, _ = residual_smoke._procedural_background(
                (64, 64), second
            )

            self.assertEqual(first.geometry_seed, second.geometry_seed)
            self.assertNotEqual(first.appearance_seed, second.appearance_seed)
            np.testing.assert_array_equal(first_depth, second_depth)
            self.assertFalse(np.array_equal(first_rgb, second_rgb))

    def test_provider_target_uses_positive_face_fit_and_records_no_inference_oracle(self):
        rows, columns = np.indices((32, 32), dtype=np.float32)
        exact = 0.2 + 0.5 * rows / 31.0 + 0.1 * columns / 31.0
        exact_signal = 1.0 - exact
        provider = (exact_signal - 0.12) / 1.7
        face = np.zeros_like(exact, dtype=bool)
        face[4:28, 4:28] = True

        target, metadata = residual_smoke._provider_space_target(exact, provider, face)

        np.testing.assert_allclose(target, provider, rtol=1e-5, atol=1e-6)
        self.assertGreater(metadata["face_fit_scale"], 0.0)
        self.assertFalse(metadata["target_used_at_inference"])

    def test_correct_scene_restores_selected_provider_values_bit_exactly(self):
        import torch

        class DeliberatelyDestructive(torch.nn.Module):
            def forward(self, inputs):
                return torch.full_like(inputs[:, 3:4], 2.0)

        provider = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
        face = np.zeros_like(provider, dtype=bool)
        face[3:13, 4:12] = True
        scene = {
            "input": np.concatenate(
                (
                    np.zeros((3, 16, 16), dtype=np.float32),
                    provider[None],
                    face.astype(np.float32)[None],
                )
            ),
            "provider_depth": provider,
            "base_normalized": provider,
            "face_mask": face,
            "calibration": {
                "provider_face_p01": 0.0,
                "provider_face_span": 1.0,
            },
        }

        corrected = residual_smoke._correct_scene(
            DeliberatelyDestructive(), scene, "cpu"
        )

        self.assertEqual(corrected[face].tobytes(), provider[face].tobytes())
        self.assertTrue(np.all(corrected[~face] > provider[~face]))

    def test_model_zero_head_starts_as_zero_residual(self):
        import torch

        torch.manual_seed(4)
        model = residual_smoke._build_model().eval()
        inputs = torch.rand((2, 5, 64, 64), dtype=torch.float32)
        with torch.inference_mode():
            output = model(inputs)

        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0.0, atol=0.0)

    def test_inference_normalization_does_not_require_exact_depth(self):
        provider = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
        face = np.zeros_like(provider, dtype=bool)
        face[4:28, 4:28] = True
        scene = {
            "source_rgb": np.full((32, 32, 3), 0.5, dtype=np.float32),
            "face_mask": face,
        }

        normalized = residual_smoke._normalize_inference_scene(scene, provider)

        self.assertNotIn("target_normalized", normalized)
        self.assertFalse(normalized["calibration"]["oracle_used"])
        self.assertTrue(np.all(normalized["input"][:4, face] == 0.0))

    def test_validation_score_is_driven_by_weakest_scene(self):
        def record(correlation, gradient, rms=1.0):
            return {
                "metrics": {
                    "background": {
                        "correlation": correlation,
                        "gradient_correlation": gradient,
                        "rms_retention": rms,
                    }
                }
            }

        balanced = [record(0.80, 0.70), record(0.82, 0.71)]
        hidden_collapse = [record(0.99, 0.99), record(0.50, 0.50)]

        self.assertGreater(
            residual_smoke._validation_score(balanced),
            residual_smoke._validation_score(hidden_collapse),
        )

    def test_amplitude_failure_cannot_pass_depth_array_gate(self):
        rows, columns = np.indices((64, 64), dtype=np.float32)
        provider = 0.2 + 0.4 * rows / 63.0 + 0.2 * columns / 63.0
        face = np.zeros_like(provider, dtype=bool)
        face[16:48, 16:48] = True
        exact = 1.0 - provider
        exaggerated = provider.copy()
        background = ~face
        center = float(np.mean(provider[background]))
        exaggerated[background] = center + 10.0 * (provider[background] - center)
        scene = {
            "exact_depth": exact,
            "face_mask": face,
            "provider_depth": provider,
        }

        metrics = residual_smoke._evaluate_corrected(scene, exaggerated)

        self.assertFalse(metrics["background"]["passed"])
        self.assertFalse(metrics["checks"]["full_background_amplitude_and_structure"])
        self.assertFalse(metrics["checks"]["passed"])


if __name__ == "__main__":
    unittest.main()
