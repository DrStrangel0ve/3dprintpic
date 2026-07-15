import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.benchmark import run_makehuman_face_provider_relief_smoke as provider_smoke


class MakeHumanFaceProviderReliefSmokeTests(unittest.TestCase):
    def test_raw_ordering_diagnostics_detect_region_specific_contract_reversal(self):
        exact = np.arange(256, dtype=np.float32).reshape(16, 16)
        face = np.zeros_like(exact, dtype=bool)
        face[:, :8] = True
        candidate = np.empty_like(exact)
        candidate[face] = -exact[face]
        candidate[~face] = exact[~face]

        diagnostics = provider_smoke._raw_depth_ordering_diagnostics(
            exact,
            candidate,
            face,
            depth_semantics="relative-near-high",
        )

        self.assertTrue(
            diagnostics["regions"]["face"]["provider_contract_sign_consistent"]
        )
        self.assertFalse(
            diagnostics["regions"]["background"]["provider_contract_sign_consistent"]
        )
        self.assertFalse(diagnostics["provider_contract_consistent_across_regions"])
        self.assertFalse(diagnostics["oracle_used_for_stl"])

    def test_paired_metrics_fail_closed_on_shape_or_transform_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oracle_path = root / "oracle.npy"
            candidate_path = root / "candidate.npy"
            np.save(oracle_path, np.zeros((3, 3), dtype=np.float32))

            oracle = {
                "surface_path": oracle_path,
                "surface_grid_transform": {"origin": [0.0, 0.0]},
                "sample_pitch_mm": 1.0,
                "face_mask": np.ones((3, 3), dtype=bool),
                "part_masks": {},
            }
            cases = (
                ("shape", np.zeros((2, 3), dtype=np.float32), oracle["surface_grid_transform"]),
                ("transform", np.zeros((3, 3), dtype=np.float32), {"origin": [1.0, 0.0]}),
            )
            for mismatch, surface, transform in cases:
                with self.subTest(mismatch=mismatch):
                    np.save(candidate_path, surface)
                    candidate = {
                        **oracle,
                        "surface_path": candidate_path,
                        "surface_grid_transform": transform,
                    }

                    result = provider_smoke._paired_metrics(oracle, candidate)

                    self.assertFalse(result["available"])
                    self.assertFalse(result["checks"][f"{mismatch}_match"])
                    self.assertFalse(result["checks"]["passed"])
                    self.assertNotIn("appearance", result)

    def test_main_parses_provider_and_scene_tuples(self):
        argv = [
            "provider-smoke",
            "--output-dir",
            "unused",
            "--providers",
            " depth-anything-v2-large, da3mono-large, da3metric-large ",
            "--scene-profiles",
            " caucasian_female_smile, asian_female_asymmetric ",
        ]
        with patch.object(sys, "argv", argv), patch.object(provider_smoke, "run") as run:
            provider_smoke.main()

        self.assertEqual(
            run.call_args.kwargs["providers"],
            (
                provider_smoke.DA2_PROVIDER,
                provider_smoke.DA3_PROVIDER,
                provider_smoke.DA3_METRIC_PROVIDER,
            ),
        )
        self.assertEqual(
            tuple(scene.profile_name for scene in run.call_args.kwargs["scenes"]),
            ("caucasian_female_smile", "asian_female_asymmetric"),
        )

    def test_main_rejects_unknown_or_duplicate_scene_profiles_before_run(self):
        for profiles in ("missing-profile", "caucasian_female_smile,caucasian_female_smile"):
            with self.subTest(profiles=profiles), patch.object(
                sys,
                "argv",
                [
                    "provider-smoke",
                    "--output-dir",
                    "unused",
                    "--scene-profiles",
                    profiles,
                ],
            ), patch.object(provider_smoke, "run") as run:
                with self.assertRaisesRegex(ValueError, "Unknown or duplicate scene profiles"):
                    provider_smoke.main()
                run.assert_not_called()

    def test_run_rejects_duplicate_providers_before_inference(self):
        with self.assertRaisesRegex(ValueError, "Duplicate providers"):
            provider_smoke.run(
                "unused",
                providers=(provider_smoke.DA2_PROVIDER, provider_smoke.DA2_PROVIDER),
            )

    def test_run_requires_positive_background_context(self):
        with self.assertRaisesRegex(ValueError, "positive background depth ratio"):
            provider_smoke.run("unused", background_depth_ratio=0.0)

    def test_da3_confidence_metadata_is_optional_without_loading_real_model(self):
        class FakeModel:
            def __init__(self, confidence):
                self.confidence = confidence

            def to(self, _device):
                return self

            def eval(self):
                return None

            def inference(self, _paths, *, process_res):
                self.process_res = process_res
                return types.SimpleNamespace(
                    depth=[np.array([[1.0, 2.0]], dtype=np.float32)],
                    conf=self.confidence,
                )

        for confidence, expected in (
            (None, {"available": False}),
            ([np.array([[0.2, 0.8]], dtype=np.float32)], {"available": True, "min": 0.2, "median": 0.5, "max": 0.8}),
        ):
            with self.subTest(confidence=confidence is not None):
                model = FakeModel(confidence)
                torch = types.ModuleType("torch")
                torch.cuda = types.SimpleNamespace(is_available=lambda: False)
                api = types.ModuleType("depth_anything_3.api")
                api.__file__ = "fake-checkout/depth_anything_3/api.py"
                api.DepthAnything3 = types.SimpleNamespace(
                    from_pretrained=lambda _snapshot: model
                )
                package = types.ModuleType("depth_anything_3")
                package.__path__ = []
                package.__file__ = "fake-checkout/depth_anything_3/__init__.py"
                package.api = api
                hub = types.ModuleType("huggingface_hub")
                hub.snapshot_download = lambda *_args, **_kwargs: "fake-snapshot"
                modules = {
                    "torch": torch,
                    "depth_anything_3": package,
                    "depth_anything_3.api": api,
                    "huggingface_hub": hub,
                }
                provider_smoke._DA3_MODEL_CACHE.clear()

                with patch.dict(sys.modules, modules), patch.object(
                    provider_smoke,
                    "_validate_da3_source",
                    return_value={"revision": "test", "clean": True},
                ):
                    depth, metadata = provider_smoke._infer_da3mono(
                        Path("unused.png"),
                        device="cpu",
                    )

                np.testing.assert_array_equal(
                    depth,
                    np.array([[1.0, 2.0]], dtype=np.float32),
                )
                self.assertEqual(model.process_res, 504)
                self.assertEqual(metadata["confidence"]["available"], expected["available"])
                self.assertFalse(metadata["prediction_is_metric"])
                self.assertFalse(metadata["intrinsics"]["available"])
                self.assertFalse(metadata["sky"]["available"])
                for key in ("min", "median", "max"):
                    if key in expected:
                        self.assertAlmostEqual(metadata["confidence"][key], expected[key])
                    else:
                        self.assertNotIn(key, metadata["confidence"])

    def test_da3_metric_uses_distinct_pin_and_records_real_standalone_contract(self):
        depth = np.array([[2.0, 4.0]], dtype=np.float32)

        class FakeModel:
            def __init__(self, snapshot):
                self.snapshot = snapshot

            def to(self, _device):
                return self

            def eval(self):
                return None

            def inference(self, _paths, *, process_res):
                self.process_res = process_res
                return types.SimpleNamespace(
                    depth=[depth],
                    conf=None,
                    intrinsics=None,
                    sky=[np.array([[False, True]])],
                    is_metric=0,
                )

        loaded = []
        downloads = []

        def load_model(snapshot):
            loaded.append(snapshot)
            return FakeModel(snapshot)

        def download(model_id, **kwargs):
            downloads.append((model_id, kwargs))
            return model_id

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        api = types.ModuleType("depth_anything_3.api")
        api.__file__ = "fake-checkout/depth_anything_3/api.py"
        api.DepthAnything3 = types.SimpleNamespace(from_pretrained=load_model)
        package = types.ModuleType("depth_anything_3")
        package.__path__ = []
        package.api = api
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = download
        provider_smoke._DA3_MODEL_CACHE.clear()

        with patch.dict(
            sys.modules,
            {
                "torch": torch,
                "depth_anything_3": package,
                "depth_anything_3.api": api,
                "huggingface_hub": hub,
            },
        ), patch.object(
            provider_smoke,
            "_validate_da3_source",
            return_value={"revision": "test", "clean": True},
        ):
            provider_smoke._infer_da3(
                Path("unused.png"),
                provider=provider_smoke.DA3_MONO_PROVIDER,
                device="cpu",
            )
            measured, metadata = provider_smoke._infer_da3(
                Path("unused.png"),
                provider=provider_smoke.DA3_METRIC_PROVIDER,
                device="cpu",
            )

        np.testing.assert_array_equal(measured, depth)
        self.assertEqual(
            downloads,
            [
                (
                    provider_smoke.DA3_MODEL_SPECS[provider_smoke.DA3_MONO_PROVIDER][
                        "model_id"
                    ],
                    {
                        "revision": provider_smoke.DA3_MODEL_SPECS[
                            provider_smoke.DA3_MONO_PROVIDER
                        ]["model_revision"],
                        "allow_patterns": ("config.json", "model.safetensors"),
                    },
                ),
                (
                    provider_smoke.DA3_MODEL_SPECS[provider_smoke.DA3_METRIC_PROVIDER][
                        "model_id"
                    ],
                    {
                        "revision": "4010e39f3634a45bc60553321fb49fb760bd594e",
                        "allow_patterns": ("config.json", "model.safetensors"),
                    },
                ),
            ],
        )
        self.assertEqual(loaded, [downloads[0][0], downloads[1][0]])
        self.assertFalse(metadata["prediction_is_metric"])
        self.assertEqual(metadata["intrinsics"], {"available": False, "focal_px": None})
        self.assertEqual(metadata["sky"], {"available": True, "fraction": 0.5})
        self.assertIn("cancels", metadata["metric_scaling"])


if __name__ == "__main__":
    unittest.main()
