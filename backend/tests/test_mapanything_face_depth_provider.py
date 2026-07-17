import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import backend.benchmark.mapanything_face_depth_provider as provider_module
from backend.benchmark.mapanything_face_depth_provider import (
    DINOV2_LICENSE,
    DINOV2_MODEL_NAME,
    DINOV2_REPOSITORY,
    DINOV2_REVISION,
    DINOV2_TORCH_HUB_REPOSITORY,
    INFERENCE_SHAPE,
    METADATA_FILENAME,
    MODEL_ID,
    MODEL_CONFIG_SHA256,
    MODEL_LICENSE,
    MODEL_REVISION,
    MODEL_SAFETENSORS_SHA256,
    MODEL_SAFETENSORS_SIZE,
    OUTPUT_DEPTH_FILENAME,
    OUTPUT_ORIENTATION,
    REQUIRED_SOURCE_FILES,
    REQUIRED_DINOV2_SOURCE_FILES,
    SOURCE_LICENSE,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    UNICEPTION_VERSION,
    MapAnythingFaceDepthProvider,
    _offline_model_load_environment,
    _reject_unverified_namespace_modules,
    _verified_local_dinov2_hub,
    mapanything_preflight,
    parse_snapshot_config,
    require_runnable_preflight,
    resize_depth_to_original,
    save_depth_and_metadata,
    select_amp_dtype,
    validate_prediction,
    validate_snapshot_config,
)


def _pinned_config() -> dict:
    return {
        "name": "mapanything",
        "encoder_config": {
            "encoder_str": "dinov2",
            "data_norm_type": "dinov2",
            "size": "giant",
            "keep_first_n_layers": 24,
            "uses_torch_hub": True,
        },
        "geometric_input_config": {},
        "info_sharing_config": {
            "module_args": {"pretrained_checkpoint_path": None},
        },
        "pred_head_config": {
            "type": "dpt+pose",
            "adaptor_type": "raydirs+depth+pose+confidence+mask",
            "adaptor_config": {
                "scene_rep_type": "raydirs+depth+pose",
                "confidence_vmax": math.inf,
            },
        },
        "use_register_tokens_from_encoder": True,
        "load_specific_pretrained_submodules": False,
        "specific_pretrained_submodules": [],
        "torch_hub_force_reload": False,
        "pretrained_checkpoint_path": None,
    }


def _prediction(height: int = 3, width: int = 4) -> dict[str, np.ndarray]:
    depth = np.linspace(1.0, 2.0, height * width, dtype=np.float32).reshape(
        height,
        width,
    )
    points = np.zeros((height, width, 3), dtype=np.float32)
    points[..., 2] = depth
    yy, xx = np.mgrid[:height, :width]
    rays = np.stack(
        (
            (xx - (width - 1) / 2.0) / max(width, 1),
            (yy - (height - 1) / 2.0) / max(height, 1),
            np.ones((height, width), dtype=np.float64),
        ),
        axis=-1,
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    intrinsics = np.array(
        [
            [5.0, 0.0, (width - 1) / 2.0],
            [0.0, 5.0, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return {
        "depth_z": depth[None, ..., None],
        "pts3d_cam": points[None],
        "ray_directions": rays.astype(np.float32)[None],
        "intrinsics": intrinsics[None],
        "camera_poses": np.eye(4, dtype=np.float32)[None],
    }


class _FakeCuda:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def synchronize(self, device) -> None:
        self.calls.append(("synchronize", str(device)))

    def reset_peak_memory_stats(self, device) -> None:
        self.calls.append(("reset", str(device)))

    def max_memory_allocated(self, device) -> int:
        self.calls.append(("peak", str(device)))
        return 123_456_789


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


class _FakeModel:
    def __init__(self, prediction: dict[str, np.ndarray]) -> None:
        self.prediction = prediction
        self.calls: list[tuple[list[dict], dict]] = []

    def infer(self, views, **kwargs):
        self.calls.append((views, kwargs))
        return [self.prediction]


class MapAnythingFaceDepthProviderTests(unittest.TestCase):
    def test_public_provenance_pins_are_exact(self):
        self.assertEqual(
            SOURCE_REPOSITORY,
            "https://github.com/facebookresearch/map-anything.git",
        )
        self.assertEqual(
            SOURCE_REVISION,
            "c845b8f4f6cde0c20aecd87573656c3f69f5b2b0",
        )
        self.assertEqual(SOURCE_LICENSE, "Apache-2.0")
        self.assertEqual(MODEL_ID, "facebook/map-anything-apache")
        self.assertEqual(
            MODEL_REVISION,
            "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a",
        )
        self.assertEqual(MODEL_LICENSE, "Apache-2.0")
        self.assertEqual(
            MODEL_CONFIG_SHA256,
            "65701d09d99ed37a21d295f0d138978b3d584ab3bccdbcb4a2853da212b676c5",
        )
        self.assertEqual(
            MODEL_SAFETENSORS_SHA256,
            "fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201",
        )
        self.assertEqual(MODEL_SAFETENSORS_SIZE, 4_914_062_480)
        self.assertEqual(
            DINOV2_REPOSITORY,
            "https://github.com/facebookresearch/dinov2.git",
        )
        self.assertEqual(
            DINOV2_TORCH_HUB_REPOSITORY,
            "facebookresearch/dinov2",
        )
        self.assertEqual(
            DINOV2_REVISION,
            "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        )
        self.assertEqual(DINOV2_LICENSE, "Apache-2.0")
        self.assertEqual(DINOV2_MODEL_NAME, "dinov2_vitg14")
        self.assertEqual(UNICEPTION_VERSION, "0.1.7")
        self.assertNotIn("torch", provider_module.__dict__)
        self.assertNotIn("cv2", provider_module.__dict__)

    def test_snapshot_config_accepts_official_infinity_but_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(_pinned_config()),
                encoding="utf-8",
            )
            parsed = parse_snapshot_config(config_path)

        self.assertTrue(
            math.isinf(
                parsed["pred_head_config"]["adaptor_config"][
                    "confidence_vmax"
                ]
            )
        )
        validate_snapshot_config(parsed)
        parsed["name"] = "mapanything-unknown"
        with self.assertRaisesRegex(ValueError, "name is not pinned"):
            validate_snapshot_config(parsed)

    def test_preflight_requires_clean_source_and_exact_apache_snapshot(self):
        weights = b"pinned-test-weights"
        expected_hash = hashlib.sha256(weights).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            for relative in REQUIRED_SOURCE_FILES:
                path = source_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            snapshot_root = root / "snapshot"
            snapshot_root.mkdir()
            config_path = snapshot_root / "config.json"
            config_path.write_text(
                json.dumps(_pinned_config()),
                encoding="utf-8",
            )
            expected_config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            (snapshot_root / "model.safetensors").write_bytes(weights)
            dinov2_source_root = root / "dinov2"
            for relative in REQUIRED_DINOV2_SOURCE_FILES:
                path = dinov2_source_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")

            def clean_git(git_root, *args):
                if args != ("rev-parse", "HEAD"):
                    return ""
                return (
                    SOURCE_REVISION
                    if Path(git_root) == source_root
                    else DINOV2_REVISION
                )

            with (
                patch.object(
                    provider_module,
                    "MODEL_SAFETENSORS_SIZE",
                    len(weights),
                ),
                patch.object(
                    provider_module,
                    "MODEL_SAFETENSORS_SHA256",
                    expected_hash,
                ),
                patch.object(
                    provider_module,
                    "MODEL_CONFIG_SHA256",
                    expected_config_hash,
                ),
                patch.object(
                    provider_module,
                    "_git_output",
                    side_effect=clean_git,
                ),
                patch.object(
                    provider_module,
                    "_installed_distribution_version",
                    return_value=UNICEPTION_VERSION,
                ),
            ):
                evidence = mapanything_preflight(
                    source_root,
                    snapshot_root,
                    dinov2_source_root,
                )
                alias_evidence = mapanything_preflight(
                    source_root,
                    snapshot_root,
                    dinov2_source_root,
                    model_id="facebook/map-anything",
                )
                with patch.object(
                    provider_module,
                    "MODEL_CONFIG_SHA256",
                    "0" * 64,
                ):
                    config_drift_evidence = mapanything_preflight(
                        source_root,
                        snapshot_root,
                        dinov2_source_root,
                    )

            with (
                patch.object(
                    provider_module,
                    "MODEL_SAFETENSORS_SIZE",
                    len(weights),
                ),
                patch.object(
                    provider_module,
                    "MODEL_SAFETENSORS_SHA256",
                    expected_hash,
                ),
                patch.object(
                    provider_module,
                    "MODEL_CONFIG_SHA256",
                    expected_config_hash,
                ),
                patch.object(
                    provider_module,
                    "_git_output",
                    side_effect=(
                        SOURCE_REVISION,
                        "?? local.patch",
                        DINOV2_REVISION,
                        "",
                    ),
                ),
                patch.object(
                    provider_module,
                    "_installed_distribution_version",
                    return_value=UNICEPTION_VERSION,
                ),
            ):
                dirty_evidence = mapanything_preflight(
                    source_root,
                    snapshot_root,
                    dinov2_source_root,
                )

        self.assertTrue(evidence["runnable"])
        require_runnable_preflight(evidence)
        self.assertFalse(alias_evidence["checks"]["model_id_pinned"])
        self.assertFalse(alias_evidence["runnable"])
        self.assertFalse(config_drift_evidence["checks"]["config_hash_pinned"])
        self.assertFalse(config_drift_evidence["runnable"])
        with self.assertRaisesRegex(RuntimeError, "model_id_pinned"):
            require_runnable_preflight(alias_evidence)
        self.assertFalse(dirty_evidence["checks"]["source_clean"])
        self.assertFalse(dirty_evidence["runnable"])

    def test_preflight_rejects_config_hash_drift(self):
        config = _pinned_config()
        config["harmless_but_unpinned_field"] = "drift"
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            digest = hashlib.sha256(config_path.read_bytes()).hexdigest()

        self.assertNotEqual(digest, MODEL_CONFIG_SHA256)
        self.assertTrue(validate_snapshot_config(config))

    def test_torch_hub_is_routed_to_verified_local_dinov2_source(self):
        class FakeHub:
            def __init__(self):
                self.calls = []

            def load(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return "local-model"

        class FakeTorch:
            def __init__(self):
                self.hub = FakeHub()

        torch = FakeTorch()
        original_load = torch.hub.load
        source_root = Path("/verified/dinov2")
        with _verified_local_dinov2_hub(torch, source_root) as calls:
            result = torch.hub.load(
                DINOV2_TORCH_HUB_REPOSITORY,
                DINOV2_MODEL_NAME,
                force_reload=False,
                pretrained=False,
            )

        self.assertEqual(result, "local-model")
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            torch.hub.calls,
            [
                (
                    (str(source_root), DINOV2_MODEL_NAME),
                    {"pretrained": False, "source": "local"},
                )
            ],
        )
        self.assertEqual(torch.hub.load, original_load)

    def test_preloaded_foreign_dinov2_module_is_rejected(self):
        foreign = types.ModuleType("dinov2")
        foreign.__file__ = str(Path("/foreign/dinov2/__init__.py"))
        with patch.dict(sys.modules, {"dinov2": foreign}):
            with self.assertRaisesRegex(RuntimeError, "unverified dinov2"):
                _reject_unverified_namespace_modules(
                    "dinov2",
                    Path("/verified/dinov2"),
                )

    def test_offline_environment_is_scoped_and_restored(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "prior"}, clear=False):
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            with _offline_model_load_environment():
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "prior")
            self.assertNotIn("TRANSFORMERS_OFFLINE", os.environ)

    def test_prediction_normalizes_official_batch_one_arrays(self):
        normalized = validate_prediction(
            _prediction(),
            expected_shape=(3, 4),
        )

        self.assertEqual(normalized["depth_z"].shape, (3, 4))
        self.assertEqual(normalized["pts3d_cam"].shape, (3, 4, 3))
        self.assertEqual(normalized["ray_directions"].shape, (3, 4, 3))
        self.assertEqual(normalized["intrinsics"].shape, (3, 3))
        self.assertEqual(normalized["camera_poses"].shape, (4, 4))
        self.assertTrue(
            all(value.dtype == np.float32 for value in normalized.values())
        )

    def test_prediction_fails_closed_on_contract_violations(self):
        cases = []

        missing = _prediction()
        del missing["camera_poses"]
        cases.append((missing, "missing required keys"))

        nonfinite = _prediction()
        nonfinite["depth_z"][0, 0, 0, 0] = np.nan
        cases.append((nonfinite, "non-finite"))

        nonpositive = _prediction()
        nonpositive["depth_z"][0, 0, 0, 0] = 0.0
        cases.append((nonpositive, "strictly positive"))

        wrong_points = _prediction()
        wrong_points["pts3d_cam"][0, 0, 0, 2] += 0.1
        cases.append((wrong_points, "does not agree"))

        zero_rays = _prediction()
        zero_rays["ray_directions"][0, 0, 0] = 0.0
        cases.append((zero_rays, "finite and nonzero"))

        long_rays = _prediction()
        long_rays["ray_directions"] *= 2.0
        cases.append((long_rays, "approximately unit"))

        bad_focal = _prediction()
        bad_focal["intrinsics"][0, 0, 0] = -1.0
        cases.append((bad_focal, "focal lengths"))

        bad_principal = _prediction()
        bad_principal["intrinsics"][0, 0, 2] = 4.0
        cases.append((bad_principal, "principal point"))

        bad_pose = _prediction()
        bad_pose["camera_poses"][0, 3, 3] = 0.0
        cases.append((bad_pose, "homogeneous row"))

        bad_shape = _prediction()
        bad_shape["pts3d_cam"] = bad_shape["pts3d_cam"][:, :, :-1, :]
        cases.append((bad_shape, "unsupported shape"))

        for prediction, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    validate_prediction(prediction, expected_shape=(3, 4))

    def test_amp_selection_uses_bf16_only_for_blackwell(self):
        self.assertEqual(select_amp_dtype("NVIDIA B200", (10, 0)), "bf16")
        self.assertEqual(select_amp_dtype("NVIDIA RTX 5090", (12, 0)), "bf16")
        self.assertEqual(
            select_amp_dtype("NVIDIA Blackwell Engineering Sample", (9, 0)),
            "bf16",
        )
        self.assertEqual(select_amp_dtype("NVIDIA H100", (9, 0)), "fp16")

    def test_depth_is_only_resized_then_saved_with_compact_metadata(self):
        raw_depth = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        resized = resize_depth_to_original(raw_depth, (3, 5))
        self.assertEqual(resized.shape, (3, 5))
        self.assertGreaterEqual(float(resized.min()), 2.0)
        self.assertLessEqual(float(resized.max()), 8.0)

        with tempfile.TemporaryDirectory() as directory:
            payload = save_depth_and_metadata(
                resized,
                directory,
                {
                    "provider": "mapanything-apache",
                    "orientation": OUTPUT_ORIENTATION,
                    "original_shape": [3, 5],
                    "inference_shape": [2, 2],
                },
            )
            root = Path(directory)
            saved = np.load(root / OUTPUT_DEPTH_FILENAME, allow_pickle=False)
            metadata_text = (root / METADATA_FILENAME).read_text(encoding="utf-8")
            metadata = json.loads(metadata_text)
            expected_hash = hashlib.sha256(
                (root / OUTPUT_DEPTH_FILENAME).read_bytes()
            ).hexdigest()

        np.testing.assert_array_equal(saved, resized)
        self.assertEqual(payload, metadata)
        self.assertEqual(metadata["output_depth_data_sha256"], expected_hash)
        self.assertEqual(metadata["output_shape"], [3, 5])
        self.assertEqual(metadata["orientation"], OUTPUT_ORIENTATION)
        self.assertTrue(metadata_text.endswith("\n"))
        self.assertNotIn(": ", metadata_text)

    def test_runtime_uses_one_official_view_and_exact_inference_contract(self):
        instance = object.__new__(MapAnythingFaceDepthProvider)
        instance.device = "cuda:3"
        instance.device_name = "NVIDIA B200"
        instance.compute_capability = (10, 0)
        instance.amp_dtype = "bf16"
        instance.preflight = {"model": {"config": {"sha256": "config-sha"}}}
        instance.dinov2_hub_call = {
            "repository": DINOV2_TORCH_HUB_REPOSITORY,
            "model": DINOV2_MODEL_NAME,
            "source_revision": DINOV2_REVISION,
            "pretrained": False,
        }
        instance._torch = _FakeTorch()
        model_prediction = _prediction(518, 518)
        model_prediction["depth_z"].fill(1.0)
        model_prediction["pts3d_cam"][..., 2] = 1.0
        instance.model = _FakeModel(model_prediction)
        preprocess_calls = []

        def fake_preprocess(views, **kwargs):
            preprocess_calls.append((views, kwargs))
            return [{"img": np.empty((1, 3, 518, 518), dtype=np.float32)}]

        instance._preprocess_inputs = fake_preprocess
        result = instance.infer(np.zeros((7, 11, 3), dtype=np.uint8))

        self.assertEqual(len(preprocess_calls), 1)
        source_views, preprocess_kwargs = preprocess_calls[0]
        self.assertEqual(len(source_views), 1)
        self.assertEqual(source_views[0]["img"].shape, (7, 11, 3))
        self.assertEqual(
            preprocess_kwargs,
            {
                "resize_mode": "fixed_size",
                "size": (518, 518),
                "norm_type": "dinov2",
                "patch_size": 14,
                "resolution_set": 518,
                "verbose": False,
            },
        )
        self.assertEqual(len(instance.model.calls), 1)
        _, inference_kwargs = instance.model.calls[0]
        self.assertEqual(
            inference_kwargs,
            {
                "memory_efficient_inference": True,
                "minibatch_size": 1,
                "use_amp": True,
                "amp_dtype": "bf16",
                "apply_mask": False,
                "mask_edges": False,
            },
        )
        self.assertEqual(
            instance._torch.cuda.calls,
            [
                ("synchronize", "cuda:3"),
                ("reset", "cuda:3"),
                ("synchronize", "cuda:3"),
                ("peak", "cuda:3"),
            ],
        )
        self.assertEqual(result["depth"].shape, (7, 11))
        self.assertTrue(np.all(result["depth"] == 1.0))
        self.assertEqual(result["metadata"]["original_shape"], [7, 11])
        self.assertEqual(result["metadata"]["inference_shape"], [518, 518])
        self.assertEqual(result["metadata"]["orientation"], OUTPUT_ORIENTATION)
        self.assertEqual(result["metadata"]["peak_vram_bytes"], 123_456_789)


if __name__ == "__main__":
    unittest.main()
