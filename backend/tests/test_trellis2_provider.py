from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from backend.benchmark import preflight_image_to_mesh_providers as preflight_module
from backend.benchmark import run_image_to_mesh_provider as provider_module
from backend.benchmark import trellis2_models
from backend.benchmark.trellis2_models import (
    DEFAULT_TRELLIS2_MODEL,
    DEFAULT_TRELLIS2_MODEL_REVISION,
    DEFAULT_TRELLIS2_RESOLUTION,
    DEFAULT_TRELLIS2_SOURCE_REVISION,
)


class Trellis2ProviderTest(unittest.TestCase):
    def test_exact_pins_and_provider_registration(self):
        self.assertEqual(
            DEFAULT_TRELLIS2_SOURCE_REVISION,
            "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
        )
        self.assertEqual(DEFAULT_TRELLIS2_MODEL, "microsoft/TRELLIS.2-4B")
        self.assertEqual(
            DEFAULT_TRELLIS2_MODEL_REVISION,
            "af44b45f2e35a493886929c6d786e563ec68364d",
        )
        self.assertEqual(DEFAULT_TRELLIS2_RESOLUTION, 512)
        self.assertIn(provider_module.TRELLIS2_PROVIDER, provider_module.PROVIDERS)
        self.assertEqual(
            provider_module.CLI_PROVIDERS[provider_module.TRELLIS2_PROVIDER]["env"],
            "TRELLIS2_DIR",
        )
        self.assertEqual(
            provider_module.CLI_PROVIDERS[provider_module.TRELLIS2_PROVIDER][
                "default_dirs"
            ],
            ("/content/TRELLIS.2",),
        )

    def test_snapshot_is_resolved_at_exact_revision_then_forced_offline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "snapshot"
            snapshot.mkdir()
            (snapshot / "pipeline.json").write_text("{}", encoding="utf-8")

            with patch(
                "huggingface_hub.snapshot_download",
                return_value=str(snapshot),
            ) as download, patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"},
                clear=False,
            ):
                resolved = trellis2_models.resolve_trellis2_model_snapshot()
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

            self.assertEqual(resolved, snapshot.resolve())
            download.assert_called_once_with(
                repo_id=DEFAULT_TRELLIS2_MODEL,
                revision=DEFAULT_TRELLIS2_MODEL_REVISION,
            )

    def test_wrapper_uses_official_512_api_and_exports_geometry_only(self):
        observed: dict[str, object] = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, path):
                observed["snapshot"] = path
                return cls()

            def cuda(self):
                observed["cuda"] = True

            def run(self, image, **kwargs):
                observed["image"] = image
                observed["run_kwargs"] = kwargs
                return [
                    SimpleNamespace(
                        vertices=np.array(
                            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                            dtype=np.float32,
                        ),
                        faces=np.array([[0, 1, 2]], dtype=np.int32),
                        attrs="must-not-be-exported",
                        coords="must-not-be-exported",
                    )
                ]

        class FakeGeometry:
            def __init__(self, vertices, faces):
                self.vertices = vertices
                self.faces = faces

            def export(self, path):
                Path(path).write_bytes(b"geometry-only")

        def fake_trimesh(**kwargs):
            observed["trimesh_kwargs"] = kwargs
            return FakeGeometry(kwargs["vertices"], kwargs["faces"])

        fake_trellis2 = types.ModuleType("trellis2")
        fake_trellis2.__path__ = []
        fake_pipelines = types.ModuleType("trellis2.pipelines")
        fake_pipelines.Trellis2ImageTo3DPipeline = FakePipeline
        fake_trimesh_module = types.ModuleType("trimesh")
        fake_trimesh_module.Trimesh = fake_trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "TRELLIS.2"
            snapshot = root / "snapshot"
            provider_dir.mkdir()
            snapshot.mkdir()
            input_image = root / "input.png"
            output_mesh = root / "raw.glb"
            Image.new("RGB", (8, 8), (40, 90, 140)).save(input_image)

            with patch.object(
                trellis2_models,
                "resolve_trellis2_model_snapshot",
                return_value=snapshot,
            ), patch.dict(
                sys.modules,
                {
                    "trellis2": fake_trellis2,
                    "trellis2.pipelines": fake_pipelines,
                    "trimesh": fake_trimesh_module,
                },
            ):
                result = trellis2_models.run_trellis2(
                    provider_dir=provider_dir,
                    input_image=input_image,
                    output_mesh=output_mesh,
                    resolution=512,
                    seed=17,
                )
                output_exists = output_mesh.is_file()

        self.assertEqual(result, output_mesh)
        self.assertEqual(observed["snapshot"], str(snapshot))
        self.assertTrue(observed["cuda"])
        self.assertEqual(
            observed["run_kwargs"],
            {"num_samples": 1, "seed": 17, "pipeline_type": "512"},
        )
        self.assertEqual(
            set(observed["trimesh_kwargs"]),
            {"vertices", "faces", "process"},
        )
        self.assertFalse(observed["trimesh_kwargs"]["process"])
        self.assertTrue(output_exists)

    def test_preflight_requires_exact_model_source_and_official_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            provider_dir = Path(temp_dir) / "TRELLIS.2"
            entrypoint = (
                provider_dir / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
            )
            entrypoint.parent.mkdir(parents=True)
            entrypoint.write_text("# pinned source fixture\n", encoding="utf-8")
            command = (
                "python -m backend.benchmark.run_image_to_mesh_provider "
                f"--provider trellis2 --provider-dir {provider_dir.as_posix()} "
                "--provider-python python "
                f"--trellis2-model-path {DEFAULT_TRELLIS2_MODEL} "
                f"--trellis2-model-revision {DEFAULT_TRELLIS2_MODEL_REVISION} "
                "--trellis2-resolution 512 --seed 9"
            )
            parsed = preflight_module.parse_provider_command(command)
            with patch.object(
                preflight_module,
                "provider_git_revision",
                return_value=DEFAULT_TRELLIS2_SOURCE_REVISION,
            ):
                row = preflight_module.provider_preflight_row(parsed)

            self.assertTrue(row["runnable"])
            self.assertEqual(
                row["provider_models"]["trellis2"]["revision"],
                DEFAULT_TRELLIS2_MODEL_REVISION,
            )
            self.assertEqual(
                row["checks"]["entrypoint"],
                "trellis2/pipelines/trellis2_image_to_3d.py",
            )
            self.assertTrue(row["checks"]["provider_source_revision_pinned"])
            self.assertTrue(row["checks"]["trellis2_resolution_supported"])

            changed_model = dict(parsed)
            changed_model["provider_models"] = {
                "trellis2": {
                    "repo_id": DEFAULT_TRELLIS2_MODEL,
                    "revision": "f" * 40,
                }
            }
            with patch.object(
                preflight_module,
                "provider_git_revision",
                return_value=DEFAULT_TRELLIS2_SOURCE_REVISION,
            ):
                changed_model_row = preflight_module.provider_preflight_row(changed_model)
            self.assertFalse(changed_model_row["runnable"])
            self.assertFalse(changed_model_row["checks"]["model_revision_pinned"])

            with patch.object(
                preflight_module,
                "provider_git_revision",
                return_value="e" * 40,
            ):
                changed_source_row = preflight_module.provider_preflight_row(parsed)
            self.assertFalse(changed_source_row["runnable"])
            self.assertFalse(
                changed_source_row["checks"]["provider_source_revision_pinned"]
            )

    def test_model_seed_source_and_wrapper_are_cache_identity_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "TRELLIS.2"
            run_entry = (
                provider_dir / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
            )
            run_entry.parent.mkdir(parents=True)
            run_entry.write_text("# source v1\n", encoding="utf-8")
            input_image = root / "input.png"
            input_image.write_bytes(b"input-v1")
            args = SimpleNamespace(
                provider=provider_module.TRELLIS2_PROVIDER,
                python="trellis2-python",
                input_image=input_image,
                provider_arg=[],
                prefetch_only=False,
                trellis2_model_path=DEFAULT_TRELLIS2_MODEL,
                trellis2_model_revision=DEFAULT_TRELLIS2_MODEL_REVISION,
                trellis2_resolution=512,
                seed=23,
            )

            with patch.object(
                provider_module,
                "provider_git_revision",
                return_value=DEFAULT_TRELLIS2_SOURCE_REVISION,
            ):
                baseline = provider_module.cli_provider_cache_payload(
                    args,
                    provider_dir,
                    run_entry,
                )
            baseline_key = provider_module.cli_provider_cache_key(baseline)

            changed_model_args = SimpleNamespace(**vars(args))
            changed_model_args.trellis2_model_revision = "f" * 40
            with patch.object(
                provider_module,
                "provider_git_revision",
                return_value=DEFAULT_TRELLIS2_SOURCE_REVISION,
            ):
                changed_model = provider_module.cli_provider_cache_payload(
                    changed_model_args,
                    provider_dir,
                    run_entry,
                )

            changed_seed_args = SimpleNamespace(**vars(args))
            changed_seed_args.seed = 24
            with patch.object(
                provider_module,
                "provider_git_revision",
                return_value=DEFAULT_TRELLIS2_SOURCE_REVISION,
            ):
                changed_seed = provider_module.cli_provider_cache_payload(
                    changed_seed_args,
                    provider_dir,
                    run_entry,
                )

            with patch.object(
                provider_module,
                "provider_git_revision",
                return_value="e" * 40,
            ):
                changed_source = provider_module.cli_provider_cache_payload(
                    args,
                    provider_dir,
                    run_entry,
                )

            self.assertNotEqual(
                provider_module.cli_provider_cache_key(changed_model), baseline_key
            )
            self.assertNotEqual(
                provider_module.cli_provider_cache_key(changed_seed), baseline_key
            )
            self.assertNotEqual(
                provider_module.cli_provider_cache_key(changed_source), baseline_key
            )
            self.assertEqual(
                baseline["expected_provider_revision"],
                DEFAULT_TRELLIS2_SOURCE_REVISION,
            )
            self.assertIn(
                "backend/benchmark/trellis2_models.py",
                baseline["provider_source_sha256"],
            )
            self.assertIn("--resolution", baseline["command"])
            self.assertIn("512", baseline["command"])

    def test_raw_and_repaired_outputs_share_one_cached_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "TRELLIS.2"
            run_entry = (
                provider_dir / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
            )
            run_entry.parent.mkdir(parents=True)
            run_entry.write_text("# pinned source fixture\n", encoding="utf-8")
            input_image = root / "input.png"
            cache_dir = root / "provider-cache"
            Image.new("RGB", (12, 12), (80, 140, 110)).save(input_image)
            invocations: list[list[str]] = []

            def fake_subprocess_run(command, cwd, check, timeout):
                invocations.append(command)
                output_path = Path(command[command.index("--output-mesh") + 1])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"trellis2-raw-geometry")
                return subprocess.CompletedProcess(command, 0)

            def fake_repair(source, target, mode, **kwargs):
                target = Path(target)
                target.write_bytes(Path(source).read_bytes())
                return target

            first_argv = [
                "run_image_to_mesh_provider",
                "--provider",
                "trellis2",
                "--provider-dir",
                str(provider_dir),
                "--provider-python",
                "trellis2-python",
                "--input-image",
                str(input_image),
                "--output-mesh",
                str(root / "raw.glb"),
                "--provider-mesh-cache-dir",
                str(cache_dir),
                "--trellis2-resolution",
                "512",
                "--seed",
                "31",
            ]
            second_argv = [
                "run_image_to_mesh_provider",
                "--provider",
                "trellis2",
                "--provider-dir",
                str(provider_dir),
                "--provider-python",
                "trellis2-python",
                "--input-image",
                str(input_image),
                "--output-mesh",
                str(root / "repaired.glb"),
                "--raw-output-mesh",
                str(root / "repaired-source.glb"),
                "--provider-mesh-cache-dir",
                str(cache_dir),
                "--trellis2-resolution",
                "512",
                "--seed",
                "31",
                "--mesh-repair",
                "printable",
            ]

            first_stderr = io.StringIO()
            second_stderr = io.StringIO()
            with patch.object(
                provider_module,
                "provider_git_revision",
                return_value=DEFAULT_TRELLIS2_SOURCE_REVISION,
            ), patch.object(
                provider_module.subprocess,
                "run",
                side_effect=fake_subprocess_run,
            ), patch.object(
                provider_module,
                "repair_mesh_for_printable_stl",
                side_effect=fake_repair,
            ):
                with patch.object(sys, "argv", first_argv), contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(first_stderr):
                    provider_module.main()
                with patch.object(sys, "argv", second_argv), contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(second_stderr):
                    provider_module.main()

            self.assertEqual(len(invocations), 1)
            self.assertIn('"status": "stored"', first_stderr.getvalue())
            self.assertIn('"status": "hit"', second_stderr.getvalue())
            self.assertEqual(
                (root / "raw.glb").read_bytes(),
                (root / "repaired-source.glb").read_bytes(),
            )
            self.assertTrue((root / "repaired.glb").is_file())
            self.assertEqual(len(list(cache_dir.glob("*.glb"))), 1)
            metadata = json.loads(
                next(cache_dir.glob("*.json")).read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["provider_models"],
                {
                    "trellis2": {
                        "repo_id": DEFAULT_TRELLIS2_MODEL,
                        "revision": DEFAULT_TRELLIS2_MODEL_REVISION,
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
