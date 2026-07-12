from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.benchmark import preflight_image_to_mesh_providers as preflight_module
from backend.benchmark import run_image_to_mesh_provider as provider_module
from backend.benchmark import step1x3d_models
from backend.benchmark.package_colab_inputs import (
    DEFAULT_STEP1X3D_COLAB_PYTHON,
    package_inputs,
)
from backend.benchmark.patch_step1x3d_sources import patch_step1x3d_sources
from backend.benchmark.run_stl_first_smoke import build_experiments, parse_args
from backend.benchmark.step1x3d_models import (
    DEFAULT_STEP1X3D_MODEL,
    DEFAULT_STEP1X3D_MODEL_REVISION,
    DEFAULT_STEP1X3D_SOURCE_REVISION,
    DEFAULT_STEP1X3D_SUBFOLDER,
)


class Step1X3DProviderTest(unittest.TestCase):
    def _write_snapshot(self, root: Path, *, omit: str | None = None) -> Path:
        spec = step1x3d_models.step1x3d_model_specs()["step1x3d"]
        for relative_path in spec["required_files"]:
            if relative_path == omit:
                continue
            path = root / str(relative_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        return root

    def test_exact_pins_registration_and_required_snapshot_files(self):
        self.assertEqual(
            DEFAULT_STEP1X3D_SOURCE_REVISION,
            "cb5ac944709c6c913109070c7b90c3447f57f3d4",
        )
        self.assertEqual(DEFAULT_STEP1X3D_MODEL, "stepfun-ai/Step1X-3D")
        self.assertEqual(
            DEFAULT_STEP1X3D_MODEL_REVISION,
            "bf7084495b3a72222f36549b7942948aa4d9daa7",
        )
        self.assertEqual(DEFAULT_STEP1X3D_SUBFOLDER, "Step1X-3D-Geometry-1300m")
        self.assertIn(provider_module.STEP1X3D_PROVIDER, provider_module.PROVIDERS)
        self.assertEqual(
            provider_module.CLI_PROVIDERS[provider_module.STEP1X3D_PROVIDER]["env"],
            "STEP1X3D_DIR",
        )
        required = step1x3d_models.step1x3d_model_specs()["step1x3d"][
            "required_files"
        ]
        self.assertIn(
            f"{DEFAULT_STEP1X3D_SUBFOLDER}/transformer/diffusion_pytorch_model.safetensors",
            required,
        )
        self.assertIn(
            f"{DEFAULT_STEP1X3D_SUBFOLDER}/vae/diffusion_pytorch_model.safetensors",
            required,
        )

    def test_snapshot_resolution_checks_every_required_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_snapshot(root)
            resolved = step1x3d_models.resolve_step1x3d_model_snapshot(str(root))
            missing = (
                f"{DEFAULT_STEP1X3D_SUBFOLDER}/visual_encoder/"
                "diffusion_pytorch_model.safetensors"
            )
            (root / missing).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "visual_encoder"):
                step1x3d_models.resolve_step1x3d_model_snapshot(str(root))
        self.assertEqual(resolved, root.resolve())

    def test_geometry_wrapper_preserves_provider_raw_mesh(self):
        observed: dict[str, object] = {}

        class FakeMesh:
            import torch

            verts = torch.zeros((4, 3), dtype=torch.float32)
            faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                observed["snapshot"] = path
                observed["load_kwargs"] = kwargs
                return cls()

            def to(self, device):
                observed["device"] = device
                return self

            def __call__(self, image, **kwargs):
                observed["image"] = image
                observed["call_kwargs"] = kwargs
                return SimpleNamespace(mesh=[FakeMesh()])

        modules = {}
        for name in (
            "step1x3d_geometry",
            "step1x3d_geometry.models",
            "step1x3d_geometry.models.pipelines",
        ):
            module = types.ModuleType(name)
            module.__path__ = []
            modules[name] = module
        pipeline_module = types.ModuleType(
            "step1x3d_geometry.models.pipelines.pipeline"
        )
        pipeline_module.Step1X3DGeometryPipeline = FakePipeline
        modules[pipeline_module.__name__] = pipeline_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_mesh = root / "raw" / "output.glb"
            input_image = root / "input.png"
            input_image.write_bytes(b"image")
            with patch.object(
                step1x3d_models,
                "resolve_step1x3d_model_snapshot",
                return_value=root,
            ), patch.object(
                step1x3d_models,
                "verify_step1x3d_source_integrity",
                return_value={"patched_source_sha256": "fixture"},
            ), patch.dict(sys.modules, modules):
                result = step1x3d_models.run_step1x3d(
                    provider_dir=root,
                    input_image=input_image,
                    output_mesh=output_mesh,
                    num_inference_steps=7,
                    guidance_scale=6.5,
                    octree_resolution=192,
                    seed=17,
                    device="cpu",
                )
                metrics = json.loads(
                    (output_mesh.parent / "provider_native_metrics.json").read_text()
                )

        self.assertEqual(result, output_mesh)
        self.assertEqual(observed["snapshot"], str(root))
        self.assertEqual(observed["device"], "cpu")
        self.assertEqual(observed["call_kwargs"]["num_inference_steps"], 7)
        self.assertEqual(observed["call_kwargs"]["octree_resolution"], 192)
        self.assertFalse(observed["call_kwargs"]["do_remove_floater"])
        self.assertFalse(observed["call_kwargs"]["do_reduce_face"])
        self.assertFalse(observed["call_kwargs"]["do_shade_smooth"])
        self.assertEqual(observed["call_kwargs"]["output_type"], "raw")
        self.assertEqual(metrics["provider_raw_face_count"], 2)
        self.assertFalse(metrics["provider_peak_cuda_vram_supported"])

    def test_preflight_requires_exact_model_source_and_cuda_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "Step1X-3D"
            entrypoint = (
                provider_dir
                / "step1x3d_geometry"
                / "models"
                / "pipelines"
                / "pipeline.py"
            )
            entrypoint.parent.mkdir(parents=True)
            entrypoint.write_text("# fixture\n", encoding="utf-8")
            (provider_dir / "step1x3d_geometry" / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            provider_python = root / "step1x3d-python"
            provider_python.write_text("", encoding="utf-8")
            command = (
                f"{provider_python.as_posix()} -m backend.benchmark.run_image_to_mesh_provider "
                f"--provider step1x3d --provider-dir {provider_dir.as_posix()} "
                f"--step1x3d-model-path {DEFAULT_STEP1X3D_MODEL} "
                f"--step1x3d-model-revision {DEFAULT_STEP1X3D_MODEL_REVISION} "
                f"--step1x3d-subfolder {DEFAULT_STEP1X3D_SUBFOLDER}"
            )
            parsed = preflight_module.parse_provider_command(command)
            ready_probe = {
                "returncode": 0,
                "pipeline_module_importable": True,
                "pipeline_class_importable": True,
                "torch_cuda_available": True,
                "torch_cuda_capability": [12, 0],
                "backend_ready": True,
                "error": "",
            }
            with patch.object(
                preflight_module,
                "provider_git_revision",
                return_value=DEFAULT_STEP1X3D_SOURCE_REVISION,
            ), patch.object(
                preflight_module,
                "verify_step1x3d_source_integrity",
                return_value={"patched_source_sha256": "fixture"},
            ), patch.object(
                preflight_module,
                "probe_step1x3d_provider_python",
                return_value=ready_probe,
            ):
                row = preflight_module.provider_preflight_row(parsed)
        self.assertTrue(row["runnable"])
        self.assertTrue(row["checks"]["model_revision_pinned"])
        self.assertTrue(row["checks"]["provider_source_revision_pinned"])
        self.assertEqual(row["checks"]["step1x3d_torch_cuda_capability"], [12, 0])
        self.assertTrue(row["checks"]["step1x3d_source_integrity_ok"])

    def test_source_patch_is_idempotent_and_geometry_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            provider_dir = Path(temp_dir)
            package_init = provider_dir / "step1x3d_geometry" / "__init__.py"
            package_init.parent.mkdir()
            package_init.write_text(
                "from . import data, models, systems\n", encoding="utf-8"
            )
            first = patch_step1x3d_sources(provider_dir)
            second = patch_step1x3d_sources(provider_dir)
            patched = package_init.read_text(encoding="utf-8")
        self.assertEqual(first, [package_init])
        self.assertEqual(second, [])
        self.assertEqual(patched, "from . import models\n")

    def test_source_integrity_allows_only_the_expected_patch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            provider_dir = Path(temp_dir)
            package_init = provider_dir / "step1x3d_geometry" / "__init__.py"
            package_init.parent.mkdir()
            package_init.write_text(
                "header\nfrom . import data, models, systems\n", encoding="utf-8"
            )
            tracked = provider_dir / "tracked.py"
            tracked.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=provider_dir, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=provider_dir, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Codex Test",
                    "-c",
                    "user.email=codex@example.invalid",
                    "commit",
                    "-m",
                    "fixture",
                ],
                cwd=provider_dir,
                check=True,
                capture_output=True,
            )
            patch_step1x3d_sources(provider_dir)
            integrity = step1x3d_models.verify_step1x3d_source_integrity(provider_dir)
            tracked.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected tracked modifications"):
                step1x3d_models.verify_step1x3d_source_integrity(provider_dir)
        self.assertEqual(
            integrity["tracked_changes"], ["step1x3d_geometry/__init__.py"]
        )

    def test_cuda_metrics_use_the_selected_device(self):
        calls: list[tuple[str, object]] = []

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def max_memory_allocated(device):
                calls.append(("allocated", device))
                return 2 * 1024**3

            @staticmethod
            def max_memory_reserved(device):
                calls.append(("reserved", device))
                return 3 * 1024**3

        metrics = step1x3d_models._cuda_metrics(SimpleNamespace(cuda=FakeCuda), "cuda:1")
        self.assertEqual(calls, [("allocated", "cuda:1"), ("reserved", "cuda:1")])
        self.assertEqual(metrics["provider_peak_cuda_vram_gib"], 3.0)

    def test_smoke_config_shares_native_cache_between_raw_and_repaired(self):
        argv = [
            "run_stl_first_smoke",
            "--include-step1x3d",
            "--step1x3d-direct-input",
            "biharmonic",
            "--mesh-target-bbox-source",
            "inferred",
            "--mesh-target-faces",
            "40000",
            "--mesh-max-normalized-face-density-log1p",
            "9.95",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()
        experiments = {row["name"]: row for row in build_experiments(args)}
        raw = experiments["step1x3d_biharmonic_prefill_raw_direct_mesh"]
        repaired = experiments[
            "step1x3d_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh"
        ]
        raw_command = raw["direct_mesh_command"]
        repaired_command = repaired["direct_mesh_command"]
        for command in (raw_command, repaired_command):
            self.assertIn("--provider step1x3d", command)
            self.assertIn(DEFAULT_STEP1X3D_MODEL_REVISION, command)
            self.assertIn("--seed 2025", command)
            self.assertIn("--provider-mesh-cache-dir /content/step1x3d-provider-cache", command)
            self.assertIn("--step1x3d-max-faces 0", command)
        self.assertNotIn("--mesh-repair", raw_command)
        self.assertNotIn("--mesh-target-bbox-extents", raw_command)
        self.assertIn("--mesh-repair printable", repaired_command)
        self.assertIn('--mesh-target-bbox-extents "{inferred_bbox_extents}"', repaired_command)
        self.assertIn("--mesh-target-faces 40000", repaired_command)
        self.assertIn("--mesh-max-normalized-face-density-log1p 9.95", repaired_command)
        self.assertIn("output_mesh_raw.glb", repaired_command)

    def test_colab_package_embeds_geometry_only_setup_and_preflight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            image = dataset / "image.png"
            image.write_bytes(b"image")
            manifest = dataset / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"id": "sample", "full_image": "dataset/image.png"})
                + "\n",
                encoding="utf-8",
            )
            archive = root / "step1x3d_bundle.tar.gz"
            report = package_inputs(
                manifest=manifest,
                output=archive,
                extract_root="/content/inputs/step1x3d",
                root=root,
                include_run_script=True,
                run_name="g4_step1x3d_setup",
                modern_config="step1x3d.json",
                skip_cache=True,
                eval_starts=[40],
                eval_limit=1,
                score_profile="stl-quality",
                include_step1x3d_setup=True,
                colab_env={"STEP1X3D_SETUP_ONLY": "1"},
            )
            with tarfile.open(archive, "r:gz") as tar:
                run_script = tar.extractfile("run_colab_eval.sh").read().decode()

        self.assertTrue(report["include_step1x3d_setup"])
        self.assertEqual(report["step1x3d_source_revision"], DEFAULT_STEP1X3D_SOURCE_REVISION)
        self.assertEqual(report["step1x3d_python"], DEFAULT_STEP1X3D_COLAB_PYTHON)
        self.assertEqual(
            report["step1x3d_model_snapshot"]["revision"],
            DEFAULT_STEP1X3D_MODEL_REVISION,
        )
        self.assertLess(
            run_script.index("--query-gpu=name,memory.total"),
            run_script.index('STEP1X3D_DIR="${STEP1X3D_DIR:-/content/Step1X-3D}"'),
        )
        self.assertIn("python -m venv --system-site-packages", run_script)
        self.assertIn("jaxtyping==0.2.28", run_script)
        self.assertIn("typeguard==2.13.3", run_script)
        self.assertNotIn("PyMCubes", run_script)
        self.assertIn("USE_SAGEATTN=0", run_script)
        self.assertIn("torch.__version__.startswith('2.11.')", run_script)
        self.assertIn("torch.version.cuda == '12.8'", run_script)
        self.assertIn("torch.cuda.get_device_capability(0) == (12, 0)", run_script)
        self.assertIn("'sm_120' in torch.cuda.get_arch_list()", run_script)
        self.assertIn("--require-runnable", run_script)
        self.assertLess(
            run_script.index("--prefetch-only"),
            run_script.index("--require-runnable"),
        )
        self.assertIn("step1x3d_provider_preflight.json", run_script)
        self.assertIn("STEP1X3D_SETUP_ONLY", run_script)
        self.assertNotIn("nvdiffrast", run_script)
        self.assertNotIn("pytorch3d", run_script.lower())
        self.assertNotIn("sageattention==", run_script.lower())


if __name__ == "__main__":
    unittest.main()
