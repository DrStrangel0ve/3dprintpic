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
    DEFAULT_TRELLIS2_DINOV3_MODEL,
    DEFAULT_TRELLIS2_DINOV3_REVISION,
    DEFAULT_TRELLIS2_MODEL,
    DEFAULT_TRELLIS2_MODEL_REVISION,
    DEFAULT_TRELLIS2_REMBG_MODEL,
    DEFAULT_TRELLIS2_REMBG_REVISION,
    DEFAULT_TRELLIS2_RESOLUTION,
    DEFAULT_TRELLIS2_SOURCE_REVISION,
    DEFAULT_TRELLIS2_SPARSE_STRUCTURE_MODEL,
    DEFAULT_TRELLIS2_SPARSE_STRUCTURE_REVISION,
)


class Trellis2ProviderTest(unittest.TestCase):
    def _pipeline_config(self) -> dict:
        return {
            "name": "Trellis2ImageTo3DPipeline",
            "args": {
                "models": {
                    "sparse_structure_decoder": (
                        "microsoft/TRELLIS-image-large/ckpts/"
                        "ss_dec_conv3d_16l8_fp16"
                    ),
                    "sparse_structure_flow_model": (
                        "ckpts/ss_flow_img_dit_1_3B_64_bf16"
                    ),
                    "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
                    "shape_slat_flow_model_512": (
                        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
                    ),
                    "shape_slat_flow_model_1024": (
                        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16"
                    ),
                    "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
                    "tex_slat_flow_model_512": (
                        "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16"
                    ),
                    "tex_slat_flow_model_1024": (
                        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16"
                    ),
                },
                "image_cond_model": {
                    "name": "DinoV3FeatureExtractor",
                    "args": {
                        "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m"
                    },
                },
                "rembg_model": {
                    "name": "BiRefNet",
                    "args": {"model_name": "briaai/RMBG-2.0"},
                },
            },
        }

    def _snapshot_fixtures(
        self,
        root: Path,
        *,
        missing: tuple[str, str] | None = None,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for name, spec in trellis2_models.trellis2_model_specs().items():
            snapshot = root / name
            snapshot.mkdir(parents=True)
            for required_file in spec["required_files"]:
                if missing == (name, required_file):
                    continue
                required_path = snapshot / required_file
                required_path.parent.mkdir(parents=True, exist_ok=True)
                required_path.write_bytes(b"snapshot fixture")
            if name == "trellis2" and missing != (name, "pipeline.json"):
                (snapshot / "pipeline.json").write_text(
                    json.dumps(self._pipeline_config()),
                    encoding="utf-8",
                )
            paths[name] = snapshot.resolve()
        return paths

    def _snapshot_download_side_effect(self, paths: dict[str, Path]):
        specs = trellis2_models.trellis2_model_specs()
        names_by_repo = {
            str(spec["repo_id"]): name for name, spec in specs.items()
        }

        def download(*, repo_id, revision):
            name = names_by_repo[repo_id]
            self.assertEqual(revision, specs[name]["revision"])
            return str(paths[name])

        return download

    def _provider_model_metadata(self) -> dict[str, dict[str, str]]:
        return {
            name: {
                "repo_id": str(spec["repo_id"]),
                "revision": str(spec["revision"]),
            }
            for name, spec in trellis2_models.trellis2_model_specs().items()
        }

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
        self.assertEqual(
            (
                DEFAULT_TRELLIS2_SPARSE_STRUCTURE_MODEL,
                DEFAULT_TRELLIS2_SPARSE_STRUCTURE_REVISION,
            ),
            (
                "microsoft/TRELLIS-image-large",
                "25e0d31ffbebe4b5a97464dd851910efc3002d96",
            ),
        )
        self.assertEqual(
            (DEFAULT_TRELLIS2_DINOV3_MODEL, DEFAULT_TRELLIS2_DINOV3_REVISION),
            (
                "camenduru/dinov3-vitl16-pretrain-lvd1689m",
                "3c276edd87d6f6e569ff0c4400e086807d0f3881",
            ),
        )
        self.assertEqual(
            (DEFAULT_TRELLIS2_REMBG_MODEL, DEFAULT_TRELLIS2_REMBG_REVISION),
            (
                "ZhengPeng7/BiRefNet",
                "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
            ),
        )
        self.assertEqual(
            set(trellis2_models.trellis2_model_specs()),
            {"trellis2", "sparse_structure_decoder", "dinov3", "rembg"},
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

    def test_all_snapshots_resolve_at_exact_revisions_then_force_offline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshots = self._snapshot_fixtures(Path(temp_dir))

            with patch(
                "huggingface_hub.snapshot_download",
                side_effect=self._snapshot_download_side_effect(snapshots),
            ) as download, patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"},
                clear=False,
            ):
                resolved = trellis2_models.resolve_trellis2_model_snapshots()
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

            self.assertEqual(resolved, snapshots)
            self.assertEqual(
                [call.kwargs for call in download.call_args_list],
                [
                    {
                        "repo_id": str(spec["repo_id"]),
                        "revision": str(spec["revision"]),
                    }
                    for spec in trellis2_models.trellis2_model_specs().values()
                ],
            )

    def test_prefetch_only_fails_when_a_transitive_asset_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshots = self._snapshot_fixtures(
                root,
                missing=("dinov3", "model.safetensors"),
            )
            argv = [
                "trellis2_models",
                "--provider-dir",
                str(root),
                "--prefetch-only",
            ]
            with patch(
                "huggingface_hub.snapshot_download",
                side_effect=self._snapshot_download_side_effect(snapshots),
            ), patch.object(sys, "argv", argv), patch.dict(
                os.environ,
                {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    r"dinov3.*model\.safetensors",
                ):
                    trellis2_models.main()
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "0")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "0")

    def test_wrapper_uses_official_512_api_and_glb_coordinates(self):
        observed: dict[str, object] = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, path):
                observed["snapshot"] = path
                manifest = json.loads(
                    (Path(path) / "pipeline.json").read_text(encoding="utf-8")
                )
                observed["pipeline_models"] = {
                    name: (Path(path) / reference).resolve()
                    for name, reference in manifest["args"]["models"].items()
                }
                observed["dinov3_path"] = Path(
                    manifest["args"]["image_cond_model"]["args"]["model_name"]
                ).resolve()
                observed["rembg_path"] = Path(
                    manifest["args"]["rembg_model"]["args"]["model_name"]
                ).resolve()
                return cls()

            def cuda(self):
                observed["cuda"] = True

            def run(self, image, **kwargs):
                observed["image"] = image
                observed["run_kwargs"] = kwargs
                return [
                    SimpleNamespace(
                        vertices=np.array(
                            [[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]],
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
            provider_dir.mkdir()
            snapshots = self._snapshot_fixtures(root / "snapshots")
            input_image = root / "input.png"
            output_mesh = root / "raw.glb"
            Image.new("RGB", (8, 8), (40, 90, 140)).save(input_image)

            with patch.object(
                trellis2_models,
                "resolve_trellis2_model_snapshots",
                return_value=snapshots,
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
        self.assertNotEqual(observed["snapshot"], str(snapshots["trellis2"]))
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
        np.testing.assert_array_equal(
            observed["trimesh_kwargs"]["vertices"],
            np.array(
                [[1.0, 3.0, -2.0], [-4.0, -6.0, -5.0], [7.0, 9.0, 8.0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            observed["trimesh_kwargs"]["faces"],
            np.array([[0, 1, 2]], dtype=np.int32),
        )
        expected_pipeline_models = {
            name: (
                snapshots["sparse_structure_decoder"]
                / reference.removeprefix(
                    f"{DEFAULT_TRELLIS2_SPARSE_STRUCTURE_MODEL}/"
                )
                if name == "sparse_structure_decoder"
                else snapshots["trellis2"] / reference
            ).resolve()
            for name, reference in self._pipeline_config()["args"]["models"].items()
        }
        self.assertEqual(observed["pipeline_models"], expected_pipeline_models)
        self.assertEqual(observed["dinov3_path"], snapshots["dinov3"])
        self.assertEqual(observed["rembg_path"], snapshots["rembg"])
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
            self.assertEqual(row["provider_models"], self._provider_model_metadata())
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
            self.assertEqual(
                baseline["provider_models"],
                self._provider_model_metadata(),
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
                self._provider_model_metadata(),
            )

    def test_stl_smoke_config_compares_raw_and_repaired_geometry(self):
        config_path = (
            Path(__file__).parents[1]
            / "benchmark"
            / "experiment_configs"
            / "modelnet10_60_balanced_stl_quality_trellis2_triposg_s40_n1.json"
        )
        experiments = json.loads(config_path.read_text(encoding="utf-8"))
        by_name = {experiment["name"]: experiment for experiment in experiments}
        raw = by_name["trellis2_biharmonic_prefill_raw_direct_mesh"]
        repaired = by_name[
            "trellis2_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh"
        ]

        for experiment in (raw, repaired):
            command = experiment["direct_mesh_command"]
            self.assertIn("--provider trellis2", command)
            self.assertIn(f"--trellis2-model-path {DEFAULT_TRELLIS2_MODEL}", command)
            self.assertIn(
                f"--trellis2-model-revision {DEFAULT_TRELLIS2_MODEL_REVISION}",
                command,
            )
            self.assertIn("--trellis2-resolution 512", command)
            self.assertIn("--seed 42", command)
            self.assertIn(
                "--provider-mesh-cache-dir /content/trellis2-provider-cache",
                command,
            )
            self.assertNotIn("to_glb", command)

        self.assertNotIn("--mesh-repair", raw["direct_mesh_command"])
        self.assertIn("--mesh-repair printable", repaired["direct_mesh_command"])
        self.assertIn(
            '--mesh-target-bbox-extents "{inferred_bbox_extents}"',
            repaired["direct_mesh_command"],
        )
        self.assertIn(
            "--mesh-max-normalized-face-density-log1p 9.95",
            repaired["direct_mesh_command"],
        )
        self.assertEqual(repaired["direct_mesh_bbox_source"], "inferred")
        self.assertEqual(repaired["direct_mesh_reference_method"], "mirror")


if __name__ == "__main__":
    unittest.main()
