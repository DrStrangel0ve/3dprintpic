from __future__ import annotations

import contextlib
import json
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from backend.benchmark import hunyuan3d_2mv_models
from backend.benchmark import preflight_image_to_mesh_providers as preflight_module
from backend.benchmark import run_image_to_mesh_provider as provider_module
from backend.benchmark.generate_rendered_dataset import (
    generate_dataset,
    iter_mesh_paths_from_manifest,
)
from backend.benchmark.package_colab_inputs import package_inputs
from backend.benchmark.run_completion_benchmark import heldout_multiview_mesh_metrics


class Hunyuan3D2mvProviderTest(unittest.TestCase):
    def _write_bundle(self, root: Path, *, primary: str = "front") -> Path:
        views = []
        for index, (label, azimuth) in enumerate(
            (("right", 270), ("front", 0), ("back", 180), ("left", 90))
        ):
            image_path = root / f"{label}.png"
            mask_path = root / f"{label}_mask.png"
            Image.new("RGB", (8, 8), (20 + index * 20, 80, 120)).save(image_path)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:7, 1:6] = 255
            Image.fromarray(mask).save(mask_path)
            views.append(
                {
                    "index": index,
                    "sample_id": label,
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "camera": {
                        "azimuth_deg": azimuth,
                        "elevation_deg": 0,
                        "roll_deg": 0,
                    },
                }
            )
        bundle = root / "multiview_input.json"
        bundle.write_text(
            json.dumps(
                {
                    "sample_id": primary,
                    "source_mesh": str(root / "diagnostic_only.ply"),
                    "views": views,
                }
            ),
            encoding="utf-8",
        )
        return bundle

    def _provider_args(self, root: Path, bundle: Path, *, input_name: str = "front.png"):
        return SimpleNamespace(
            provider=provider_module.HUNYUAN3D_2MV_PROVIDER,
            python="hunyuan-python",
            input_image=root / input_name,
            input_bundle=bundle,
            provider_arg=[],
            prefetch_only=False,
            hunyuan3d_2mv_model_path=hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_MODEL,
            hunyuan3d_2mv_model_revision=(
                hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION
            ),
            hunyuan3d_2mv_subfolder=(
                hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_SUBFOLDER
            ),
            hunyuan3d_2mv_required_view=["front", "left", "back", "right"],
            hunyuan3d_2mv_max_view_angle_error=5.0,
            num_inference_steps=50,
            guidance_scale=5.0,
            octree_resolution=380,
            num_chunks=20000,
            seed=42,
            provider_device="cuda",
            disable_progress=False,
        )

    def test_exact_pins_and_provider_registration(self):
        self.assertEqual(
            hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION,
            "f8db63096c8282cb27354314d896feba5ba6ff8a",
        )
        self.assertEqual(
            hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION,
            "3a761b539b29fe4ff64714813aa9560fd66f5de0",
        )
        self.assertIn(provider_module.HUNYUAN3D_2MV_PROVIDER, provider_module.PROVIDERS)
        self.assertEqual(
            provider_module.CLI_PROVIDERS[provider_module.HUNYUAN3D_2MV_PROVIDER],
            {
                "env": "HUNYUAN3D_2MV_DIR",
                "default_dirs": ("/content/Hunyuan3D-2",),
                "runner": "hunyuan3d-2mv-wrapper",
                "supports_low_vram": False,
                "supports_device": True,
                "supports_remesh": False,
                "supports_texture_resolution": False,
            },
        )

    def test_view_selection_is_keyed_by_pose_and_rgba_uses_silhouette(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = self._write_bundle(root)
            images, manifest = hunyuan3d_2mv_models.prepare_hunyuan3d_2mv_images(
                bundle,
                root / "output",
                max_angle_error=5,
                required_views=("front", "left", "back", "right"),
            )

            self.assertEqual(list(images), ["front", "left", "back", "right"])
            selected = hunyuan3d_2mv_models.select_hunyuan3d_2mv_views(
                json.loads(bundle.read_text(encoding="utf-8")),
                max_angle_error=5,
                required_views=("front", "left", "back", "right"),
            )
            self.assertEqual(selected["left"]["camera"]["azimuth_deg"], 270)
            self.assertEqual(selected["right"]["camera"]["azimuth_deg"], 90)
            self.assertTrue(all(image.mode == "RGBA" for image in images.values()))
            self.assertEqual(images["front"].getpixel((0, 0)), (255, 255, 255, 0))
            self.assertGreater(images["front"].getpixel((2, 2))[3], 0)
            self.assertFalse(manifest["source_geometry_used"])
            self.assertEqual(
                [row["angle_error_deg"] for row in manifest["views"]],
                [0.0, 0.0, 0.0, 0.0],
            )

    def test_missing_required_cardinal_view_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing required.*right"):
            hunyuan3d_2mv_models.select_hunyuan3d_2mv_views(
                {
                    "views": [
                        {"camera": {"azimuth_deg": azimuth}}
                        for azimuth in (0, 90, 180)
                    ]
                },
                max_angle_error=5,
                required_views=("front", "left", "back", "right"),
            )

    def test_wrapper_calls_official_pipeline_with_keyed_pil_views_and_writes_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "Hunyuan3D-2"
            entrypoint = provider_dir / "hy3dgen" / "shapegen" / "pipelines.py"
            entrypoint.parent.mkdir(parents=True)
            entrypoint.write_text("# fixture\n", encoding="utf-8")
            bundle = self._write_bundle(root)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            output_mesh = root / "result" / "output.glb"
            observed = {}

            class FakeMesh:
                def export(self, path):
                    Path(path).write_bytes(b"fake-glb")

            class FakePipeline:
                @classmethod
                def from_pretrained(cls, model_path, **kwargs):
                    observed["model_path"] = model_path
                    observed["load_kwargs"] = kwargs
                    return cls()

                def __call__(self, **kwargs):
                    observed["call_kwargs"] = kwargs
                    return [FakeMesh()]

            class FakeCuda:
                @staticmethod
                def is_available():
                    return True

                @staticmethod
                def empty_cache():
                    return None

                @staticmethod
                def reset_peak_memory_stats(device=None):
                    return None

                @staticmethod
                def synchronize(device=None):
                    return None

                @staticmethod
                def max_memory_allocated(device=None):
                    return 3 * 1024**3

                @staticmethod
                def max_memory_reserved(device=None):
                    return 4 * 1024**3

            fake_torch = types.ModuleType("torch")
            fake_torch.float16 = "float16"
            fake_torch.float32 = "float32"
            fake_torch.cuda = FakeCuda()
            fake_torch.manual_seed = lambda seed: f"generator:{seed}"
            fake_torch.inference_mode = contextlib.nullcontext
            fake_hy3dgen = types.ModuleType("hy3dgen")
            fake_hy3dgen.__path__ = []
            fake_shapegen = types.ModuleType("hy3dgen.shapegen")
            fake_shapegen.Hunyuan3DDiTFlowMatchingPipeline = FakePipeline

            with patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "hy3dgen": fake_hy3dgen,
                    "hy3dgen.shapegen": fake_shapegen,
                },
            ), patch.object(
                hunyuan3d_2mv_models,
                "resolve_hunyuan3d_2mv_model_snapshot",
                return_value=snapshot,
            ):
                metrics = hunyuan3d_2mv_models.run_hunyuan3d_2mv(
                    provider_dir=provider_dir,
                    input_bundle=bundle,
                    output_mesh=output_mesh,
                    required_views=("front", "left", "back", "right"),
                    max_view_angle_error=5,
                )

            self.assertTrue(output_mesh.is_file())
            self.assertEqual(observed["model_path"], str(snapshot))
            self.assertEqual(observed["load_kwargs"]["variant"], "fp16")
            self.assertTrue(observed["load_kwargs"]["use_safetensors"])
            self.assertEqual(
                list(observed["call_kwargs"]["image"]),
                ["front", "left", "back", "right"],
            )
            self.assertTrue(
                all(
                    isinstance(image, Image.Image) and image.mode == "RGBA"
                    for image in observed["call_kwargs"]["image"].values()
                )
            )
            self.assertEqual(metrics["provider_peak_cuda_vram_gib"], 4.0)
            self.assertEqual(metrics["provider_peak_cuda_allocated_gib"], 3.0)
            self.assertEqual(metrics["provider_peak_cuda_vram_measurement"], "torch_peak_reserved")
            self.assertFalse(metrics["source_geometry_used"])
            self.assertTrue(
                (output_mesh.parent / "provider_native_metrics.json").is_file()
            )

    def test_cache_hashes_every_view_and_ignores_unused_primary_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = self._write_bundle(root)
            provider_dir = root / "Hunyuan3D-2"
            run_entry = provider_dir / "hy3dgen" / "shapegen" / "pipelines.py"
            run_entry.parent.mkdir(parents=True)
            run_entry.write_text("# official source\n", encoding="utf-8")
            args = self._provider_args(root, bundle)
            with patch.object(
                provider_module,
                "provider_git_revision",
                return_value=hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION,
            ):
                baseline = provider_module.cli_provider_cache_payload(
                    args, provider_dir, run_entry
                )
                changed_primary_args = self._provider_args(
                    root, bundle, input_name="back.png"
                )
                changed_primary = provider_module.cli_provider_cache_payload(
                    changed_primary_args, provider_dir, run_entry
                )
                changed_repair_args = self._provider_args(root, bundle)
                changed_repair_args.mesh_repair_preconditioner = "voxel-close"
                changed_repair_args.mesh_repair_component_area_ratio = 0.01
                changed_repair_args.mesh_repair_voxel_resolution = 256
                changed_repair_args.mesh_repair_voxel_fill_method = "orthographic"
                changed_repair = provider_module.cli_provider_cache_payload(
                    changed_repair_args,
                    provider_dir,
                    run_entry,
                )
                mask = Image.open(root / "left_mask.png").convert("L")
                mask.putpixel((0, 0), 255)
                mask.save(root / "left_mask.png")
                changed_mask = provider_module.cli_provider_cache_payload(
                    args, provider_dir, run_entry
                )

            baseline_key = provider_module.cli_provider_cache_key(baseline)
            self.assertEqual(
                provider_module.cli_provider_cache_key(changed_primary), baseline_key
            )
            self.assertEqual(
                provider_module.cli_provider_cache_key(changed_repair), baseline_key
            )
            self.assertNotEqual(
                provider_module.cli_provider_cache_key(changed_mask), baseline_key
            )
            self.assertEqual(baseline["input_sha256"], "")
            self.assertEqual(len(baseline["multiview_bundle_identity"]["views"]), 4)
            self.assertIn(
                "backend/benchmark/hunyuan3d_2mv_models.py",
                baseline["provider_source_sha256"],
            )

    def test_preflight_requires_exact_source_model_bundle_and_provider_python(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "Hunyuan3D-2"
            entrypoint = provider_dir / "hy3dgen" / "shapegen" / "pipelines.py"
            entrypoint.parent.mkdir(parents=True)
            entrypoint.write_text("# fixture\n", encoding="utf-8")
            (entrypoint.parent / "__init__.py").write_text("", encoding="utf-8")
            provider_python = Path(sys.executable).as_posix()
            command = (
                f"{provider_python} -m backend.benchmark.run_image_to_mesh_provider "
                f"--provider hunyuan3d-2mv --provider-dir {provider_dir.as_posix()} "
                f"--provider-python {provider_python} --input-image '{{input_image}}' "
                "--input-bundle '{input_bundle}' --output-mesh '{output_mesh}' "
                f"--hunyuan3d-2mv-model-path {hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_MODEL} "
                f"--hunyuan3d-2mv-model-revision {hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION} "
                f"--hunyuan3d-2mv-subfolder {hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_SUBFOLDER}"
            )
            parsed = preflight_module.parse_provider_command(command)
            ready_probe = {
                "python_executable": sys.executable,
                "returncode": 0,
                "shapegen_importable": True,
                "pipeline_class_importable": True,
                "backend_ready": True,
                "error": "",
            }
            with patch.object(
                preflight_module,
                "provider_git_revision",
                return_value=hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION,
            ), patch.object(
                preflight_module,
                "probe_hunyuan3d_2mv_provider_python",
                return_value=ready_probe,
            ):
                row = preflight_module.provider_preflight_row(parsed)

            changed_model = dict(parsed)
            changed_model["provider_models"] = {
                "hunyuan3d_2mv": {
                    "repo_id": hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_MODEL,
                    "revision": "f" * 40,
                    "subfolder": hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_SUBFOLDER,
                }
            }
            with patch.object(
                preflight_module,
                "provider_git_revision",
                return_value=hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION,
            ), patch.object(
                preflight_module,
                "probe_hunyuan3d_2mv_provider_python",
                return_value=ready_probe,
            ):
                changed_model_row = preflight_module.provider_preflight_row(changed_model)

        self.assertTrue(row["runnable"])
        self.assertTrue(row["checks"]["model_revision_pinned"])
        self.assertTrue(row["checks"]["provider_source_revision_pinned"])
        self.assertTrue(row["checks"]["hunyuan3d_2mv_backend_ready"])
        self.assertFalse(changed_model_row["runnable"])

    def test_cardinal_renderer_emits_exact_four_view_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = generate_dataset(
                output_dir=Path(temp_dir),
                source="procedural",
                count=4,
                size=32,
                seed=2026,
                views_per_asset=4,
                camera_layout="cardinal",
            )
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]

        self.assertEqual(
            [row["camera"]["azimuth_deg"] for row in rows],
            [0.0, 90.0, 180.0, 270.0],
        )
        self.assertTrue(all(len(row["multiview_images"]) == 4 for row in rows))
        self.assertTrue(all(len(row["multiview_masks"]) == 4 for row in rows))

    def test_asset_manifest_selects_exact_heldout_meshes_in_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meshes = []
            for index in range(3):
                path = root / f"mesh_{index}.off"
                path.write_text("fixture", encoding="utf-8")
                meshes.append(path)
            manifest = root / "source.jsonl"
            manifest.write_text(
                "\n".join(json.dumps({"asset_path": str(path)}) for path in meshes)
                + "\n",
                encoding="utf-8",
            )
            selected = iter_mesh_paths_from_manifest(
                manifest,
                start_index=1,
                limit=2,
            )

        self.assertEqual(selected, [meshes[1].resolve(), meshes[2].resolve()])

    def test_octants_renderer_reserves_four_unseen_views_for_mesh_agreement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = generate_dataset(
                output_dir=Path(temp_dir),
                source="procedural",
                count=8,
                size=64,
                seed=2026,
                views_per_asset=8,
                camera_layout="octants",
                anchor_views_only=True,
            )
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            sample = rows[0]
            selected = [
                {"source_image": sample["multiview_images"][index]}
                for index in (0, 2, 4, 6)
            ]
            metrics = heldout_multiview_mesh_metrics(
                sample,
                sample["mesh"],
                {"provider_native_metrics": {"selected_views": selected}},
                render_size=64,
            )

        self.assertEqual(
            [camera["azimuth_deg"] for camera in sample["multiview_cameras"]],
            [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(sample["view_index"], 0)
        self.assertTrue(metrics["heldout_view_agreement_supported"])
        self.assertEqual(metrics["heldout_view_selected_count"], 4)
        self.assertEqual(metrics["heldout_view_count"], 4)
        self.assertGreater(metrics["heldout_view_silhouette_iou_min"], 0.99)

    def test_colab_package_embeds_isolated_shape_only_setup_and_exact_pins(self):
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
            archive = root / "hunyuan3d_2mv_bundle.tar.gz"
            report = package_inputs(
                manifest=manifest,
                output=archive,
                extract_root="/content/inputs/hunyuan3d_2mv",
                root=root,
                include_run_script=True,
                colab_archive_path="/content/hunyuan3d_2mv_bundle.tar.gz",
                run_name="g4_hunyuan3d_2mv_setup",
                modern_config=(
                    "backend/benchmark/experiment_configs/"
                    "modelnet10_60_balanced_stl_quality_hunyuan3d_2mv_triposg_s40_n1.json"
                ),
                skip_cache=True,
                eval_starts=[0],
                eval_limit=1,
                score_profile="stl-quality",
                include_hunyuan3d_2mv_setup=True,
                colab_env={"HUNYUAN3D_2MV_SETUP_ONLY": "1"},
            )
            with tarfile.open(archive, "r:gz") as tar:
                run_script = tar.extractfile("run_colab_eval.sh").read().decode("utf-8")

        self.assertTrue(report["include_hunyuan3d_2mv_setup"])
        self.assertEqual(
            report["hunyuan3d_2mv_source_revision"],
            hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION,
        )
        self.assertEqual(
            report["hunyuan3d_2mv_model_snapshot"]["revision"],
            hunyuan3d_2mv_models.DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION,
        )
        self.assertIn("/content/hunyuan3d-2mv-venv", run_script)
        self.assertIn("transformers==4.49.0", run_script)
        self.assertIn("pymeshlab==2025.7.post1", run_script)
        self.assertIn("torch.cuda.get_arch_list()", run_script)
        self.assertIn("provider preflight passed", run_script)
        self.assertIn("HUNYUAN3D_2MV_PREFLIGHT_PATH", run_script)
        self.assertIn("hunyuan3d_2mv_provider_preflight.json", run_script)
        self.assertIn("'hunyuan3d_2mv_preflight'", run_script)
        self.assertIn("--prefetch-only", run_script)
        self.assertIn("HUNYUAN3D_2MV_SETUP_ONLY", run_script)
        self.assertNotIn("custom_rasterizer", run_script)
        self.assertNotIn("nvdiffrast", run_script)

    def test_repair_ablation_config_keeps_inference_fixed_and_sweeps_only_repair(self):
        config_path = Path(
            "backend/benchmark/experiment_configs/"
            "modelnet10_60_balanced_stl_quality_hunyuan3d_2mv_repair_ablation_s40_n4.json"
        )
        experiments = json.loads(config_path.read_text(encoding="utf-8"))
        by_name = {experiment["name"]: experiment for experiment in experiments}
        voxel_rows = [
            experiment
            for experiment in experiments
            if "_voxel_" in experiment["name"]
        ]

        self.assertEqual(len(voxel_rows), 4)
        self.assertIn("masked", by_name)
        self.assertIn("mirror", by_name)
        self.assertIn("hunyuan3d_2mv_cardinal4_raw_direct_mesh", by_name)
        self.assertIn(
            "hunyuan3d_2mv_cardinal4_legacy_repaired_stl_inferred_bbox_direct_mesh",
            by_name,
        )
        self.assertFalse(any(row.get("oracle_diagnostic", False) for row in experiments))
        for experiment in experiments:
            self.assertNotIn("{source_bbox", experiment.get("direct_mesh_command", ""))
        for experiment in voxel_rows:
            command = experiment["direct_mesh_command"]
            self.assertIn("--mesh-repair-preconditioner voxel-close", command)
            self.assertIn("--mesh-repair-voxel-fill-method orthographic", command)
            self.assertIn("--num-inference-steps 50", command)
            self.assertIn("--guidance-scale 5.0", command)
            self.assertIn("--octree-resolution 380", command)
            self.assertIn("--seed 42", command)
            self.assertIn("--mesh-max-normalized-face-density-log1p 9.95", command)
        resolutions = {
            token
            for experiment in voxel_rows
            for token in experiment["direct_mesh_command"].split()
            if token in {"192", "256"}
        }
        ratios = {
            token
            for experiment in voxel_rows
            for token in experiment["direct_mesh_command"].split()
            if token in {"0", "0.01"}
        }
        self.assertEqual(resolutions, {"192", "256"})
        self.assertEqual(ratios, {"0", "0.01"})


if __name__ == "__main__":
    unittest.main()
