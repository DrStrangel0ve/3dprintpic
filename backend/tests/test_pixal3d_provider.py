from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from backend.benchmark import run_image_to_mesh_provider as provider_module
from backend.benchmark.mesh_rendering import load_mesh
from backend.benchmark.pixal3d_models import (
    DEFAULT_PIXAL3D_DINOV3_MODEL,
    DEFAULT_PIXAL3D_DINOV3_REVISION,
    DEFAULT_PIXAL3D_MODEL,
    DEFAULT_PIXAL3D_MODEL_REVISION,
    DEFAULT_PIXAL3D_MOGE_MODEL,
    DEFAULT_PIXAL3D_MOGE_REVISION,
    DEFAULT_PIXAL3D_REMBG_MODEL,
    DEFAULT_PIXAL3D_REMBG_REVISION,
)
from backend.benchmark.preflight_image_to_mesh_providers import parse_provider_command, provider_preflight_row


class Pixal3DProviderTest(unittest.TestCase):
    def test_provider_registration_and_environment_resolution(self):
        self.assertIn("pixal3d", provider_module.PROVIDERS)
        self.assertEqual(provider_module.CLI_PROVIDERS["pixal3d"]["env"], "PIXAL3D_DIR")
        self.assertEqual(
            provider_module.CLI_PROVIDERS["pixal3d"]["default_dirs"],
            ("/content/Pixal3D",),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            provider_dir = Path(temp_dir) / "Pixal3D"
            provider_dir.mkdir()
            with patch.dict(os.environ, {"PIXAL3D_DIR": str(provider_dir)}):
                resolved = provider_module.resolve_provider_dir("pixal3d", None)

        self.assertEqual(resolved, provider_dir)

    def test_preflight_accepts_official_inference_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            provider_dir = Path(temp_dir) / "Pixal3D"
            provider_dir.mkdir()
            (provider_dir / "inference.py").write_text("# official CLI entrypoint\n", encoding="utf-8")
            row = provider_preflight_row(
                {
                    "provider": "pixal3d",
                    "provider_dir": str(provider_dir),
                    "provider_python": sys.executable,
                    "wrapper_python": sys.executable,
                },
                experiment_names=["pixal3d_raw"],
            )

        self.assertTrue(row["runnable"])
        self.assertEqual(row["readiness"], "ready")
        self.assertEqual(row["checks"]["entrypoint"], "inference.py")
        self.assertTrue(row["checks"]["entrypoint_found"])

    def test_preflight_preserves_pinned_model_identity(self):
        command = (
            f'python -m backend.benchmark.run_image_to_mesh_provider --provider pixal3d '
            f'--pixal3d-model-path {DEFAULT_PIXAL3D_MODEL} '
            f'--pixal3d-model-revision {DEFAULT_PIXAL3D_MODEL_REVISION} '
            f'--pixal3d-moge-revision {DEFAULT_PIXAL3D_MOGE_REVISION} '
            f'--pixal3d-dinov3-revision {DEFAULT_PIXAL3D_DINOV3_REVISION} '
            f'--pixal3d-rembg-model {DEFAULT_PIXAL3D_REMBG_MODEL} '
            f'--pixal3d-rembg-revision {DEFAULT_PIXAL3D_REMBG_REVISION}'
        )

        parsed = parse_provider_command(command)

        self.assertIsNotNone(parsed)
        self.assertEqual(
            parsed["provider_models"]["moge"],
            {
                "repo_id": DEFAULT_PIXAL3D_MOGE_MODEL,
                "revision": DEFAULT_PIXAL3D_MOGE_REVISION,
            },
        )

    def test_parser_builds_exact_pixal3d_cli(self):
        observed = {}

        def fake_run_provider(args):
            observed["args"] = args
            return Path(args.output_mesh), None

        argv = [
            "run_image_to_mesh_provider",
            "--provider",
            "pixal3d",
            "--input-image",
            "input.png",
            "--output-mesh",
            "normalized.glb",
            "--provider-python",
            "pixal-python",
            "--low-vram",
            "--pixal3d-resolution",
            "1536",
            "--seed",
            "17",
            "--pixal3d-fov",
            "0.25",
            "--pixal3d-model-path",
            "unit/Pixal3D",
            "--pixal3d-model-revision",
            DEFAULT_PIXAL3D_MODEL_REVISION,
            "--pixal3d-moge-revision",
            DEFAULT_PIXAL3D_MOGE_REVISION,
            "--pixal3d-dinov3-revision",
            DEFAULT_PIXAL3D_DINOV3_REVISION,
            "--pixal3d-rembg-revision",
            DEFAULT_PIXAL3D_REMBG_REVISION,
            "--provider-arg=--custom-flag",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            provider_module,
            "run_provider",
            side_effect=fake_run_provider,
        ), contextlib.redirect_stdout(io.StringIO()):
            provider_module.main()

        args = observed["args"]
        command = provider_module.cli_provider_command(
            args,
            Path("provider-repo"),
            Path("provider-raw"),
        )
        self.assertEqual(
            command,
            [
                "pixal-python",
                str(Path("provider-repo") / "inference.py"),
                "--image",
                "input.png",
                "--output",
                str(Path("provider-raw") / "output.glb"),
                "--low_vram",
                "--resolution",
                "1536",
                "--seed",
                "17",
                "--fov",
                "0.25",
                "--model_path",
                "unit/Pixal3D",
                "--custom-flag",
            ],
        )

    def test_fake_provider_glb_is_discovered_normalized_and_exported_to_stl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "Pixal3D"
            provider_dir.mkdir()
            input_image = root / "input.png"
            output_mesh = root / "normalized.glb"
            raw_output_mesh = root / "raw_provider.glb"
            output_stl = root / "normalized.stl"
            provider_raw_dir = root / "pixal3d_raw"
            Image.new("RGB", (12, 12), (120, 80, 160)).save(input_image)
            (provider_dir / "inference.py").write_text(
                "\n".join(
                    [
                        "import argparse, json, sys",
                        "from pathlib import Path",
                        "import trimesh",
                        "parser = argparse.ArgumentParser()",
                        "parser.add_argument('--image', required=True)",
                        "parser.add_argument('--output', required=True)",
                        "parser.add_argument('--low_vram', action='store_true')",
                        "parser.add_argument('--resolution', type=int, choices=(1024, 1536))",
                        "parser.add_argument('--seed', type=int)",
                        "parser.add_argument('--fov', type=float)",
                        "parser.add_argument('--model_path')",
                        "parser.add_argument('--custom-flag', action='store_true')",
                        "args = parser.parse_args()",
                        "output = Path(args.output)",
                        "output.parent.mkdir(parents=True, exist_ok=True)",
                        "(output.parent / 'argv.json').write_text(json.dumps(sys.argv[1:]), encoding='utf-8')",
                        "trimesh.creation.box(extents=(2.0, 1.0, 0.25)).export(output)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            argv = [
                "run_image_to_mesh_provider",
                "--provider",
                "pixal3d",
                "--provider-dir",
                str(provider_dir),
                "--provider-output-dir",
                str(provider_raw_dir),
                "--input-image",
                str(input_image),
                "--output-mesh",
                str(output_mesh),
                "--raw-output-mesh",
                str(raw_output_mesh),
                "--output-stl",
                str(output_stl),
                "--provider-python",
                sys.executable,
                "--low-vram",
                "--pixal3d-resolution",
                "1024",
                "--seed",
                "23",
                "--pixal3d-fov",
                "0.2",
                "--pixal3d-model-path",
                "unit/Pixal3D",
                "--provider-arg=--custom-flag",
                "--mesh-target-max-dimension",
                "40",
            ]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                provider_module.main()

            provider_glb = provider_raw_dir / "output.glb"
            recorded_argv = json.loads((provider_raw_dir / "argv.json").read_text(encoding="utf-8"))
            self.assertEqual(
                recorded_argv,
                [
                    "--image",
                    str(input_image),
                    "--output",
                    str(provider_glb),
                    "--low_vram",
                    "--resolution",
                    "1024",
                    "--seed",
                    "23",
                    "--fov",
                    "0.2",
                    "--model_path",
                    "unit/Pixal3D",
                    "--custom-flag",
                ],
            )
            self.assertTrue(provider_glb.exists())
            self.assertTrue(output_mesh.exists())
            self.assertTrue(raw_output_mesh.exists())
            self.assertTrue(output_stl.exists())
            self.assertEqual(raw_output_mesh.read_bytes(), provider_glb.read_bytes())

            raw_mesh = load_mesh(raw_output_mesh)
            normalized_mesh = load_mesh(output_mesh)
            stl_mesh = load_mesh(output_stl)
            self.assertAlmostEqual(float(raw_mesh.extents.max()), 2.0, places=4)
            self.assertAlmostEqual(float(normalized_mesh.extents.max()), 40.0, places=4)
            self.assertAlmostEqual(float(stl_mesh.extents.max()), 40.0, places=4)
            self.assertTrue(stl_mesh.is_watertight)
            self.assertGreater(float(stl_mesh.volume), 0.0)

    def test_content_addressed_cache_reuses_identical_mesh_for_repair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "Pixal3D"
            provider_dir.mkdir()
            input_image = root / "input.png"
            cache_dir = root / "provider_cache"
            Image.new("RGB", (12, 12), (80, 140, 110)).save(input_image)
            (provider_dir / "inference.py").write_text(
                "\n".join(
                    [
                        "import argparse",
                        "from pathlib import Path",
                        "import trimesh",
                        "parser = argparse.ArgumentParser()",
                        "parser.add_argument('--image', required=True)",
                        "parser.add_argument('--output', required=True)",
                        "args = parser.parse_args()",
                        "counter = Path(__file__).with_name('invocations.txt')",
                        "count = int(counter.read_text()) if counter.exists() else 0",
                        "counter.write_text(str(count + 1))",
                        "output = Path(args.output)",
                        "output.parent.mkdir(parents=True, exist_ok=True)",
                        "trimesh.creation.box(extents=(1.0, 0.75, 0.5)).export(output)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            first_argv = [
                "run_image_to_mesh_provider",
                "--provider",
                "pixal3d",
                "--provider-dir",
                str(provider_dir),
                "--input-image",
                str(input_image),
                "--output-mesh",
                str(root / "raw.glb"),
                "--output-stl",
                str(root / "raw.stl"),
                "--provider-python",
                sys.executable,
                "--provider-mesh-cache-dir",
                str(cache_dir),
                "--pixal3d-model-revision",
                DEFAULT_PIXAL3D_MODEL_REVISION,
                "--pixal3d-moge-revision",
                DEFAULT_PIXAL3D_MOGE_REVISION,
                "--pixal3d-dinov3-revision",
                DEFAULT_PIXAL3D_DINOV3_REVISION,
                "--pixal3d-rembg-revision",
                DEFAULT_PIXAL3D_REMBG_REVISION,
            ]
            first_stderr = io.StringIO()
            with patch.object(sys, "argv", first_argv), contextlib.redirect_stdout(
                io.StringIO()
            ), contextlib.redirect_stderr(first_stderr), patch.object(
                provider_module, "resolve_pixal3d_model_snapshots", return_value={}
            ):
                provider_module.main()

            second_argv = [
                "run_image_to_mesh_provider",
                "--provider",
                "pixal3d",
                "--provider-dir",
                str(provider_dir),
                "--input-image",
                str(input_image),
                "--output-mesh",
                str(root / "repaired.glb"),
                "--raw-output-mesh",
                str(root / "repaired_source.glb"),
                "--output-stl",
                str(root / "repaired.stl"),
                "--provider-python",
                sys.executable,
                "--provider-mesh-cache-dir",
                str(cache_dir),
                "--pixal3d-model-revision",
                DEFAULT_PIXAL3D_MODEL_REVISION,
                "--pixal3d-moge-revision",
                DEFAULT_PIXAL3D_MOGE_REVISION,
                "--pixal3d-dinov3-revision",
                DEFAULT_PIXAL3D_DINOV3_REVISION,
                "--pixal3d-rembg-revision",
                DEFAULT_PIXAL3D_REMBG_REVISION,
                "--mesh-repair",
                "printable",
            ]
            second_stderr = io.StringIO()
            with patch.object(sys, "argv", second_argv), contextlib.redirect_stdout(
                io.StringIO()
            ), contextlib.redirect_stderr(second_stderr):
                provider_module.main()

            self.assertEqual((provider_dir / "invocations.txt").read_text(), "1")
            self.assertIn('"status": "stored"', first_stderr.getvalue())
            self.assertIn('"status": "hit"', second_stderr.getvalue())
            self.assertEqual(load_mesh(root / "raw.glb").vertices.shape, load_mesh(root / "repaired_source.glb").vertices.shape)
            self.assertTrue((root / "raw.stl").exists())
            self.assertTrue((root / "repaired.stl").exists())
            self.assertEqual(len(list(cache_dir.glob("*.glb"))), 1)
            self.assertEqual(len(list(cache_dir.glob("*.json"))), 1)
            cache_metadata = json.loads(next(cache_dir.glob("*.json")).read_text(encoding="utf-8"))
            self.assertIn("inference.py", cache_metadata["provider_source_sha256"])
            self.assertIn("provider_environment", cache_metadata)
            self.assertEqual(
                cache_metadata["provider_models"],
                {
                    "pixal3d": {
                        "repo_id": DEFAULT_PIXAL3D_MODEL,
                        "revision": DEFAULT_PIXAL3D_MODEL_REVISION,
                    },
                    "moge": {
                        "repo_id": DEFAULT_PIXAL3D_MOGE_MODEL,
                        "revision": DEFAULT_PIXAL3D_MOGE_REVISION,
                    },
                    "dinov3": {
                        "repo_id": DEFAULT_PIXAL3D_DINOV3_MODEL,
                        "revision": DEFAULT_PIXAL3D_DINOV3_REVISION,
                    },
                    "rembg": {
                        "repo_id": DEFAULT_PIXAL3D_REMBG_MODEL,
                        "revision": DEFAULT_PIXAL3D_REMBG_REVISION,
                    },
                },
            )

    def test_pinned_snapshot_resolution_uses_local_layouts_and_offline_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshots = {}
            for name, required_file in (
                ("pixal3d", "pipeline.json"),
                ("moge", "model.pt"),
                ("dinov3", "config.json"),
                ("rembg", "config.json"),
            ):
                path = root / name
                path.mkdir()
                (path / required_file).write_text("fixture", encoding="utf-8")
                snapshots[name] = path
            by_repo = {
                DEFAULT_PIXAL3D_MODEL: snapshots["pixal3d"],
                DEFAULT_PIXAL3D_MOGE_MODEL: snapshots["moge"],
                DEFAULT_PIXAL3D_DINOV3_MODEL: snapshots["dinov3"],
                DEFAULT_PIXAL3D_REMBG_MODEL: snapshots["rembg"],
            }
            args = SimpleNamespace(
                pixal3d_model_path=DEFAULT_PIXAL3D_MODEL,
                pixal3d_model_revision=DEFAULT_PIXAL3D_MODEL_REVISION,
                pixal3d_moge_revision=DEFAULT_PIXAL3D_MOGE_REVISION,
                pixal3d_dinov3_revision=DEFAULT_PIXAL3D_DINOV3_REVISION,
                pixal3d_rembg_model=DEFAULT_PIXAL3D_REMBG_MODEL,
                pixal3d_rembg_revision=DEFAULT_PIXAL3D_REMBG_REVISION,
            )

            with patch(
                "huggingface_hub.snapshot_download",
                side_effect=lambda repo_id, revision: str(by_repo[repo_id]),
            ) as download, patch.dict(os.environ, {}, clear=False):
                resolved = provider_module.resolve_pixal3d_model_snapshots(args)
                self.assertEqual(Path(args.pixal3d_model_path), snapshots["pixal3d"].resolve())
                self.assertEqual(
                    Path(os.environ["PIXAL3D_MOGE_MODEL_PATH"]),
                    (snapshots["moge"] / "model.pt").resolve(),
                )
                self.assertEqual(Path(os.environ["PIXAL3D_DINOV3_MODEL_PATH"]), snapshots["dinov3"].resolve())
                self.assertEqual(Path(os.environ["PIXAL3D_REMBG_MODEL"]), snapshots["rembg"].resolve())
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

            self.assertEqual(set(resolved), {"pixal3d", "moge", "dinov3", "rembg"})
            self.assertEqual(download.call_count, 4)
            download.assert_any_call(
                repo_id=DEFAULT_PIXAL3D_MOGE_MODEL,
                revision=DEFAULT_PIXAL3D_MOGE_REVISION,
            )

    def test_each_pixal3d_revision_changes_cache_identity(self):
        base = SimpleNamespace(
            pixal3d_model_path=DEFAULT_PIXAL3D_MODEL,
            pixal3d_model_revision=DEFAULT_PIXAL3D_MODEL_REVISION,
            pixal3d_moge_revision=DEFAULT_PIXAL3D_MOGE_REVISION,
            pixal3d_dinov3_revision=DEFAULT_PIXAL3D_DINOV3_REVISION,
            pixal3d_rembg_model=DEFAULT_PIXAL3D_REMBG_MODEL,
            pixal3d_rembg_revision=DEFAULT_PIXAL3D_REMBG_REVISION,
        )
        baseline_key = provider_module.cli_provider_cache_key(
            {"provider_models": provider_module.pixal3d_provider_models(base)}
        )

        for field in (
            "pixal3d_model_revision",
            "pixal3d_moge_revision",
            "pixal3d_dinov3_revision",
            "pixal3d_rembg_revision",
        ):
            changed = SimpleNamespace(**vars(base))
            setattr(changed, field, "f" * 40)
            changed_key = provider_module.cli_provider_cache_key(
                {"provider_models": provider_module.pixal3d_provider_models(changed)}
            )
            self.assertNotEqual(changed_key, baseline_key, field)

    def test_partial_pixal3d_revision_set_is_rejected(self):
        args = SimpleNamespace(
            pixal3d_model_revision=DEFAULT_PIXAL3D_MODEL_REVISION,
            pixal3d_moge_revision=None,
            pixal3d_dinov3_revision=None,
            pixal3d_rembg_revision=None,
        )
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            provider_module.pixal3d_model_revisions_pinned(args)


if __name__ == "__main__":
    unittest.main()
