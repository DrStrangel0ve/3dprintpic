from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend.benchmark import run_image_to_mesh_provider as provider_module
from backend.benchmark.mesh_rendering import load_mesh
from backend.benchmark.preflight_image_to_mesh_providers import provider_preflight_row


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
            ]
            first_stderr = io.StringIO()
            with patch.object(sys, "argv", first_argv), contextlib.redirect_stdout(
                io.StringIO()
            ), contextlib.redirect_stderr(first_stderr):
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


if __name__ == "__main__":
    unittest.main()
