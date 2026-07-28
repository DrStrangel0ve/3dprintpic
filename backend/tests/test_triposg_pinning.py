from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.benchmark import run_image_to_mesh_provider as provider_module
from backend.benchmark.patch_triposg_sources import patch_triposg_sources
from backend.benchmark.preflight_image_to_mesh_providers import parse_provider_command
from backend.benchmark.run_stl_first_smoke import build_experiments, parse_args
from backend.benchmark.triposg_models import (
    DEFAULT_TRIPOSG_MODEL,
    DEFAULT_TRIPOSG_MODEL_REVISION,
    DEFAULT_TRIPOSG_REMBG_MODEL,
    DEFAULT_TRIPOSG_REMBG_REVISION,
)


class TripoSGPinningTest(unittest.TestCase):
    def test_local_launcher_avoids_installed_scripts_package_shadowing(self):
        args = SimpleNamespace(
            provider="triposg",
            python="provider-python",
            input_image=Path("input.png"),
            num_inference_steps=12,
            guidance_scale=5.5,
            seed=7,
            mesh_target_faces=1234,
            provider_arg=[],
        )
        provider_dir = Path("provider-repo")
        command = provider_module.cli_provider_command(
            args,
            provider_dir,
            Path("provider-output"),
        )
        env = provider_module.cli_provider_subprocess_env(args, provider_dir)

        self.assertEqual(command[0], "provider-python")
        self.assertEqual(
            command[1],
            str(provider_dir / "scripts" / "inference_triposg.py"),
        )
        self.assertNotIn("scripts.inference_triposg", command)
        self.assertIsNotNone(env)
        self.assertEqual(
            env["PYTHONPATH"].split(os.pathsep)[0],
            str(provider_dir.resolve()),
        )

    def test_official_source_snapshot_calls_are_pinned_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "TripoSG"
            inference = root / "scripts" / "inference_triposg.py"
            inference.parent.mkdir(parents=True)
            inference.write_text(
                "import os\n"
                "from huggingface_hub import snapshot_download\n"
                "triposg_weights_dir = 'pretrained_weights/TripoSG'\n"
                "rmbg_weights_dir = 'pretrained_weights/RMBG-1.4'\n"
                f'snapshot_download(repo_id="{DEFAULT_TRIPOSG_MODEL}", local_dir=triposg_weights_dir)\n'
                f'snapshot_download(repo_id="{DEFAULT_TRIPOSG_REMBG_MODEL}", local_dir=rmbg_weights_dir)\n',
                encoding="utf-8",
            )

            first = patch_triposg_sources(root)
            second = patch_triposg_sources(root)
            text = inference.read_text(encoding="utf-8")

        self.assertEqual(first["inference_sha256"], second["inference_sha256"])
        self.assertIn("TRIPOSG_MODEL_REVISION", text)
        self.assertIn("TRIPOSG_REMBG_REVISION", text)
        self.assertEqual(text.count("local_files_only"), 2)

    def test_smoke_command_and_preflight_preserve_both_revisions(self):
        argv = [
            "run_stl_first_smoke",
            "--include-triposg",
            "--triposg-direct-input",
            "biharmonic",
            "--mesh-target-bbox-source",
            "inferred",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        experiments = {row["name"]: row for row in build_experiments(args)}
        experiment = experiments[
            "triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh"
        ]
        command = experiment["direct_mesh_command"]
        parsed = parse_provider_command(command)

        self.assertIn(f"--triposg-model-revision {DEFAULT_TRIPOSG_MODEL_REVISION}", command)
        self.assertIn(f"--triposg-rembg-revision {DEFAULT_TRIPOSG_REMBG_REVISION}", command)
        self.assertIn("--seed 42", command)
        self.assertIn(
            "--provider-mesh-cache-dir /content/triposg-provider-cache",
            command,
        )
        self.assertEqual(
            parsed["provider_models"],
            {
                "triposg": {
                    "repo_id": DEFAULT_TRIPOSG_MODEL,
                    "revision": DEFAULT_TRIPOSG_MODEL_REVISION,
                },
                "rembg": {
                    "repo_id": DEFAULT_TRIPOSG_REMBG_MODEL,
                    "revision": DEFAULT_TRIPOSG_REMBG_REVISION,
                },
            },
        )

    def test_revisions_change_cache_identity_and_partial_sets_fail(self):
        base = SimpleNamespace(
            triposg_model_revision=DEFAULT_TRIPOSG_MODEL_REVISION,
            triposg_rembg_revision=DEFAULT_TRIPOSG_REMBG_REVISION,
        )
        baseline_key = provider_module.cli_provider_cache_key(
            {"provider_models": provider_module.triposg_provider_models(base)}
        )
        changed = SimpleNamespace(**vars(base))
        changed.triposg_model_revision = "f" * 40
        changed_key = provider_module.cli_provider_cache_key(
            {"provider_models": provider_module.triposg_provider_models(changed)}
        )
        self.assertNotEqual(changed_key, baseline_key)

        partial = SimpleNamespace(
            triposg_model_revision=DEFAULT_TRIPOSG_MODEL_REVISION,
            triposg_rembg_revision=None,
        )
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            provider_module.triposg_model_revisions_pinned(partial)

    def test_comparison_configs_pin_seed_and_raw_mesh_cache(self):
        config_root = Path(__file__).parents[1] / "benchmark" / "experiment_configs"
        for filename in (
            "modelnet10_60_balanced_stl_quality_pixal3d_triposg_s40_n1.json",
            "modelnet10_60_balanced_stl_quality_trellis2_triposg_s40_n1.json",
        ):
            experiments = json.loads(
                (config_root / filename).read_text(encoding="utf-8")
            )
            triposg = next(
                experiment
                for experiment in experiments
                if experiment["name"]
                == "triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh"
            )
            command = triposg["direct_mesh_command"]
            self.assertIn("--seed 42", command, filename)
            self.assertIn(
                "--provider-mesh-cache-dir /content/triposg-provider-cache",
                command,
                filename,
            )


if __name__ == "__main__":
    unittest.main()
