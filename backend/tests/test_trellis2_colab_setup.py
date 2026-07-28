from __future__ import annotations

import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.benchmark import preflight_image_to_mesh_providers as preflight_module
from backend.benchmark.package_colab_inputs import (
    DEFAULT_TRELLIS2_COLAB_PYTHON,
    TRELLIS2_XFORMERS_VERSION,
    package_inputs,
)
from backend.benchmark.trellis2_models import (
    DEFAULT_TRELLIS2_MODEL,
    DEFAULT_TRELLIS2_MODEL_REVISION,
    DEFAULT_TRELLIS2_SOURCE_REVISION,
)


class Trellis2ColabSetupTest(unittest.TestCase):
    def test_package_embeds_dedicated_pinned_setup_and_preflight(self):
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
            archive = root / "trellis2_bundle.tar.gz"

            report = package_inputs(
                manifest=manifest,
                output=archive,
                extract_root="/content/inputs/trellis2",
                root=root,
                include_run_script=True,
                colab_archive_path="/content/trellis2_bundle.tar.gz",
                run_name="g4_trellis2_setup",
                modern_config=(
                    "backend/benchmark/experiment_configs/"
                    "modelnet10_60_balanced_stl_quality_trellis2_triposg_s40_n1.json"
                ),
                skip_cache=True,
                eval_starts=[40],
                eval_limit=1,
                score_profile="stl-quality",
                include_trellis2_setup=True,
                colab_env={"TRELLIS2_SETUP_ONLY": "1"},
            )
            with tarfile.open(archive, "r:gz") as tar:
                run_script = tar.extractfile("run_colab_eval.sh").read().decode(
                    "utf-8"
                )

        self.assertTrue(report["include_trellis2_setup"])
        self.assertEqual(report["trellis2_source_revision"], DEFAULT_TRELLIS2_SOURCE_REVISION)
        self.assertEqual(report["trellis2_attention_backend"], "xformers")
        self.assertEqual(report["trellis2_python"], DEFAULT_TRELLIS2_COLAB_PYTHON)
        self.assertEqual(
            report["trellis2_model_snapshots"]["trellis2"],
            {
                "repo_id": DEFAULT_TRELLIS2_MODEL,
                "revision": DEFAULT_TRELLIS2_MODEL_REVISION,
            },
        )
        self.assertLess(
            run_script.index("--query-gpu=name,memory.total"),
            run_script.index('TRELLIS2_DIR="${TRELLIS2_DIR:-/content/TRELLIS.2}"'),
        )
        self.assertIn(
            'TRELLIS2_VENV="${TRELLIS2_VENV:-/content/trellis2-venv}"',
            run_script,
        )
        self.assertIn(DEFAULT_TRELLIS2_SOURCE_REVISION, run_script)
        self.assertIn(DEFAULT_TRELLIS2_MODEL_REVISION, run_script)
        self.assertIn(
            f'"xformers==$TRELLIS2_XFORMERS_VERSION"',
            run_script,
        )
        self.assertIn(
            f'TRELLIS2_XFORMERS_VERSION="${{TRELLIS2_XFORMERS_VERSION:-{TRELLIS2_XFORMERS_VERSION}}}"',
            run_script,
        )
        self.assertIn('export ATTN_BACKEND="$TRELLIS2_ATTN_BACKEND"', run_script)
        self.assertIn(
            'export SPARSE_ATTN_BACKEND="$TRELLIS2_ATTN_BACKEND"', run_script
        )
        self.assertNotIn('TRELLIS2_ATTN_BACKEND="${TRELLIS2_ATTN_BACKEND:-sdpa}"', run_script)
        self.assertIn("from trellis2.pipelines import Trellis2ImageTo3DPipeline", run_script)
        self.assertIn("xops.fmha.BlockDiagonalMask.from_seqlens", run_script)
        self.assertIn("xops.memory_efficient_attention", run_script)
        self.assertLess(
            run_script.index("provider preflight passed"),
            run_script.index("--prefetch-only"),
        )
        self.assertIn("TRELLIS2_SETUP_ONLY", run_script)
        self.assertIn("trellis2_provider_preflight.json", run_script)
        self.assertIn('TRELLIS2_HAD_ATTN_BACKEND="${ATTN_BACKEND+x}"', run_script)
        self.assertIn(
            'TRELLIS2_HAD_SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND+x}"',
            run_script,
        )
        self.assertIn("caller attention environment restored", run_script)

    def test_probe_uses_provider_python_and_rejects_sdpa_for_sparse_attention(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            provider_dir = Path(temp_dir)
            files = {
                "trellis2/__init__.py": "",
                "trellis2/pipelines/__init__.py": (
                    "class Trellis2ImageTo3DPipeline:\n"
                    "    pass\n"
                ),
                "trellis2/modules/__init__.py": "",
                "trellis2/modules/sparse/__init__.py": "",
                "trellis2/modules/sparse/config.py": (
                    "import os\n"
                    "accepted = ('xformers', 'flash_attn', 'flash_attn_3')\n"
                    "requested = os.environ.get('SPARSE_ATTN_BACKEND') or os.environ.get('ATTN_BACKEND')\n"
                    "ATTN = requested if requested in accepted else 'flash_attn'\n"
                ),
                "xformers/__init__.py": "",
                "xformers/ops.py": (
                    "def memory_efficient_attention(*args):\n"
                    "    return args[0]\n"
                    "class BlockDiagonalMask:\n"
                    "    @classmethod\n"
                    "    def from_seqlens(cls, lengths):\n"
                    "        return cls()\n"
                    "class fmha:\n"
                    "    pass\n"
                    "fmha.BlockDiagonalMask = BlockDiagonalMask\n"
                ),
            }
            for relative, content in files.items():
                path = provider_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            ready = preflight_module.probe_trellis2_provider_python(
                sys.executable,
                provider_dir,
            )
            invalid = preflight_module.probe_trellis2_provider_python(
                sys.executable,
                provider_dir,
                attention_backend="sdpa",
            )

        self.assertTrue(ready["pipelines_importable"])
        self.assertTrue(ready["pipeline_class_importable"])
        self.assertEqual(ready["attention_backend"], "xformers")
        self.assertEqual(ready["attention_backend_module"], "xformers.ops")
        self.assertTrue(ready["attention_backend_api_ready"])
        self.assertTrue(ready["backend_ready"])
        self.assertEqual(invalid["attention_backend_requested"], "sdpa")
        self.assertEqual(invalid["attention_backend"], "flash_attn")
        self.assertFalse(invalid["attention_backend_accepted"])
        self.assertFalse(invalid["backend_ready"])

    def test_provider_preflight_reports_pipeline_and_backend_readiness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider_dir = root / "TRELLIS.2"
            entrypoint = (
                provider_dir / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
            )
            entrypoint.parent.mkdir(parents=True)
            entrypoint.write_text("# fixture\n", encoding="utf-8")
            (provider_dir / "trellis2" / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            (provider_dir / ".git").mkdir()
            provider_python = root / "trellis2-python"
            provider_python.write_text("", encoding="utf-8")
            command = (
                f"{provider_python.as_posix()} -m "
                "backend.benchmark.run_image_to_mesh_provider --provider trellis2 "
                f"--provider-dir {provider_dir.as_posix()} "
                f"--trellis2-model-path {DEFAULT_TRELLIS2_MODEL} "
                f"--trellis2-model-revision {DEFAULT_TRELLIS2_MODEL_REVISION} "
                "--trellis2-resolution 512"
            )
            parsed = preflight_module.parse_provider_command(command)
            ready_probe = {
                "python_executable": str(provider_python),
                "returncode": 0,
                "pipelines_importable": True,
                "pipeline_class_importable": True,
                "attention_backend_requested": "xformers",
                "attention_backend": "xformers",
                "attention_backend_accepted": True,
                "attention_backend_module": "xformers.ops",
                "attention_backend_importable": True,
                "attention_backend_api_ready": True,
                "backend_ready": True,
                "error": "",
            }
            with patch.object(
                preflight_module,
                "provider_git_revision",
                return_value=DEFAULT_TRELLIS2_SOURCE_REVISION,
            ), patch.object(
                preflight_module,
                "probe_trellis2_provider_python",
                return_value=ready_probe,
            ):
                row = preflight_module.provider_preflight_row(parsed)

            invalid_probe = dict(ready_probe)
            invalid_probe.update(
                {
                    "returncode": 1,
                    "attention_backend_requested": "sdpa",
                    "attention_backend": "flash_attn",
                    "attention_backend_accepted": False,
                    "attention_backend_module": "flash_attn",
                    "attention_backend_importable": False,
                    "backend_ready": False,
                }
            )
            with patch.object(
                preflight_module,
                "provider_git_revision",
                return_value=DEFAULT_TRELLIS2_SOURCE_REVISION,
            ), patch.object(
                preflight_module,
                "probe_trellis2_provider_python",
                return_value=invalid_probe,
            ):
                invalid_row = preflight_module.provider_preflight_row(parsed)

        self.assertTrue(row["runnable"])
        self.assertTrue(row["checks"]["trellis2_pipelines_importable"])
        self.assertEqual(
            row["checks"]["trellis2_attention_backend"], "xformers"
        )
        self.assertTrue(row["checks"]["trellis2_attention_backend_ready"])
        self.assertFalse(invalid_row["runnable"])
        self.assertTrue(
            any("xformers, flash_attn, flash_attn_3" in error for error in invalid_row["setup_errors"])
        )

    def test_n1_config_uses_only_the_dedicated_trellis2_python(self):
        config_path = (
            Path(__file__).parents[1]
            / "benchmark"
            / "experiment_configs"
            / "modelnet10_60_balanced_stl_quality_trellis2_triposg_s40_n1.json"
        )
        experiments = json.loads(config_path.read_text(encoding="utf-8"))
        commands = [
            experiment["direct_mesh_command"]
            for experiment in experiments
            if experiment.get("direct_mesh_command")
            and "--provider trellis2" in experiment["direct_mesh_command"]
        ]

        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertTrue(command.startswith(DEFAULT_TRELLIS2_COLAB_PYTHON + " "))
            self.assertNotIn("/content/pixal3d-venv/", command)
            self.assertIn(f"--trellis2-model-path {DEFAULT_TRELLIS2_MODEL}", command)
            self.assertIn(
                f"--trellis2-model-revision {DEFAULT_TRELLIS2_MODEL_REVISION}",
                command,
            )

    def test_component_close_config_is_cached_bounded_and_hull_free(self):
        config_path = (
            Path(__file__).parents[1]
            / "benchmark"
            / "experiment_configs"
            / "modelnet10_60_balanced_stl_quality_trellis2_component_close_s40_n5.json"
        )
        experiments = json.loads(config_path.read_text(encoding="utf-8"))
        by_name = {experiment["name"]: experiment for experiment in experiments}
        candidate_name = (
            "trellis2_biharmonic_prefill_component_close_d995_no_hull_"
            "stl_inferred_bbox_direct_mesh"
        )

        self.assertEqual(len(experiments), 6)
        self.assertEqual(set(by_name) & {"masked", "mirror"}, {"masked", "mirror"})
        self.assertNotIn("source_mesh_oracle", by_name)
        candidate = by_name[candidate_name]
        command = candidate["direct_mesh_command"]
        self.assertIn("--mesh-repair basic", command)
        self.assertIn("--mesh-repair-preconditioner component-close", command)
        self.assertIn("--mesh-repair-component-area-ratio 0.01", command)
        self.assertIn("--mesh-repair-hole-face-addition-ratio 0.02", command)
        self.assertIn("--mesh-repair-simplify-placement optimal", command)
        self.assertIn("--mesh-max-normalized-face-density-log1p 9.95", command)
        self.assertNotIn("--mesh-repair printable", command)
        self.assertNotIn("{source_bbox", command)
        self.assertEqual(candidate["direct_mesh_bbox_source"], "inferred")
        self.assertFalse(candidate["oracle_diagnostic"])

        trellis_commands = [
            experiment["direct_mesh_command"]
            for experiment in experiments
            if "--provider trellis2" in experiment.get("direct_mesh_command", "")
        ]
        self.assertEqual(len(trellis_commands), 3)
        for trellis_command in trellis_commands:
            self.assertIn("--provider-mesh-cache-dir /content/trellis2-provider-cache", trellis_command)
            self.assertIn(f"--trellis2-model-path {DEFAULT_TRELLIS2_MODEL}", trellis_command)
            self.assertIn(
                f"--trellis2-model-revision {DEFAULT_TRELLIS2_MODEL_REVISION}",
                trellis_command,
            )
            self.assertIn("--seed 42", trellis_command)


if __name__ == "__main__":
    unittest.main()
