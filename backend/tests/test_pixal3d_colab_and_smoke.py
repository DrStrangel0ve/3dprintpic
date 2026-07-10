from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.benchmark.package_colab_inputs import package_inputs
from backend.benchmark.patch_pixal3d_sources import patch_pixal3d_sources
from backend.benchmark.pixal3d_models import (
    DEFAULT_PIXAL3D_DINOV3_REVISION,
    DEFAULT_PIXAL3D_MODEL_REVISION,
    DEFAULT_PIXAL3D_MOGE_REVISION,
    DEFAULT_PIXAL3D_REMBG_REVISION,
)
from backend.benchmark.run_stl_first_smoke import build_experiments, parse_args


class Pixal3DColabSetupTest(unittest.TestCase):
    def test_package_embeds_pinned_blackwell_setup_and_setup_only_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            full = dataset / "full.png"
            masked = dataset / "masked.png"
            mask = dataset / "mask.png"
            for path in (full, masked, mask):
                path.write_bytes(b"asset")
            manifest = dataset / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "id": "sample",
                        "full_image": str(full.relative_to(root)),
                        "masked_image": str(masked.relative_to(root)),
                        "mask": str(mask.relative_to(root)),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            archive = root / "pixal3d_bundle.tar.gz"

            report = package_inputs(
                manifest=manifest,
                output=archive,
                extract_root="/content/inputs/pixal3d",
                root=root,
                include_run_script=True,
                colab_archive_path="/content/pixal3d_bundle.tar.gz",
                run_name="g4_pixal3d_setup",
                modern_config="backend/benchmark/experiment_configs/pixal3d_smoke.json",
                skip_cache=True,
                eval_starts=[0],
                eval_limit=1,
                score_profile="stl-quality",
                candidate_method="pixal3d_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh",
                current_method="triposg_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh",
                colab_require_gpu_name_regex="RTX PRO 6000|Blackwell|G4",
                colab_min_gpu_memory_gb=90,
                include_pixal3d_setup=True,
                colab_env={"PIXAL3D_SETUP_ONLY": "1"},
            )
            with tarfile.open(archive, "r:gz") as tar:
                run_script = tar.extractfile("run_colab_eval.sh").read().decode("utf-8")

        self.assertTrue(report["include_pixal3d_setup"])
        self.assertEqual(report["colab_env"], {"PIXAL3D_SETUP_ONLY": "1"})
        self.assertEqual(
            report["pixal3d_model_snapshots"]["pixal3d"]["revision"],
            DEFAULT_PIXAL3D_MODEL_REVISION,
        )
        self.assertEqual(
            report["pixal3d_model_snapshots"]["rembg"]["revision"],
            DEFAULT_PIXAL3D_REMBG_REVISION,
        )
        self.assertLess(
            run_script.index("--query-gpu=name,memory.total"),
            run_script.index('PIXAL3D_DIR="${PIXAL3D_DIR:-/content/Pixal3D}"'),
        )
        self.assertIn("https://github.com/TencentARC/Pixal3D", run_script)
        self.assertIn("cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af", run_script)
        self.assertIn("07444410f1e33f402353b99d6ccd26bd31e469e8", run_script)
        self.assertIn("https://github.com/microsoft/TRELLIS.2", run_script)
        self.assertIn("75fbf0183001ed9876c8dbb35de6b68552ee08bd", run_script)
        self.assertIn("12289e1062f0603f2f0d0771b02e1395d247f26f", run_script)
        self.assertIn("6dd94a859c26ee8246888502eada3dd8ad85532e", run_script)
        self.assertIn("253ac4fcea7de5f396371124af597e6cc957bfae", run_script)
        self.assertIn('PIXAL3D_VENV="${PIXAL3D_VENV:-/content/pixal3d-venv}"', run_script)
        self.assertIn("python -m virtualenv --system-site-packages", run_script)
        self.assertIn('export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"', run_script)
        for revision in (
            DEFAULT_PIXAL3D_MODEL_REVISION,
            DEFAULT_PIXAL3D_MOGE_REVISION,
            DEFAULT_PIXAL3D_DINOV3_REVISION,
            DEFAULT_PIXAL3D_REMBG_REVISION,
        ):
            self.assertIn(revision, run_script)
        self.assertIn("snapshot_download(repo_id=repo_id, revision=revision)", run_script)
        self.assertNotIn("snapshot_download(repo_id=repo_id)\n", run_script)
        self.assertIn("source /tmp/pixal3d_model_paths.sh", run_script)
        self.assertIn("'PIXAL3D_MOGE_MODEL_PATH': resolved['moge'] / 'model.pt'", run_script)
        self.assertIn("HF_HUB_OFFLINE", run_script)
        self.assertIn("runtime_exports = {", run_script)
        self.assertIn("offline_exports = {", run_script)
        self.assertIn("for name, value in runtime_exports.items()", run_script)
        self.assertNotIn("for name, value in exports.items()", run_script)
        self.assertIn(
            'python -m backend.benchmark.patch_pixal3d_sources --pixal3d-dir "$PIXAL3D_DIR"',
            run_script,
        )
        self.assertIn("Pixal3D rembg model load verified", run_script)
        self.assertNotIn("briaai/RMBG-2.0", run_script)
        self.assertIn("nvcc --version", run_script)
        self.assertIn("release 12\\.8", run_script)
        self.assertIn("torch.version.cuda != '12.8'", run_script)
        self.assertIn("torch.cuda.get_device_capability(0)", run_script)
        self.assertIn('export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$PIXAL3D_TORCH_CUDA_ARCH}"', run_script)
        self.assertIn('test "$(git -C "$PIXAL3D_DIR" rev-parse HEAD)" = "$PIXAL3D_REF"', run_script)
        self.assertIn("/tmp/pixal3d_requirements_colab.txt", run_script)
        self.assertIn('git+https://github.com/microsoft/MoGe.git@$MOGE_REF', run_script)
        self.assertIn("natten==0.21.6+torch2110cu128", run_script)
        self.assertIn("https://whl.natten.org", run_script)
        self.assertIn("utils3d-0.0.2-py3-none-any.whl", run_script)
        self.assertIn('pip install -v --no-build-isolation "$NVDIFFRAST_DIR"', run_script)
        self.assertIn('pip install -v --no-build-isolation "$CUMESH_DIR"', run_script)
        self.assertIn('pip install -v --no-build-isolation "$FLEXGEMM_DIR"', run_script)
        self.assertIn('pip install -v --no-build-isolation "$TRELLIS2_DIR/o-voxel"', run_script)
        self.assertIn("Pixal3DImageTo3DPipeline", run_script)
        self.assertIn("torch.cuda.is_available()", run_script)
        self.assertIn("inference.py --help", run_script)
        self.assertIn("TencentARC/Pixal3D", run_script)
        self.assertIn("Ruicheng/moge-2-vitl", run_script)
        self.assertIn("camenduru/dinov3-vitl16-pretrain-lvd1689m", run_script)
        self.assertIn("ZhengPeng7/BiRefNet", run_script)
        self.assertIn("valeoai/NAF", run_script)
        self.assertIn("PIXAL3D_SETUP_ONLY", run_script)
        self.assertIn("Provider setup only requested; skipping benchmark stages", run_script)


class Pixal3DSourcePatchTest(unittest.TestCase):
    def test_official_source_adaptation_is_complete_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "Pixal3D"
            pipeline = root / "pixal3d" / "pipelines" / "pixal3d_image_to_3d.py"
            pipeline.parent.mkdir(parents=True)
            pipeline.write_text(
                "from typing import *\n"
                "class Pipeline:\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, args):\n"
                "        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])\n",
                encoding="utf-8",
            )
            inference = root / "inference.py"
            inference.write_text(
                "import os\n"
                "MOGE_MODEL_NAME = \"Ruicheng/moge-2-vitl\"\n"
                + "\n".join(
                    '    \"model_name\": \"camenduru/dinov3-vitl16-pretrain-lvd1689m\",'
                    for _ in range(4)
                )
                + "\n",
                encoding="utf-8",
            )

            first = patch_pixal3d_sources(root)
            second = patch_pixal3d_sources(root)
            pipeline_text = pipeline.read_text(encoding="utf-8")
            inference_text = inference.read_text(encoding="utf-8")

        self.assertEqual(first["pipeline_sha256"], second["pipeline_sha256"])
        self.assertEqual(first["inference_sha256"], second["inference_sha256"])
        self.assertIn("rembg_args['model_name'] = rembg_model_override", pipeline_text)
        self.assertIn("PIXAL3D_MOGE_MODEL_PATH", inference_text)
        self.assertEqual(inference_text.count("PIXAL3D_DINOV3_MODEL_PATH"), 4)


class Pixal3DSTLSmokeConfigTest(unittest.TestCase):
    def test_pixal3d_emits_separate_raw_and_repaired_stl_candidates(self):
        argv = [
            "run_stl_first_smoke",
            "--include-pixal3d",
            "--pixal3d-direct-input",
            "biharmonic",
            "--pixal3d-python",
            "/content/pixal3d-venv/bin/python",
            "--pixal3d-dir",
            "/content/Pixal3D",
            "--pixal3d-resolution",
            "1024",
            "--pixal3d-seed",
            "17",
            "--pixal3d-fov",
            "0.2",
            "--pixal3d-model-path",
            "TencentARC/Pixal3D",
            "--mesh-target-bbox-source",
            "inferred",
            "--mesh-target-faces",
            "250000",
            "--mesh-max-normalized-face-density-log1p",
            "9.95",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        experiments = {row["name"]: row for row in build_experiments(args)}
        raw = experiments["pixal3d_biharmonic_prefill_raw_direct_mesh"]
        repaired = experiments[
            "pixal3d_biharmonic_prefill_repaired_stl_inferred_bbox_direct_mesh"
        ]

        self.assertEqual(raw["direct_mesh_input"], "biharmonic")
        self.assertEqual(repaired["direct_mesh_input"], "biharmonic")
        self.assertNotIn("direct_mesh_bbox_source", raw)
        self.assertEqual(repaired["direct_mesh_bbox_source"], "inferred")
        self.assertEqual(repaired["direct_mesh_reference_method"], "mirror")
        self.assertFalse(repaired["oracle_diagnostic"])

        raw_command = raw["direct_mesh_command"]
        repaired_command = repaired["direct_mesh_command"]
        for command in (raw_command, repaired_command):
            self.assertIn("--provider pixal3d", command)
            self.assertIn("--pixal3d-resolution 1024", command)
            self.assertIn("--seed 17", command)
            self.assertIn("--pixal3d-fov 0.2", command)
            self.assertIn("--pixal3d-model-path TencentARC/Pixal3D", command)
            self.assertIn(f"--pixal3d-model-revision {DEFAULT_PIXAL3D_MODEL_REVISION}", command)
            self.assertIn(f"--pixal3d-moge-revision {DEFAULT_PIXAL3D_MOGE_REVISION}", command)
            self.assertIn(f"--pixal3d-dinov3-revision {DEFAULT_PIXAL3D_DINOV3_REVISION}", command)
            self.assertIn(f"--pixal3d-rembg-revision {DEFAULT_PIXAL3D_REMBG_REVISION}", command)
            self.assertIn("--provider-mesh-cache-dir /content/pixal3d-provider-cache", command)
        self.assertNotIn("--mesh-repair", raw_command)
        self.assertNotIn("--mesh-target-bbox-extents", raw_command)
        self.assertNotIn("--mesh-target-faces", raw_command)
        self.assertNotIn("--mesh-max-normalized-face-density-log1p", raw_command)
        self.assertIn("--mesh-repair printable", repaired_command)
        self.assertIn('--mesh-target-bbox-extents "{inferred_bbox_extents}"', repaired_command)
        self.assertIn("--mesh-target-faces 250000", repaired_command)
        self.assertIn("--mesh-max-normalized-face-density-log1p 9.95", repaired_command)
        self.assertIn('output_mesh_raw.glb', repaired_command)


if __name__ == "__main__":
    unittest.main()
