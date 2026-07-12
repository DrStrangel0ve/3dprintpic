from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from backend.benchmark.rank_methods import SCORE_PROFILES


DEFAULT_COLAB_NOTEBOOK_URL = "https://colab.research.google.com/drive/1SuilhFuF5L3ELkEy2rnEsTmKAL19ob60"
DEFAULT_REPO_URL = "https://github.com/jennyzzt/3dprintpic.git"
DEFAULT_PROMPT = (
    "Complete the missing half of the same {category} with matching geometry, lighting, "
    "viewpoint, and background. The masked half must contain the missing {category}, "
    "not an empty background."
)
DEFAULT_MODERN_CONFIG = "backend/benchmark/experiment_configs/modelnet10_60_balanced_modern_sdxl_depth_stl.json"
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_CACHE_PROVIDERS = ("sdxl-inpaint", "dreamshaper-inpaint", "amused-inpaint")
ALL_STAGES = ("setup", "dataset", "cache", "calibrate", "weight", "train", "eval", "combine")
LORA_ADAPTER_FILES = ("pytorch_lora_weights.safetensors", "adapter_model.safetensors", "adapter_model.bin")


class CommandLogger:
    def __init__(self, path: Path, *, dry_run: bool = False) -> None:
        self.path = path
        self.dry_run = dry_run
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

    def run(self, command: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
        printable = shlex.join(str(part) for part in command)
        print(f"\n$ {printable}", flush=True)
        started = time.time()
        row = {
            "command": list(map(str, command)),
            "cwd": str(cwd) if cwd else "",
            "started_at": started,
            "dry_run": self.dry_run,
        }
        if self.dry_run:
            row.update({"returncode": 0, "duration_seconds": 0.0})
            self.write(row)
            return
        completed = subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=False)
        row.update({"returncode": completed.returncode, "duration_seconds": round(time.time() - started, 3)})
        self.write(row)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)


def here_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def python_cmd(args: Iterable[str], python: str) -> list[str]:
    return [python, *args]


def repo_python(repo_root: Path, args: Iterable[str], python: str) -> list[str]:
    return python_cmd(["-m", *args], python)


def relative_to_repo(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def require_existing_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_report(path: Path) -> dict:
    report = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
    }
    if path.is_file():
        report.update(
            {
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return report


def git_state(repo_root: Path) -> dict:
    state = {"repo_root": str(repo_root)}
    commands = {
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "remote_origin": ["git", "remote", "get-url", "origin"],
        "status_short": ["git", "status", "--short"],
    }
    for key, command in commands.items():
        try:
            state[key] = subprocess.check_output(command, cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception as exc:
            state[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
    state["dirty"] = bool(state.get("status_short"))
    return state


def validate_lora_weights(path: Path) -> Path:
    require_existing_file(path, "--existing-lora-weights")
    if path.is_file():
        if path.suffix in {".safetensors", ".bin"}:
            return path
        raise ValueError(f"--existing-lora-weights file must be a safetensors or bin adapter: {path}")
    for filename in LORA_ADAPTER_FILES:
        if (path / filename).exists():
            return path
    expected = ", ".join(LORA_ADAPTER_FILES)
    raise FileNotFoundError(f"--existing-lora-weights directory has no adapter file ({expected}): {path}")


def stable_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return label.strip("._-") or "adapter"


def lora_experiment_label(args: argparse.Namespace, lora_weights: Path) -> str:
    if args.existing_lora_weights:
        label = lora_weights.stem if lora_weights.is_file() else lora_weights.name
        if label.startswith("weighted_"):
            label = label[len("weighted_") :]
        return stable_label(label)
    return stable_label(f"{args.score_profile}_s{args.train_steps}")


def stage_enabled(args: argparse.Namespace, stage: str) -> bool:
    stages = set(args.stage or ["all"])
    return "all" in stages or stage in stages


def needs_training_pairs(args: argparse.Namespace) -> bool:
    if stage_enabled(args, "weight"):
        return True
    return stage_enabled(args, "train") and not args.existing_lora_weights


def resolve_repo(args: argparse.Namespace, logger: CommandLogger) -> Path:
    if args.use_current_repo:
        return here_repo_root()

    workdir = Path(args.workdir)
    if not workdir.exists():
        logger.run(["git", "clone", args.repo_url, str(workdir.parent / workdir.name)])
    else:
        logger.run(["git", "fetch", "--all", "--tags"], cwd=workdir)

    if args.repo_ref:
        logger.run(["git", "checkout", args.repo_ref], cwd=workdir)
    return workdir


def write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gpu_probe(logger: CommandLogger) -> dict:
    probe = {
        "notebook_url": DEFAULT_COLAB_NOTEBOOK_URL,
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    try:
        import torch

        probe.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            }
        )
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            probe.update(
                {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                    "vram_total_gb": round(total_bytes / 1024**3, 3),
                    "vram_free_gb": round(free_bytes / 1024**3, 3),
                }
            )
    except Exception as exc:
        probe.update({"torch_probe_error": f"{type(exc).__name__}: {exc}"})

    try:
        logger.run(["nvidia-smi"])
    except Exception as exc:
        probe["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return probe


def install_requirements(args: argparse.Namespace, repo_root: Path, logger: CommandLogger) -> None:
    if args.skip_install:
        return
    logger.run([args.python, "-m", "pip", "install", "-U", "pip"], cwd=repo_root)
    requirements = repo_root / "backend" / "requirements-cuda.txt"
    if not requirements.exists():
        requirements = repo_root / "backend" / "requirements.txt"
    logger.run([args.python, "-m", "pip", "install", "-r", str(requirements)], cwd=repo_root)


def ensure_dataset(args: argparse.Namespace, repo_root: Path, output_root: Path, logger: CommandLogger) -> Path:
    if args.manifest:
        manifest = relative_to_repo(repo_root, args.manifest)
        if manifest.exists():
            return manifest
        if args.dataset_source == "existing":
            raise FileNotFoundError(f"--manifest does not exist: {manifest}")

    manifest = output_root / "dataset" / "manifest.jsonl"
    if manifest.exists() and not args.force_dataset:
        return manifest

    if args.dataset_source == "existing":
        raise FileNotFoundError(
            "No manifest was provided and the default Colab dataset is missing. "
            "Use --dataset-source procedural or --dataset-source mesh-dir."
        )

    command = repo_python(
        repo_root,
        [
            "backend.benchmark.generate_rendered_dataset",
            "--output-dir",
            str(manifest.parent),
            "--source",
            args.dataset_source,
            "--count",
            str(args.dataset_count),
            "--size",
            str(args.size),
            "--seed",
            str(args.seed),
        ],
        args.python,
    )
    if args.dataset_source == "mesh-dir":
        if not args.asset_root:
            raise ValueError("--asset-root is required for --dataset-source mesh-dir")
        command.extend(
            [
                "--asset-root",
                args.asset_root,
                "--asset-glob",
                args.asset_glob,
                "--views-per-asset",
                str(args.views_per_asset),
                "--mesh-sample-strategy",
                args.mesh_sample_strategy,
                "--continue-on-error",
            ]
        )
    logger.run(command, cwd=repo_root)
    return manifest


def cache_modern_providers(args: argparse.Namespace, repo_root: Path, output_root: Path, logger: CommandLogger) -> None:
    if args.skip_cache:
        return
    cache_dir = output_root / "cache_plans"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for provider in args.cache_provider:
        command = repo_python(
            repo_root,
            [
                "backend.benchmark.cache_provider",
                provider,
                "--download",
                "--download-mode",
                args.cache_download_mode,
                "--max-workers",
                str(args.cache_max_workers),
                "--output",
                str(cache_dir / f"{provider}.json"),
            ],
            args.python,
        )
        if args.cache_full:
            command.append("--full")
        logger.run(command, cwd=repo_root)


def export_training_pairs(
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    manifest: Path,
    logger: CommandLogger,
) -> Path:
    pair_dir = output_root / "training_pairs" / f"train_s{args.train_start}_n{args.train_limit}"
    metadata = pair_dir / "metadata.jsonl"
    if metadata.exists() and not args.force_pairs:
        return metadata
    logger.run(
        repo_python(
            repo_root,
            [
                "backend.benchmark.export_training_pairs",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(pair_dir),
                "--start-index",
                str(args.train_start),
                "--limit",
                str(args.train_limit),
                "--prompt",
                args.prompt,
            ],
            args.python,
        ),
        cwd=repo_root,
    )
    return metadata


def run_calibration(
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    manifest: Path,
    logger: CommandLogger,
) -> Path:
    calibration_dir = output_root / "runs" / f"mirror_calibration_s{args.calibration_start}_n{args.calibration_limit}"
    logger.run(
        repo_python(
            repo_root,
            [
                "backend.benchmark.run_completion_benchmark",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(calibration_dir),
                "--methods",
                "mirror",
                "--start-index",
                str(args.calibration_start),
                "--limit",
                str(args.calibration_limit),
                "--depth-provider",
                args.depth_provider,
                "--depth-model",
                args.depth_model,
                "--device",
                "auto",
                "--emit-stl",
                "--stl-target-dimension",
                str(args.stl_target_dimension),
                "--resume",
                "--continue-on-error",
            ],
            args.python,
        ),
        cwd=repo_root,
    )
    logger.run(
        repo_python(
            repo_root,
            [
                "backend.benchmark.report_run",
                str(calibration_dir),
                "--baseline-method",
                "mirror",
                "--score-profile",
                args.score_profile,
                "--output",
                str(calibration_dir / "benchmark_report.md"),
            ],
            args.python,
        ),
        cwd=repo_root,
    )
    return calibration_dir


def weight_pairs(
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    metadata: Path,
    calibration_dir: Path,
    logger: CommandLogger,
) -> Path:
    weighted = metadata.with_name(f"{metadata.stem}_weighted_mirror_{args.score_profile}.jsonl")
    logger.run(
        repo_python(
            repo_root,
            [
                "backend.benchmark.weight_training_pairs",
                "--metadata",
                str(metadata),
                "--benchmark-dir",
                str(calibration_dir),
                "--method",
                "mirror",
                "--score-profile",
                args.score_profile,
                "--base-weight",
                str(args.base_weight),
                "--scale",
                str(args.weight_scale),
                "--max-weight",
                str(args.max_weight),
                "--output",
                str(weighted),
                "--report",
                str(weighted.with_suffix(".report.json")),
                "--details-csv",
                str(weighted.with_suffix(".details.csv")),
            ],
            args.python,
        ),
        cwd=repo_root,
    )
    return weighted


def train_weighted_lora(
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    metadata: Path | None,
    logger: CommandLogger,
) -> Path:
    if args.existing_lora_weights:
        return validate_lora_weights(relative_to_repo(repo_root, args.existing_lora_weights))
    if metadata is None:
        raise ValueError("The train stage requires exported metadata unless --existing-lora-weights is supplied.")

    lora_dir = output_root / "lora" / f"weighted_{args.score_profile}_s{args.train_steps}"
    logger.run(
        repo_python(
            repo_root,
            [
                "backend.benchmark.train_inpainting_lora",
                "--metadata",
                str(metadata),
                "--output-dir",
                str(lora_dir),
                "--base-model",
                args.base_model,
                "--variant",
                args.variant,
                "--resolution",
                str(args.train_resolution),
                "--train-batch-size",
                str(args.train_batch_size),
                "--gradient-accumulation-steps",
                str(args.gradient_accumulation_steps),
                "--max-train-steps",
                str(args.train_steps),
                "--learning-rate",
                str(args.learning_rate),
                "--mask-loss-weight",
                str(args.mask_loss_weight),
                "--seam-loss-weight",
                str(args.seam_loss_weight),
                "--object-loss-weight",
                str(args.object_loss_weight),
                "--sample-weight-field",
                "sample_weight",
                "--mixed-precision",
                args.mixed_precision,
                "--num-workers",
                str(args.num_workers),
                "--allow-tf32",
            ],
            args.python,
        ),
        cwd=repo_root,
    )
    return lora_dir


def load_config(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Experiment config must be a JSON list: {path}")
    return payload


def write_eval_config(
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    lora_weights: Path | None,
) -> Path:
    config_path = relative_to_repo(repo_root, args.modern_config)
    experiments = load_config(config_path)
    if lora_weights:
        experiments.append(
            {
                "name": f"dreamshaper_weighted_lora_{lora_experiment_label(args, lora_weights)}_scale{args.lora_scale:g}",
                "method": "dreamshaper-inpaint",
                "model_name": args.base_model,
                "lora_weights": str(lora_weights),
                "lora_scale": args.lora_scale,
                "prompt": args.prompt,
                "steps": args.eval_steps,
                "guidance": args.eval_guidance,
                "seed": args.seed,
                "inpaint_max_dimension": args.eval_inpaint_max_dimension,
            }
        )
    rendered = output_root / "configs" / "modern_plus_weighted_lora.json"
    rendered.parent.mkdir(parents=True, exist_ok=True)
    rendered.write_text(json.dumps(experiments, indent=2) + "\n", encoding="utf-8")
    return rendered


def run_eval_slices(
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    manifest: Path,
    config: Path,
    logger: CommandLogger,
) -> list[Path]:
    eval_dirs = []
    for start in args.eval_start:
        eval_dir = output_root / "experiments" / f"modern_weighted_eval_s{start}_n{args.eval_limit}"
        command = repo_python(
            repo_root,
            [
                "backend.benchmark.optimize_completion",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(eval_dir),
                "--config",
                str(config),
                "--start-index",
                str(start),
                "--limit",
                str(args.eval_limit),
                "--depth-provider",
                args.depth_provider,
                "--depth-model",
                args.depth_model,
                "--device",
                "auto",
                "--emit-stl",
                "--stl-target-dimension",
                str(args.stl_target_dimension),
                "--score-mode",
                "baseline-delta",
                "--score-profile",
                args.score_profile,
                "--baseline-method",
                "masked",
                "--select-candidate",
                "--min-paired-n",
                str(args.min_paired_n),
                "--max-mesh-surface-chamfer-ratio-vs-current",
                str(getattr(args, "max_mesh_surface_chamfer_ratio_vs_current", 1.1)),
                "--max-mesh-surface-hausdorff95-ratio-vs-current",
                str(getattr(args, "max_mesh_surface_hausdorff95_ratio_vs_current", 1.1)),
                "--max-method-failures",
                str(args.max_method_failures),
                "--contact-sheet",
                "--contact-sheet-max-samples",
                str(args.contact_sheet_max_samples),
                "--resume",
                "--continue-on-error",
            ],
            args.python,
        )
        if args.candidate_method:
            command.extend(["--candidate-method", args.candidate_method])
        if args.current_method:
            command.extend(["--current-method", args.current_method])
        if args.contact_sheet_methods:
            command.extend(["--contact-sheet-methods", args.contact_sheet_methods])
        if args.require_modern_cache:
            command.append("--require-modern-cache")
        if args.require_image_to_mesh_providers:
            command.append("--require-image-to-mesh-providers")
        if args.allow_missing_split_audit:
            command.append("--allow-missing-split-audit")
        logger.run(command, cwd=repo_root)
        eval_dirs.append(eval_dir)
    return eval_dirs


def combine_eval_slices(
    args: argparse.Namespace,
    repo_root: Path,
    output_root: Path,
    eval_dirs: list[Path],
    logger: CommandLogger,
) -> Path | None:
    if len(eval_dirs) < 2:
        return None
    combined_dir = output_root / "combined" / f"modern_weighted_{args.score_profile}_n{len(eval_dirs) * args.eval_limit}"
    command = repo_python(
        repo_root,
        [
            "backend.benchmark.combine_optimize_runs",
            "--output-dir",
            str(combined_dir),
            "--score-mode",
            "baseline-delta",
            "--score-profile",
            args.score_profile,
            "--baseline-method",
            "masked",
            "--select-candidate",
            "--min-paired-n",
            str(args.min_paired_n),
            "--max-mesh-surface-chamfer-ratio-vs-current",
            str(getattr(args, "max_mesh_surface_chamfer_ratio_vs_current", 1.1)),
            "--max-mesh-surface-hausdorff95-ratio-vs-current",
            str(getattr(args, "max_mesh_surface_hausdorff95_ratio_vs_current", 1.1)),
        ],
        args.python,
    )
    if args.candidate_method:
        command.extend(["--candidate-method", args.candidate_method])
    if args.current_method:
        command.extend(["--current-method", args.current_method])
    for eval_dir in eval_dirs:
        command.extend(["--run", f"{eval_dir.name}={eval_dir}"])
    logger.run(command, cwd=repo_root)
    return combined_dir


def expected_eval_dirs(args: argparse.Namespace, output_root: Path) -> list[Path]:
    eval_dirs = [output_root / "experiments" / f"modern_weighted_eval_s{start}_n{args.eval_limit}" for start in args.eval_start]
    missing = [path for path in eval_dirs if not (path / "aggregate_summary.csv").exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Cannot combine without completed eval dirs: {missing_text}")
    return eval_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 3dprintpic completion-to-depth-to-STL experiment loop on a single Colab G4 runtime. "
            "The default stages cache modern inpainting models, derive metric weights from a mirror calibration "
            "slice, train a weighted DreamShaper LoRA, and rank it against modern baselines."
        )
    )
    parser.add_argument("--stage", action="append", default=None, choices=("all", *ALL_STAGES))
    parser.add_argument("--workdir", default="/content/3dprintpic")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-ref", default=None)
    parser.add_argument("--use-current-repo", action="store_true", help="Use the checkout containing this script.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--run-name", default="g4_object_surface")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dataset-source", choices=("existing", "procedural", "mesh-dir"), default="procedural")
    parser.add_argument("--dataset-count", type=int, default=60)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--asset-glob", default="**/*.off")
    parser.add_argument("--views-per-asset", type=int, default=1)
    parser.add_argument("--mesh-sample-strategy", choices=("random", "balanced"), default="balanced")
    parser.add_argument("--force-dataset", action="store_true")

    parser.add_argument("--cache-provider", action="append", default=None)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--cache-full", action="store_true")
    parser.add_argument("--cache-download-mode", choices=("files", "snapshot"), default="snapshot")
    parser.add_argument("--cache-max-workers", type=int, default=8)
    parser.add_argument("--require-modern-cache", action="store_true")
    parser.add_argument(
        "--require-image-to-mesh-providers",
        action="store_true",
        help="Fail eval slices before sample work if any direct image-to-mesh provider command is not runnable.",
    )

    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=40)
    parser.add_argument("--force-pairs", action="store_true")
    parser.add_argument("--calibration-start", type=int, default=0)
    parser.add_argument("--calibration-limit", type=int, default=10)
    parser.add_argument("--score-profile", choices=tuple(SCORE_PROFILES), default="object-surface")
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--weight-scale", type=float, default=2.0)
    parser.add_argument("--max-weight", type=float, default=4.0)

    parser.add_argument("--base-model", default="Lykon/dreamshaper-8-inpainting")
    parser.add_argument("--variant", default="fp16")
    parser.add_argument("--train-resolution", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mask-loss-weight", type=float, default=3.0)
    parser.add_argument("--seam-loss-weight", type=float, default=4.0)
    parser.add_argument("--object-loss-weight", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--existing-lora-weights", default=None)

    parser.add_argument("--modern-config", default=DEFAULT_MODERN_CONFIG)
    parser.add_argument("--eval-start", type=int, action="append", default=None)
    parser.add_argument("--eval-limit", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--eval-guidance", type=float, default=7.5)
    parser.add_argument("--eval-inpaint-max-dimension", type=int, default=256)
    parser.add_argument("--lora-scale", type=float, default=0.75)
    parser.add_argument("--depth-provider", default="depth-anything-v2")
    parser.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL)
    parser.add_argument("--stl-target-dimension", type=int, default=96)
    parser.add_argument("--candidate-method", default=None)
    parser.add_argument("--current-method", default="mirror")
    parser.add_argument("--min-paired-n", type=int, default=5)
    parser.add_argument("--max-mesh-surface-chamfer-ratio-vs-current", type=float, default=1.1)
    parser.add_argument("--max-mesh-surface-hausdorff95-ratio-vs-current", type=float, default=1.1)
    parser.add_argument(
        "--max-method-failures",
        type=int,
        default=2,
        help="Stop each method/config after this many failures during continue-on-error G4 evals.",
    )
    parser.add_argument("--allow-missing-split-audit", action="store_true")
    parser.add_argument("--contact-sheet-methods", default=None)
    parser.add_argument("--contact-sheet-max-samples", type=int, default=6)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    args.stage = args.stage or ["all"]
    args.cache_provider = args.cache_provider or list(DEFAULT_CACHE_PROVIDERS)
    args.eval_start = args.eval_start or [40, 50]
    if args.max_method_failures < 0:
        raise ValueError("--max-method-failures must be non-negative")
    return args


def main() -> None:
    started_at = utc_now()
    started = time.time()
    args = parse_args()
    logger_root = Path(args.output_root or Path(args.workdir) / "backend/output/completion-benchmark/colab_g4" / args.run_name)
    logger = CommandLogger(logger_root / "command_log.jsonl", dry_run=args.dry_run)

    probe = gpu_probe(logger)
    write_metadata(logger_root / "gpu_probe.json", probe)

    repo_root = resolve_repo(args, logger) if stage_enabled(args, "setup") else here_repo_root()
    output_root = relative_to_repo(repo_root, args.output_root) if args.output_root else repo_root / "backend/output/completion-benchmark/colab_g4" / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root != logger_root:
        logger = CommandLogger(output_root / "command_log.jsonl", dry_run=args.dry_run)
        write_metadata(output_root / "gpu_probe.json", probe)

    source_modern_config = relative_to_repo(repo_root, args.modern_config)
    write_metadata(
        output_root / "orchestrator_config.json",
        {
            "started_at": started_at,
            "notebook_url": DEFAULT_COLAB_NOTEBOOK_URL,
            "repo_root": str(repo_root),
            "repo_state": git_state(repo_root),
            "output_root": str(output_root),
            "argv": sys.argv[1:],
            "args": vars(args),
            "source_modern_config": file_report(source_modern_config),
        },
    )

    if stage_enabled(args, "setup"):
        install_requirements(args, repo_root, logger)

    if stage_enabled(args, "dataset"):
        manifest = ensure_dataset(args, repo_root, output_root, logger)
    elif args.manifest:
        manifest = require_existing_file(relative_to_repo(repo_root, args.manifest), "--manifest")
    else:
        raise ValueError("--manifest is required when the dataset stage is disabled.")

    if stage_enabled(args, "cache"):
        cache_modern_providers(args, repo_root, output_root, logger)

    metadata = export_training_pairs(args, repo_root, output_root, manifest, logger) if needs_training_pairs(args) else None
    calibration_dir = run_calibration(args, repo_root, output_root, manifest, logger) if stage_enabled(args, "calibrate") else output_root / "runs" / f"mirror_calibration_s{args.calibration_start}_n{args.calibration_limit}"
    weighted_metadata = weight_pairs(args, repo_root, output_root, metadata, calibration_dir, logger) if stage_enabled(args, "weight") and metadata else metadata
    if stage_enabled(args, "train"):
        lora_weights = train_weighted_lora(args, repo_root, output_root, weighted_metadata, logger)
    elif args.existing_lora_weights:
        lora_weights = validate_lora_weights(relative_to_repo(repo_root, args.existing_lora_weights))
    else:
        lora_weights = None

    eval_config = write_eval_config(args, repo_root, output_root, lora_weights)
    if stage_enabled(args, "eval"):
        eval_dirs = run_eval_slices(args, repo_root, output_root, manifest, eval_config, logger)
    elif stage_enabled(args, "combine"):
        eval_dirs = expected_eval_dirs(args, output_root)
    else:
        eval_dirs = []
    combined_dir = combine_eval_slices(args, repo_root, output_root, eval_dirs, logger) if stage_enabled(args, "combine") else None

    write_metadata(
        output_root / "orchestrator_result.json",
        {
            "started_at": started_at,
            "finished_at": utc_now(),
            "duration_seconds": round(time.time() - started, 3),
            "manifest": str(manifest),
            "manifest_report": file_report(manifest),
            "training_metadata": str(metadata) if metadata else "",
            "weighted_metadata": str(weighted_metadata) if weighted_metadata else "",
            "lora_weights": str(lora_weights) if lora_weights else "",
            "eval_config": str(eval_config),
            "eval_config_report": file_report(eval_config),
            "source_modern_config": file_report(source_modern_config),
            "repo_state": git_state(repo_root),
            "eval_dirs": [str(path) for path in eval_dirs],
            "combined_dir": str(combined_dir) if combined_dir else "",
        },
    )
    print(f"\nColab G4 orchestration complete. Results: {output_root}")


if __name__ == "__main__":
    main()
