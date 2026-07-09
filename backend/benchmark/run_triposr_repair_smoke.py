from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_TRIPOSR_PYTHON = "/content/triposr-venv/bin/python"
DEFAULT_TRIPOSR_DIR = "/content/TripoSR"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def shell_token(value: str | Path) -> str:
    return shlex.quote(str(value))


def triposr_provider_command(args: argparse.Namespace, repaired: bool) -> str:
    command = [
        shell_token(args.triposr_python),
        "-m",
        "backend.benchmark.run_image_to_mesh_provider",
        "--provider",
        "triposr-api",
        "--provider-dir",
        shell_token(args.provider_dir),
        "--input-image",
        '"{input_image}"',
        "--output-mesh",
        '"{output_mesh}"',
    ]
    if repaired:
        command.extend(["--raw-output-mesh", '"{output_dir}/output_mesh_raw.obj"'])
    command.extend(
        [
            "--output-stl",
            '"{output_stl}"',
            "--timeout",
            str(args.direct_mesh_timeout),
            "--provider-device",
            args.provider_device,
            "--chunk-size",
            str(args.chunk_size),
            "--mc-resolution",
            str(args.mc_resolution),
        ]
    )
    if repaired:
        command.extend(["--mesh-repair", args.mesh_repair])
    return " ".join(command)


def build_experiments(args: argparse.Namespace) -> list[dict]:
    raw_command = triposr_provider_command(args, repaired=False)
    repaired_command = triposr_provider_command(args, repaired=True)
    return [
        {"name": "masked", "method": "masked"},
        {"name": "mirror", "method": "mirror"},
        {"name": "biharmonic", "method": "biharmonic"},
        {
            "name": "triposr_api_masked_direct_mesh",
            "method": "external-image-to-mesh",
            "skip_depth": True,
            "emit_stl": True,
            "direct_mesh_input": "masked",
            "direct_mesh_output_ext": "obj",
            "direct_mesh_timeout": args.direct_mesh_timeout,
            "direct_mesh_command": raw_command,
        },
        {
            "name": "triposr_api_masked_repaired_direct_mesh",
            "method": "external-image-to-mesh",
            "skip_depth": True,
            "emit_stl": True,
            "direct_mesh_input": "masked",
            "direct_mesh_output_ext": "obj",
            "direct_mesh_timeout": args.direct_mesh_timeout,
            "direct_mesh_command": repaired_command,
        },
    ]


def rows_from_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def run_command(command: list[str], cwd: Path, timeout: int) -> dict:
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    output = proc.stdout or ""
    return {
        "command": command,
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 2),
        "tail": output[-9000:],
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_optimize_command(args: argparse.Namespace, manifest_path: Path, experiment_dir: Path, config_path: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "backend.benchmark.optimize_completion",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(experiment_dir),
        "--config",
        str(config_path),
        "--start-index",
        str(args.start_index),
        "--limit",
        str(args.limit),
        "--depth-provider",
        args.depth_provider,
        "--depth-model",
        args.depth_model,
        "--device",
        args.device,
        "--emit-stl",
        "--stl-target-dimension",
        str(args.stl_target_dimension),
        "--score-mode",
        "baseline-delta",
        "--score-profile",
        "stl-quality",
        "--baseline-method",
        "masked",
        "--contact-sheet",
        "--contact-sheet-methods",
        "masked,mirror,biharmonic,triposr_api_masked_direct_mesh,triposr_api_masked_repaired_direct_mesh",
        "--contact-sheet-max-samples",
        str(args.contact_sheet_max_samples),
        "--continue-on-error",
    ]
    if args.resume:
        command.append("--resume")
    return command


def run_smoke(args: argparse.Namespace) -> dict:
    repo_dir = Path(args.repo_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_dir = output_dir / "dataset"
    experiment_dir = output_dir / "experiment"
    config_path = output_dir / "triposr_raw_vs_repaired_config.json"

    summary: dict = {
        "marker": "TRIPOSR_REPAIRED_SMOKE",
        "started_at": utc_now(),
        "repo_dir": str(repo_dir),
        "output_dir": str(output_dir),
        "triposr_python": str(args.triposr_python),
        "provider_dir": str(args.provider_dir),
        "mesh_repair": args.mesh_repair,
    }
    experiments = build_experiments(args)
    write_json(config_path, experiments)
    summary["config"] = str(config_path)

    dataset_command = [
        sys.executable,
        "-m",
        "backend.benchmark.generate_rendered_dataset",
        "--output-dir",
        str(dataset_dir),
        "--source",
        "procedural",
        "--count",
        str(args.dataset_count),
        "--size",
        str(args.size),
    ]
    summary["dataset"] = run_command(dataset_command, repo_dir, timeout=args.dataset_timeout)

    manifest_path = dataset_dir / "manifest.jsonl"
    optimize_command = build_optimize_command(args, manifest_path, experiment_dir, config_path)
    summary["benchmark"] = run_command(optimize_command, repo_dir, timeout=args.benchmark_timeout)
    summary["aggregate_summary"] = rows_from_csv(experiment_dir / "aggregate_summary.csv")
    summary["ranked_experiments"] = rows_from_csv(experiment_dir / "ranked_experiments.csv")
    summary["raw_triposr_rows"] = rows_from_csv(
        experiment_dir / "triposr_api_masked_direct_mesh" / "per_sample_metrics.csv"
    )
    summary["repaired_triposr_rows"] = rows_from_csv(
        experiment_dir / "triposr_api_masked_repaired_direct_mesh" / "per_sample_metrics.csv"
    )
    summary["finished_at"] = utc_now()
    write_json(output_dir / "triposr_repaired_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible one-command Colab/local smoke comparing raw TripoSR API output "
            "against the printable mesh-repair path under the STL-quality benchmark profile."
        )
    )
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument(
        "--output-dir",
        default="backend/output/completion-benchmark/colab_g4/g4_triposr_repaired_s0_n1",
    )
    parser.add_argument("--triposr-python", default=DEFAULT_TRIPOSR_PYTHON)
    parser.add_argument("--provider-dir", default=DEFAULT_TRIPOSR_DIR)
    parser.add_argument("--provider-device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--mesh-repair", choices=("basic", "convex-hull", "printable"), default="printable")
    parser.add_argument("--direct-mesh-timeout", type=int, default=3600)
    parser.add_argument("--dataset-count", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--depth-provider", default="depth-anything-v2")
    parser.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stl-target-dimension", type=int, default=96)
    parser.add_argument("--contact-sheet-max-samples", type=int, default=1)
    parser.add_argument("--dataset-timeout", type=int, default=300)
    parser.add_argument("--benchmark-timeout", type=int, default=5400)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    summary = run_smoke(parse_args())
    print(json.dumps(summary, indent=2))
    if summary.get("dataset", {}).get("returncode") or summary.get("benchmark", {}).get("returncode"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
