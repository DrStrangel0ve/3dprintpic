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


def image_to_mesh_command(
    *,
    python: str,
    provider: str,
    provider_dir: str | None,
    provider_device: str,
    timeout: int,
    output_mesh_repair: str = "none",
    output_mesh_raw: bool = False,
    raw_output_ext: str = "obj",
    output_extra: list[str] | None = None,
) -> str:
    command = [
        shell_token(python),
        "-m",
        "backend.benchmark.run_image_to_mesh_provider",
        "--provider",
        provider,
        "--input-image",
        '"{input_image}"',
        "--output-mesh",
        '"{output_mesh}"',
        "--output-stl",
        '"{output_stl}"',
        "--timeout",
        str(timeout),
        "--provider-device",
        provider_device,
    ]
    if provider_dir:
        command.extend(["--provider-dir", shell_token(provider_dir)])
    if output_mesh_raw:
        raw_ext = raw_output_ext.lstrip(".") or "mesh"
        command.extend(["--raw-output-mesh", f'"{{output_dir}}/output_mesh_raw.{raw_ext}"'])
    if output_mesh_repair != "none":
        command.extend(["--mesh-repair", output_mesh_repair])
    command.extend(output_extra or [])
    return " ".join(command)


def triposr_api_command(args: argparse.Namespace, repaired: bool) -> str:
    return image_to_mesh_command(
        python=args.triposr_python,
        provider="triposr-api",
        provider_dir=args.triposr_dir,
        provider_device=args.provider_device,
        timeout=args.direct_mesh_timeout,
        output_mesh_repair=args.mesh_repair if repaired else "none",
        output_mesh_raw=repaired,
        raw_output_ext="obj",
        output_extra=["--chunk-size", str(args.chunk_size), "--mc-resolution", str(args.mc_resolution)],
    )


def hunyuan3d_command(args: argparse.Namespace, repaired: bool) -> str:
    return image_to_mesh_command(
        python=args.provider_python,
        provider="hunyuan3d-shape",
        provider_dir=args.hunyuan3d_dir,
        provider_device=args.provider_device,
        timeout=args.direct_mesh_timeout,
        output_mesh_repair=args.mesh_repair if repaired else "none",
        output_mesh_raw=repaired,
        raw_output_ext="glb",
    )


def build_experiments(args: argparse.Namespace) -> list[dict]:
    experiments = [
        {"name": "masked", "method": "masked"},
        {"name": "mirror", "method": "mirror"},
        {"name": "biharmonic", "method": "biharmonic"},
    ]
    if args.include_source_oracle:
        experiments.append(
            {
                "name": "source_mesh_oracle",
                "method": "source-mesh-oracle",
                "skip_depth": True,
                "emit_stl": True,
                "direct_mesh_input": "full",
                "direct_mesh_output_ext": "ply",
                "source_mesh_repair": args.mesh_repair,
            }
        )
    if args.include_triposr_api:
        if args.include_raw_direct_mesh:
            experiments.append(
                {
                    "name": "triposr_api_masked_direct_mesh",
                    "method": "external-image-to-mesh",
                    "skip_depth": True,
                    "emit_stl": True,
                    "direct_mesh_input": "masked",
                    "direct_mesh_output_ext": "obj",
                    "direct_mesh_timeout": args.direct_mesh_timeout,
                    "direct_mesh_command": triposr_api_command(args, repaired=False),
                }
            )
        for direct_input in args.triposr_direct_inputs:
            input_suffix = "masked" if direct_input == "masked" else f"{direct_input}_prefill"
            experiments.append(
                {
                    "name": f"triposr_api_{input_suffix}_repaired_direct_mesh",
                    "method": "external-image-to-mesh",
                    "skip_depth": True,
                    "emit_stl": True,
                    "direct_mesh_input": direct_input,
                    "direct_mesh_output_ext": "obj",
                    "direct_mesh_timeout": args.direct_mesh_timeout,
                    "direct_mesh_command": triposr_api_command(args, repaired=True),
                }
            )
    if args.include_hunyuan3d_shape:
        experiments.append(
            {
                "name": "hunyuan3d_shape_masked_repaired_direct_mesh",
                "method": "external-image-to-mesh",
                "skip_depth": True,
                "emit_stl": True,
                "direct_mesh_input": "masked",
                "direct_mesh_output_ext": "glb",
                "direct_mesh_timeout": args.direct_mesh_timeout,
                "direct_mesh_command": hunyuan3d_command(args, repaired=True),
            }
        )
    if args.multiview_command:
        experiments.append(
            {
                "name": args.multiview_name,
                "method": "external-multiview-to-mesh",
                "skip_depth": True,
                "emit_stl": True,
                "direct_mesh_input": args.multiview_primary_input,
                "direct_mesh_output_ext": args.multiview_output_ext,
                "direct_mesh_timeout": args.direct_mesh_timeout,
                "direct_mesh_command": args.multiview_command,
            }
        )
    return experiments


def rows_from_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


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


def ensure_manifest(args: argparse.Namespace, repo_dir: Path, output_dir: Path) -> tuple[Path, dict | None]:
    if args.manifest:
        return Path(args.manifest), None
    dataset_dir = output_dir / "dataset"
    command = [
        sys.executable,
        "-m",
        "backend.benchmark.generate_rendered_dataset",
        "--output-dir",
        str(dataset_dir),
        "--source",
        args.dataset_source,
        "--count",
        str(args.dataset_count),
        "--size",
        str(args.size),
        "--seed",
        str(args.seed),
        "--views-per-asset",
        str(args.views_per_asset),
        "--mesh-sample-strategy",
        args.mesh_sample_strategy,
    ]
    if args.asset_root:
        command.extend(["--asset-root", args.asset_root])
    if args.asset_glob:
        command.extend(["--asset-glob", args.asset_glob])
    if args.continue_on_error:
        command.append("--continue-on-error")
    result = run_command(command, repo_dir, timeout=args.dataset_timeout)
    return dataset_dir / "manifest.jsonl", result


def build_optimize_command(args: argparse.Namespace, manifest_path: Path, experiment_dir: Path, config_path: Path, experiments: list[dict]) -> list[str]:
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
        ",".join(experiment["name"] for experiment in experiments),
        "--contact-sheet-max-samples",
        str(args.contact_sheet_max_samples),
    ]
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.resume:
        command.append("--resume")
    if getattr(args, "select_candidate", False):
        command.append("--select-candidate")
        if args.candidate_method:
            command.extend(["--candidate-method", args.candidate_method])
        if args.current_method:
            command.extend(["--current-method", args.current_method])
        command.extend(["--min-paired-n", str(args.min_paired_n)])
        command.extend(["--min-win-rate", str(args.min_win_rate)])
        command.extend(["--min-ci95-low", str(args.min_ci95_low)])
        command.extend(["--min-score-margin", str(args.min_score_margin)])
        if args.allow_missing_split_audit:
            command.append("--allow-missing-split-audit")
    return command


def run_smoke(args: argparse.Namespace) -> dict:
    repo_dir = Path(args.repo_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    experiment_dir = output_dir / "experiment"
    config_path = output_dir / "stl_first_reconstruction_config.json"
    experiments = build_experiments(args)
    write_json(config_path, experiments)
    manifest_path, dataset_result = ensure_manifest(args, repo_dir, output_dir)
    summary: dict = {
        "marker": "STL_FIRST_RECONSTRUCTION_SMOKE",
        "started_at": utc_now(),
        "repo_dir": str(repo_dir),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "config": str(config_path),
        "experiments": [experiment["name"] for experiment in experiments],
    }
    if dataset_result is not None:
        summary["dataset"] = dataset_result
    optimize_command = build_optimize_command(args, manifest_path, experiment_dir, config_path, experiments)
    summary["benchmark"] = run_command(optimize_command, repo_dir, timeout=args.benchmark_timeout)
    summary["aggregate_summary"] = rows_from_csv(experiment_dir / "aggregate_summary.csv")
    summary["ranked_experiments"] = rows_from_csv(experiment_dir / "ranked_experiments.csv")
    summary["selection_decision"] = read_json(experiment_dir / "selection_decision.json")
    if (experiment_dir / "selection_decision.md").exists():
        summary["selection_decision_md"] = str(experiment_dir / "selection_decision.md")
    summary["finished_at"] = utc_now()
    write_json(output_dir / "stl_first_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an STL-first reconstruction smoke: depth-relief baselines plus opt-in single-image "
            "and multiview mesh providers, all ranked by the stl-quality objective."
        )
    )
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--output-dir", default="backend/output/completion-benchmark/experiments/stl_first_reconstruction_smoke")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dataset-source", choices=("procedural", "mesh-dir"), default="procedural")
    parser.add_argument("--dataset-count", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--asset-glob", default="**/*.glb")
    parser.add_argument("--views-per-asset", type=int, default=1)
    parser.add_argument("--mesh-sample-strategy", choices=("random", "balanced"), default="balanced")
    parser.add_argument("--include-source-oracle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-triposr-api", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-hunyuan3d-shape", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-raw-direct-mesh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--triposr-direct-input",
        action="append",
        choices=("masked", "full", "mirror", "biharmonic"),
        dest="triposr_direct_inputs",
        default=None,
        help=(
            "Direct image input mode for repaired TripoSR API candidates. Repeat to compare "
            "masked, full, mirror-prefill, and biharmonic-prefill variants."
        ),
    )
    parser.add_argument("--provider-python", default=sys.executable)
    parser.add_argument("--provider-device", default="cuda")
    parser.add_argument("--triposr-python", default=DEFAULT_TRIPOSR_PYTHON)
    parser.add_argument("--triposr-dir", default=DEFAULT_TRIPOSR_DIR)
    parser.add_argument("--hunyuan3d-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--mesh-repair", choices=("basic", "convex-hull", "printable"), default="printable")
    parser.add_argument("--direct-mesh-timeout", type=int, default=3600)
    parser.add_argument("--multiview-command", default=None)
    parser.add_argument("--multiview-name", default="external_multiview_reconstruction")
    parser.add_argument("--multiview-primary-input", choices=("masked", "full", "mirror", "biharmonic"), default="masked")
    parser.add_argument("--multiview-output-ext", default="glb")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--depth-provider", default="depth-anything-v2")
    parser.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stl-target-dimension", type=int, default=96)
    parser.add_argument("--contact-sheet-max-samples", type=int, default=2)
    parser.add_argument("--select-candidate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidate-method", default=None)
    parser.add_argument("--current-method", default="mirror")
    parser.add_argument("--min-paired-n", type=int, default=5)
    parser.add_argument("--min-win-rate", type=float, default=0.8)
    parser.add_argument("--min-ci95-low", type=float, default=0.0)
    parser.add_argument("--min-score-margin", type=float, default=0.0)
    parser.add_argument("--allow-missing-split-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-timeout", type=int, default=600)
    parser.add_argument("--benchmark-timeout", type=int, default=7200)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.triposr_direct_inputs = args.triposr_direct_inputs or ["masked"]
    return args


def main() -> None:
    summary = run_smoke(parse_args())
    print(json.dumps(summary, indent=2))
    if summary.get("dataset", {}).get("returncode") or summary.get("benchmark", {}).get("returncode"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
