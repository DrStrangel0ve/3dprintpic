import importlib.util
import json
import os
import subprocess
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/jennyzzt/3dprintpic.git"
MODERN_METHODS = {"sdxl-inpaint", "dreamshaper-inpaint", "amused-inpaint", "flux-fill", "qwen-image-inpaint", "qwen-image-edit"}
MINIMAL_MODULE_PACKAGES = {
    "numpy": "numpy",
    "PIL": "pillow",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "stl": "numpy-stl",
}
RENDERED_MODULE_PACKAGES = {"trimesh": "trimesh"}


def env(name, default):
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


REPO_SOURCE = env("THREEDPRINTPIC_REPO", DEFAULT_REPO_URL)
REPO_REF = env("THREEDPRINTPIC_REF", "")
METHODS = env("COMPLETION_METHODS", "mirror,biharmonic")
COUNT = env("COMPLETION_BENCHMARK_COUNT", "20")
START_INDEX = env("COMPLETION_START_INDEX", "0")
DATASET = env("COMPLETION_DATASET", "synthetic").lower()
ASSET_ROOT = env("COMPLETION_ASSET_ROOT", "")
ASSET_GLOB = env("COMPLETION_ASSET_GLOB", "**/*.glb")
VIEWS_PER_ASSET = env("COMPLETION_VIEWS_PER_ASSET", "1")
SKIP_DEPTH = env("COMPLETION_SKIP_DEPTH", "1") == "1"
TRAIN_LORA = env("COMPLETION_TRAIN_LORA", "0") == "1"
TRAIN_LORA_BASE_MODEL = env("COMPLETION_TRAIN_LORA_BASE_MODEL", "Lykon/dreamshaper-8-inpainting")
TRAIN_LORA_METHOD = env("COMPLETION_TRAIN_LORA_METHOD", "dreamshaper-inpaint")
TRAIN_LORA_VARIANT = env("COMPLETION_TRAIN_LORA_VARIANT", "fp16")
TRAIN_LORA_STEPS = env("COMPLETION_TRAIN_LORA_STEPS", "200")
TRAIN_LORA_LIMIT = env("COMPLETION_TRAIN_LORA_LIMIT", COUNT)
TRAIN_LORA_START_INDEX = env("COMPLETION_TRAIN_LORA_START_INDEX", "0")
TRAIN_LORA_EVAL_START_INDEX = env("COMPLETION_TRAIN_LORA_EVAL_START_INDEX", START_INDEX)
TRAIN_LORA_RESOLUTION = env("COMPLETION_TRAIN_LORA_RESOLUTION", "256")
TRAIN_LORA_BATCH_SIZE = env("COMPLETION_TRAIN_LORA_BATCH_SIZE", "1")
TRAIN_LORA_GRAD_ACCUM = env("COMPLETION_TRAIN_LORA_GRAD_ACCUM", "4")
TRAIN_LORA_PROMPT = env(
    "COMPLETION_TRAIN_LORA_PROMPT",
    "Complete the missing half of the same object with matching geometry, lighting, viewpoint, and background. The masked half must contain the missing object, not an empty background.",
)


def run(command, cwd=None):
    command = [str(part) for part in command]
    print("+", " ".join(command))
    subprocess.check_call(command, cwd=cwd)


def command_succeeds(command, cwd=None):
    return subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def split_methods(methods):
    return [method.strip() for method in methods.split(",") if method.strip()]


def is_repo_tree(path):
    return (path / "backend" / "benchmark").exists()


def local_repo_path(source):
    path = Path(source).expanduser()
    return path.resolve() if path.exists() else None


def ensure_origin(repo_dir, source):
    if command_succeeds(["git", "remote", "get-url", "origin"], cwd=repo_dir):
        run(["git", "remote", "set-url", "origin", source], cwd=repo_dir)
    else:
        run(["git", "remote", "add", "origin", source], cwd=repo_dir)


def checkout_ref(repo_dir, ref):
    run(["git", "fetch", "origin", "--tags", "--prune"], cwd=repo_dir)
    if ref:
        if command_succeeds(["git", "rev-parse", "--verify", f"origin/{ref}"], cwd=repo_dir):
            run(["git", "checkout", "-B", ref, f"origin/{ref}"], cwd=repo_dir)
        elif command_succeeds(["git", "rev-parse", "--verify", ref], cwd=repo_dir):
            run(["git", "checkout", ref], cwd=repo_dir)
        elif command_succeeds(["git", "fetch", "origin", ref, "--tags"], cwd=repo_dir):
            run(["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)
        else:
            raise ValueError(f"THREEDPRINTPIC_REF was not found after fetch: {ref}")
        return

    branch = default_branch(repo_dir)
    run(["git", "checkout", "-B", branch, f"origin/{branch}"], cwd=repo_dir)


def default_branch(repo_dir):
    try:
        remote_head = subprocess.check_output(
            ["git", "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"],
            cwd=repo_dir,
            text=True,
        ).strip()
        return remote_head.split("/", 1)[1] if remote_head.startswith("origin/") else remote_head
    except subprocess.CalledProcessError:
        for branch in ("main", "master"):
            if command_succeeds(["git", "rev-parse", "--verify", f"origin/{branch}"], cwd=repo_dir):
                return branch
        raise


def clone_or_update_repo(repo_dir, source, ref):
    if repo_dir.exists() and not (repo_dir / ".git").exists():
        if is_repo_tree(repo_dir) and not ref:
            print(f"Using existing non-git repo tree at {repo_dir}")
            return repo_dir
        raise ValueError(f"{repo_dir} exists but is not a git checkout")

    if not repo_dir.exists():
        run(["git", "clone", source, str(repo_dir)])
    else:
        ensure_origin(repo_dir, source)

    checkout_ref(repo_dir, ref)
    return repo_dir


def prepare_repo(workdir):
    repo_dir = workdir / "3dprintpic"
    local_source = local_repo_path(REPO_SOURCE)
    if local_source:
        if not is_repo_tree(local_source):
            raise ValueError(f"THREEDPRINTPIC_REPO does not look like a 3dprintpic checkout: {local_source}")
        if not REPO_REF:
            print(f"Using local repo tree from THREEDPRINTPIC_REPO={local_source}")
            return local_source
        if not (local_source / ".git").exists():
            raise ValueError("THREEDPRINTPIC_REF requires a git checkout when THREEDPRINTPIC_REPO is a local path")
        return clone_or_update_repo(repo_dir, str(local_source), REPO_REF)

    return clone_or_update_repo(repo_dir, REPO_SOURCE, REPO_REF)


def missing_packages(module_packages):
    packages = []
    for module_name, package_name in module_packages.items():
        if importlib.util.find_spec(module_name) is None:
            packages.append(package_name)
    return packages


def install_dependencies(repo_dir, dataset, methods, skip_depth):
    parsed_methods = split_methods(methods)
    needs_heavy = TRAIN_LORA or not skip_depth or any(method in MODERN_METHODS for method in parsed_methods)
    if needs_heavy:
        print("Depth or modern diffusion completion requested; installing CUDA benchmark requirements.")
        run(["python", "-m", "pip", "install", "-r", "backend/requirements-cuda.txt"], cwd=repo_dir)
        return

    module_packages = dict(MINIMAL_MODULE_PACKAGES)
    if dataset in ("rendered-procedural", "mesh-dir"):
        module_packages.update(RENDERED_MODULE_PACKAGES)

    packages = missing_packages(module_packages)
    if packages:
        print("Installing missing lightweight benchmark dependencies:", ", ".join(packages))
        run(["python", "-m", "pip", "install", *packages], cwd=repo_dir)
    else:
        print("Lightweight benchmark dependencies are already available; skipping pip install.")


def generate_manifest(repo_dir, workdir):
    benchmark_dir = workdir / "completion-benchmark"
    if DATASET == "synthetic":
        output_dir = benchmark_dir / "synthetic"
        run(
            [
                "python",
                "-m",
                "backend.benchmark.generate_synthetic_dataset",
                "--output-dir",
                output_dir,
                "--count",
                COUNT,
                "--size",
                "256",
            ],
            cwd=repo_dir,
        )
        return output_dir / "manifest.jsonl"

    if DATASET == "rendered-procedural":
        output_dir = benchmark_dir / "rendered-procedural"
        run(
            [
                "python",
                "-m",
                "backend.benchmark.generate_rendered_dataset",
                "--output-dir",
                output_dir,
                "--source",
                "procedural",
                "--count",
                COUNT,
                "--size",
                "256",
            ],
            cwd=repo_dir,
        )
        return output_dir / "manifest.jsonl"

    if DATASET == "mesh-dir":
        if not ASSET_ROOT:
            raise ValueError("COMPLETION_ASSET_ROOT is required when COMPLETION_DATASET=mesh-dir")
        output_dir = benchmark_dir / "mesh-dir"
        run(
            [
                "python",
                "-m",
                "backend.benchmark.generate_rendered_dataset",
                "--output-dir",
                output_dir,
                "--source",
                "mesh-dir",
                "--count",
                COUNT,
                "--size",
                "256",
                "--asset-root",
                ASSET_ROOT,
                "--asset-glob",
                ASSET_GLOB,
                "--views-per-asset",
                VIEWS_PER_ASSET,
            ],
            cwd=repo_dir,
        )
        return output_dir / "manifest.jsonl"

    raise ValueError("COMPLETION_DATASET must be one of: synthetic, rendered-procedural, mesh-dir")


def train_and_evaluate_lora(repo_dir, workdir, manifest_path):
    train_dir = workdir / "completion-benchmark" / "training-pairs"
    lora_dir = workdir / "completion-benchmark" / "lora"
    experiment_config = workdir / "completion-benchmark" / "trained-lora-config.json"
    prompt = TRAIN_LORA_PROMPT

    run(
        [
            "python",
            "-m",
            "backend.benchmark.export_training_pairs",
            "--manifest",
            manifest_path,
            "--output-dir",
            train_dir,
            "--limit",
            TRAIN_LORA_LIMIT,
            "--start-index",
            TRAIN_LORA_START_INDEX,
            "--prompt",
            prompt,
        ],
        cwd=repo_dir,
    )
    run(
        [
            "python",
            "-m",
            "backend.benchmark.train_inpainting_lora",
            "--metadata",
            train_dir / "metadata.jsonl",
            "--output-dir",
            lora_dir,
            "--base-model",
            TRAIN_LORA_BASE_MODEL,
            "--variant",
            TRAIN_LORA_VARIANT,
            "--resolution",
            TRAIN_LORA_RESOLUTION,
            "--train-batch-size",
            TRAIN_LORA_BATCH_SIZE,
            "--gradient-accumulation-steps",
            TRAIN_LORA_GRAD_ACCUM,
            "--max-train-steps",
            TRAIN_LORA_STEPS,
            "--mixed-precision",
            "fp16",
            "--allow-tf32",
        ],
        cwd=repo_dir,
    )

    experiment_config.parent.mkdir(parents=True, exist_ok=True)
    experiment_config.write_text(
        json.dumps(
            [
                {"name": "mirror", "method": "mirror"},
                {"name": "biharmonic", "method": "biharmonic"},
                {
                    "name": f"{TRAIN_LORA_METHOD}_base",
                    "method": TRAIN_LORA_METHOD,
                    "model_name": TRAIN_LORA_BASE_MODEL,
                    "prompt": prompt,
                    "steps": 20,
                    "guidance": 7.5,
                    "seed": 1234,
                    "inpaint_max_dimension": int(TRAIN_LORA_RESOLUTION),
                },
                {
                    "name": f"{TRAIN_LORA_METHOD}_lora",
                    "method": TRAIN_LORA_METHOD,
                    "model_name": TRAIN_LORA_BASE_MODEL,
                    "lora_weights": str(lora_dir),
                    "lora_scale": 1.0,
                    "prompt": prompt,
                    "steps": 20,
                    "guidance": 7.5,
                    "seed": 1234,
                    "inpaint_max_dimension": int(TRAIN_LORA_RESOLUTION),
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    optimize_command = [
        "python",
        "-m",
        "backend.benchmark.optimize_completion",
        "--manifest",
        manifest_path,
        "--output-dir",
        workdir / "completion-benchmark" / "lora-eval",
        "--config",
        experiment_config,
        "--limit",
        COUNT,
        "--start-index",
        TRAIN_LORA_EVAL_START_INDEX,
        "--device",
        "auto",
        "--resume",
        "--continue-on-error",
    ]
    if SKIP_DEPTH:
        optimize_command.append("--skip-depth")
    run(optimize_command, cwd=repo_dir)


def main():
    workdir = Path("/kaggle/working")
    repo_dir = prepare_repo(workdir)
    print(
        "Benchmark config:",
        f"repo={repo_dir}",
        f"ref={REPO_REF or 'default'}",
        f"dataset={DATASET}",
        f"methods={METHODS}",
        f"count={COUNT}",
        f"start_index={START_INDEX}",
        f"skip_depth={SKIP_DEPTH}",
        f"train_lora={TRAIN_LORA}",
    )

    install_dependencies(repo_dir, DATASET, METHODS, SKIP_DEPTH)
    manifest_path = generate_manifest(repo_dir, workdir)

    if TRAIN_LORA:
        train_and_evaluate_lora(repo_dir, workdir, manifest_path)
        return

    benchmark_command = [
        "python",
        "-m",
        "backend.benchmark.run_completion_benchmark",
        "--manifest",
        manifest_path,
        "--output-dir",
        workdir / "completion-benchmark" / "run",
        "--methods",
        METHODS,
        "--limit",
        COUNT,
        "--start-index",
        START_INDEX,
    ]
    if SKIP_DEPTH:
        benchmark_command.append("--skip-depth")
    run(benchmark_command, cwd=repo_dir)


if __name__ == "__main__":
    main()
