from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_TRELLIS2_SOURCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
DEFAULT_TRELLIS2_SOURCE_COMMIT = DEFAULT_TRELLIS2_SOURCE_REVISION
DEFAULT_TRELLIS2_MODEL = "microsoft/TRELLIS.2-4B"
DEFAULT_TRELLIS2_MODEL_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
DEFAULT_TRELLIS2_RESOLUTION = 512
DEFAULT_TRELLIS2_SEED = 42
TRELLIS2_RESOLUTIONS = (DEFAULT_TRELLIS2_RESOLUTION,)


def trellis2_model_specs(
    *,
    model_repo: str = DEFAULT_TRELLIS2_MODEL,
    model_revision: str = DEFAULT_TRELLIS2_MODEL_REVISION,
) -> dict[str, dict[str, str]]:
    return {
        "trellis2": {
            "repo_id": model_repo,
            "revision": model_revision,
            "required_file": "pipeline.json",
        }
    }


def resolve_trellis2_model_snapshot(
    model_repo: str = DEFAULT_TRELLIS2_MODEL,
    model_revision: str = DEFAULT_TRELLIS2_MODEL_REVISION,
) -> Path:
    from huggingface_hub import snapshot_download

    configured_path = Path(model_repo).expanduser()
    if configured_path.exists():
        snapshot_path = configured_path.resolve()
    else:
        snapshot_path = Path(
            snapshot_download(repo_id=model_repo, revision=model_revision)
        ).resolve()

    required_path = snapshot_path / "pipeline.json"
    if not required_path.is_file():
        raise FileNotFoundError(
            f"Pinned TRELLIS.2 snapshot is missing pipeline.json: {required_path}"
        )

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    print(
        json.dumps(
            {
                "event": "trellis2_model_snapshot",
                "model": {
                    "repo_id": model_repo,
                    "revision": model_revision,
                },
                "path": str(snapshot_path),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    return snapshot_path


def _as_numpy(tensor):
    value = tensor.detach() if hasattr(tensor, "detach") else tensor
    value = value.cpu() if hasattr(value, "cpu") else value
    return value.numpy() if hasattr(value, "numpy") else value


def run_trellis2(
    *,
    provider_dir: Path,
    input_image: Path,
    output_mesh: Path,
    model_repo: str = DEFAULT_TRELLIS2_MODEL,
    model_revision: str = DEFAULT_TRELLIS2_MODEL_REVISION,
    resolution: int = DEFAULT_TRELLIS2_RESOLUTION,
    seed: int = DEFAULT_TRELLIS2_SEED,
) -> Path:
    if resolution not in TRELLIS2_RESOLUTIONS:
        supported = ", ".join(str(value) for value in TRELLIS2_RESOLUTIONS)
        raise ValueError(f"TRELLIS.2 resolution must be one of: {supported}")

    snapshot_path = resolve_trellis2_model_snapshot(model_repo, model_revision)
    provider_dir = provider_dir.resolve()
    sys.path.insert(0, str(provider_dir))

    import trimesh
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(snapshot_path))
    pipeline.cuda()
    with Image.open(input_image) as image_file:
        image = image_file.copy()
    meshes = pipeline.run(
        image,
        num_samples=1,
        seed=int(seed),
        pipeline_type=str(int(resolution)),
    )
    if len(meshes) != 1:
        raise RuntimeError(f"TRELLIS.2 returned {len(meshes)} meshes; expected exactly one")

    mesh = meshes[0]
    geometry = trimesh.Trimesh(
        vertices=_as_numpy(mesh.vertices),
        faces=_as_numpy(mesh.faces),
        process=False,
    )
    if not len(geometry.vertices) or not len(geometry.faces):
        raise ValueError("TRELLIS.2 returned an empty mesh")
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    geometry.export(output_mesh)
    return output_mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pinned TRELLIS.2 image-to-mesh inference and export geometry only."
    )
    parser.add_argument("--provider-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, default=None)
    parser.add_argument("--output-mesh", type=Path, default=None)
    parser.add_argument("--model-path", default=DEFAULT_TRELLIS2_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_TRELLIS2_MODEL_REVISION)
    parser.add_argument(
        "--resolution",
        type=int,
        choices=TRELLIS2_RESOLUTIONS,
        default=DEFAULT_TRELLIS2_RESOLUTION,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_TRELLIS2_SEED)
    parser.add_argument("--prefetch-only", action="store_true")
    args = parser.parse_args()

    if args.prefetch_only:
        snapshot_path = resolve_trellis2_model_snapshot(
            args.model_path,
            args.model_revision,
        )
        print(f"snapshot={snapshot_path}")
        return
    if args.input_image is None or args.output_mesh is None:
        raise ValueError("--input-image and --output-mesh are required unless --prefetch-only is set")

    output_mesh = run_trellis2(
        provider_dir=args.provider_dir,
        input_image=args.input_image,
        output_mesh=args.output_mesh,
        model_repo=args.model_path,
        model_revision=args.model_revision,
        resolution=args.resolution,
        seed=args.seed,
    )
    print(f"mesh={output_mesh}")


if __name__ == "__main__":
    main()
