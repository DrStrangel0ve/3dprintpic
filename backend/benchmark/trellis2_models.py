from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator


DEFAULT_TRELLIS2_SOURCE_REVISION = "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
DEFAULT_TRELLIS2_SOURCE_COMMIT = DEFAULT_TRELLIS2_SOURCE_REVISION
DEFAULT_TRELLIS2_MODEL = "microsoft/TRELLIS.2-4B"
DEFAULT_TRELLIS2_MODEL_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
DEFAULT_TRELLIS2_SPARSE_STRUCTURE_MODEL = "microsoft/TRELLIS-image-large"
DEFAULT_TRELLIS2_SPARSE_STRUCTURE_REVISION = (
    "25e0d31ffbebe4b5a97464dd851910efc3002d96"
)
DEFAULT_TRELLIS2_DINOV3_MODEL = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_TRELLIS2_DINOV3_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
DEFAULT_TRELLIS2_REMBG_MODEL = "ZhengPeng7/BiRefNet"
DEFAULT_TRELLIS2_REMBG_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
DEFAULT_TRELLIS2_RESOLUTION = 512
DEFAULT_TRELLIS2_SEED = 42
DEFAULT_TRELLIS2_ATTENTION_BACKEND = "xformers"
TRELLIS2_RESOLUTIONS = (DEFAULT_TRELLIS2_RESOLUTION,)

_OFFICIAL_TRELLIS2_DINOV3_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
_OFFICIAL_TRELLIS2_REMBG_MODEL = "briaai/RMBG-2.0"
_TRELLIS2_PIPELINE_MODEL_REFERENCES = {
    "sparse_structure_decoder": (
        "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
    ),
    "sparse_structure_flow_model": "ckpts/ss_flow_img_dit_1_3B_64_bf16",
    "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
    "shape_slat_flow_model_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
    "shape_slat_flow_model_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
    "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
    "tex_slat_flow_model_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
    "tex_slat_flow_model_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
}


def _checkpoint_files(reference: str) -> tuple[str, str]:
    return (f"{reference}.json", f"{reference}.safetensors")


_TRELLIS2_PIPELINE_REQUIRED_FILES = (
    "pipeline.json",
    *(
        required_file
        for name, reference in _TRELLIS2_PIPELINE_MODEL_REFERENCES.items()
        if name != "sparse_structure_decoder"
        for required_file in _checkpoint_files(reference)
    ),
)


def trellis2_model_specs(
    *,
    model_repo: str = DEFAULT_TRELLIS2_MODEL,
    model_revision: str = DEFAULT_TRELLIS2_MODEL_REVISION,
) -> dict[str, dict[str, str | tuple[str, ...]]]:
    return {
        "trellis2": {
            "repo_id": model_repo,
            "revision": model_revision,
            "required_file": "pipeline.json",
            "required_files": _TRELLIS2_PIPELINE_REQUIRED_FILES,
        },
        "sparse_structure_decoder": {
            "repo_id": DEFAULT_TRELLIS2_SPARSE_STRUCTURE_MODEL,
            "revision": DEFAULT_TRELLIS2_SPARSE_STRUCTURE_REVISION,
            "required_file": "ckpts/ss_dec_conv3d_16l8_fp16.json",
            "required_files": _checkpoint_files(
                "ckpts/ss_dec_conv3d_16l8_fp16"
            ),
        },
        "dinov3": {
            "repo_id": DEFAULT_TRELLIS2_DINOV3_MODEL,
            "revision": DEFAULT_TRELLIS2_DINOV3_REVISION,
            "required_file": "config.json",
            "required_files": ("config.json", "model.safetensors"),
        },
        "rembg": {
            "repo_id": DEFAULT_TRELLIS2_REMBG_MODEL,
            "revision": DEFAULT_TRELLIS2_REMBG_REVISION,
            "required_file": "config.json",
            "required_files": (
                "config.json",
                "model.safetensors",
                "BiRefNet_config.py",
                "birefnet.py",
            ),
        },
    }


def _load_pinned_pipeline_config(snapshot_path: Path) -> dict:
    pipeline_path = snapshot_path / "pipeline.json"
    with pipeline_path.open("r", encoding="utf-8") as pipeline_file:
        config = json.load(pipeline_file)

    if config.get("name") != "Trellis2ImageTo3DPipeline":
        raise ValueError(
            f"Pinned TRELLIS.2 pipeline has an unexpected entrypoint: {pipeline_path}"
        )
    args = config.get("args") or {}
    if args.get("models") != _TRELLIS2_PIPELINE_MODEL_REFERENCES:
        raise ValueError(
            f"Pinned TRELLIS.2 pipeline has unexpected model references: {pipeline_path}"
        )
    image_cond_model = ((args.get("image_cond_model") or {}).get("args") or {}).get(
        "model_name"
    )
    if image_cond_model != _OFFICIAL_TRELLIS2_DINOV3_MODEL:
        raise ValueError(
            f"Pinned TRELLIS.2 pipeline has an unexpected DINOv3 model: {pipeline_path}"
        )
    rembg_model = ((args.get("rembg_model") or {}).get("args") or {}).get(
        "model_name"
    )
    if rembg_model != _OFFICIAL_TRELLIS2_REMBG_MODEL:
        raise ValueError(
            f"Pinned TRELLIS.2 pipeline has an unexpected rembg model: {pipeline_path}"
        )
    return config


def resolve_trellis2_model_snapshots(
    model_repo: str = DEFAULT_TRELLIS2_MODEL,
    model_revision: str = DEFAULT_TRELLIS2_MODEL_REVISION,
) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    specs = trellis2_model_specs(
        model_repo=model_repo,
        model_revision=model_revision,
    )
    paths: dict[str, Path] = {}
    for name, spec in specs.items():
        repo_id = str(spec["repo_id"])
        revision = str(spec["revision"])
        configured_path = Path(repo_id).expanduser()
        if configured_path.exists():
            snapshot_path = configured_path.resolve()
        else:
            snapshot_path = Path(
                snapshot_download(repo_id=repo_id, revision=revision)
            ).resolve()

        required_files = tuple(spec.get("required_files") or (spec["required_file"],))
        missing = [
            snapshot_path / str(required_file)
            for required_file in required_files
            if not (snapshot_path / str(required_file)).is_file()
        ]
        if missing:
            missing_paths = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Pinned TRELLIS.2 {name} snapshot is missing required files: "
                f"{missing_paths}"
            )
        paths[name] = snapshot_path

    _load_pinned_pipeline_config(paths["trellis2"])

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    print(
        json.dumps(
            {
                "event": "trellis2_model_snapshots",
                "models": {
                    name: {
                        "repo_id": str(spec["repo_id"]),
                        "revision": str(spec["revision"]),
                    }
                    for name, spec in specs.items()
                },
                "paths": {name: str(path) for name, path in paths.items()},
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    return paths


def resolve_trellis2_model_snapshot(
    model_repo: str = DEFAULT_TRELLIS2_MODEL,
    model_revision: str = DEFAULT_TRELLIS2_MODEL_REVISION,
) -> Path:
    return resolve_trellis2_model_snapshots(model_repo, model_revision)["trellis2"]


def _local_checkpoint_path(
    snapshots: dict[str, Path],
    name: str,
    reference: str,
) -> Path:
    if name == "sparse_structure_decoder":
        prefix = f"{DEFAULT_TRELLIS2_SPARSE_STRUCTURE_MODEL}/"
        return (snapshots[name] / reference.removeprefix(prefix)).resolve()
    return (snapshots["trellis2"] / reference).resolve()


@contextlib.contextmanager
def _offline_pipeline_snapshot(
    snapshots: dict[str, Path],
) -> Iterator[Path]:
    config = _load_pinned_pipeline_config(snapshots["trellis2"])
    with tempfile.TemporaryDirectory(prefix="trellis2-offline-") as temp_dir:
        pipeline_path = Path(temp_dir).resolve()
        for name, reference in _TRELLIS2_PIPELINE_MODEL_REFERENCES.items():
            checkpoint_path = _local_checkpoint_path(snapshots, name, reference)
            try:
                local_reference = os.path.relpath(checkpoint_path, pipeline_path)
            except ValueError:
                local_reference = str(checkpoint_path)
            config["args"]["models"][name] = Path(local_reference).as_posix()
        config["args"]["image_cond_model"]["args"]["model_name"] = str(
            snapshots["dinov3"]
        )
        config["args"]["rembg_model"]["args"]["model_name"] = str(
            snapshots["rembg"]
        )
        (pipeline_path / "pipeline.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        yield pipeline_path


def _as_numpy(tensor):
    value = tensor.detach() if hasattr(tensor, "detach") else tensor
    value = value.cpu() if hasattr(value, "cpu") else value
    return value.numpy() if hasattr(value, "numpy") else value


def _vertices_in_glb_coordinates(vertices):
    converted = vertices[:, [0, 2, 1]].copy()
    converted[:, 2] *= -1
    return converted


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

    snapshots = resolve_trellis2_model_snapshots(model_repo, model_revision)
    provider_dir = provider_dir.resolve()
    sys.path.insert(0, str(provider_dir))
    os.environ["ATTN_BACKEND"] = DEFAULT_TRELLIS2_ATTENTION_BACKEND
    os.environ["SPARSE_ATTN_BACKEND"] = DEFAULT_TRELLIS2_ATTENTION_BACKEND

    import trimesh
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    with _offline_pipeline_snapshot(snapshots) as snapshot_path:
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
        vertices=_vertices_in_glb_coordinates(_as_numpy(mesh.vertices)),
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
        snapshot_paths = resolve_trellis2_model_snapshots(
            args.model_path,
            args.model_revision,
        )
        for name, snapshot_path in snapshot_paths.items():
            print(f"snapshot[{name}]={snapshot_path}")
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
