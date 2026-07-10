from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION = "f8db63096c8282cb27354314d896feba5ba6ff8a"
DEFAULT_HUNYUAN3D_2MV_MODEL = "tencent/Hunyuan3D-2mv"
DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION = "3a761b539b29fe4ff64714813aa9560fd66f5de0"
DEFAULT_HUNYUAN3D_2MV_SUBFOLDER = "hunyuan3d-dit-v2-mv"
DEFAULT_HUNYUAN3D_2MV_STEPS = 50
DEFAULT_HUNYUAN3D_2MV_GUIDANCE = 5.0
DEFAULT_HUNYUAN3D_2MV_OCTREE_RESOLUTION = 380
DEFAULT_HUNYUAN3D_2MV_NUM_CHUNKS = 20000
DEFAULT_HUNYUAN3D_2MV_SEED = 42
DEFAULT_HUNYUAN3D_2MV_MAX_VIEW_ANGLE_ERROR = 60.0
DEFAULT_HUNYUAN3D_2MV_REQUIRED_VIEWS = ("front", "left", "back")
HUNYUAN3D_2MV_VIEW_ORDER = ("front", "left", "back", "right")
# Positive benchmark yaw rotates the object, so its equivalent camera orbit has the opposite sign.
HUNYUAN3D_2MV_VIEW_AZIMUTHS = {
    "front": 0.0,
    "left": 270.0,
    "back": 180.0,
    "right": 90.0,
}
PROVIDER_NATIVE_METRICS_FILENAME = "provider_native_metrics.json"


def hunyuan3d_2mv_model_specs(
    *,
    model_repo: str = DEFAULT_HUNYUAN3D_2MV_MODEL,
    model_revision: str = DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION,
    subfolder: str = DEFAULT_HUNYUAN3D_2MV_SUBFOLDER,
) -> dict[str, dict[str, str | tuple[str, ...]]]:
    return {
        "hunyuan3d_2mv": {
            "repo_id": model_repo,
            "revision": model_revision,
            "subfolder": subfolder,
            "required_file": f"{subfolder}/config.yaml",
            "required_files": (
                f"{subfolder}/config.yaml",
                f"{subfolder}/model.fp16.safetensors",
            ),
        }
    }


def resolve_hunyuan3d_2mv_model_snapshot(
    model_repo: str = DEFAULT_HUNYUAN3D_2MV_MODEL,
    model_revision: str = DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION,
    subfolder: str = DEFAULT_HUNYUAN3D_2MV_SUBFOLDER,
) -> Path:
    from huggingface_hub import snapshot_download

    configured_path = Path(model_repo).expanduser()
    if configured_path.exists():
        snapshot_path = configured_path.resolve()
    else:
        snapshot_path = Path(
            snapshot_download(
                repo_id=model_repo,
                revision=model_revision,
                allow_patterns=[f"{subfolder}/*"],
            )
        ).resolve()

    spec = hunyuan3d_2mv_model_specs(
        model_repo=model_repo,
        model_revision=model_revision,
        subfolder=subfolder,
    )["hunyuan3d_2mv"]
    missing = [
        snapshot_path / str(required_file)
        for required_file in spec["required_files"]
        if not (snapshot_path / str(required_file)).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Pinned Hunyuan3D-2mv snapshot is missing required files: "
            + ", ".join(str(path) for path in missing)
        )

    print(
        json.dumps(
            {
                "event": "hunyuan3d_2mv_model_snapshot",
                "model": {
                    "repo_id": model_repo,
                    "revision": model_revision,
                    "subfolder": subfolder,
                },
                "path": str(snapshot_path),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    return snapshot_path


def _resolve_bundle_path(value: str | None, bundle_path: Path) -> Path | None:
    if not value:
        return None
    raw = Path(str(value)).expanduser()
    candidates = [raw] if raw.is_absolute() else [bundle_path.parent / raw, Path.cwd() / raw, raw]
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), None)


def _normalized_view_label(view: dict) -> str | None:
    camera = view.get("camera") if isinstance(view.get("camera"), dict) else {}
    for key in ("hunyuan_view", "view_tag", "view_name", "view"):
        value = view.get(key) or camera.get(key)
        if str(value or "").strip().lower() in HUNYUAN3D_2MV_VIEW_ORDER:
            return str(value).strip().lower()
    return None


def _angular_distance_degrees(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def select_hunyuan3d_2mv_views(
    bundle: dict,
    *,
    max_angle_error: float = DEFAULT_HUNYUAN3D_2MV_MAX_VIEW_ANGLE_ERROR,
    required_views: Iterable[str] = DEFAULT_HUNYUAN3D_2MV_REQUIRED_VIEWS,
) -> dict[str, dict]:
    views = bundle.get("views") or []
    if not isinstance(views, list) or not views:
        raise ValueError("Multiview bundle has no views")
    if not math.isfinite(float(max_angle_error)) or float(max_angle_error) < 0:
        raise ValueError("max view angle error must be finite and non-negative")

    candidates: dict[str, list[tuple[float, int, dict]]] = {
        label: [] for label in HUNYUAN3D_2MV_VIEW_ORDER
    }
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            continue
        explicit_label = _normalized_view_label(view)
        if explicit_label:
            candidates[explicit_label].append((0.0, index, view))
            continue
        camera = view.get("camera") if isinstance(view.get("camera"), dict) else {}
        try:
            azimuth = float(camera["azimuth_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        label, error = min(
            (
                (label, _angular_distance_degrees(azimuth, expected_azimuth))
                for label, expected_azimuth in HUNYUAN3D_2MV_VIEW_AZIMUTHS.items()
            ),
            key=lambda item: item[1],
        )
        if error <= float(max_angle_error):
            candidates[label].append((error, index, view))

    selected: dict[str, dict] = {}
    used_indices: set[int] = set()
    for label in HUNYUAN3D_2MV_VIEW_ORDER:
        for error, index, view in sorted(candidates[label], key=lambda item: (item[0], item[1])):
            if index in used_indices:
                continue
            selected[label] = {
                **view,
                "hunyuan_view": label,
                "hunyuan_view_angle_error_deg": float(error),
                "hunyuan_bundle_view_index": index,
            }
            used_indices.add(index)
            break

    required = tuple(str(label).strip().lower() for label in required_views)
    invalid = [label for label in required if label not in HUNYUAN3D_2MV_VIEW_ORDER]
    if invalid:
        raise ValueError(f"Unsupported required Hunyuan3D-2mv views: {', '.join(invalid)}")
    missing = [label for label in required if label not in selected]
    if missing:
        available = ", ".join(selected) or "none"
        raise ValueError(
            "Multiview bundle is missing required Hunyuan3D-2mv views "
            f"{', '.join(missing)}; selected: {available}"
        )
    return selected


def _rgba_from_view(view: dict, bundle_path: Path) -> Image.Image:
    image_path = _resolve_bundle_path(view.get("image"), bundle_path)
    if image_path is None:
        raise FileNotFoundError(f"Hunyuan3D-2mv view image does not exist: {view.get('image')}")
    source = Image.open(image_path)
    rgba = source.convert("RGBA")
    mask_path = _resolve_bundle_path(view.get("mask"), bundle_path)
    if mask_path is not None:
        alpha = Image.open(mask_path).convert("L")
        if alpha.size != rgba.size:
            alpha = alpha.resize(rgba.size, Image.Resampling.NEAREST)
        rgba.putalpha(alpha)
    alpha_array = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    if not np.any(alpha_array > 0):
        raise ValueError(f"Hunyuan3D-2mv view mask is empty: {mask_path or image_path}")
    pixels = np.asarray(rgba, dtype=np.uint8).copy()
    pixels[alpha_array == 0, :3] = 255
    return Image.fromarray(pixels, mode="RGBA")


def prepare_hunyuan3d_2mv_images(
    bundle_path: Path,
    output_dir: Path,
    *,
    max_angle_error: float = DEFAULT_HUNYUAN3D_2MV_MAX_VIEW_ANGLE_ERROR,
    required_views: Iterable[str] = DEFAULT_HUNYUAN3D_2MV_REQUIRED_VIEWS,
) -> tuple[dict[str, Image.Image], dict]:
    bundle_path = Path(bundle_path).resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Multiview input bundle does not exist: {bundle_path}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    selected = select_hunyuan3d_2mv_views(
        bundle,
        max_angle_error=max_angle_error,
        required_views=required_views,
    )
    prepared_dir = Path(output_dir) / "hunyuan3d_2mv_inputs"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, Image.Image] = {}
    rows = []
    for label in HUNYUAN3D_2MV_VIEW_ORDER:
        if label not in selected:
            continue
        image = _rgba_from_view(selected[label], bundle_path)
        prepared_path = prepared_dir / f"{label}.png"
        image.save(prepared_path)
        images[label] = image
        rows.append(
            {
                "view": label,
                "bundle_view_index": selected[label]["hunyuan_bundle_view_index"],
                "angle_error_deg": selected[label]["hunyuan_view_angle_error_deg"],
                "source_image": str(
                    _resolve_bundle_path(selected[label].get("image"), bundle_path) or ""
                ),
                "source_mask": str(
                    _resolve_bundle_path(selected[label].get("mask"), bundle_path) or ""
                ),
                "prepared_image": str(prepared_path),
            }
        )
    manifest = {
        "bundle": str(bundle_path),
        "source_geometry_used": False,
        "view_order": list(images),
        "views": rows,
    }
    (prepared_dir / "selected_views.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return images, manifest


def run_hunyuan3d_2mv(
    *,
    provider_dir: Path,
    input_bundle: Path,
    output_mesh: Path,
    model_path: str = DEFAULT_HUNYUAN3D_2MV_MODEL,
    model_revision: str = DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION,
    subfolder: str = DEFAULT_HUNYUAN3D_2MV_SUBFOLDER,
    num_inference_steps: int = DEFAULT_HUNYUAN3D_2MV_STEPS,
    guidance_scale: float = DEFAULT_HUNYUAN3D_2MV_GUIDANCE,
    octree_resolution: int = DEFAULT_HUNYUAN3D_2MV_OCTREE_RESOLUTION,
    num_chunks: int = DEFAULT_HUNYUAN3D_2MV_NUM_CHUNKS,
    seed: int = DEFAULT_HUNYUAN3D_2MV_SEED,
    device: str = "cuda",
    max_view_angle_error: float = DEFAULT_HUNYUAN3D_2MV_MAX_VIEW_ANGLE_ERROR,
    required_views: Iterable[str] = DEFAULT_HUNYUAN3D_2MV_REQUIRED_VIEWS,
    disable_progress: bool = False,
) -> dict:
    provider_dir = Path(provider_dir).resolve()
    output_mesh = Path(output_mesh).resolve()
    if not (provider_dir / "hy3dgen" / "shapegen" / "pipelines.py").is_file():
        raise FileNotFoundError(f"Hunyuan3D-2 provider source is incomplete: {provider_dir}")
    if str(provider_dir) not in sys.path:
        sys.path.insert(0, str(provider_dir))

    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Hunyuan3D-2mv requested CUDA but torch.cuda.is_available() is false")
    snapshot_path = resolve_hunyuan3d_2mv_model_snapshot(
        model_repo=model_path,
        model_revision=model_revision,
        subfolder=subfolder,
    )
    images, view_manifest = prepare_hunyuan3d_2mv_images(
        Path(input_bundle),
        output_mesh.parent,
        max_angle_error=max_view_angle_error,
        required_views=required_views,
    )

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(snapshot_path),
        subfolder=subfolder,
        variant="fp16",
        use_safetensors=True,
        device=device,
        dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
    )
    load_runtime = time.perf_counter() - load_started
    generator = torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        mesh = pipeline(
            image=images,
            num_inference_steps=max(1, int(num_inference_steps)),
            guidance_scale=float(guidance_scale),
            octree_resolution=max(16, int(octree_resolution)),
            num_chunks=max(1, int(num_chunks)),
            generator=generator,
            output_type="trimesh",
            enable_pbar=not disable_progress,
        )[0]
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    inference_runtime = time.perf_counter() - inference_started
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_mesh)
    peak_gib = None
    peak_allocated_gib = None
    peak_reserved_gib = None
    peak_measurement = "unsupported"
    peak_supported = False
    if str(device).startswith("cuda"):
        peak_allocated_gib = float(torch.cuda.max_memory_allocated(device)) / float(
            1024**3
        )
        peak_reserved_gib = float(torch.cuda.max_memory_reserved(device)) / float(
            1024**3
        )
        peak_gib = peak_reserved_gib
        peak_measurement = "torch_peak_reserved"
        peak_supported = True
    metrics = {
        "provider": "hunyuan3d-2mv",
        "model": model_path,
        "model_revision": model_revision,
        "model_snapshot": str(snapshot_path),
        "subfolder": subfolder,
        "source_revision_expected": DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION,
        "source_geometry_used": False,
        "selected_views": view_manifest["views"],
        "model_load_runtime_seconds": load_runtime,
        "model_inference_runtime_seconds": inference_runtime,
        "provider_peak_cuda_vram_gib": peak_gib,
        "provider_peak_cuda_vram_supported": peak_supported,
        "provider_peak_cuda_vram_measurement": peak_measurement,
        "provider_peak_cuda_allocated_gib": peak_allocated_gib,
        "provider_peak_cuda_reserved_gib": peak_reserved_gib,
        "output_mesh": str(output_mesh),
    }
    (output_mesh.parent / PROVIDER_NATIVE_METRICS_FILENAME).write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official pinned Hunyuan3D-2mv multiview shape pipeline."
    )
    parser.add_argument("--provider-dir", required=True)
    parser.add_argument("--input-bundle")
    parser.add_argument("--output-mesh")
    parser.add_argument("--model-path", default=DEFAULT_HUNYUAN3D_2MV_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION)
    parser.add_argument("--subfolder", default=DEFAULT_HUNYUAN3D_2MV_SUBFOLDER)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_HUNYUAN3D_2MV_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_HUNYUAN3D_2MV_GUIDANCE)
    parser.add_argument("--octree-resolution", type=int, default=DEFAULT_HUNYUAN3D_2MV_OCTREE_RESOLUTION)
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_HUNYUAN3D_2MV_NUM_CHUNKS)
    parser.add_argument("--seed", type=int, default=DEFAULT_HUNYUAN3D_2MV_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--required-view",
        action="append",
        choices=HUNYUAN3D_2MV_VIEW_ORDER,
        default=None,
    )
    parser.add_argument(
        "--max-view-angle-error",
        type=float,
        default=DEFAULT_HUNYUAN3D_2MV_MAX_VIEW_ANGLE_ERROR,
    )
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--prefetch-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prefetch_only:
        path = resolve_hunyuan3d_2mv_model_snapshot(
            model_repo=args.model_path,
            model_revision=args.model_revision,
            subfolder=args.subfolder,
        )
        print(f"snapshot={path}")
        return
    if not args.input_bundle or not args.output_mesh:
        raise ValueError("--input-bundle and --output-mesh are required unless --prefetch-only is set")
    metrics = run_hunyuan3d_2mv(
        provider_dir=Path(args.provider_dir),
        input_bundle=Path(args.input_bundle),
        output_mesh=Path(args.output_mesh),
        model_path=args.model_path,
        model_revision=args.model_revision,
        subfolder=args.subfolder,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        seed=args.seed,
        device=args.device,
        max_view_angle_error=args.max_view_angle_error,
        required_views=args.required_view or DEFAULT_HUNYUAN3D_2MV_REQUIRED_VIEWS,
        disable_progress=args.disable_progress,
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
