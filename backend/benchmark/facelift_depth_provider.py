"""Preflight FaceLift and normalize its Gaussian output to front-view depth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


FACELIFT_SOURCE_URL = "https://github.com/weijielyu/FaceLift.git"
FACELIFT_SOURCE_REVISION = "0a9420c1480db1e22daa6dee192b76a013295992"
FACELIFT_MODEL_ID = "wlyu/OpenFaceLift"
FACELIFT_MODEL_REVISION = "3ebac9f8e2d791f02507b8aa61520e1ba379a115"
FACELIFT_FRONT_CAMERA_INDEX = 2
FACELIFT_WEIGHT_LICENSE = (
    "Adobe Research License v1.2; noncommercial research only"
)
FACELIFT_REQUIRED_SOURCE_FILES = (
    "Adobe Research License v1.2.txt",
    "LICENSE",
    "inference.py",
    "configs/gslrm.yaml",
    "gslrm/model/gaussians_renderer.py",
    "utils_folder/opencv_cameras.json",
)
FACELIFT_REQUIRED_MODEL_FILES = (
    "LICENSE",
    "gslrm/ckpt_0000000000021125.pt",
    "mvdiffusion/pipeckpts/feature_extractor/preprocessor_config.json",
    "mvdiffusion/pipeckpts/image_encoder/config.json",
    "mvdiffusion/pipeckpts/image_noising_scheduler/scheduler_config.json",
    "mvdiffusion/pipeckpts/image_normalizer/config.json",
    "mvdiffusion/pipeckpts/image_normalizer/diffusion_pytorch_model.safetensors",
    "mvdiffusion/pipeckpts/model_index.json",
    "mvdiffusion/pipeckpts/image_encoder/model.safetensors",
    "mvdiffusion/pipeckpts/scheduler/scheduler_config.json",
    "mvdiffusion/pipeckpts/text_encoder/config.json",
    "mvdiffusion/pipeckpts/text_encoder/model.safetensors",
    "mvdiffusion/pipeckpts/tokenizer/merges.txt",
    "mvdiffusion/pipeckpts/tokenizer/special_tokens_map.json",
    "mvdiffusion/pipeckpts/tokenizer/tokenizer_config.json",
    "mvdiffusion/pipeckpts/tokenizer/vocab.json",
    "mvdiffusion/pipeckpts/unet/config.json",
    "mvdiffusion/pipeckpts/unet/diffusion_pytorch_model.safetensors",
    "mvdiffusion/pipeckpts/vae/config.json",
    "mvdiffusion/pipeckpts/vae/diffusion_pytorch_model.safetensors",
)
FACELIFT_MODEL_FILE_SHA256 = {
    "gslrm/ckpt_0000000000021125.pt": (
        "705e9aa37c7ea964bb06f25db060c12676a93d9ce6a365ac8b50128a3dc12662"
    ),
    "mvdiffusion/pipeckpts/image_encoder/model.safetensors": (
        "ae616c24393dd1854372b0639e5541666f7521cbe219669255e865cb7f89466a"
    ),
    (
        "mvdiffusion/pipeckpts/image_normalizer/"
        "diffusion_pytorch_model.safetensors"
    ): "a726e7159498c008a99fad02cb52280065ee93432f2af192a8511b7ddcf997a3",
    "mvdiffusion/pipeckpts/text_encoder/model.safetensors": (
        "bc1827c465450322616f06dea41596eac7d493f4e95904dcb51f0fc745c4e13f"
    ),
    "mvdiffusion/pipeckpts/unet/diffusion_pytorch_model.safetensors": (
        "2389a09216381a2020ac751c58735408feb4e76797c6f2393e66c7a0cedc1629"
    ),
    "mvdiffusion/pipeckpts/vae/diffusion_pytorch_model.safetensors": (
        "3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def facelift_preflight(
    provider_root: str | Path,
    model_root: str | Path | None = None,
) -> dict:
    provider_root = Path(provider_root).resolve()
    model_root = Path(model_root).resolve() if model_root else None
    missing_source = [
        relative
        for relative in FACELIFT_REQUIRED_SOURCE_FILES
        if not (provider_root / relative).is_file()
    ]
    source_revision = None
    source_status = None
    source_error = None
    try:
        source_revision = _git_output(provider_root, "rev-parse", "HEAD")
        source_status = _git_output(
            provider_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        source_error = f"{type(exc).__name__}: {exc}"

    missing_model = list(FACELIFT_REQUIRED_MODEL_FILES)
    model_files = {}
    model_hashes_pinned = False
    if model_root is not None:
        missing_model = []
        for relative in FACELIFT_REQUIRED_MODEL_FILES:
            path = model_root / relative
            if not path.is_file():
                missing_model.append(relative)
                continue
            actual_sha256 = _sha256(path)
            expected_sha256 = FACELIFT_MODEL_FILE_SHA256.get(relative)
            model_files[relative] = {
                "size_bytes": int(path.stat().st_size),
                "sha256": actual_sha256,
                "expected_sha256": expected_sha256,
                "hash_pinned": (
                    expected_sha256 is None
                    or actual_sha256 == expected_sha256
                ),
            }
        model_hashes_pinned = bool(
            not missing_model
            and all(
                model_files[relative]["hash_pinned"]
                for relative in FACELIFT_MODEL_FILE_SHA256
            )
        )

    checks = {
        "source_exists": provider_root.is_dir(),
        "source_revision_pinned": (
            source_revision == FACELIFT_SOURCE_REVISION
        ),
        "source_clean": source_status == "",
        "source_files_complete": not missing_source,
        "model_root_supplied": model_root is not None,
        "model_files_complete": not missing_model,
        "model_weight_hashes_pinned": model_hashes_pinned,
    }
    return {
        "schema_version": 1,
        "provider": "facelift",
        "source": {
            "url": FACELIFT_SOURCE_URL,
            "expected_revision": FACELIFT_SOURCE_REVISION,
            "actual_revision": source_revision,
            "status": source_status,
            "error": source_error,
            "root": str(provider_root),
            "missing_files": missing_source,
        },
        "model": {
            "id": FACELIFT_MODEL_ID,
            "expected_revision": FACELIFT_MODEL_REVISION,
            "root": str(model_root) if model_root else None,
            "missing_files": missing_model,
            "files": model_files,
        },
        "license": {
            "source_code": "Apache-2.0",
            "weights": FACELIFT_WEIGHT_LICENSE,
            "production_eligible": False,
            "research_only": True,
        },
        "checks": checks,
        "runnable": bool(all(checks.values())),
    }


def load_facelift_camera(
    provider_root: str | Path,
    camera_index: int = FACELIFT_FRONT_CAMERA_INDEX,
) -> dict:
    provider_root = Path(provider_root)
    camera_path = provider_root / "utils_folder" / "opencv_cameras.json"
    payload = json.loads(camera_path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not 0 <= int(camera_index) < len(frames):
        raise ValueError("FaceLift camera file has no requested frame")
    frame = frames[int(camera_index)]
    w2c = np.asarray(frame["w2c"], dtype=np.float64)
    if w2c.shape != (4, 4) or not np.all(np.isfinite(w2c)):
        raise ValueError("FaceLift camera transform is invalid")
    intrinsics = np.asarray(
        [frame["fx"], frame["fy"], frame["cx"], frame["cy"]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(intrinsics)) or np.any(intrinsics[:2] <= 0):
        raise ValueError("FaceLift camera intrinsics are invalid")
    source_width = float(frame.get("w", 2.0 * intrinsics[2]))
    source_height = float(frame.get("h", 2.0 * intrinsics[3]))
    if (
        not np.isfinite(source_width)
        or not np.isfinite(source_height)
        or source_width <= 0
        or source_height <= 0
    ):
        raise ValueError("FaceLift camera resolution is invalid")
    return {
        "index": int(camera_index),
        "w2c": w2c,
        "fx": float(intrinsics[0]),
        "fy": float(intrinsics[1]),
        "cx": float(intrinsics[2]),
        "cy": float(intrinsics[3]),
        "source_width": source_width,
        "source_height": source_height,
        "source_path": str(camera_path.resolve()),
    }


def scale_camera_intrinsics(
    camera: dict,
    *,
    height: int,
    width: int,
) -> dict:
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError("Depth dimensions must be positive")
    source_width = float(camera.get("source_width", width))
    source_height = float(camera.get("source_height", height))
    if (
        not np.isfinite(source_width)
        or not np.isfinite(source_height)
        or source_width <= 0
        or source_height <= 0
    ):
        raise ValueError("FaceLift camera resolution is invalid")
    scale_x = float(width) / source_width
    scale_y = float(height) / source_height
    scaled = dict(camera)
    scaled.update(
        {
            "fx": float(camera["fx"]) * scale_x,
            "fy": float(camera["fy"]) * scale_y,
            "cx": float(camera["cx"]) * scale_x,
            "cy": float(camera["cy"]) * scale_y,
            "render_width": width,
            "render_height": height,
            "intrinsics_scale_x": scale_x,
            "intrinsics_scale_y": scale_y,
        }
    )
    return scaled


def load_facelift_gaussians(path: str | Path) -> dict[str, np.ndarray]:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError(
            "Reading FaceLift Gaussian PLY requires the optional "
            "'plyfile' dependency"
        ) from exc

    path = Path(path)
    vertices = PlyData.read(path)["vertex"].data
    names = set(vertices.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            "FaceLift Gaussian PLY is missing properties: "
            + ", ".join(missing)
        )
    xyz = np.column_stack(
        [np.asarray(vertices[name], dtype=np.float64) for name in ("x", "y", "z")]
    )
    opacity_logits = np.asarray(vertices["opacity"], dtype=np.float64)
    log_scales = np.column_stack(
        [
            np.asarray(vertices[name], dtype=np.float64)
            for name in ("scale_0", "scale_1", "scale_2")
        ]
    )
    finite = (
        np.all(np.isfinite(xyz), axis=1)
        & np.isfinite(opacity_logits)
        & np.all(np.isfinite(log_scales), axis=1)
    )
    xyz = xyz[finite]
    opacity = 1.0 / (1.0 + np.exp(-np.clip(opacity_logits[finite], -30, 30)))
    scales = np.exp(np.clip(log_scales[finite], -20, 5))
    if not xyz.size:
        raise ValueError("FaceLift Gaussian PLY has no finite points")
    return {
        "xyz": xyz,
        "opacity": opacity,
        "scales": scales,
    }


def rasterize_gaussian_center_depth(
    xyz: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    camera: dict,
    *,
    height: int = 512,
    width: int = 512,
    minimum_opacity: float = 0.04,
    minimum_accumulated_alpha: float = 0.05,
    maximum_radius_pixels: float = 24.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    xyz = np.asarray(xyz, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64).reshape(-1)
    scales = np.asarray(scales, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Gaussian positions must have shape (N, 3)")
    if opacity.shape != (xyz.shape[0],) or scales.shape != xyz.shape:
        raise ValueError("Gaussian opacity or scale shape does not match positions")
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError("Depth dimensions must be positive")

    homogeneous = np.concatenate(
        [xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    camera_xyz = homogeneous @ np.asarray(camera["w2c"], dtype=np.float64).T
    camera_xyz = camera_xyz[:, :3]
    depth = camera_xyz[:, 2]
    valid = (
        np.all(np.isfinite(camera_xyz), axis=1)
        & np.all(np.isfinite(scales), axis=1)
        & np.isfinite(opacity)
        & (depth > 1e-6)
        & (opacity >= float(minimum_opacity))
    )
    camera_xyz = camera_xyz[valid]
    depth = depth[valid]
    opacity = opacity[valid]
    scales = scales[valid]
    if not depth.size:
        raise ValueError("No FaceLift Gaussians project in front of the camera")

    u = float(camera["fx"]) * camera_xyz[:, 0] / depth + float(camera["cx"])
    v = float(camera["fy"]) * camera_xyz[:, 1] / depth + float(camera["cy"])
    focal = math.sqrt(float(camera["fx"]) * float(camera["fy"]))
    sigma = focal * np.max(scales, axis=1) / depth
    sigma = np.clip(sigma, 0.5, float(maximum_radius_pixels) / 3.0)
    visible = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= -3.0 * sigma)
        & (u < width + 3.0 * sigma)
        & (v >= -3.0 * sigma)
        & (v < height + 3.0 * sigma)
    )
    order = np.argsort(depth[visible])
    u = u[visible][order]
    v = v[visible][order]
    depth = depth[visible][order]
    opacity = opacity[visible][order]
    sigma = sigma[visible][order]

    transmittance = np.ones((height, width), dtype=np.float64)
    accumulated_alpha = np.zeros((height, width), dtype=np.float64)
    weighted_depth = np.zeros((height, width), dtype=np.float64)
    rendered = 0
    for center_x, center_y, center_depth, alpha_base, spread in zip(
        u,
        v,
        depth,
        opacity,
        sigma,
        strict=True,
    ):
        radius = min(
            float(maximum_radius_pixels),
            max(1.0, math.ceil(3.0 * float(spread))),
        )
        left = max(0, int(math.floor(center_x - radius)))
        right = min(width, int(math.ceil(center_x + radius)) + 1)
        top = max(0, int(math.floor(center_y - radius)))
        bottom = min(height, int(math.ceil(center_y + radius)) + 1)
        if left >= right or top >= bottom:
            continue
        yy, xx = np.mgrid[top:bottom, left:right]
        distance_squared = (
            np.square(xx - center_x) + np.square(yy - center_y)
        )
        local_alpha = float(alpha_base) * np.exp(
            -0.5 * distance_squared / max(float(spread) ** 2, 1e-8)
        )
        local_alpha = np.clip(local_alpha, 0.0, 0.995)
        local_transmittance = transmittance[top:bottom, left:right]
        contribution = local_transmittance * local_alpha
        weighted_depth[top:bottom, left:right] += contribution * center_depth
        accumulated_alpha[top:bottom, left:right] += contribution
        transmittance[top:bottom, left:right] = (
            local_transmittance * (1.0 - local_alpha)
        )
        rendered += 1

    valid_pixels = accumulated_alpha >= float(minimum_accumulated_alpha)
    output = np.full((height, width), np.nan, dtype=np.float32)
    output[valid_pixels] = (
        weighted_depth[valid_pixels] / accumulated_alpha[valid_pixels]
    ).astype(np.float32)
    stats = {
        "input_gaussians": int(xyz.shape[0]),
        "front_facing_gaussians": int(np.count_nonzero(valid)),
        "rendered_gaussians": int(rendered),
        "finite_pixels": int(np.count_nonzero(valid_pixels)),
        "coverage_ratio": float(np.mean(valid_pixels)),
        "minimum_opacity": float(minimum_opacity),
        "minimum_accumulated_alpha": float(minimum_accumulated_alpha),
        "maximum_radius_pixels": float(maximum_radius_pixels),
        "approximation": (
            "front-to-back alpha composition of projected Gaussian centers; "
            "ellipsoid rotation is diagnostic-only and not modeled"
        ),
    }
    return output, accumulated_alpha.astype(np.float32), stats


def normalize_depth(depth: np.ndarray) -> tuple[np.ndarray, dict]:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if np.count_nonzero(finite) < 16:
        raise ValueError("FaceLift depth has insufficient finite coverage")
    low, high = np.percentile(depth[finite], (1.0, 99.0))
    span = float(high - low)
    if not np.isfinite(span) or span <= 1e-8:
        raise ValueError("FaceLift depth has no usable span")
    normalized = np.full(depth.shape, np.nan, dtype=np.float32)
    normalized[finite] = np.clip(
        (depth[finite] - float(low)) / span,
        0.0,
        1.0,
    )
    return normalized, {
        "method": "finite-p01-p99",
        "near_is_smaller": True,
        "p01": float(low),
        "p99": float(high),
        "span": span,
    }


def write_depth_outputs(
    provider_root: str | Path,
    gaussian_path: str | Path,
    output_depth: str | Path,
    *,
    output_alpha: str | Path | None = None,
    output_preview: str | Path | None = None,
    output_metadata: str | Path | None = None,
    height: int = 512,
    width: int = 512,
    camera_index: int = FACELIFT_FRONT_CAMERA_INDEX,
) -> dict:
    gaussian_path = Path(gaussian_path)
    output_depth = Path(output_depth)
    native_camera = load_facelift_camera(provider_root, camera_index)
    camera = scale_camera_intrinsics(
        native_camera,
        height=height,
        width=width,
    )
    gaussians = load_facelift_gaussians(gaussian_path)
    raw_depth, alpha, render_stats = rasterize_gaussian_center_depth(
        gaussians["xyz"],
        gaussians["opacity"],
        gaussians["scales"],
        camera,
        height=height,
        width=width,
    )
    depth, normalization = normalize_depth(raw_depth)
    output_depth.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_depth, depth)
    if output_alpha:
        alpha_path = Path(output_alpha)
        alpha_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(alpha_path, alpha)
    if output_preview:
        preview_path = Path(output_preview)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview = np.zeros(depth.shape, dtype=np.uint8)
        finite = np.isfinite(depth)
        preview[finite] = np.clip(
            np.rint((1.0 - depth[finite]) * 255.0),
            0,
            255,
        ).astype(np.uint8)
        Image.fromarray(preview, mode="L").save(preview_path)
    metadata = {
        "schema_version": 1,
        "provider": "facelift",
        "source_revision": FACELIFT_SOURCE_REVISION,
        "model_id": FACELIFT_MODEL_ID,
        "model_revision": FACELIFT_MODEL_REVISION,
        "weights_license": FACELIFT_WEIGHT_LICENSE,
        "production_eligible": False,
        "gaussian_path": str(gaussian_path.resolve()),
        "gaussian_sha256": _sha256(gaussian_path),
        "camera": {
            "index": camera["index"],
            "fx": camera["fx"],
            "fy": camera["fy"],
            "cx": camera["cx"],
            "cy": camera["cy"],
            "w2c": camera["w2c"].tolist(),
            "source_width": camera["source_width"],
            "source_height": camera["source_height"],
            "render_width": camera["render_width"],
            "render_height": camera["render_height"],
            "intrinsics_scale_x": camera["intrinsics_scale_x"],
            "intrinsics_scale_y": camera["intrinsics_scale_y"],
        },
        "render": render_stats,
        "normalization": normalization,
        "output_depth": str(output_depth.resolve()),
        "output_depth_sha256": _sha256(output_depth),
    }
    if output_metadata:
        metadata_path = Path(output_metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--provider-root", required=True)
    preflight_parser.add_argument("--model-root")
    preflight_parser.add_argument("--output")
    preflight_parser.add_argument("--require-runnable", action="store_true")

    depth_parser = subparsers.add_parser("depth")
    depth_parser.add_argument("--provider-root", required=True)
    depth_parser.add_argument("--gaussians", required=True)
    depth_parser.add_argument("--output-depth", required=True)
    depth_parser.add_argument("--output-alpha")
    depth_parser.add_argument("--output-preview")
    depth_parser.add_argument("--output-metadata")
    depth_parser.add_argument("--height", type=int, default=512)
    depth_parser.add_argument("--width", type=int, default=512)
    depth_parser.add_argument(
        "--camera-index",
        type=int,
        default=FACELIFT_FRONT_CAMERA_INDEX,
    )

    args = parser.parse_args()
    if args.command == "preflight":
        evidence = facelift_preflight(args.provider_root, args.model_root)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(evidence, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(evidence, indent=2))
        if args.require_runnable and not evidence["runnable"]:
            raise SystemExit(2)
        return
    evidence = write_depth_outputs(
        args.provider_root,
        args.gaussians,
        args.output_depth,
        output_alpha=args.output_alpha,
        output_preview=args.output_preview,
        output_metadata=args.output_metadata,
        height=args.height,
        width=args.width,
        camera_index=args.camera_index,
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
