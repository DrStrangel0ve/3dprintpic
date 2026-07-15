"""Compare monocular depth providers on a perspective CC0 face oracle."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from backend.benchmark.face_part_metrics import (
    FACE_PART_AFFINE_MM_GATES,
    FACE_PART_GATES,
    face_part_affine_surface_error_metrics,
    face_part_cross_height_metrics,
)
from backend.benchmark.makehuman_face_fixture import (
    FACE_PART_NAMES,
    load_makehuman_face_fixture,
    make_profile_vertex_colors,
)
from backend.benchmark.mesh_rendering import CameraSpec, RenderConfig, render_mesh
from backend.benchmark.metrics import fit_scale_shift
from backend.pic_to_3d import (
    _resize_nan_aware,
    process_image_get_depth_data_transformers,
)


DEFAULT_ASSET_DIR = Path(__file__).parent / "assets" / "makehuman_cc0_heads"
DEPTH_ANYTHING_V2_LARGE = "depth-anything/Depth-Anything-V2-Large-hf"
MOGE2_MODEL_ID = "Ruicheng/moge-2-vitb-normal"
MOGE2_SOURCE_COMMIT = "07444410f1e33f402353b99d6ccd26bd31e469e8"
PROVIDER_NAMES = ("oracle", "depth-anything-v2-large", "moge2-vitb-normal")
PROVENANCE_PATHS = (
    "backend/pic_to_3d.py",
    "backend/benchmark/face_part_metrics.py",
    "backend/benchmark/makehuman_face_fixture.py",
    "backend/benchmark/mesh_rendering.py",
    "backend/benchmark/run_makehuman_face_depth_smoke.py",
    "backend/benchmark/assets/makehuman_cc0_heads",
)
BACKGROUND_GATES = {
    "minimum_correlation": 0.75,
    "minimum_gradient_correlation": 0.65,
    "minimum_rms_retention": 0.45,
    "maximum_rms_retention": 2.20,
    "minimum_span_retention": 0.45,
    "maximum_span_retention": 2.20,
    "minimum_coverage_ratio": 0.995,
}
_MOGE_CACHE = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--short",
                "--untracked-files=all",
                "--",
                *PROVENANCE_PATHS,
            ],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "available": False,
            "clean": False,
            "revision": None,
            "paths": list(PROVENANCE_PATHS),
            "status": [],
            "error": type(exc).__name__,
        }
    return {
        "available": True,
        "clean": not bool(status),
        "revision": revision,
        "paths": list(PROVENANCE_PATHS),
        "status": status.splitlines() if status else [],
    }


def _correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    reference -= float(np.mean(reference))
    candidate -= float(np.mean(candidate))
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(reference, candidate) / denominator)


def _make_scene(
    rendered_depth: np.ndarray,
    face_mask: np.ndarray,
    face_rgb: np.ndarray,
    phase: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.indices(rendered_depth.shape, dtype=np.float32)
    x = 2.0 * columns / max(rendered_depth.shape[1] - 1, 1) - 1.0
    y = 2.0 * rows / max(rendered_depth.shape[0] - 1, 1) - 1.0
    background_depth = 0.72 + 0.06 * x + 0.045 * y
    background_depth += 0.035 * np.sin(2.4 * np.pi * x + phase)
    background_depth += 0.025 * np.cos(1.8 * np.pi * y - 0.3 * x)
    panel = (x > 0.35) & (y > -0.65) & (y < 0.35)
    shelf = (y > 0.36) & (y < 0.46) & (x < 0.38)
    background_depth = np.where(panel, background_depth - 0.055, background_depth)
    background_depth = np.where(shelf, background_depth - 0.035, background_depth)
    background_depth = np.clip(background_depth, 0.60, 0.92)

    background_rgb = np.stack(
        (
            0.40 + 0.18 * (1.0 - background_depth) + 0.08 * np.sin(3.1 * x),
            0.46 + 0.16 * x + 0.05 * np.cos(4.0 * y + phase),
            0.52 + 0.14 * y + 0.06 * np.sin(2.3 * x - 1.7 * y),
        ),
        axis=-1,
    )
    background_rgb[panel] = 0.75 * background_rgb[panel] + 0.25 * np.asarray(
        (0.63, 0.36, 0.30), dtype=np.float32
    )
    background_rgb[shelf] *= np.asarray((0.62, 0.56, 0.50), dtype=np.float32)
    texture = 0.012 * np.sin(71.0 * x + 43.0 * y + phase)
    background_rgb += texture[..., None]

    exact_depth = np.where(
        face_mask,
        0.08 + 0.50 * np.asarray(rendered_depth, dtype=np.float32),
        background_depth,
    ).astype(np.float32)
    source_rgb = np.clip(background_rgb, 0.0, 1.0).astype(np.float32)
    source_rgb[face_mask] = face_rgb[face_mask]
    return exact_depth, source_rgb


def _save_masks(
    row_dir: Path,
    face_mask: np.ndarray,
    part_masks: dict[str, np.ndarray],
) -> Path:
    mask_dir = row_dir / "face_parts"
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(face_mask.astype(np.uint8) * 255).save(mask_dir / "face.png")
    files = {}
    for name in FACE_PART_NAMES:
        path = mask_dir / f"{name}.png"
        Image.fromarray(np.asarray(part_masks[name], dtype=np.uint8) * 255).save(path)
        files[name] = path.relative_to(row_dir).as_posix()
    metadata = {
        "part_mask_schema_version": 1,
        "part_names": list(FACE_PART_NAMES),
        "faces": [
            {
                "index": 0,
                "detector": "makehuman-target-weight-perspective-z-buffer",
                "part_masks": {
                    "schema_version": 1,
                    "coordinate_space": "source-depth",
                    "face_file": "face_parts/face.png",
                    "files": files,
                    "complete": True,
                },
            }
        ],
    }
    path = row_dir / "face_part_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _infer_moge2(
    image_path: Path,
    *,
    device: str,
    resolution_level: int,
) -> tuple[np.ndarray, dict]:
    import torch
    from moge.model.v2 import MoGeModel

    resolved = _resolve_device(device)
    key = (MOGE2_MODEL_ID, resolved)
    load_started = time.perf_counter()
    if key not in _MOGE_CACHE:
        _MOGE_CACHE[key] = MoGeModel.from_pretrained(MOGE2_MODEL_ID).to(resolved).eval()
    model = _MOGE_CACHE[key]
    load_seconds = time.perf_counter() - load_started
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).to(resolved)
    if resolved.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(resolved)
    started = time.perf_counter()
    output = model.infer(
        tensor,
        resolution_level=int(resolution_level),
        use_fp16=resolved.startswith("cuda"),
        apply_mask=False,
    )
    inference_seconds = time.perf_counter() - started
    depth = output["depth"].detach().float().cpu().numpy()
    peak_vram = (
        float(torch.cuda.max_memory_allocated(resolved) / 1024**3)
        if resolved.startswith("cuda")
        else 0.0
    )
    return depth, {
        "provider": "moge2",
        "model": MOGE2_MODEL_ID,
        "source_repository": "https://github.com/microsoft/MoGe",
        "source_commit": MOGE2_SOURCE_COMMIT,
        "model_license": "MIT",
        "depth_semantics": "metric-distance-far-high",
        "relief_transform": "inverse-depth",
        "resolution_level": int(resolution_level),
        "device": resolved,
        "model_load_seconds": float(load_seconds),
        "inference_seconds": float(inference_seconds),
        "peak_vram_gb": peak_vram,
    }


def _infer_depth_anything(
    image_path: Path,
    output_dir: Path,
    *,
    device: str,
    model_name: str = DEPTH_ANYTHING_V2_LARGE,
) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    depth_path = process_image_get_depth_data_transformers(
        image_path,
        output_dir=output_dir,
        model_name=model_name,
        device=device,
        requested_model_name=DEPTH_ANYTHING_V2_LARGE,
    )
    elapsed = time.perf_counter() - started
    metadata_path = output_dir / "output_depth_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "provider": "depth-anything-v2",
            "model": model_name,
            "depth_semantics": "relative-near-high",
            "relief_transform": "linear",
            "inference_seconds": float(elapsed),
        }
    )
    return np.load(depth_path), metadata


def _relief_signal(depth: np.ndarray, transform: str) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if transform == "linear":
        return depth.copy()
    if transform == "one-minus-depth":
        return 1.0 - depth
    if transform != "inverse-depth":
        raise ValueError(f"Unsupported relief transform: {transform}")
    positive = np.isfinite(depth) & (depth > 0)
    if not np.any(positive):
        return np.full_like(depth, np.nan)
    floor = max(float(np.percentile(depth[positive], 0.1)), np.finfo(np.float32).tiny)
    signal = np.full_like(depth, np.nan)
    signal[positive] = 1.0 / np.maximum(depth[positive], floor)
    return signal


def _background_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    face_mask: np.ndarray,
    sample_pitch_mm: float,
) -> dict:
    expected = (~face_mask) & np.isfinite(reference)
    measured = expected & np.isfinite(candidate)
    coverage = float(np.count_nonzero(measured) / max(np.count_nonzero(expected), 1))
    if np.count_nonzero(measured) < 64:
        return {"available": False, "passed": False, "coverage_ratio": coverage}
    ref = reference[measured]
    pred = candidate[measured]
    ref_rms = float(np.std(ref))
    pred_rms = float(np.std(pred))
    ref_span = float(np.percentile(ref, 95) - np.percentile(ref, 5))
    pred_span = float(np.percentile(pred, 95) - np.percentile(pred, 5))
    sigma = 0.8 / max(float(sample_pitch_mm), 1e-6)
    reference_smooth = gaussian_filter(reference, sigma=sigma)
    candidate_filled = np.where(np.isfinite(candidate), candidate, 0.0)
    candidate_weight = gaussian_filter(np.isfinite(candidate).astype(np.float32), sigma=sigma)
    candidate_smooth = gaussian_filter(candidate_filled, sigma=sigma) / np.maximum(
        candidate_weight, 1e-6
    )
    reference_gradients = np.gradient(reference_smooth, sample_pitch_mm)
    candidate_gradients = np.gradient(candidate_smooth, sample_pitch_mm)
    gradient_correlations = [
        _correlation(reference_gradient[measured], candidate_gradient[measured])
        for reference_gradient, candidate_gradient in zip(
            reference_gradients, candidate_gradients
        )
    ]
    metrics = {
        "available": True,
        "coverage_ratio": coverage,
        "correlation": _correlation(ref, pred),
        "gradient_correlation": float(min(gradient_correlations)),
        "rms_retention": pred_rms / max(ref_rms, 1e-12),
        "span_retention": pred_span / max(ref_span, 1e-12),
        "reference_span_mm": ref_span,
        "candidate_span_mm": pred_span,
        "gates": BACKGROUND_GATES,
    }
    checks = {
        "coverage": metrics["coverage_ratio"] >= BACKGROUND_GATES["minimum_coverage_ratio"],
        "correlation": metrics["correlation"] >= BACKGROUND_GATES["minimum_correlation"],
        "gradient_correlation": metrics["gradient_correlation"]
        >= BACKGROUND_GATES["minimum_gradient_correlation"],
        "rms_retention": BACKGROUND_GATES["minimum_rms_retention"]
        <= metrics["rms_retention"]
        <= BACKGROUND_GATES["maximum_rms_retention"],
        "span_retention": BACKGROUND_GATES["minimum_span_retention"]
        <= metrics["span_retention"]
        <= BACKGROUND_GATES["maximum_span_retention"],
    }
    return {**metrics, "checks": checks, "passed": bool(all(checks.values()))}


def _evaluate_prediction(
    exact_depth: np.ndarray,
    predicted_depth: np.ndarray,
    face_mask: np.ndarray,
    part_masks: dict[str, np.ndarray],
    *,
    transform: str,
    relief_height_mm: float,
    physical_size_mm: float,
) -> tuple[np.ndarray, dict]:
    predicted = np.asarray(predicted_depth, dtype=np.float32)
    if predicted.shape != exact_depth.shape:
        predicted = _resize_nan_aware(predicted, exact_depth.shape)
    exact_signal = 1.0 - np.asarray(exact_depth, dtype=np.float32)
    exact_min = float(np.nanmin(exact_signal))
    exact_span = float(np.nanmax(exact_signal) - exact_min)
    reference = (exact_signal - exact_min) * (
        float(relief_height_mm) / max(exact_span, 1e-6)
    )
    signal = _relief_signal(predicted, transform)
    fit_mask = face_mask & np.isfinite(signal) & np.isfinite(reference)
    if np.count_nonzero(fit_mask) < 64:
        aligned = np.full_like(reference, np.nan)
        scale, shift = np.nan, np.nan
    else:
        scale, shift = fit_scale_shift(signal, reference, fit_mask)
        aligned = signal * float(scale) + float(shift)
    pitch = float(physical_size_mm) / max(exact_depth.shape[1] - 1, 1)
    named = face_part_cross_height_metrics(
        reference,
        aligned,
        face_mask,
        part_masks,
        sample_pitch_mm=pitch,
    )
    affine = face_part_affine_surface_error_metrics(
        reference,
        aligned,
        face_mask,
        part_masks,
    )
    background = _background_metrics(reference, aligned, face_mask, pitch)
    checks = {
        "named_parts": bool(named["passed"]),
        "affine_mm": bool(affine["passed"]),
        "background": bool(background["passed"]),
    }
    return aligned.astype(np.float32), {
        "fit": {"scale": float(scale), "shift": float(shift)},
        "named_parts": named,
        "affine_mm": affine,
        "background": background,
        "checks": {**checks, "passed": bool(all(checks.values()))},
    }


def run(
    output_dir: str | Path,
    *,
    asset_dir: str | Path = DEFAULT_ASSET_DIR,
    profile_name: str = "caucasian_female_smile",
    providers: tuple[str, ...] = PROVIDER_NAMES,
    relief_height_mm: float = 30.0,
    physical_size_mm: float = 96.0,
    render_size: int = 384,
    device: str = "auto",
    moge_resolution_level: int = 7,
    allow_failures: bool = False,
) -> dict:
    unknown = sorted(set(providers) - set(PROVIDER_NAMES))
    if unknown:
        raise ValueError(f"Unknown providers: {unknown}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = load_makehuman_face_fixture(asset_dir)
    if profile_name not in fixture["profiles"]:
        raise ValueError(f"Unknown MakeHuman profile: {profile_name}")
    profile = fixture["profiles"][profile_name]
    colors = make_profile_vertex_colors(
        profile["mesh"].vertices,
        profile["skin_tone"],
        fixture["part_weights"],
        fixture["surface_weights"],
    )
    rendered = render_mesh(
        profile["mesh"],
        CameraSpec(azimuth_deg=0.0, elevation_deg=0.0),
        RenderConfig(
            size=int(render_size),
            projection="perspective",
            perspective_fov_y_deg=32.0,
            camera_distance=3.2,
            background_rgb=(0.84, 0.87, 0.91),
            ambient=0.42,
            diffuse=0.53,
            specular=0.035,
            shininess=48.0,
            light_direction=(-0.35, -0.20, 0.90),
        ),
        (180, 120, 100),
        vertex_part_weights=fixture["part_weights"],
        vertex_colors=colors,
    )
    face_mask = np.asarray(rendered.silhouette, dtype=bool)
    part_masks = {
        name: np.asarray(rendered.part_masks[name], dtype=bool)
        for name in FACE_PART_NAMES
    }
    exact_depth, source_rgb = _make_scene(
        rendered.depth,
        face_mask,
        rendered.rgb,
        phase=0.73,
    )
    source_path = output_dir / "source.png"
    exact_path = output_dir / "exact_depth.npy"
    Image.fromarray(np.clip(source_rgb * 255.0, 0, 255).astype(np.uint8)).save(source_path)
    np.save(exact_path, exact_depth)
    mask_metadata_path = _save_masks(output_dir, face_mask, part_masks)

    rows = []
    for provider in providers:
        provider_dir = output_dir / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        if provider == "oracle":
            raw_depth = exact_depth
            metadata = {
                "provider": "oracle",
                "model": "perspective-z-buffer",
                "depth_semantics": "distance-far-high",
                "relief_transform": "one-minus-depth",
                "inference_seconds": 0.0,
                "peak_vram_gb": 0.0,
            }
        elif provider == "depth-anything-v2-large":
            raw_depth, metadata = _infer_depth_anything(
                source_path, provider_dir, device=device
            )
        else:
            raw_depth, metadata = _infer_moge2(
                source_path,
                device=device,
                resolution_level=moge_resolution_level,
            )
        raw_depth = np.asarray(raw_depth, dtype=np.float32)
        raw_path = provider_dir / "raw_depth.npy"
        np.save(raw_path, raw_depth)
        aligned, metrics = _evaluate_prediction(
            exact_depth,
            raw_depth,
            face_mask,
            part_masks,
            transform=metadata["relief_transform"],
            relief_height_mm=relief_height_mm,
            physical_size_mm=physical_size_mm,
        )
        aligned_path = provider_dir / "aligned_relief_mm.npy"
        np.save(aligned_path, aligned)
        metadata_path = provider_dir / "provider.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        rows.append(
            {
                "provider": provider,
                "metadata": metadata,
                "metrics": metrics,
                "artifacts": {
                    path.name: {
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in (raw_path, aligned_path, metadata_path)
                },
            }
        )

    non_oracle = [row for row in rows if row["provider"] != "oracle"]
    provenance = _git_provenance()
    checks = {
        "oracle_passed": bool(
            next(row for row in rows if row["provider"] == "oracle")["metrics"][
                "checks"
            ]["passed"]
        )
        if "oracle" in providers
        else True,
        "all_requested_completed": len(rows) == len(providers),
        "implementation_provenance_clean": bool(
            provenance.get("available") and provenance.get("clean")
        ),
        "at_least_one_challenger_passed": bool(non_oracle)
        and any(row["metrics"]["checks"]["passed"] for row in non_oracle),
    }
    summary = {
        "schema_version": 1,
        "run_kind": "makehuman_cc0_perspective_face_depth_provider_smoke",
        "privacy": "CC0 parametric synthetic heads and deterministic procedural materials only",
        "implementation_provenance": provenance,
        "fixture": fixture["manifest"],
        "profile": profile_name,
        "matrix": {
            "providers": list(providers),
            "relief_height_mm": float(relief_height_mm),
            "physical_size_mm": float(physical_size_mm),
            "render_size": int(render_size),
            "moge_resolution_level": int(moge_resolution_level),
        },
        "gates": {
            "named_parts": FACE_PART_GATES,
            "affine_mm": FACE_PART_AFFINE_MM_GATES,
            "background": BACKGROUND_GATES,
        },
        "source_artifacts": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (source_path, exact_path, mask_metadata_path)
        },
        "checks": {
            **checks,
            "passed": bool(
                checks["oracle_passed"]
                and checks["all_requested_completed"]
                and checks["implementation_provenance_clean"]
                and checks["at_least_one_challenger_passed"]
            ),
        },
        "rows": rows,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not summary["checks"]["passed"] and not allow_failures:
        raise RuntimeError("MakeHuman face depth smoke failed its expansion gate")
    gc.collect()
    return summary


def _tuple(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one provider is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="backend/output/makehuman-face-depth-smoke"
    )
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--profile", default="caucasian_female_smile")
    parser.add_argument("--providers", type=_tuple, default=PROVIDER_NAMES)
    parser.add_argument("--relief-height-mm", type=float, default=30.0)
    parser.add_argument("--physical-size-mm", type=float, default=96.0)
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--moge-resolution-level", type=int, default=7)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    summary = run(
        args.output_dir,
        asset_dir=args.asset_dir,
        profile_name=args.profile,
        providers=args.providers,
        relief_height_mm=args.relief_height_mm,
        physical_size_mm=args.physical_size_mm,
        render_size=args.render_size,
        device=args.device,
        moge_resolution_level=args.moge_resolution_level,
        allow_failures=args.allow_failures,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
