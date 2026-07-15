"""Emit paired oracle and production-depth STLs for varied CC0 face scenes."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.face_part_metrics import (
    FACE_PART_AFFINE_MM_GATES,
    FACE_PART_GATES,
    face_part_affine_surface_error_metrics,
    face_part_cross_height_metrics,
)
from backend.benchmark.makehuman_face_fixture import FACE_PART_NAMES, load_makehuman_face_fixture
from backend.benchmark.run_makehuman_face_depth_smoke import (
    BACKGROUND_GATES,
    DEFAULT_ASSET_DIR,
    DEPTH_ANYTHING_V2_LARGE,
    _background_metrics,
    _infer_depth_anything,
    _make_scene,
)
from backend.benchmark.run_makehuman_face_relief_smoke import (
    DEFAULT_SCENES,
    PROVENANCE_PATHS,
    SceneSpec,
    _cross_height,
    _emit_row,
    _render_scene,
    _sha256,
)
from backend.benchmark.run_relief_scene_regression import _git_provenance
from backend.benchmark.run_relief_visual_sweep import (
    FACE_APPEARANCE_GATES,
    _appearance_checks,
)
from backend.pic_to_3d import (
    DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
    _resize_nan_aware,
    _surface_lighting_agreement_metrics,
)


DA2_PROVIDER = "depth-anything-v2-large"
DA2_MODEL_REVISION = "7581137eff8d4e94f6e796d3baea0e9fa79b22d2"
DA3_PROVIDER = "da3mono-large"
PROVIDER_NAMES = (DA2_PROVIDER, DA3_PROVIDER)
DA3_MODEL_ID = "depth-anything/DA3MONO-LARGE"
DA3_MODEL_REVISION = "f465978e618db8cc79c83b8bbf24964857db1875"
DA3_SOURCE_REPOSITORY = "https://github.com/ByteDance-Seed/Depth-Anything-3"
DA3_SOURCE_COMMIT = "3fe327a6abe2e5db95b54444ea95463dbfef5610"
DA3_MAX_PEAK_VRAM_GB = 10.5
PREDICTED_SCENES = (DEFAULT_SCENES[0], DEFAULT_SCENES[2])
PREDICTED_PROVENANCE_PATHS = (
    *PROVENANCE_PATHS,
    "backend/benchmark/run_makehuman_face_provider_relief_smoke.py",
)
_DA3_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _validate_da3_source(module_file: str | Path) -> dict:
    module_path = Path(module_file).resolve()
    repository = module_path.parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("DA3 source must be an exact clean git checkout") from exc
    if revision != DA3_SOURCE_COMMIT or status:
        raise RuntimeError(
            "DA3 source checkout does not match the clean pinned official commit"
        )
    return {
        "repository_path": str(repository),
        "revision": revision,
        "clean": True,
    }


def _prepare_inference_scene(
    output_dir: Path,
    fixture: dict,
    spec: SceneSpec,
    *,
    render_size: int,
    crop_size: int,
) -> dict:
    scene_dir = output_dir / "inference" / f"{spec.profile_name}_{spec.framing}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    rendered, framing = _render_scene(
        fixture,
        spec,
        render_size=render_size,
        crop_size=crop_size,
    )
    face_mask = np.asarray(rendered.silhouette, dtype=bool)
    part_masks = {
        name: np.asarray(rendered.part_masks[name], dtype=bool) & face_mask
        for name in FACE_PART_NAMES
    }
    exact_depth, source_rgb = _make_scene(
        rendered.depth,
        face_mask,
        rendered.rgb,
        phase=spec.phase_rad,
    )
    source_path = scene_dir / "source.png"
    exact_path = scene_dir / "exact_depth.npy"
    Image.fromarray(np.clip(source_rgb * 255.0, 0, 255).astype(np.uint8)).save(source_path)
    np.save(exact_path, exact_depth.astype(np.float32, copy=False))
    return {
        "scene_dir": scene_dir,
        "source_path": source_path,
        "source_sha256": _sha256(source_path),
        "exact_path": exact_path,
        "exact_depth": exact_depth,
        "face_mask": face_mask,
        "part_masks": part_masks,
        "framing": framing,
    }


def _infer_da3mono(source_path: Path, *, device: str) -> tuple[np.ndarray, dict]:
    try:
        import torch
        import depth_anything_3.api as da3_api
        from depth_anything_3.api import DepthAnything3
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "DA3Mono requires the pinned official depth-anything-3 package"
        ) from exc
    source_checkout = _validate_da3_source(da3_api.__file__)
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    snapshot = snapshot_download(
        DA3_MODEL_ID,
        revision=DA3_MODEL_REVISION,
        allow_patterns=("config.json", "model.safetensors"),
    )
    cache_key = (str(snapshot), str(resolved_device))
    load_started = time.perf_counter()
    if cache_key not in _DA3_MODEL_CACHE:
        model = DepthAnything3.from_pretrained(snapshot).to(resolved_device)
        model.eval()
        _DA3_MODEL_CACHE[cache_key] = model
    model = _DA3_MODEL_CACHE[cache_key]
    load_seconds = time.perf_counter() - load_started
    inference_started = time.perf_counter()
    prediction = model.inference([str(source_path)], process_res=504)
    inference_seconds = time.perf_counter() - inference_started
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    confidence = (
        np.asarray(prediction.conf[0], dtype=np.float32)
        if prediction.conf is not None
        else None
    )
    metadata = {
        "provider": DA3_PROVIDER,
        "model": DA3_MODEL_ID,
        "model_revision": DA3_MODEL_REVISION,
        "source_repository": DA3_SOURCE_REPOSITORY,
        "source_commit": DA3_SOURCE_COMMIT,
        "source_checkout": source_checkout,
        "license": "Apache-2.0",
        "depth_semantics": "relative-distance-far-high",
        "relief_transform": "inverse-depth",
        "process_res": 504,
        "model_load_seconds": float(load_seconds),
        "inference_seconds": float(inference_seconds),
    }
    if confidence is not None:
        metadata["confidence"] = {
            "available": True,
            "min": float(np.nanmin(confidence)),
            "median": float(np.nanmedian(confidence)),
            "max": float(np.nanmax(confidence)),
        }
    else:
        metadata["confidence"] = {"available": False}
    return depth, metadata


def _infer_cached_provider(
    scene: dict,
    *,
    provider: str,
    device: str,
) -> tuple[np.ndarray, dict]:
    if provider not in PROVIDER_NAMES:
        raise ValueError(f"Unsupported provider: {provider}")
    provider_dir = scene["scene_dir"] / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    peak_vram_gb = None
    resident_vram_gb = None
    try:
        import torch

        if device != "cpu" and torch.cuda.is_available():
            resident_vram_gb = float(torch.cuda.memory_allocated() / (1024**3))
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    if provider == DA2_PROVIDER:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            DEPTH_ANYTHING_V2_LARGE,
            revision=DA2_MODEL_REVISION,
        )
        raw_depth, metadata = _infer_depth_anything(
            scene["source_path"],
            provider_dir,
            device=device,
            model_name=snapshot,
        )
        value_transform = "linear"
        depth_semantics = "relative-near-high"
        model_id = DEPTH_ANYTHING_V2_LARGE
        model_revision = DA2_MODEL_REVISION
    else:
        raw_depth, metadata = _infer_da3mono(scene["source_path"], device=device)
        value_transform = "inverse-depth"
        depth_semantics = "relative-distance-far-high"
        model_id = DA3_MODEL_ID
        model_revision = DA3_MODEL_REVISION
    if torch is not None and device != "cpu" and torch.cuda.is_available():
        peak_vram_gb = float(
            max(torch.cuda.max_memory_allocated() / (1024**3) - (resident_vram_gb or 0.0), 0.0)
        )
    native = np.asarray(raw_depth, dtype=np.float32)
    resized = (
        native
        if native.shape == scene["exact_depth"].shape
        else _resize_nan_aware(native, scene["exact_depth"].shape)
    )
    native_path = provider_dir / "native_depth.npy"
    resized_path = provider_dir / "resized_depth.npy"
    manifest_path = provider_dir / "prediction_manifest.json"
    np.save(native_path, native)
    np.save(resized_path, resized.astype(np.float32, copy=False))
    if (
        provider == DA3_PROVIDER
        and peak_vram_gb is not None
        and peak_vram_gb > DA3_MAX_PEAK_VRAM_GB
    ):
        raise RuntimeError(
            f"{provider} exceeded the bounded VRAM guard: {peak_vram_gb:.3f} GB"
        )
    manifest = {
        "schema_version": 1,
        "source_sha256": scene["source_sha256"],
        "model_id": model_id,
        "model_revision": model_revision,
        "provider": provider,
        "depth_semantics": depth_semantics,
        "value_transform": value_transform,
        "normalization": "provider-native then selection-only p01-p99 at STL emission",
        "native_shape": list(native.shape),
        "resized_shape": list(resized.shape),
        "native_sha256": _sha256(native_path),
        "resized_sha256": _sha256(resized_path),
        "inference_seconds": float(time.perf_counter() - started),
        "peak_vram_gb": peak_vram_gb,
        "resident_vram_before_inference_gb": resident_vram_gb,
        "python": platform.python_version(),
        "provider_metadata": metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return resized.astype(np.float32, copy=False), manifest


def _paired_metrics(oracle_context: dict, candidate_context: dict) -> dict:
    oracle = np.load(oracle_context["surface_path"]).astype(np.float64)
    candidate = np.load(candidate_context["surface_path"]).astype(np.float64)
    shape_match = oracle.shape == candidate.shape
    transform_match = (
        oracle_context["surface_grid_transform"]
        == candidate_context["surface_grid_transform"]
    )
    pitch_match = bool(
        np.isclose(
            oracle_context["sample_pitch_mm"],
            candidate_context["sample_pitch_mm"],
            rtol=0.0,
            atol=1e-9,
        )
    )
    mask_match = bool(
        shape_match
        and np.array_equal(oracle_context["face_mask"], candidate_context["face_mask"])
    )
    if not (shape_match and transform_match and pitch_match and mask_match):
        return {
            "available": False,
            "checks": {
                "shape_match": shape_match,
                "transform_match": transform_match,
                "pitch_match": pitch_match,
                "mask_match": mask_match,
                "passed": False,
            },
        }
    face_mask = oracle_context["face_mask"]
    part_masks = oracle_context["part_masks"]
    pitch = oracle_context["sample_pitch_mm"]
    appearance = _surface_lighting_agreement_metrics(
        oracle,
        candidate,
        face_mask,
        sample_pitch_mm=pitch,
        component_metrics=False,
    )
    appearance_checks = _appearance_checks(appearance, FACE_APPEARANCE_GATES)
    named = face_part_cross_height_metrics(
        oracle,
        candidate,
        face_mask,
        part_masks,
        sample_pitch_mm=pitch,
    )
    affine = face_part_affine_surface_error_metrics(
        oracle,
        candidate,
        face_mask,
        part_masks,
    )
    background = _background_metrics(oracle, candidate, face_mask, pitch)
    checks = {
        "shape_match": shape_match,
        "transform_match": transform_match,
        "pitch_match": pitch_match,
        "mask_match": mask_match,
        "face_appearance": bool(appearance_checks["passed"]),
        "named_parts": bool(named["passed"]),
        "affine_mm": bool(affine["passed"]),
        "background": bool(background["passed"]),
    }
    return {
        "available": True,
        "appearance": appearance,
        "appearance_checks": appearance_checks,
        "named_parts": named,
        "affine_mm": affine,
        "background": background,
        "checks": {**checks, "passed": bool(all(checks.values()))},
    }


def run(
    output_dir: str | Path,
    *,
    asset_dir: str | Path = DEFAULT_ASSET_DIR,
    scenes: tuple[SceneSpec, ...] = PREDICTED_SCENES,
    relief_heights_mm: tuple[float, ...] = (30.0, 40.0),
    render_size: int = 384,
    crop_size: int = 256,
    physical_size_mm: float = 96.0,
    background_depth_ratio: float = DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
    providers: tuple[str, ...] = (DA2_PROVIDER,),
    device: str = "cuda",
    allow_dirty: bool = False,
    allow_failures: bool = False,
) -> dict:
    if len(relief_heights_mm) != 2 or len(set(relief_heights_mm)) != 2:
        raise ValueError("Exactly two distinct relief heights are required")
    unknown_providers = sorted(set(providers) - set(PROVIDER_NAMES))
    if unknown_providers or not providers:
        raise ValueError(f"Unsupported providers: {unknown_providers}")
    if len(set(providers)) != len(providers):
        raise ValueError("Duplicate providers are not allowed")
    if not 0 < float(background_depth_ratio) <= 1:
        raise ValueError("Predicted-depth smoke requires a positive background depth ratio")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = load_makehuman_face_fixture(asset_dir)
    provenance = _git_provenance(PREDICTED_PROVENANCE_PATHS)
    rows: list[dict] = []
    contexts: dict[str, dict] = {}
    paired: list[dict] = []
    inference_manifests: list[dict] = []
    for spec in scenes:
        scene = _prepare_inference_scene(
            output_dir,
            fixture,
            spec,
            render_size=render_size,
            crop_size=crop_size,
        )
        predictions = {}
        for provider in providers:
            prediction, manifest = _infer_cached_provider(
                scene,
                provider=provider,
                device=device,
            )
            predictions[provider] = (prediction, manifest)
            inference_manifests.append({"scene": spec.__dict__, **manifest})
        for height in relief_heights_mm:
            oracle_row, oracle_context = _emit_row(
                output_dir,
                fixture,
                spec,
                relief_height_mm=float(height),
                render_size=render_size,
                crop_size=crop_size,
                physical_size_mm=physical_size_mm,
                background_depth_ratio=background_depth_ratio,
            )
            oracle_source = output_dir / oracle_row["row_id"] / "source.png"
            rows.append(oracle_row)
            contexts[oracle_row["row_id"]] = oracle_context
            for provider in providers:
                prediction, manifest = predictions[provider]
                value_transform = manifest["value_transform"]
                candidate_row, candidate_context = _emit_row(
                    output_dir,
                    fixture,
                    spec,
                    relief_height_mm=float(height),
                    render_size=render_size,
                    crop_size=crop_size,
                    physical_size_mm=physical_size_mm,
                    background_depth_ratio=background_depth_ratio,
                    provider_name=provider,
                    input_depth=prediction,
                    value_transform=value_transform,
                    invert=value_transform == "inverse-depth",
                    low_percentile=1.0,
                    high_percentile=99.0,
                    use_part_feature_weight=False,
                )
                candidate_source = output_dir / candidate_row["row_id"] / "source.png"
                source_match = (
                    _sha256(oracle_source)
                    == _sha256(candidate_source)
                    == scene["source_sha256"]
                )
                pair = _paired_metrics(oracle_context, candidate_context)
                pair.update(
                    {
                        "provider": provider,
                        "profile_name": spec.profile_name,
                        "framing": spec.framing,
                        "relief_height_mm": float(height),
                        "oracle_row_id": oracle_row["row_id"],
                        "candidate_row_id": candidate_row["row_id"],
                        "source_match": source_match,
                    }
                )
                pair["checks"]["source_match"] = source_match
                pair["checks"]["passed"] = bool(all(pair["checks"].values()))
                paired.append(pair)
                rows.append(candidate_row)
                contexts[candidate_row["row_id"]] = candidate_context

    oracle_rows = [row for row in rows if row["provider"] == "oracle"]
    oracle_cross_height = _cross_height(oracle_rows, contexts)
    candidate_cross_height = {
        provider: _cross_height(
            [row for row in rows if row["provider"] == provider],
            contexts,
        )
        for provider in providers
    }
    expected_rows = len(scenes) * len(relief_heights_mm) * (1 + len(providers))
    checks = {
        "expected_rows": len(rows) == expected_rows,
        "unique_rows": len({row["row_id"] for row in rows}) == len(rows),
        "one_inference_per_scene_provider": len(inference_manifests)
        == len(scenes) * len(providers),
        "all_downstream_rows_passed": bool(rows)
        and all(row["checks"]["passed"] for row in rows),
        "all_paired_gates_passed": bool(paired)
        and all(record["checks"]["passed"] for record in paired),
        "oracle_cross_height_passed": bool(oracle_cross_height["passed"]),
        "candidate_cross_height_passed": all(
            record["passed"] for record in candidate_cross_height.values()
        ),
        "implementation_provenance_clean": bool(
            provenance.get("available") and provenance.get("clean")
        ),
    }
    summary = {
        "schema_version": 1,
        "run_kind": "makehuman_cc0_predicted_depth_to_high_relief_stl",
        "privacy": "checksum-pinned CC0 generated heads and deterministic analytic backgrounds only",
        "implementation_provenance": provenance,
        "matrix": {
            "scenes": [spec.__dict__ for spec in scenes],
            "providers": ["oracle", *providers],
            "relief_heights_mm": [float(value) for value in relief_heights_mm],
            "render_size": int(render_size),
            "crop_size": int(crop_size),
            "physical_size_mm": float(physical_size_mm),
            "background_depth_ratio": float(background_depth_ratio),
            "expected_rows": expected_rows,
            "completed_rows": len(rows),
        },
        "normalization_policy": {
            "oracle": "far-high exact depth, invert=true, p00-p100",
            "challengers": "provider-native semantics, selection-only p01-p99",
            "oracle_fit_used_for_stl": False,
            "oracle_part_masks_used_for_candidate_stl": False,
            "perfect_selection_control": True,
        },
        "gates": {
            "face_appearance": FACE_APPEARANCE_GATES,
            "named_parts": FACE_PART_GATES,
            "affine_mm": FACE_PART_AFFINE_MM_GATES,
            "background": BACKGROUND_GATES,
        },
        "inference": inference_manifests,
        "paired": paired,
        "cross_height": {
            "oracle": oracle_cross_height,
            **candidate_cross_height,
        },
        "rows": rows,
        "checks": {
            **checks,
            "passed": bool(
                checks["expected_rows"]
                and checks["unique_rows"]
                and checks["one_inference_per_scene_provider"]
                and checks["all_downstream_rows_passed"]
                and checks["all_paired_gates_passed"]
                and checks["oracle_cross_height_passed"]
                and checks["candidate_cross_height_passed"]
                and (allow_dirty or checks["implementation_provenance_clean"])
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["checks"]["passed"] and not allow_failures:
        raise RuntimeError("Predicted-depth face relief smoke failed")
    return summary


def _float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _string_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--relief-heights-mm", type=_float_tuple, default=(30.0, 40.0))
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--physical-size-mm", type=float, default=96.0)
    parser.add_argument("--providers", type=_string_tuple, default=(DA2_PROVIDER,))
    parser.add_argument(
        "--scene-profiles",
        type=_string_tuple,
        default=tuple(spec.profile_name for spec in PREDICTED_SCENES),
    )
    parser.add_argument(
        "--background-depth-ratio",
        type=float,
        default=DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    selected = tuple(
        spec for spec in PREDICTED_SCENES if spec.profile_name in args.scene_profiles
    )
    if len(selected) != len(args.scene_profiles):
        raise ValueError(f"Unknown or duplicate scene profiles: {args.scene_profiles}")
    run(
        args.output_dir,
        asset_dir=args.asset_dir,
        scenes=selected,
        relief_heights_mm=args.relief_heights_mm,
        render_size=args.render_size,
        crop_size=args.crop_size,
        physical_size_mm=args.physical_size_mm,
        background_depth_ratio=args.background_depth_ratio,
        providers=args.providers,
        device=args.device,
        allow_dirty=args.allow_dirty,
        allow_failures=args.allow_failures,
    )


if __name__ == "__main__":
    main()
