"""Train a bounded background-depth correction without modifying selected faces."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.metrics import fit_scale_shift
from backend.benchmark.run_makehuman_face_depth_smoke import (
    BACKGROUND_GATES,
    DEFAULT_ASSET_DIR,
    _background_metrics,
)
from backend.benchmark.run_makehuman_face_provider_relief_smoke import (
    DA2_MODEL_REVISION,
    DA2_PROVIDER,
    _infer_cached_provider,
    _prepare_inference_scene,
)
from backend.benchmark.run_makehuman_face_relief_smoke import (
    DEFAULT_SCENES,
    SceneSpec,
    _render_scene,
    _sha256,
)
from backend.benchmark.makehuman_face_fixture import load_makehuman_face_fixture


CORPUS_SEED = 20260715
MODEL_SEED = 80431
BACKGROUND_RESIDUAL_BOUND = 3.0
PHYSICAL_SIZE_MM = 96.0
DEPTH_ARRAY_GATES = {
    "minimum_correlation": 0.75,
    "minimum_gradient_correlation": 0.65,
    "minimum_coverage_ratio": 0.995,
}
PROVENANCE_PATHS = (
    "backend/benchmark/run_background_residual_training_smoke.py",
    "backend/benchmark/run_makehuman_face_provider_relief_smoke.py",
    "backend/benchmark/run_makehuman_face_relief_smoke.py",
    "backend/benchmark/run_makehuman_face_depth_smoke.py",
    "backend/benchmark/assets/makehuman_cc0_heads",
)


@dataclass(frozen=True)
class CorpusSceneSpec:
    scene_id: str
    split: str
    geometry_seed: int
    appearance_seed: int
    profile_name: str
    framing: str
    geometry_family: str


def _build_corpus_specs() -> tuple[CorpusSceneSpec, ...]:
    train_families = (
        "legacy_room",
        "tilted_panels",
        "radial_objects",
        "crossed_shelves",
    )
    validation_families = (
        "legacy_room",
        "tilted_panels",
        "radial_objects",
        "crossed_shelves",
    )
    specs = []
    for index in range(16):
        geometry_index = index // 2
        specs.append(
            CorpusSceneSpec(
                scene_id=f"train_{index:02d}",
                split="train",
                geometry_seed=CORPUS_SEED + 17 * geometry_index,
                appearance_seed=CORPUS_SEED + 4001 + 31 * index,
                profile_name="african_male_neutral",
                framing="left_frame",
                geometry_family=train_families[geometry_index % len(train_families)],
            )
        )
    for index in range(4):
        specs.append(
            CorpusSceneSpec(
                scene_id=f"validation_{index:02d}",
                split="validation",
                geometry_seed=CORPUS_SEED + 1009 + 29 * index,
                appearance_seed=CORPUS_SEED + 8009 + 37 * index,
                profile_name="african_male_neutral",
                framing="left_frame",
                geometry_family=validation_families[(index + 2) % len(validation_families)],
            )
        )
    return tuple(specs)


CORPUS_SPECS = _build_corpus_specs()
FINAL_HELD_OUT_SCENES = (DEFAULT_SCENES[0], DEFAULT_SCENES[2])


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _git_provenance() -> dict:
    import subprocess

    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all", "--", *PROVENANCE_PATHS],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "clean": False, "error": type(exc).__name__}
    return {
        "available": True,
        "clean": not bool(status),
        "revision": revision,
        "paths": list(PROVENANCE_PATHS),
        "status": status.splitlines() if status else [],
    }


def _procedural_background(
    shape: tuple[int, int],
    spec: CorpusSceneSpec,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return far-high depth and RGB with randomized, depth-consistent lighting."""
    geometry_rng = np.random.default_rng(spec.geometry_seed)
    appearance_rng = np.random.default_rng(spec.appearance_seed)
    rows, columns = np.indices(shape, dtype=np.float32)
    x = 2.0 * columns / max(shape[1] - 1, 1) - 1.0
    y = 2.0 * rows / max(shape[0] - 1, 1) - 1.0
    phase = float(geometry_rng.uniform(-np.pi, np.pi))
    slope_x = float(geometry_rng.uniform(-0.075, 0.075))
    slope_y = float(geometry_rng.uniform(-0.060, 0.060))
    depth = 0.74 + slope_x * x + slope_y * y
    depth += float(geometry_rng.uniform(0.018, 0.045)) * np.sin(
        float(geometry_rng.uniform(1.4, 3.2)) * np.pi * x + phase
    )
    depth += float(geometry_rng.uniform(0.012, 0.032)) * np.cos(
        float(geometry_rng.uniform(1.2, 2.8)) * np.pi * y - 0.37 * x - phase
    )

    geometry = {}
    if spec.geometry_family == "legacy_room":
        panel_x = float(geometry_rng.uniform(-0.50, 0.42))
        panel_y0 = float(geometry_rng.uniform(-0.72, -0.30))
        panel_y1 = float(geometry_rng.uniform(0.16, 0.58))
        panel = (x > panel_x) & (y > panel_y0) & (y < panel_y1)
        shelf_y = float(geometry_rng.uniform(0.22, 0.58))
        shelf = (np.abs(y - shelf_y) < float(geometry_rng.uniform(0.035, 0.075))) & (
            x < float(geometry_rng.uniform(0.15, 0.72))
        )
        depth = np.where(panel, depth - float(geometry_rng.uniform(0.035, 0.085)), depth)
        depth = np.where(shelf, depth - float(geometry_rng.uniform(0.025, 0.065)), depth)
        geometry = {"panel_x": panel_x, "panel_y": [panel_y0, panel_y1], "shelf_y": shelf_y}
    elif spec.geometry_family == "tilted_panels":
        diagonal = x + float(geometry_rng.uniform(-0.8, 0.8)) * y
        split = float(geometry_rng.uniform(-0.35, 0.35))
        depth += np.where(diagonal > split, float(geometry_rng.uniform(-0.09, -0.04)), 0.0)
        strip = np.abs(diagonal - float(geometry_rng.uniform(-0.6, 0.6))) < float(
            geometry_rng.uniform(0.035, 0.08)
        )
        depth = np.where(strip, depth + float(geometry_rng.uniform(0.025, 0.060)), depth)
        geometry = {"diagonal_split": split}
    elif spec.geometry_family == "radial_objects":
        centers = []
        for _ in range(3):
            cx, cy = geometry_rng.uniform(-0.85, 0.85, size=2)
            radius = float(geometry_rng.uniform(0.16, 0.42))
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mound = np.clip(1.0 - distance / radius, 0.0, 1.0)
            depth -= float(geometry_rng.uniform(0.035, 0.095)) * mound**2
            centers.append([float(cx), float(cy), radius])
        geometry = {"radial_objects": centers}
    elif spec.geometry_family == "crossed_shelves":
        horizontal_y = float(geometry_rng.uniform(-0.55, 0.55))
        vertical_x = float(geometry_rng.uniform(-0.65, 0.65))
        width = float(geometry_rng.uniform(0.035, 0.075))
        horizontal = np.abs(y - horizontal_y) < width
        vertical = np.abs(x - vertical_x) < width
        depth = np.where(horizontal, depth - float(geometry_rng.uniform(0.025, 0.070)), depth)
        depth = np.where(vertical, depth + float(geometry_rng.uniform(0.020, 0.055)), depth)
        geometry = {"horizontal_y": horizontal_y, "vertical_x": vertical_x, "width": width}
    else:
        raise ValueError(f"Unsupported geometry family: {spec.geometry_family}")
    depth = np.clip(depth, 0.56, 0.94).astype(np.float32)

    # Random albedo and randomized lighting prevent a fixed color channel from being
    # a shortcut while retaining a real image cue for local surface orientation.
    gy, gx = np.gradient(depth)
    normal = np.stack((-3.2 * gx, -3.2 * gy, np.ones_like(depth)), axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    light = appearance_rng.normal(size=3).astype(np.float32)
    light[2] = abs(float(light[2])) + 0.75
    light /= np.linalg.norm(light)
    shading = 0.52 + 0.38 * np.maximum(np.sum(normal * light, axis=-1), 0.0)
    appearance_phase = float(appearance_rng.uniform(-np.pi, np.pi))
    base_color = appearance_rng.uniform(0.30, 0.72, size=3).astype(np.float32)
    color_wave = (
        0.09 * np.sin(float(appearance_rng.uniform(1.0, 4.0)) * x + appearance_phase)
        + 0.07 * np.cos(float(appearance_rng.uniform(1.0, 4.0)) * y - 0.7 * appearance_phase)
    )
    texture = 0.018 * np.sin(
        float(appearance_rng.uniform(31.0, 79.0)) * x
        + float(appearance_rng.uniform(29.0, 73.0)) * y
        + appearance_phase
    )
    rgb = base_color[None, None, :] * shading[..., None]
    rgb += color_wave[..., None] * appearance_rng.uniform(-0.7, 0.7, size=3)
    rgb += texture[..., None]
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    metadata = {
        "geometry_seed": int(spec.geometry_seed),
        "appearance_seed": int(spec.appearance_seed),
        "geometry_family": spec.geometry_family,
        "slope_x": slope_x,
        "slope_y": slope_y,
        "phase": phase,
        "light_direction": [float(value) for value in light],
        "base_color": [float(value) for value in base_color],
        "appearance_phase": appearance_phase,
        "direct_depth_color_cue": False,
        "geometry": geometry,
    }
    return depth, rgb, metadata


def _compose_corpus_scene(rendered, spec: CorpusSceneSpec) -> tuple[np.ndarray, np.ndarray, dict]:
    face_mask = np.asarray(rendered.silhouette, dtype=bool)
    background_depth, background_rgb, metadata = _procedural_background(
        rendered.depth.shape,
        spec,
    )
    exact_depth = np.where(
        face_mask,
        0.08 + 0.50 * np.asarray(rendered.depth, dtype=np.float32),
        background_depth,
    ).astype(np.float32)
    source_rgb = background_rgb.copy()
    source_rgb[face_mask] = np.asarray(rendered.rgb, dtype=np.float32)[face_mask]
    return exact_depth, source_rgb, metadata


def _prepare_corpus_scene(
    root: Path,
    fixture: dict,
    spec: CorpusSceneSpec,
    *,
    render_size: int,
    crop_size: int,
) -> dict:
    scene_dir = root / "corpus" / spec.split / spec.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    render_spec = SceneSpec(
        profile_name=spec.profile_name,
        framing=spec.framing,
        phase_rad=float(spec.geometry_seed % 997) / 997.0 * 2.0 * np.pi,
    )
    rendered, framing = _render_scene(
        fixture,
        render_spec,
        render_size=render_size,
        crop_size=crop_size,
    )
    exact_depth, source_rgb, generator = _compose_corpus_scene(rendered, spec)
    face_mask = np.asarray(rendered.silhouette, dtype=bool)
    source_path = scene_dir / "source.png"
    exact_path = scene_dir / "exact_depth.npy"
    mask_path = scene_dir / "selection_mask.npy"
    Image.fromarray(np.clip(source_rgb * 255.0, 0, 255).astype(np.uint8)).save(source_path)
    np.save(exact_path, exact_depth)
    np.save(mask_path, face_mask)
    return {
        "scene_id": spec.scene_id,
        "split": spec.split,
        "spec": asdict(spec),
        "scene_dir": scene_dir,
        "source_path": source_path,
        "source_sha256": _sha256(source_path),
        "exact_path": exact_path,
        "exact_depth": exact_depth,
        "face_mask": face_mask,
        "source_rgb": np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32) / 255.0,
        "framing": framing,
        "generator": generator,
        "artifacts": {
            "exact_depth_sha256": _sha256_array(exact_depth),
            "selection_mask_sha256": _sha256_array(face_mask),
        },
    }


def _provider_space_target(
    exact_depth: np.ndarray,
    provider_depth: np.ndarray,
    face_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Map exact near-high depth into the unchanged provider face's affine space."""
    exact_signal = 1.0 - np.asarray(exact_depth, dtype=np.float32)
    provider = np.asarray(provider_depth, dtype=np.float32)
    face = np.asarray(face_mask, dtype=bool)
    fit = face & np.isfinite(exact_signal) & np.isfinite(provider)
    if np.count_nonzero(fit) < 64:
        raise ValueError("At least 64 finite selected pixels are required")
    scale, shift = fit_scale_shift(provider, exact_signal, fit)
    if not np.isfinite(scale) or scale <= 1e-6 or not np.isfinite(shift):
        raise ValueError("Provider face fit must have a finite positive scale")
    target = (exact_signal - float(shift)) / float(scale)
    low, high = np.percentile(provider[fit], (1.0, 99.0))
    span = max(float(high - low), 1e-6)
    return target.astype(np.float32), {
        "face_fit_scale": float(scale),
        "face_fit_shift": float(shift),
        "provider_face_p01": float(low),
        "provider_face_p99": float(high),
        "provider_face_span": span,
        "target_used_at_inference": False,
    }


def _provider_calibration(provider_depth: np.ndarray, face_mask: np.ndarray) -> dict:
    provider = np.asarray(provider_depth, dtype=np.float32)
    face = np.asarray(face_mask, dtype=bool)
    fit = face & np.isfinite(provider)
    if np.count_nonzero(fit) < 64:
        raise ValueError("At least 64 finite selected provider pixels are required")
    low, high = np.percentile(provider[fit], (1.0, 99.0))
    return {
        "provider_face_p01": float(low),
        "provider_face_p99": float(high),
        "provider_face_span": max(float(high - low), 1e-6),
        "oracle_used": False,
    }


def _normalize_inference_scene(scene: dict, provider_depth: np.ndarray) -> dict:
    calibration = _provider_calibration(provider_depth, scene["face_mask"])
    low = calibration["provider_face_p01"]
    span = calibration["provider_face_span"]
    base_normalized = (np.asarray(provider_depth, dtype=np.float32) - low) / span
    rgb = np.asarray(scene["source_rgb"], dtype=np.float32)
    face = np.asarray(scene["face_mask"], dtype=bool)
    inputs = np.concatenate(
        (
            rgb.transpose(2, 0, 1),
            np.clip(base_normalized, -3.0, 4.0)[None],
            face.astype(np.float32)[None],
        ),
        axis=0,
    ).astype(np.float32)
    inputs[:4, face] = 0.0
    return {
        **scene,
        "provider_depth": np.asarray(provider_depth, dtype=np.float32),
        "input": inputs,
        "base_normalized": base_normalized.astype(np.float32),
        "calibration": calibration,
    }


def _normalize_training_scene(scene: dict, provider_depth: np.ndarray) -> dict:
    normalized = _normalize_inference_scene(scene, provider_depth)
    target, target_calibration = _provider_space_target(
        scene["exact_depth"], provider_depth, scene["face_mask"]
    )
    low = normalized["calibration"]["provider_face_p01"]
    span = normalized["calibration"]["provider_face_span"]
    target_normalized = (target - low) / span
    delta = target_normalized - normalized["base_normalized"]
    background = ~np.asarray(scene["face_mask"], dtype=bool)
    reachable = background & np.isfinite(delta) & (
        np.abs(delta) <= BACKGROUND_RESIDUAL_BOUND
    )
    normalized.update(
        {
            "target_normalized": target_normalized.astype(np.float32),
            "target_calibration": target_calibration,
            "target_reachable_fraction": float(
                np.count_nonzero(reachable) / max(np.count_nonzero(background), 1)
            ),
        }
    )
    return normalized


def _build_model():
    import torch

    class ResidualBlock(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(channels, channels, 3, padding=1),
                torch.nn.GroupNorm(4, channels),
                torch.nn.GELU(),
                torch.nn.Conv2d(channels, channels, 3, padding=1),
                torch.nn.GroupNorm(4, channels),
            )

        def forward(self, value):
            return torch.nn.functional.gelu(value + self.layers(value))

    class BackgroundResidualNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_conv = torch.nn.Sequential(
                torch.nn.Conv2d(5, 16, 5, padding=2),
                torch.nn.GELU(),
            )
            self.enc1 = ResidualBlock(16)
            self.down1 = torch.nn.Conv2d(16, 24, 4, stride=2, padding=1)
            self.enc2 = ResidualBlock(24)
            self.down2 = torch.nn.Conv2d(24, 32, 4, stride=2, padding=1)
            self.bottleneck = torch.nn.Sequential(ResidualBlock(32), ResidualBlock(32))
            self.up2 = torch.nn.Conv2d(32 + 24, 24, 3, padding=1)
            self.dec2 = ResidualBlock(24)
            self.up1 = torch.nn.Conv2d(24 + 16, 16, 3, padding=1)
            self.dec1 = ResidualBlock(16)
            self.out_conv = torch.nn.Conv2d(16, 1, 3, padding=1)
            torch.nn.init.zeros_(self.out_conv.weight)
            torch.nn.init.zeros_(self.out_conv.bias)

        def forward(self, value):
            import torch.nn.functional as functional

            first = self.enc1(self.in_conv(value))
            second = self.enc2(functional.gelu(self.down1(first)))
            bottleneck = self.bottleneck(functional.gelu(self.down2(second)))
            up_second = functional.interpolate(
                bottleneck, size=second.shape[-2:], mode="bilinear", align_corners=False
            )
            up_second = self.dec2(functional.gelu(self.up2(torch.cat((up_second, second), dim=1))))
            up_first = functional.interpolate(
                up_second, size=first.shape[-2:], mode="bilinear", align_corners=False
            )
            up_first = self.dec1(functional.gelu(self.up1(torch.cat((up_first, first), dim=1))))
            return BACKGROUND_RESIDUAL_BOUND * torch.tanh(self.out_conv(up_first))

    return BackgroundResidualNet()


def _masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _correlation_loss(prediction, target, mask):
    import torch

    losses = []
    for index in range(prediction.shape[0]):
        selected = mask[index, 0] > 0.5
        pred = prediction[index, 0][selected]
        truth = target[index, 0][selected]
        pred = pred - pred.mean()
        truth = truth - truth.mean()
        denominator = torch.linalg.vector_norm(pred) * torch.linalg.vector_norm(truth)
        losses.append(1.0 - (pred * truth).sum() / denominator.clamp_min(1e-6))
    return torch.stack(losses).mean()


def _gradient_and_rank_loss(prediction, target, background):
    import torch

    gradient_losses = []
    rank_losses = []
    for axis in (-1, -2):
        pred_delta = torch.diff(prediction, dim=axis)
        target_delta = torch.diff(target, dim=axis)
        if axis == -1:
            valid = background[..., :, 1:] * background[..., :, :-1]
        else:
            valid = background[..., 1:, :] * background[..., :-1, :]
        gradient_losses.append(_masked_mean(torch.abs(pred_delta - target_delta), valid))
        ordered = valid * (torch.abs(target_delta) >= 0.015)
        rank = torch.nn.functional.softplus(
            -4.0 * torch.sign(target_delta) * pred_delta
        )
        rank_losses.append(_masked_mean(rank, ordered))
    return torch.stack(gradient_losses).mean(), torch.stack(rank_losses).mean()


def _loss_terms(prediction, target, face_mask):
    import torch

    background = 1.0 - face_mask
    robust = _masked_mean(
        torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none", beta=0.08),
        background,
    )
    gradient, rank = _gradient_and_rank_loss(prediction, target, background)
    correlation = _correlation_loss(prediction, target, background)
    total = robust + 0.70 * gradient + 0.35 * correlation + 0.12 * rank
    return total, {
        "robust": robust,
        "gradient": gradient,
        "correlation": correlation,
        "rank": rank,
    }


def _correct_scene(model, scene: dict, device: str) -> np.ndarray:
    import torch

    model.eval()
    tensor = torch.from_numpy(scene["input"][None]).to(device)
    with torch.inference_mode():
        residual = model(tensor)[0, 0].float().cpu().numpy()
    normalized = scene["base_normalized"] + residual
    calibration = scene["calibration"]
    corrected = normalized * calibration["provider_face_span"] + calibration["provider_face_p01"]
    face = np.asarray(scene["face_mask"], dtype=bool)
    corrected[face] = scene["provider_depth"][face]
    return corrected.astype(np.float32)


def _evaluate_corrected(scene: dict, corrected: np.ndarray) -> dict:
    exact_signal = 1.0 - np.asarray(scene["exact_depth"], dtype=np.float32)
    face = np.asarray(scene["face_mask"], dtype=bool)
    provider = np.asarray(scene["provider_depth"], dtype=np.float32)
    scale, shift = fit_scale_shift(corrected, exact_signal, face)
    aligned = corrected * float(scale) + float(shift)
    pitch = PHYSICAL_SIZE_MM / max(corrected.shape[1] - 1, 1)
    background = _background_metrics(exact_signal, aligned, face, pitch)
    before = provider[face]
    after = corrected[face]
    face_exact = bool(
        before.dtype == after.dtype
        and before.shape == after.shape
        and before.tobytes() == after.tobytes()
    )
    finite = bool(np.all(np.isfinite(corrected)))
    checks = {
        "selected_face_bit_exact": face_exact,
        "finite": finite,
        "coverage": bool(
            background.get("coverage_ratio", 0.0)
            >= DEPTH_ARRAY_GATES["minimum_coverage_ratio"]
        ),
        "background_correlation": bool(
            background.get("correlation", -1.0)
            >= DEPTH_ARRAY_GATES["minimum_correlation"]
        ),
        "background_gradient_correlation": bool(
            background.get("gradient_correlation", -1.0)
            >= DEPTH_ARRAY_GATES["minimum_gradient_correlation"]
        ),
        "full_background_amplitude_and_structure": bool(
            background.get("passed", False)
        ),
    }
    return {
        "background": background,
        "face_sha256_before": _sha256_array(before),
        "face_sha256_after": _sha256_array(after),
        "checks": {**checks, "passed": bool(all(checks.values()))},
    }


def _validation_score(records: list[dict]) -> float:
    if not records:
        return -float("inf")
    values = []
    for record in records:
        metrics = record["metrics"]["background"]
        values.append(
            float(metrics.get("correlation", -1.0))
            + float(metrics.get("gradient_correlation", -1.0))
            - 0.10 * abs(float(metrics.get("rms_retention", 1.0)) - 1.0)
        )
    # A strong mean must not hide one collapsed scene. The mean is only a
    # deterministic tie-break after the weakest scene.
    return float(min(values) + 1e-3 * np.mean(values))


def _checkpoint_is_eligible(records: list[dict]) -> bool:
    return bool(records) and all(
        record["metrics"]["checks"]["passed"]
        and record["metrics"]["background"].get("passed", False)
        for record in records
    )


def _train(
    train_scenes: list[dict],
    validation_scenes: list[dict],
    *,
    output_dir: Path,
    device: str,
    epochs: int,
) -> tuple[object, dict]:
    import torch

    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(MODEL_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = _build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    generator = torch.Generator(device="cpu").manual_seed(MODEL_SEED)
    history = []
    best_score = -float("inf")
    best_epoch = None
    best_state = None
    best_checkpoint_gate_eligible = False
    started = time.perf_counter()
    peak_vram_gb = 0.0
    cuda_device_index = None
    resident_vram_gb = 0.0
    if device.startswith("cuda"):
        resolved_device = torch.device(device)
        cuda_device_index = int(
            resolved_device.index if resolved_device.index is not None else 0
        )
        with torch.cuda.device(cuda_device_index):
            resident_vram_gb = float(torch.cuda.memory_allocated() / 1024**3)
            torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_scenes), generator=generator).tolist()
        totals = []
        for offset in range(0, len(order), 4):
            batch = [train_scenes[index] for index in order[offset : offset + 4]]
            inputs = torch.from_numpy(np.stack([scene["input"] for scene in batch])).to(device)
            targets = torch.from_numpy(
                np.stack([scene["target_normalized"][None] for scene in batch])
            ).to(device)
            bases = torch.from_numpy(
                np.stack([scene["base_normalized"][None] for scene in batch])
            ).to(device)
            masks = torch.from_numpy(
                np.stack([scene["face_mask"][None].astype(np.float32) for scene in batch])
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = bases + model(inputs)
            total, terms = _loss_terms(prediction, targets, masks)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            totals.append(
                {"total": float(total.detach().cpu()), **{key: float(value.detach().cpu()) for key, value in terms.items()}}
            )
        scheduler.step()
        validation = []
        for scene in validation_scenes:
            corrected = _correct_scene(model, scene, device)
            validation.append(
                {"scene_id": scene["scene_id"], "metrics": _evaluate_corrected(scene, corrected)}
            )
        score = _validation_score(validation)
        eligible = _checkpoint_is_eligible(validation)
        epoch_record = {
            "epoch": epoch + 1,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train": {
                key: float(np.mean([record[key] for record in totals]))
                for key in totals[0]
            },
            "validation_score": score,
            "checkpoint_gate_eligible": eligible,
            "validation": validation,
        }
        history.append(epoch_record)
        if (eligible and not best_checkpoint_gate_eligible) or (
            eligible == best_checkpoint_gate_eligible and score > best_score
        ):
            best_score = score
            best_epoch = epoch + 1
            best_checkpoint_gate_eligible = eligible
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    best_checkpoint_evaluation = {}
    for split_name, scenes in (
        ("train", train_scenes),
        ("validation", validation_scenes),
    ):
        records = []
        for scene in scenes:
            corrected = _correct_scene(model, scene, device)
            records.append(
                {
                    "scene_id": scene["scene_id"],
                    "metrics": _evaluate_corrected(scene, corrected),
                }
            )
        correlations = [
            float(record["metrics"]["background"].get("correlation", -1.0))
            for record in records
        ]
        gradients = [
            float(
                record["metrics"]["background"].get(
                    "gradient_correlation", -1.0
                )
            )
            for record in records
        ]
        best_checkpoint_evaluation[split_name] = {
            "records": records,
            "minimum_correlation": float(min(correlations)),
            "minimum_gradient_correlation": float(min(gradients)),
            "all_full_background_gates_passed": all(
                record["metrics"]["background"].get("passed", False)
                and record["metrics"]["checks"]["selected_face_bit_exact"]
                for record in records
            ),
        }
    train_validation_gap = max(
        best_checkpoint_evaluation["train"]["minimum_correlation"]
        - best_checkpoint_evaluation["validation"]["minimum_correlation"],
        0.0,
    )
    checkpoint_path = output_dir / "background_residual_checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_seed": MODEL_SEED,
            "corpus_seed": CORPUS_SEED,
            "da2_model_revision": DA2_MODEL_REVISION,
            "residual_bound": BACKGROUND_RESIDUAL_BOUND,
            "best_epoch": best_epoch,
            "state_dict": best_state,
        },
        checkpoint_path,
    )
    absolute_peak_vram_gb = 0.0
    if cuda_device_index is not None:
        with torch.cuda.device(cuda_device_index):
            absolute_peak_vram_gb = float(torch.cuda.max_memory_allocated() / 1024**3)
        peak_vram_gb = max(absolute_peak_vram_gb - resident_vram_gb, 0.0)
    return model, {
        "epochs": int(epochs),
        "batch_size": 4,
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_score),
        "best_checkpoint_gate_eligible": bool(best_checkpoint_gate_eligible),
        "runtime_seconds": float(time.perf_counter() - started),
        "incremental_peak_vram_gb": peak_vram_gb,
        "absolute_process_peak_vram_gb": absolute_peak_vram_gb,
        "resident_vram_before_training_gb": resident_vram_gb,
        "cuda_device_index": cuda_device_index,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_checkpoint_evaluation": best_checkpoint_evaluation,
        "train_validation_minimum_correlation_gap": float(train_validation_gap),
        "maximum_allowed_train_validation_gap": 0.15,
        "checkpoint": {
            "path": checkpoint_path.name,
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256(checkpoint_path),
        },
        "history": history,
    }


def run(
    output_dir: str | Path,
    *,
    asset_dir: str | Path = DEFAULT_ASSET_DIR,
    render_size: int = 384,
    crop_size: int = 256,
    device: str = "cuda",
    epochs: int = 80,
    allow_dirty: bool = False,
    allow_failures: bool = False,
) -> dict:
    import torch

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = load_makehuman_face_fixture(asset_dir)
    corpus = []
    inference = []
    for spec in CORPUS_SPECS:
        scene = _prepare_corpus_scene(
            output_dir,
            fixture,
            spec,
            render_size=render_size,
            crop_size=crop_size,
        )
        provider_depth, manifest = _infer_cached_provider(
            scene,
            provider=DA2_PROVIDER,
            device=device,
        )
        scene = _normalize_training_scene(scene, provider_depth)
        corpus.append(scene)
        inference.append({"scene_id": spec.scene_id, **manifest})

    train_scenes = [scene for scene in corpus if scene["split"] == "train"]
    validation_scenes = [scene for scene in corpus if scene["split"] == "validation"]
    model, training = _train(
        train_scenes,
        validation_scenes,
        output_dir=output_dir,
        device=device,
        epochs=epochs,
    )

    held_out = []
    held_out_root = output_dir / "held_out"
    for spec in FINAL_HELD_OUT_SCENES:
        scene = _prepare_inference_scene(
            held_out_root,
            fixture,
            spec,
            render_size=render_size,
            crop_size=crop_size,
        )
        scene.update(
            {
                "scene_id": f"{spec.profile_name}_{spec.framing}",
                "split": "final_held_out",
                "source_rgb": np.asarray(
                    Image.open(scene["source_path"]).convert("RGB"), dtype=np.float32
                )
                / 255.0,
            }
        )
        provider_depth, manifest = _infer_cached_provider(
            scene,
            provider=DA2_PROVIDER,
            device=device,
        )
        normalized = _normalize_inference_scene(scene, provider_depth)
        corrected = _correct_scene(model, normalized, device)
        corrected_path = scene["scene_dir"] / "background_residual_depth.npy"
        np.save(corrected_path, corrected)
        baseline_metrics = _evaluate_corrected(normalized, provider_depth)
        corrected_metrics = _evaluate_corrected(normalized, corrected)
        corrected_background = corrected_metrics["background"]
        baseline_background = baseline_metrics["background"]
        non_regression = {
            "correlation": bool(
                corrected_background.get("correlation", -1.0)
                >= baseline_background.get("correlation", -1.0) - 0.03
            ),
            "gradient_correlation": bool(
                corrected_background.get("gradient_correlation", -1.0)
                >= baseline_background.get("gradient_correlation", -1.0) - 0.03
            ),
        }
        held_out.append(
            {
                "scene": asdict(spec),
                "source_sha256": scene["source_sha256"],
                "prediction_manifest": manifest,
                "baseline": baseline_metrics,
                "corrected": corrected_metrics,
                "baseline_non_regression": {
                    **non_regression,
                    "maximum_allowed_regression": 0.03,
                    "passed": bool(all(non_regression.values())),
                },
                "artifact": {
                    "path": corrected_path.relative_to(output_dir).as_posix(),
                    "size_bytes": corrected_path.stat().st_size,
                    "sha256": _sha256(corrected_path),
                },
            }
        )

    corpus_manifest = []
    for scene in corpus:
        corpus_manifest.append(
            {
                "scene_id": scene["scene_id"],
                "split": scene["split"],
                "spec": scene["spec"],
                "source_sha256": scene["source_sha256"],
                "generator": scene["generator"],
                "artifacts": scene["artifacts"],
                "calibration": scene["calibration"],
                "target_calibration": scene["target_calibration"],
                "target_reachable_fraction": scene["target_reachable_fraction"],
                "provider_depth_sha256": _sha256_array(scene["provider_depth"]),
                "target_normalized_sha256": _sha256_array(
                    scene["target_normalized"]
                ),
            }
        )
    no_source_duplicates = len({row["source_sha256"] for row in corpus_manifest}) == len(
        corpus_manifest
    )
    held_out_hashes = {row["source_sha256"] for row in held_out}
    no_held_out_leakage = not held_out_hashes.intersection(
        row["source_sha256"] for row in corpus_manifest
    )
    held_out_passed = bool(held_out) and all(
        row["corrected"]["checks"]["passed"]
        and row["corrected"]["background"].get("passed", False)
        and row["baseline_non_regression"]["passed"]
        for row in held_out
    )
    validation_passed = bool(validation_scenes) and bool(
        training["best_checkpoint_evaluation"]["validation"][
            "all_full_background_gates_passed"
        ]
    )
    overfit_audit_passed = bool(
        training["train_validation_minimum_correlation_gap"]
        <= training["maximum_allowed_train_validation_gap"]
    )
    targets_reachable = all(
        scene["target_reachable_fraction"] >= 0.995 for scene in corpus
    )
    provenance = _git_provenance()
    implementation_clean = bool(provenance.get("available") and provenance.get("clean"))
    expansion_prerequisites = {
        "train_count": len(train_scenes) == 16,
        "validation_count": len(validation_scenes) == 4,
        "unique_corpus_sources": no_source_duplicates,
        "final_held_out_not_in_corpus": no_held_out_leakage,
        "training_targets_reachable": targets_reachable,
        "validation_background_gates_passed": validation_passed,
        "best_checkpoint_gate_eligible": bool(
            training["best_checkpoint_gate_eligible"]
        ),
        "train_validation_gap_passed": overfit_audit_passed,
        "held_out_depth_array_gates_passed": held_out_passed,
        "implementation_provenance_clean": implementation_clean,
    }
    checks = {
        **expansion_prerequisites,
        "all_selected_faces_bit_exact": all(
            row["corrected"]["checks"]["selected_face_bit_exact"] for row in held_out
        ),
        "stl_expansion_allowed": bool(all(expansion_prerequisites.values())),
    }
    summary = {
        "schema_version": 1,
        "run_kind": "face_protected_background_residual_training_smoke",
        "privacy": "checksum-pinned CC0 generated heads and deterministic procedural scenes only",
        "implementation_provenance": provenance,
        "matrix": {
            "train_scenes": 16,
            "validation_scenes": 4,
            "final_held_out_scenes": [asdict(spec) for spec in FINAL_HELD_OUT_SCENES],
            "render_size": int(render_size),
            "crop_size": int(crop_size),
            "physical_size_mm": PHYSICAL_SIZE_MM,
            "device": device,
            "corpus_seed": CORPUS_SEED,
            "model_seed": MODEL_SEED,
            "da2_provider": DA2_PROVIDER,
            "da2_model_revision": DA2_MODEL_REVISION,
        },
        "policy": {
            "model_inputs": ["rgb", "pinned_da2_depth", "selection_mask"],
            "exact_depth_at_inference": False,
            "exact_depth_usage": "training supervision and evaluation only",
            "checkpoint_selection": "four validation scenes only",
            "final_held_out_used_for_checkpoint_selection": False,
            "selected_face_composition": "bit-exact copy from pinned DA2",
            "stl_emission": "disabled until every final depth-array gate passes",
        },
        "gates": {"depth_array": DEPTH_ARRAY_GATES, "full_background": BACKGROUND_GATES},
        "corpus": corpus_manifest,
        "inference": inference,
        "training": training,
        "held_out": held_out,
        "checks": {
            **checks,
            "passed": bool(
                checks["train_count"]
                and checks["validation_count"]
                and checks["unique_corpus_sources"]
                and checks["final_held_out_not_in_corpus"]
                and checks["training_targets_reachable"]
                and checks["all_selected_faces_bit_exact"]
                and checks["validation_background_gates_passed"]
                and checks["best_checkpoint_gate_eligible"]
                and checks["train_validation_gap_passed"]
                and checks["held_out_depth_array_gates_passed"]
                and (allow_dirty or checks["implementation_provenance_clean"])
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not summary["checks"]["passed"] and not allow_failures:
        raise RuntimeError("Background residual learner failed its held-out expansion gate")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    run(
        args.output_dir,
        asset_dir=args.asset_dir,
        render_size=args.render_size,
        crop_size=args.crop_size,
        device=args.device,
        epochs=args.epochs,
        allow_dirty=args.allow_dirty,
        allow_failures=args.allow_failures,
    )


if __name__ == "__main__":
    main()
