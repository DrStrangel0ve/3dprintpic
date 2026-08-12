from __future__ import annotations

import base64
import gc
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageOps


PROJECT_REPOSITORY = "https://github.com/DrStrangel0ve/3dprintpic.git"
PROJECT_REVISION = "65f330a7cb57241c654c6aa30c3503fa7a8f4eb7"
TRIPOSG_REPOSITORY = "https://github.com/VAST-AI-Research/TripoSG.git"
TRIPOSG_SOURCE_REVISION = "fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c"
TRIPOSG_MODEL_REVISION = "2c1c516d22d58db486a058d98d31bb6177344e06"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
DEPTH_MODEL_REVISION = "7581137eff8d4e94f6e796d3baea0e9fa79b22d2"
SAM3_MODEL = "facebook/sam3"
SAM3_MODEL_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"

SPACE_DIR = Path(__file__).resolve().parent
RUNTIME_ROOT = Path(os.getenv("THREEDPRINTPIC_RUNTIME_DIR", Path(tempfile.gettempdir()) / "3dprintpic-space"))
OUTPUT_DIR = RUNTIME_ROOT / "output"
SOURCE_DIR = RUNTIME_ROOT / "source"
TRIPOSG_DIR = RUNTIME_ROOT / "TripoSG"

_SOURCE_LOCK = Lock()
_TRIPOSG_LOCK = Lock()

SAM3_HOVER_MAP_MAX_DIMENSION = 1024
SAM3_HOVER_MAP_VERSION = 1
SAM3_PRECOMPUTE_STATE_VERSION = 2
SAM3_PRECOMPUTE_TTL_SECONDS = 20 * 60
SAM3_PRECOMPUTE_MAX_SOURCE_PIXELS = 40_000_000
SAM3_PRECOMPUTE_MAX_MASKS = 256
SAM3_PRECOMPUTE_MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
LOCAL_RELIEF_DETAIL_MULTIPLIERS = {
    384: 1.5,
    512: 2.0,
    768: 3.0,
    900: 4.0,
}
LOCAL_RELIEF_PRINTER_EDGE_MM = 256


def _run_git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return completed.stdout.strip()


def _repository_root_if_local() -> Path | None:
    explicit = os.getenv("THREEDPRINTPIC_SOURCE_DIR")
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.extend((SPACE_DIR.parent, Path.cwd()))
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "backend" / "main.py").is_file():
            return resolved
    return None


def _clone_exact_repository(repository: str, revision: str, destination: Path) -> Path:
    with _SOURCE_LOCK:
        if (destination / ".git").is_dir():
            current = _run_git("rev-parse", "HEAD", cwd=destination)
            if current == revision:
                return destination
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run_git("clone", "--filter=blob:none", "--no-checkout", repository, str(destination))
        _run_git("checkout", "--detach", revision, cwd=destination)
        current = _run_git("rev-parse", "HEAD", cwd=destination)
        if current != revision:
            raise RuntimeError(f"Source checkout mismatch: expected {revision}, got {current}")
        return destination


def ensure_project_source() -> Path:
    local = _repository_root_if_local()
    if local is not None:
        return local
    return _clone_exact_repository(PROJECT_REPOSITORY, PROJECT_REVISION, SOURCE_DIR)


PROJECT_DIR = ensure_project_source()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OUTPUT_DIR", str(OUTPUT_DIR))
os.environ.setdefault("DEPTH_PROVIDER", "transformers")
os.environ.setdefault("DEPTH_MODEL", DEPTH_MODEL)
os.environ.setdefault("SELECTION_ALLOW_MODEL_DOWNLOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from backend import main as backend_main  # noqa: E402
from backend.benchmark.direct_mesh import (  # noqa: E402
    convert_mesh_to_stl,
    postprocess_mesh_for_stl,
    repair_mesh_for_printable_stl,
)
from backend.stl_diagnostics import json_safe_stl_diagnostics, stl_diagnostics  # noqa: E402


BACKEND_CLIENT = TestClient(backend_main.app)


def image_dimensions_mm(image_path: str | Path | None, long_edge_mm: float) -> tuple[float, float]:
    if not image_path:
        return float(long_edge_mm), float(long_edge_mm)
    with Image.open(image_path) as image:
        width, height = ImageOps.exif_transpose(image).size
    scale = float(long_edge_mm) / float(max(width, height, 1))
    return round(width * scale, 2), round(height * scale, 2)


def local_relief_dimensions_mm(
    image_path: str | Path | None,
    print_scale_percent: float,
) -> tuple[float, float]:
    scale_percent = float(print_scale_percent)
    if scale_percent < 10.0 or scale_percent > 100.0 or scale_percent % 5.0 != 0.0:
        raise ValueError("Print size must match the local 10-100% production scale")
    long_edge_mm = int(LOCAL_RELIEF_PRINTER_EDGE_MM * scale_percent / 100.0)
    return image_dimensions_mm(image_path, long_edge_mm)


def _safe_file(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected output was not created: {resolved.name}")
    return str(resolved)


def _response_error(response) -> RuntimeError:
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload
    else:
        detail = payload
    return RuntimeError(f"Generation failed ({response.status_code}): {detail}")


def _diagnostic_summary(diagnostics: dict, *, model: str) -> dict:
    aliases = {
        "is_watertight": ("stl_is_watertight", "is_watertight"),
        "is_volume": ("stl_is_volume", "is_volume"),
        "is_manifold": ("stl_is_manifold", "is_manifold"),
        "winding_consistent": ("stl_winding_consistent", "winding_consistent"),
        "component_count": ("stl_component_count", "component_count"),
        "nonmanifold_edge_count": ("stl_nonmanifold_edge_count", "nonmanifold_edge_count"),
        "degenerate_face_count": ("stl_degenerate_face_count", "degenerate_face_count"),
        "positive_volume": ("stl_positive_volume", "positive_volume"),
        "face_count": ("stl_faces", "face_count"),
        "vertex_count": ("stl_vertices", "vertex_count"),
        "normalized_bbox_complexity_log1p": (
            "stl_faces_per_normalized_bbox_volume_log1p",
            "normalized_bbox_complexity_log1p",
        ),
    }
    summary = {"model": model}
    for output_key, candidates in aliases.items():
        for candidate in candidates:
            if candidate in diagnostics:
                summary[output_key] = diagnostics[candidate]
                break
    bbox = [diagnostics.get(f"stl_bbox_{axis}") for axis in "xyz"]
    if all(value is not None for value in bbox):
        summary["bbox_extents"] = bbox
    hard_checks = {
        "watertight": summary.get("is_watertight") is True,
        "volume": summary.get("is_volume") is True,
        "manifold": summary.get("is_manifold") is True,
        "winding": summary.get("winding_consistent") is True,
        "single_component": summary.get("component_count") == 1,
        "nonmanifold_edges": summary.get("nonmanifold_edge_count") == 0,
        "degenerate_faces": summary.get("degenerate_face_count") == 0,
        "positive_volume": summary.get("positive_volume") is True,
    }
    summary["stl_passes_hard_checks"] = all(hard_checks.values())
    summary["stl_failed_checks"] = [name for name, passed in hard_checks.items() if not passed]
    return summary


def _require_sam3_token() -> None:
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError(
            "SAM 3 needs the Space owner's read-only HF_TOKEN secret before object selection can run."
        )


def _load_selection_image(image_path: str | Path) -> Image.Image:
    with Image.open(image_path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def _save_selection_job(
    image_path: str | Path,
    image: Image.Image,
    mask: Image.Image,
    *,
    resolved_model: str,
    model_status: str,
    labels: list[str],
    mask_count: int = 1,
) -> dict:
    width, height = image.size
    if resolved_model != SAM3_MODEL or not model_status.startswith("sam3"):
        raise RuntimeError("SAM 3 did not produce a verified selection mask")

    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / "selection" / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    selected = backend_main.selected_image_from_mask(image, mask, background_mode="neutral")
    overlay = backend_main.selection_overlay(image, mask)
    mask_pixels = int(np.count_nonzero(np.asarray(mask.convert("L")) > 0))
    metadata = {
        "job_id": job_id,
        "source_filename": Path(image_path).name,
        "source_fingerprint": backend_main.selection_source_fingerprint(image),
        "mask_count": int(mask_count),
        "model_id": resolved_model,
        "selection_model_status": model_status,
        "model_status": "composed-clicked-masks",
        "selection_labels": labels,
        "background_mode": "neutral",
        "selection_infill_mode": "none",
        "selection_infill": {"mode": "none", "enabled": False},
        "face_detection_source": "neutral-selection-cutout",
        "image_size": {"width": width, "height": height},
        "mask_pixels": mask_pixels,
        "mask_coverage": mask_pixels / float(max(1, width * height)),
    }
    image.save(job_dir / "source.png")
    selected.save(job_dir / "selected_image.png")
    selected.save(job_dir / "selection_face_detection.png")
    mask.save(job_dir / "selection_mask.png")
    overlay.save(job_dir / "selection_overlay.png")
    backend_main.selection_tint(mask).save(job_dir / "selection_tint.png")
    (job_dir / "selection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "job_id": job_id,
        "selected": _safe_file(job_dir / "selected_image.png"),
        "overlay": _safe_file(job_dir / "selection_overlay.png"),
        "mask": backend_main.output_relative_path(job_dir / "selection_mask.png"),
        "labels": labels,
        "model": resolved_model,
        "model_status": model_status,
        "mask_coverage": metadata["mask_coverage"],
    }


def _selection_job(
    image_path: str | Path,
    point_x: float,
    point_y: float,
) -> dict:
    _require_sam3_token()
    image = _load_selection_image(image_path)
    width, height = image.size
    normalized_point = {
        "x": float(np.clip(point_x / max(width, 1), 0.0, 1.0)),
        "y": float(np.clip(point_y / max(height, 1), 0.0, 1.0)),
    }
    mask, resolved_model, model_status, labels = backend_main.sam3_person_aware_selection_mask(
        image,
        [normalized_point],
        device="cuda",
    )
    return _save_selection_job(
        image_path,
        image,
        mask,
        resolved_model=resolved_model,
        model_status=model_status,
        labels=labels,
    )


def select_object(image_path: str | Path | None, point_x: float, point_y: float) -> dict:
    if not image_path:
        raise ValueError("Upload a photo before selecting an object")
    return _selection_job(image_path, point_x, point_y)


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def cleanup_expired_selection_precomputes(
    max_age_seconds: int = SAM3_PRECOMPUTE_TTL_SECONDS,
    *,
    now: float | None = None,
) -> int:
    cutoff = (time.time() if now is None else float(now)) - float(max_age_seconds)
    removed = 0
    with backend_main.SELECTION_PRECOMPUTE_LOCK:
        expired = [
            cache_id
            for cache_id, entry in backend_main.SELECTION_PRECOMPUTE_CACHE.items()
            if float(entry.get("created_at_epoch", 0.0)) < cutoff
        ]
        for cache_id in expired:
            backend_main.SELECTION_PRECOMPUTE_CACHE.pop(cache_id, None)
            removed += 1
    return removed


def _discard_precompute_source_pixels(precompute_id: str) -> None:
    with backend_main.SELECTION_PRECOMPUTE_LOCK:
        cached = backend_main.SELECTION_PRECOMPUTE_CACHE.get(precompute_id)
        if cached is not None:
            cached.pop("image", None)


def _sam3_hover_region_map(
    masks: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    image_size: tuple[int, int],
    *,
    max_dimension: int = SAM3_HOVER_MAP_MAX_DIMENSION,
) -> tuple[Image.Image, dict[str, dict], tuple[int, int]]:
    import cv2

    width, height = image_size
    scale = min(1.0, float(max_dimension) / float(max(width, height, 1)))
    map_width = max(1, round(width * scale))
    map_height = max(1, round(height * scale))
    mask_array = np.asarray(masks, dtype=bool)
    score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(score_array) != len(mask_array) or len(labels) != len(mask_array):
        raise RuntimeError("SAM 3 returned inconsistent instance metadata")
    if not len(mask_array):
        raise RuntimeError("SAM 3 found no selectable objects in this image")

    winner_scores = np.full((map_height, map_width), -np.inf, dtype=np.float32)
    winners = np.full((map_height, map_width), -1, dtype=np.int32)
    for instance_index, mask in enumerate(mask_array):
        if (map_width, map_height) == (width, height):
            resized = mask
        else:
            resized = cv2.resize(
                mask.astype(np.uint8),
                (map_width, map_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        update = resized & (score_array[instance_index] > winner_scores)
        winner_scores[update] = score_array[instance_index]
        winners[update] = instance_index
    covered = winners >= 0

    region_map = np.zeros((map_height, map_width), dtype=np.uint32)
    regions: dict[str, dict] = {}
    minimum_region_pixels = max(4, round(map_width * map_height * 0.00001))
    next_region_id = 1
    for instance_index in range(len(mask_array)):
        winning_pixels = (winners == instance_index) & covered
        component_count, component_labels, stats, _centroids = cv2.connectedComponentsWithStats(
            winning_pixels.astype(np.uint8),
            connectivity=8,
        )
        for component_index in range(1, component_count):
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            if area < minimum_region_pixels:
                continue
            if next_region_id >= 2**24:
                raise RuntimeError("SAM 3 produced too many hover regions")
            region_map[component_labels == component_index] = next_region_id
            regions[str(next_region_id)] = {
                "label": str(labels[instance_index]),
                "score": round(float(score_array[instance_index]), 6),
                "pixels": area,
                "instance_index": instance_index,
            }
            next_region_id += 1
    if not regions:
        raise RuntimeError("SAM 3 found no selectable objects in this image")

    encoded = np.stack(
        (
            region_map & 255,
            (region_map >> 8) & 255,
            (region_map >> 16) & 255,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(encoded, mode="RGB"), regions, (map_width, map_height)


def prepare_object_selection(image_path: str | Path | None) -> tuple[str, dict]:
    if not image_path:
        raise ValueError("Upload a photo before selecting an object")
    _require_sam3_token()
    cleanup_expired_selection_precomputes()
    image = _load_selection_image(image_path)
    masks = np.empty((0, image.height, image.width), dtype=bool)
    scores = np.empty((0,), dtype=np.float32)
    labels: list[str] = []
    resolved_model = ""
    try:
        masks, scores, labels, resolved_model = backend_main.compute_sam3_selection_instances(
            image,
            device="cuda",
        )
        if resolved_model != SAM3_MODEL:
            raise RuntimeError("SAM 3 did not produce verified selection instances")
    finally:
        release_gpu_models()

    hover_map, regions, hover_size = _sam3_hover_region_map(
        masks,
        scores,
        labels,
        image.size,
    )
    source_fingerprint = backend_main.selection_source_fingerprint(image)
    public_regions = {
        region_id: {
            "label": metadata["label"],
            "score": metadata["score"],
            "pixels": metadata["pixels"],
        }
        for region_id, metadata in regions.items()
    }
    used_instance_indices = sorted(
        {int(metadata["instance_index"]) for metadata in regions.values()}
    )
    if image.width * image.height > SAM3_PRECOMPUTE_MAX_SOURCE_PIXELS:
        raise RuntimeError("The uploaded image is too large for cached object selection")
    if len(used_instance_indices) > SAM3_PRECOMPUTE_MAX_MASKS:
        raise RuntimeError("SAM 3 found too many objects to cache safely")
    compact_instance_indices = {
        instance_index: compact_index
        for compact_index, instance_index in enumerate(used_instance_indices)
    }
    compressed_masks: list[bytes] = []
    compressed_bytes = 0
    for instance_index in used_instance_indices:
        packed_mask = np.packbits(
            np.asarray(masks[instance_index], dtype=bool),
            axis=1,
        )
        payload = zlib.compress(packed_mask.tobytes(order="C"), level=6)
        compressed_bytes += len(payload)
        if compressed_bytes > SAM3_PRECOMPUTE_MAX_COMPRESSED_BYTES:
            raise RuntimeError("The SAM 3 object map is too large to cache safely")
        compressed_masks.append(payload)
    precompute_id = uuid4().hex
    manifest = {
        "version": SAM3_HOVER_MAP_VERSION,
        "width": hover_size[0],
        "height": hover_size[1],
        "source_width": image.width,
        "source_height": image.height,
        "hit_map": _png_data_url(hover_map),
        "regions": public_regions,
    }
    state = {
        "state_version": SAM3_PRECOMPUTE_STATE_VERSION,
        "precompute_id": precompute_id,
        "source_fingerprint": source_fingerprint,
        "image_size": [image.width, image.height],
        "hover_size": [hover_size[0], hover_size[1]],
        "region_count": len(regions),
        "region_instances": {
            region_id: compact_instance_indices[int(metadata["instance_index"])]
            for region_id, metadata in regions.items()
        },
        "model": resolved_model,
        # ZeroGPU executes GPU-decorated functions in forked workers. Returning
        # packed masks in gr.State is what carries them back to the app process;
        # process-local globals disappear when the worker exits.
        "selection_precompute": {
            "kind": "sam3-hover-session-v2",
            "compressed_masks": compressed_masks,
            "compressed_bytes": compressed_bytes,
            "mask_width": image.width,
            "labels": [str(labels[index]) for index in used_instance_indices],
            "model_id": resolved_model,
            "image_size": [image.width, image.height],
            "created_at_epoch": time.time(),
        },
    }
    return json.dumps(manifest, separators=(",", ":")), state


def _inline_selection_precompute(precompute_state: dict) -> dict | None:
    cached = precompute_state.get("selection_precompute")
    if not isinstance(cached, dict):
        return None
    if int(precompute_state.get("state_version", 0)) != SAM3_PRECOMPUTE_STATE_VERSION:
        raise ValueError("The object map uses an unsupported session format; refresh it")
    created_at = float(cached.get("created_at_epoch", 0.0))
    if not np.isfinite(created_at) or time.time() - created_at > SAM3_PRECOMPUTE_TTL_SECONDS:
        raise ValueError("The object map expired; wait for SAM 3 to refresh it")
    image_size = cached.get("image_size") or []
    if len(image_size) != 2:
        raise ValueError("The cached SAM 3 object map has invalid dimensions")
    width, height = (int(image_size[0]), int(image_size[1]))
    mask_width = int(cached.get("mask_width", 0))
    compressed_masks = cached.get("compressed_masks")
    labels = cached.get("labels") or []
    expected_bytes = (width + 7) // 8
    valid_compressed_masks = isinstance(compressed_masks, (list, tuple)) and all(
        isinstance(payload, bytes) for payload in compressed_masks
    )
    compressed_bytes = (
        sum(len(payload) for payload in compressed_masks)
        if valid_compressed_masks
        else SAM3_PRECOMPUTE_MAX_COMPRESSED_BYTES + 1
    )
    if (
        width <= 0
        or height <= 0
        or width * height > SAM3_PRECOMPUTE_MAX_SOURCE_PIXELS
        or mask_width != width
        or not valid_compressed_masks
        or len(compressed_masks) > SAM3_PRECOMPUTE_MAX_MASKS
        or compressed_bytes > SAM3_PRECOMPUTE_MAX_COMPRESSED_BYTES
        or len(labels) != len(compressed_masks)
        or str(cached.get("model_id")) != SAM3_MODEL
    ):
        raise ValueError("The cached SAM 3 object map is inconsistent")
    return {
        "compressed_masks": list(compressed_masks),
        "packed_row_bytes": expected_bytes,
        "mask_width": mask_width,
        "labels": [str(label) for label in labels],
        "model_id": str(cached["model_id"]),
        "image_size": (width, height),
    }


def _selection_candidate(cached: dict, instance_index: int) -> np.ndarray:
    mask_count = (
        len(cached["compressed_masks"])
        if "compressed_masks" in cached
        else len(cached.get("packed_masks", []))
    )
    if not 0 <= instance_index < mask_count:
        raise ValueError("The highlighted SAM 3 region has an invalid instance index")
    width, height = cached["image_size"]
    if "compressed_masks" in cached:
        try:
            raw_mask = zlib.decompress(cached["compressed_masks"][instance_index])
        except zlib.error as exc:
            raise ValueError("The cached SAM 3 object mask is corrupt") from exc
        expected_size = height * int(cached["packed_row_bytes"])
        if len(raw_mask) != expected_size:
            raise ValueError("The cached SAM 3 object mask has an invalid size")
        packed_mask = np.frombuffer(raw_mask, dtype=np.uint8).reshape(
            height,
            int(cached["packed_row_bytes"]),
        )
    else:
        packed_mask = cached["packed_masks"][instance_index]
    return np.unpackbits(
        packed_mask,
        axis=1,
        count=int(cached["mask_width"]),
    ).astype(bool)


def _selection_component_at_point(
    candidate: np.ndarray,
    point: dict[str, float],
    precompute_state: dict,
    width: int,
    height: int,
) -> np.ndarray:
    pixel_point = backend_main.selection_points_to_pixels([point], width, height)[0]
    px = int(np.clip(round(pixel_point[0]), 0, max(0, width - 1)))
    py = int(np.clip(round(pixel_point[1]), 0, max(0, height - 1)))
    if not candidate[py, px]:
        hover_width, hover_height = precompute_state.get("hover_size") or [width, height]
        source_pixels_per_hover_pixel = max(
            width / max(1, int(hover_width)),
            height / max(1, int(hover_height)),
        )
        radius = max(2, int(np.ceil(source_pixels_per_hover_pixel)) + 1)
        left = max(0, px - radius)
        right = min(width, px + radius + 1)
        top = max(0, py - radius)
        bottom = min(height, py + radius + 1)
        local_y, local_x = np.nonzero(candidate[top:bottom, left:right])
        if not len(local_x):
            raise ValueError("No cached SAM 3 object covers that point; hover over a highlighted object")
        distances = (local_x + left - px) ** 2 + (local_y + top - py) ** 2
        nearest = int(np.argmin(distances))
        pixel_point = [float(local_x[nearest] + left), float(local_y[nearest] + top)]
    return backend_main.seeded_sam2_component(candidate, [pixel_point])


def select_precomputed_objects(
    image_path: str | Path | None,
    precompute_state: dict | None,
    selections: list[dict] | None,
) -> dict:
    if not image_path:
        raise ValueError("Upload a photo before selecting an object")
    if not precompute_state or not (
        precompute_state.get("selection_precompute")
        or precompute_state.get("precompute_id")
    ):
        raise ValueError("Wait for SAM 3 to finish finding objects")
    if not isinstance(selections, list) or not selections:
        raise ValueError("Click at least one highlighted object before finishing")
    if len(selections) > SAM3_PRECOMPUTE_MAX_MASKS:
        raise ValueError("Too many highlighted objects were selected")
    image = _load_selection_image(image_path)
    fingerprint = backend_main.selection_source_fingerprint(image)
    if fingerprint != precompute_state.get("source_fingerprint"):
        raise ValueError("The uploaded image changed; wait for SAM 3 to refresh the object map")
    cached = _inline_selection_precompute(precompute_state)
    if cached is None:
        cleanup_expired_selection_precomputes()
        cached = backend_main.get_selection_precompute(str(precompute_state["precompute_id"]))
    region_instances = precompute_state.get("region_instances") or {}
    width, height = cached["image_size"]
    combined = np.zeros((height, width), dtype=bool)
    labels: list[str] = []
    candidates: dict[int, np.ndarray] = {}
    seen_regions: set[int] = set()
    for selection in selections:
        try:
            region_id = int(selection["region_id"])
            coordinates = np.asarray([selection["x"], selection["y"]], dtype=np.float64)
            instance_index = int(region_instances[str(region_id)])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A highlighted SAM 3 region is no longer available") from exc
        if region_id in seen_regions:
            continue
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Selection coordinates must be finite")
        seen_regions.add(region_id)
        point = {
            "x": float(np.clip(coordinates[0], 0.0, 1.0)),
            "y": float(np.clip(coordinates[1], 0.0, 1.0)),
        }
        if instance_index not in candidates:
            candidates[instance_index] = _selection_candidate(cached, instance_index)
        component = _selection_component_at_point(
            candidates[instance_index],
            point,
            precompute_state,
            width,
            height,
        )
        combined |= component
        label = str(cached["labels"][instance_index])
        if label not in labels:
            labels.append(label)
    if not np.any(combined):
        raise ValueError("The selected SAM 3 objects produced an empty mask")
    mask = Image.fromarray((combined.astype(np.uint8) * 255), mode="L")
    model_status = (
        "sam3-concept-precomputed-point"
        if len(seen_regions) == 1
        else "sam3-concept-precomputed-multi-point"
    )
    return _save_selection_job(
        image_path,
        image,
        mask,
        resolved_model=str(cached["model_id"]),
        model_status=model_status,
        labels=labels,
        mask_count=len(seen_regions),
    )


def select_precomputed_object(
    image_path: str | Path | None,
    precompute_state: dict | None,
    normalized_x: float,
    normalized_y: float,
    region_id: int | None = None,
) -> dict:
    return select_precomputed_objects(
        image_path,
        precompute_state,
        [
            {
                "x": normalized_x,
                "y": normalized_y,
                "region_id": region_id,
            }
        ],
    )


def release_gpu_models() -> None:
    try:
        backend_main.release_selection_models()
    except Exception:
        pass


def _depth_model_source() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=DEPTH_MODEL,
        revision=DEPTH_MODEL_REVISION,
        token=os.getenv("HF_TOKEN") or None,
    )
    try:
        backend_main.release_depth_pipelines()
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def generate_relief(
    image_path: str | Path | None,
    scope: str,
    selection: dict | None,
    print_scale_percent: float,
    relief_height_mm: float,
    base_thickness_mm: float,
    detail_samples: int,
) -> tuple[str, str, str, dict]:
    if not image_path:
        raise ValueError("Upload a photo before generating a relief")
    selected = str(scope).lower().startswith("select")
    if selected and not selection:
        raise ValueError("Click the object in the image before generating the selected-object relief")
    detail_samples = int(detail_samples)
    if detail_samples not in LOCAL_RELIEF_DETAIL_MULTIPLIERS:
        raise ValueError("Mesh detail must match a local frontend production preset")
    mesh_resolution_multiplier = LOCAL_RELIEF_DETAIL_MULTIPLIERS[detail_samples]
    x_mm, y_mm = local_relief_dimensions_mm(image_path, print_scale_percent)
    depth_model_source = _depth_model_source()
    data = {
        # Keep this request contract aligned with the local Next.js frontend.
        # Selection identifies the protected subject; it does not remove depth
        # samples or change the rectangular relief topology.
        "selection_mode": "context",
        "selection_subject_lock": "true" if selected else "false",
        "depth_provider": "transformers",
        "depth_model": depth_model_source,
        "device": "auto",
        "depth_downsample_sharpening": "0.35",
        "target_dimension": str(int(detail_samples)),
        "z_scale": str(float(relief_height_mm)),
        "base_thickness_mm": str(float(base_thickness_mm)),
        "max_xy_size": str(max(x_mm, y_mm)),
        "invert": "false",
        "relief_polarity": "raised-print",
        "mesh_resolution_multiplier": str(mesh_resolution_multiplier),
        "printer_profile": "Bambu Lab P1S",
        "printer_max_x_mm": str(LOCAL_RELIEF_PRINTER_EDGE_MM),
        "printer_max_y_mm": str(LOCAL_RELIEF_PRINTER_EDGE_MM),
        "printer_max_z_mm": str(LOCAL_RELIEF_PRINTER_EDGE_MM),
        "printer_clearance_mm": "0",
        "print_scale_percent": str(float(print_scale_percent)),
        "sigma": "0.35",
        "detail_boost": "0.8",
        "printable_feature_depth_mm": "0.4",
        "feature_bridge_depth_mm": "0.8",
        "background_detail_boost": "2.4",
        "background_photo_detail_mm": "0.60",
        "trim_top_background": "true",
        "relief_gamma": "0.75",
        "base_border_px": "2",
        "detail_radius": "2.0",
        "low_percentile": "1.0",
        "high_percentile": "99.0",
        "max_relief_slope": "2.0",
        "nozzle_diameter_mm": "0.4",
        "minimum_feature_mm": "0.8",
        "face_refinement_mode": "auto",
        "face_detail_strength": "1.0",
        "face_feather_ratio": "0.20",
        "face_max_correction_ratio": "0.08",
    }
    if selected:
        data["selection_job_id"] = selection["job_id"]
    release_gpu_models()
    request_image_path = (selection.get("selected") or image_path) if selected else image_path
    with open(request_image_path, "rb") as image_file:
        response = BACKEND_CLIENT.post(
            "/process_image",
            files={"file": (Path(request_image_path).name, image_file, "image/png")},
            data=data,
        )
    if response.status_code != 200:
        raise _response_error(response)
    result = response.json()
    if result.get("selection_mode") != data["selection_mode"]:
        raise RuntimeError("Relief backend did not honor the local frontend selection mode")
    if selected:
        selection_context = result.get("selection_depth_context")
        if not isinstance(selection_context, dict) or not (
            selection_context.get("method") == "full_scene_subject_locked_background_v1"
            and selection_context.get("subject_surface_locked") is True
            and result.get("selection_subject_lock") is True
            and result.get("selection_crop") is None
        ):
            raise RuntimeError("Selected relief did not preserve the local full-scene subject-lock contract")
    job_dir = OUTPUT_DIR / result["job_id"]
    stl_path = OUTPUT_DIR / result["stl_model"]
    preview_path = job_dir / "output_relief_preview.png"
    if not preview_path.is_file():
        preview_path = job_dir / "output_depth_preview.png"
    diagnostics_path = OUTPUT_DIR / result["diagnostics"]
    diagnostics = result.get("stl_diagnostics", {})
    summary = _diagnostic_summary(diagnostics, model=DEPTH_MODEL)
    summary["dimensions_mm"] = {
        "x": x_mm,
        "y": y_mm,
        "z": float(relief_height_mm) + float(base_thickness_mm),
    }
    summary["relief_height_mm"] = float(relief_height_mm)
    summary["base_thickness_mm"] = float(base_thickness_mm)
    summary["scope"] = "selected-objects-local-context" if selected else "full-scene"
    summary["selection_mode"] = result.get("selection_mode", data["selection_mode"])
    summary["local_relief_parity"] = "full-scene-subject-lock-v1"
    summary["inpainting"] = False
    return (
        _safe_file(stl_path),
        _safe_file(stl_path),
        _safe_file(preview_path),
        {"summary": summary, "full_report": _safe_file(diagnostics_path)},
    )


def generate_diorama(
    image_path: str | Path | None,
    scope: str,
    selection: dict | None,
    max_size_mm: float,
    scene_depth_mm: float,
    base_thickness_mm: float,
) -> tuple[str, str, str, str, dict]:
    if not image_path:
        raise ValueError("Upload a photo before generating a scene diorama")
    selected = str(scope).lower().startswith("select")
    if selected and not selection:
        raise ValueError("Click the object in the image before building selected diorama layers")
    release_gpu_models()
    depth_model_source = _depth_model_source()
    data = {
        "mask_paths_json": json.dumps([selection["mask"]] if selection else []),
        "selection_labels_json": json.dumps([selection.get("labels", [])] if selection else []),
        "depth_provider": "transformers",
        "depth_model": depth_model_source,
        "device": "cuda",
        "max_size_mm": str(float(max_size_mm)),
        "scene_depth_mm": str(float(scene_depth_mm)),
        "base_thickness_mm": str(float(base_thickness_mm)),
    }
    with open(image_path, "rb") as image_file:
        response = BACKEND_CLIENT.post(
            "/process_scene_diorama",
            files={"file": (Path(image_path).name, image_file, "image/png")},
            data=data,
        )
    if response.status_code != 200:
        raise _response_error(response)
    result = response.json()
    stl_path = OUTPUT_DIR / result["stl_model"]
    glb_path = OUTPUT_DIR / result["scene_model"]
    preview_path = OUTPUT_DIR / result["preview"]
    diagnostics = result.get("stl_diagnostics", {})
    summary = _diagnostic_summary(diagnostics, model=DEPTH_MODEL)
    summary["scene_mode"] = "layered-camera-free-diorama"
    summary["selected_layers"] = 1 if selected else 0
    summary["inpainting"] = False
    return (
        _safe_file(glb_path),
        _safe_file(stl_path),
        _safe_file(glb_path),
        _safe_file(preview_path),
        summary,
    )


def _patch_triposg_source(source_dir: Path) -> None:
    inference_path = source_dir / "triposg" / "inference_utils.py"
    vae_path = source_dir / "triposg" / "models" / "autoencoders" / "autoencoder_kl_triposg.py"
    inference_text = inference_path.read_text(encoding="utf-8")
    inference_text = inference_text.replace(
        "from diso import DiffDMC",
        "try:\n    from diso import DiffDMC\nexcept ImportError:\n    DiffDMC = None",
        1,
    )
    inference_path.write_text(inference_text, encoding="utf-8")
    vae_text = vae_path.read_text(encoding="utf-8")
    vae_text = vae_text.replace(
        "            q = self.proj_query(q)",
        "            q = self.proj_query(q.to(dtype=self.proj_query.weight.dtype))",
        1,
    )
    vae_path.write_text(vae_text, encoding="utf-8")


def ensure_triposg_source() -> Path:
    with _TRIPOSG_LOCK:
        source = _clone_exact_repository(TRIPOSG_REPOSITORY, TRIPOSG_SOURCE_REVISION, TRIPOSG_DIR)
        _patch_triposg_source(source)
        return source


def _load_triposg_models():
    import torch
    from huggingface_hub import snapshot_download

    source = ensure_triposg_source()
    scripts_dir = source / "scripts"
    for path in (source, scripts_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    triposg_weights = Path(
        snapshot_download(
            repo_id="VAST-AI/TripoSG",
            revision=TRIPOSG_MODEL_REVISION,
            token=os.getenv("HF_TOKEN") or None,
        )
    )
    pipeline = TripoSGPipeline.from_pretrained(
        triposg_weights,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    return pipeline


def _prepare_selected_mesh_image(selection: dict, output_path: Path, canvas_size: int = 512) -> Image.Image:
    selection_dir = OUTPUT_DIR / "selection" / selection["job_id"]
    source_path = selection_dir / "source.png"
    mask_path = selection_dir / "selection_mask.png"
    _safe_file(source_path)
    _safe_file(mask_path)
    with Image.open(source_path) as source_file, Image.open(mask_path) as mask_file:
        source = ImageOps.exif_transpose(source_file).convert("RGB")
        mask = mask_file.convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("The selected object mask is empty")
    left, top, right, bottom = bbox
    margin = max(2, int(round(max(right - left, bottom - top) * 0.08)))
    crop_box = (
        max(0, left - margin),
        max(0, top - margin),
        min(source.width, right + margin),
        min(source.height, bottom + margin),
    )
    source_crop = source.crop(crop_box)
    mask_crop = mask.crop(crop_box)
    selected_crop = Image.composite(source_crop, Image.new("RGB", source_crop.size, "white"), mask_crop)
    selected_crop.thumbnail((int(canvas_size * 0.9), int(canvas_size * 0.9)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_size, canvas_size), "white")
    canvas.paste(
        selected_crop,
        ((canvas_size - selected_crop.width) // 2, (canvas_size - selected_crop.height) // 2),
    )
    canvas.save(output_path)
    return canvas


def generate_full_mesh(
    image_path: str | Path | None,
    scope: str,
    selection: dict | None,
    max_dimension_mm: float,
    seed: int,
) -> tuple[str, str, str, dict]:
    if not image_path:
        raise ValueError("Upload a photo before generating a full mesh")
    if not str(scope).lower().startswith("select") or not selection:
        raise ValueError("Full Mesh requires Select object; click the object before generating")
    import torch
    import trimesh

    release_gpu_models()
    job_id = uuid4().hex
    job_dir = OUTPUT_DIR / "full-mesh" / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    raw_glb = job_dir / "output_mesh_raw.glb"
    repaired_mesh = job_dir / "output_mesh_repaired.ply"
    final_glb = job_dir / "output_mesh.glb"
    final_stl = job_dir / "output_model.stl"
    metrics: dict = {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    pipeline = None
    try:
        pipeline = _load_triposg_models()
        prepared = _prepare_selected_mesh_image(selection, job_dir / "mesh_input.png")
        with torch.inference_mode():
            samples = pipeline(
                image=prepared,
                generator=torch.Generator(device="cuda").manual_seed(int(seed)),
                num_inference_steps=50,
                guidance_scale=7.0,
                use_flash_decoder=False,
            ).samples[0]
        mesh = trimesh.Trimesh(
            samples[0].astype(np.float32),
            np.ascontiguousarray(samples[1]),
            process=False,
        )
        if not len(mesh.vertices) or not len(mesh.faces):
            raise RuntimeError("TripoSG returned an empty mesh")
        mesh.export(raw_glb)
        repair_mesh_for_printable_stl(
            raw_glb,
            repaired_mesh,
            "printable",
            max_normalized_face_density_log1p=9.95,
            preconditioner="adaptive-voxel-close",
            voxel_resolution=128,
            voxel_fill_method="orthographic",
            smoothing_iterations=2,
            allow_convex_hull_fallback=False,
            metrics=metrics,
        )
        postprocess_mesh_for_stl(
            repaired_mesh,
            final_glb,
            target_max_dimension=float(max_dimension_mm),
            target_faces=10000,
            max_normalized_face_density_log1p=9.95,
            preserve_printability=True,
        )
        convert_mesh_to_stl(final_glb, final_stl)
        diagnostics = json_safe_stl_diagnostics(stl_diagnostics(final_stl))
        summary = _diagnostic_summary(diagnostics, model="VAST-AI/TripoSG")
        summary.update(
            {
                "source_revision": TRIPOSG_SOURCE_REVISION,
                "model_revision": TRIPOSG_MODEL_REVISION,
                "foreground_model": SAM3_MODEL,
                "foreground_model_revision": SAM3_MODEL_REVISION,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
                "repair": metrics,
                "scope": "selected-object",
                "inpainting": False,
            }
        )
        (job_dir / "diagnostics.json").write_text(
            json.dumps({"stl": diagnostics, "summary": summary}, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return _safe_file(final_stl), _safe_file(final_stl), _safe_file(final_glb), summary
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def cleanup_expired_outputs(max_age_seconds: int = 6 * 60 * 60) -> int:
    cleanup_expired_selection_precomputes()
    if not OUTPUT_DIR.is_dir():
        return 0
    now = time.time()
    removed = 0
    for child in OUTPUT_DIR.iterdir():
        if not child.is_dir():
            continue
        candidates = list(child.iterdir()) if child.name in {"selection", "full-mesh"} else [child]
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            try:
                if now - candidate.stat().st_mtime > max_age_seconds:
                    shutil.rmtree(candidate)
                    removed += 1
            except OSError:
                continue
    return removed
