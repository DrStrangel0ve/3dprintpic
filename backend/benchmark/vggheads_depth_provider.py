"""Run pinned VGGHeads inference and normalize its projected FLAME mesh."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


VGGHEADS_SOURCE_URL = "https://github.com/KupynOrest/head_detector.git"
VGGHEADS_SOURCE_REVISION = "fc41ffdec189a983d39eba5d208b586d2204d507"
VGGHEADS_MODEL_ID = "okupyn/vgg_heads"
VGGHEADS_MODEL_REVISION = "22672222d2631d8095d01afdf92cc8537e7433cd"
VGGHEADS_MODEL_FILENAME = "vgg_heads_l.trcd"
VGGHEADS_MODEL_SIZE_BYTES = 417_153_502
VGGHEADS_MODEL_SHA256 = (
    "18acb79c53032db11e8f502c12fdd34b5f642e9bc9041bce152c7b716c1b6f74"
)
VGGHEADS_IMAGE_SIZE = 640
VGGHEADS_SMALL_FACE_MAX_HEIGHT_PX = 96
VGGHEADS_REQUIRED_SOURCE_FILES = (
    "LICENSE",
    "head_detector/detector.py",
    "head_detector/flame.py",
    "head_detector/generic_model.pkl",
    "head_detector/head_info.py",
    "head_detector/utils.py",
    "head_detector/assets/full_faces.npy",
    "head_detector/assets/flame_indices/face.npy",
    "head_detector/assets/flame_indices/head_indices.npy",
    "head_detector/assets/triangles.txt",
)
VGGHEADS_OPTIONAL_DEPENDENCIES = (
    "einops",
    "smplx",
    "torch",
    "torchvision",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
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


def vggheads_preflight(
    provider_root: str | Path,
    model_path: str | Path | None = None,
) -> dict:
    provider_root = Path(provider_root).resolve()
    model_path = Path(model_path).resolve() if model_path else None
    missing_source = [
        relative
        for relative in VGGHEADS_REQUIRED_SOURCE_FILES
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

    dependency_status = {
        name: importlib.util.find_spec(name) is not None
        for name in VGGHEADS_OPTIONAL_DEPENDENCIES
    }
    model_record = {
        "path": str(model_path) if model_path else None,
        "exists": bool(model_path and model_path.is_file()),
        "size_bytes": None,
        "expected_size_bytes": VGGHEADS_MODEL_SIZE_BYTES,
        "sha256": None,
        "expected_sha256": VGGHEADS_MODEL_SHA256,
        "size_pinned": False,
        "hash_pinned": False,
    }
    if model_path and model_path.is_file():
        model_record["size_bytes"] = int(model_path.stat().st_size)
        model_record["sha256"] = _sha256(model_path)
        model_record["size_pinned"] = (
            model_record["size_bytes"] == VGGHEADS_MODEL_SIZE_BYTES
        )
        model_record["hash_pinned"] = (
            model_record["sha256"] == VGGHEADS_MODEL_SHA256
        )

    checks = {
        "source_exists": provider_root.is_dir(),
        "source_revision_pinned": source_revision == VGGHEADS_SOURCE_REVISION,
        "source_clean": source_status == "",
        "source_files_complete": not missing_source,
        "model_supplied": model_path is not None,
        "model_exists": model_record["exists"],
        "model_size_pinned": model_record["size_pinned"],
        "model_hash_pinned": model_record["hash_pinned"],
        "optional_dependencies_available": all(dependency_status.values()),
    }
    return {
        "schema_version": 1,
        "provider": "vggheads",
        "source": {
            "url": VGGHEADS_SOURCE_URL,
            "expected_revision": VGGHEADS_SOURCE_REVISION,
            "actual_revision": source_revision,
            "status": source_status,
            "error": source_error,
            "root": str(provider_root),
            "missing_files": missing_source,
        },
        "model": {
            "id": VGGHEADS_MODEL_ID,
            "expected_revision": VGGHEADS_MODEL_REVISION,
            "filename": VGGHEADS_MODEL_FILENAME,
            **model_record,
        },
        "dependencies": dependency_status,
        "license": {
            "source_code": "MIT",
            "weights": "Hugging Face model card has no explicit license field",
            "bundled_flame_asset": (
                "upstream repository asset; redistribution terms require "
                "separate audit"
            ),
            "production_eligible": False,
            "research_only": True,
        },
        "checks": checks,
        "runnable": bool(all(checks.values())),
    }


def small_face_policy(
    face_height_pixels: int | float,
    *,
    occluded: bool = False,
    maximum_height_pixels: int = VGGHEADS_SMALL_FACE_MAX_HEIGHT_PX,
) -> dict:
    face_height_pixels = float(face_height_pixels)
    maximum_height_pixels = int(maximum_height_pixels)
    if not math.isfinite(face_height_pixels) or face_height_pixels <= 0:
        raise ValueError("Face height must be positive and finite")
    if maximum_height_pixels < 1:
        raise ValueError("Maximum face height must be positive")
    failures = []
    if face_height_pixels > maximum_height_pixels:
        failures.append("face_too_large")
    if occluded:
        failures.append("occluded_face")
    return {
        "eligible": not failures,
        "face_height_pixels": face_height_pixels,
        "maximum_height_pixels": maximum_height_pixels,
        "occluded": bool(occluded),
        "failures": failures,
        "bypass_reason": ",".join(failures) if failures else None,
    }


def subject_interior_taper(
    selection_mask: np.ndarray,
    *,
    taper_pixels: float = 4.0,
    boundary_pixels: float = 1.0,
) -> np.ndarray:
    taper_pixels = float(taper_pixels)
    boundary_pixels = float(boundary_pixels)
    if not math.isfinite(taper_pixels) or taper_pixels <= 0:
        raise ValueError("Subject taper must be positive and finite")
    if not math.isfinite(boundary_pixels) or boundary_pixels < 0:
        raise ValueError("Boundary width must be non-negative and finite")
    binary = (np.asarray(selection_mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or not np.any(binary):
        raise ValueError("Subject taper expects a non-empty 2D mask")
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    weight = np.clip(
        (distance - boundary_pixels) / taper_pixels,
        0.0,
        1.0,
    )
    weight = weight * weight * (3.0 - 2.0 * weight)
    weight *= binary
    weight[distance <= boundary_pixels] = 0.0
    return weight.astype(np.float32)


def fill_depth_nearest(depth: np.ndarray) -> tuple[np.ndarray, dict]:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("Depth fill expects a 2D array")
    finite = np.isfinite(depth)
    finite_pixels = int(np.count_nonzero(finite))
    if finite_pixels == 0:
        raise ValueError("Depth fill has no finite samples")
    if finite_pixels == depth.size:
        return depth.copy(), {
            "finite_pixels": finite_pixels,
            "filled_pixels": 0,
            "method": "identity",
        }
    _distance, indices = distance_transform_edt(
        ~finite,
        return_indices=True,
    )
    filled = depth[tuple(indices)]
    return filled.astype(np.float32), {
        "finite_pixels": finite_pixels,
        "filled_pixels": int(depth.size - finite_pixels),
        "method": "nearest-finite-euclidean",
    }


def normalize_front_depth(
    depth: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[np.ndarray, dict]:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not np.all(np.isfinite(depth)):
        raise ValueError("Front-depth normalization expects finite 2D depth")
    lower_percentile = float(lower_percentile)
    upper_percentile = float(upper_percentile)
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Depth normalization percentiles are invalid")
    low, high = np.percentile(
        depth.astype(np.float64),
        (lower_percentile, upper_percentile),
    )
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Depth normalization has no usable span")
    normalized = 1.0 - np.clip((depth - low) / (high - low), 0.0, 1.0)
    return normalized.astype(np.float32), {
        "method": "filled-depth-percentile-normalization",
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "depth_low": float(low),
        "depth_high": float(high),
        "near_is_larger": True,
    }


def rasterize_projected_mesh_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    height: int,
    width: int,
    front_surface: str = "minimum-z",
) -> tuple[np.ndarray, dict]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    height = int(height)
    width = int(width)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError("Projected vertices must have shape [N, >=3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Mesh faces must have shape [M, 3]")
    if height <= 0 or width <= 0:
        raise ValueError("Depth dimensions must be positive")
    if front_surface not in {"minimum-z", "maximum-z"}:
        raise ValueError("Front surface must be 'minimum-z' or 'maximum-z'")
    if faces.size and (
        int(np.min(faces)) < 0 or int(np.max(faces)) >= len(vertices)
    ):
        raise ValueError("Mesh face index is out of bounds")

    initial = np.inf if front_surface == "minimum-z" else -np.inf
    depth = np.full((height, width), initial, dtype=np.float64)
    rendered_faces = 0
    degenerate_faces = 0
    clipped_faces = 0
    for face in faces:
        triangle = vertices[face, :3]
        if not np.all(np.isfinite(triangle)):
            continue
        x = triangle[:, 0]
        y = triangle[:, 1]
        min_x = max(0, int(math.floor(float(np.min(x)))))
        max_x = min(width - 1, int(math.ceil(float(np.max(x)))))
        min_y = max(0, int(math.floor(float(np.min(y)))))
        max_y = min(height - 1, int(math.ceil(float(np.max(y)))))
        if min_x > max_x or min_y > max_y:
            clipped_faces += 1
            continue
        denominator = (
            (y[1] - y[2]) * (x[0] - x[2])
            + (x[2] - x[1]) * (y[0] - y[2])
        )
        if abs(float(denominator)) <= 1e-12:
            degenerate_faces += 1
            continue
        grid_y, grid_x = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        # VGGHeads vertices are image-space samples at integer pixel centers.
        sample_x = grid_x.astype(np.float64)
        sample_y = grid_y.astype(np.float64)
        weight_0 = (
            (y[1] - y[2]) * (sample_x - x[2])
            + (x[2] - x[1]) * (sample_y - y[2])
        ) / denominator
        weight_1 = (
            (y[2] - y[0]) * (sample_x - x[2])
            + (x[0] - x[2]) * (sample_y - y[2])
        ) / denominator
        weight_2 = 1.0 - weight_0 - weight_1
        inside = (
            (weight_0 >= -1e-9)
            & (weight_1 >= -1e-9)
            & (weight_2 >= -1e-9)
        )
        if not np.any(inside):
            continue
        triangle_depth = (
            weight_0 * triangle[0, 2]
            + weight_1 * triangle[1, 2]
            + weight_2 * triangle[2, 2]
        )
        target = depth[min_y : max_y + 1, min_x : max_x + 1]
        if front_surface == "minimum-z":
            update = inside & (triangle_depth < target)
        else:
            update = inside & (triangle_depth > target)
        target[update] = triangle_depth[update]
        rendered_faces += 1

    finite = np.isfinite(depth)
    depth[~finite] = np.nan
    return depth.astype(np.float32), {
        "front_surface": front_surface,
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "rendered_faces": int(rendered_faces),
        "degenerate_faces": int(degenerate_faces),
        "clipped_faces": int(clipped_faces),
        "finite_pixels": int(np.count_nonzero(finite)),
        "coverage_ratio": float(np.count_nonzero(finite) / depth.size),
    }


def _install_legacy_dependency_compatibility() -> dict:
    aliases = {
        "bool": bool,
        "complex": complex,
        "float": float,
        "int": int,
        "object": object,
        "str": str,
        "unicode": str,
    }
    installed_aliases = []
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
            installed_aliases.append(name)
    installed_getargspec = False
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
        installed_getargspec = True
    return {
        "numpy_aliases_installed": installed_aliases,
        "inspect_getargspec_installed": installed_getargspec,
    }


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load VGGHeads module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_vggheads_modules(provider_root: Path) -> tuple[dict, dict]:
    compatibility = _install_legacy_dependency_compatibility()
    package_root = provider_root / "head_detector"
    package = types.ModuleType("head_detector")
    package.__path__ = [str(package_root)]
    package.__file__ = str(package_root / "__init__.py")
    sys.modules["head_detector"] = package
    head_info = _load_module(
        "head_detector.head_info",
        package_root / "head_info.py",
    )
    utils = _load_module(
        "head_detector.utils",
        package_root / "utils.py",
    )
    flame = _load_module(
        "head_detector.flame",
        package_root / "flame.py",
    )
    return {
        "head_info": head_info,
        "utils": utils,
        "flame": flame,
    }, compatibility


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0,
        float(box_a[3] - box_a[1]),
    )
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0,
        float(box_b[3] - box_b[1]),
    )
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def flame_expression_summary(parsed_params) -> dict:
    expression = (
        parsed_params.expression.detach().float().cpu().numpy().reshape(-1)
    )
    jaw = parsed_params.jaw.detach().float().cpu().numpy().reshape(-1)
    return {
        "expression_l2": float(np.linalg.norm(expression)),
        "expression_max_abs": float(
            np.max(np.abs(expression), initial=0.0)
        ),
        "jaw_l2": float(np.linalg.norm(jaw)),
        "jaw": [float(value) for value in jaw],
    }


class VGGHeadsProvider:
    """Pinned, local-path VGGHeads inference without the optional Sim3DR import."""

    def __init__(
        self,
        provider_root: str | Path,
        model_path: str | Path,
        *,
        device: str = "cuda",
        image_size: int = VGGHEADS_IMAGE_SIZE,
    ):
        import torch

        self.provider_root = Path(provider_root).resolve()
        self.model_path = Path(model_path).resolve()
        self.device = str(device)
        self.image_size = int(image_size)
        if self.image_size < 64:
            raise ValueError("VGGHeads image size is too small")
        preflight = vggheads_preflight(self.provider_root, self.model_path)
        if not preflight["runnable"]:
            failed = [
                name
                for name, passed in preflight["checks"].items()
                if not passed
            ]
            raise RuntimeError(
                "VGGHeads preflight failed: " + ", ".join(failed)
            )
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("VGGHeads requested CUDA but CUDA is unavailable")

        modules, compatibility = _load_vggheads_modules(self.provider_root)
        self.preflight = preflight
        self.compatibility = compatibility
        self.head_info = modules["head_info"]
        self.utils = modules["utils"]
        self.flame_module = modules["flame"]
        self.flame = self.flame_module.FLAMELayer().to(self.device).eval()
        self.model = torch.jit.load(
            str(self.model_path),
            map_location=self.device,
        )
        self.model.to(self.device).eval()
        self.faces = np.asarray(self.flame.faces, dtype=np.int32)
        self.calls: list[dict] = []
        self.allocated_before_inference = (
            int(torch.cuda.memory_allocated(self.device))
            if self.device.startswith("cuda")
            else 0
        )
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

    def _preprocess(self, image_rgb: np.ndarray):
        import torch

        height, width = image_rgb.shape[:2]
        scale = self.image_size / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        resized = cv2.resize(
            image_rgb,
            (new_width, new_height),
            interpolation=cv2.INTER_LANCZOS4,
        )
        pad_width = self.image_size - new_width
        pad_height = self.image_size - new_height
        left = pad_width // 2
        top = pad_height // 2
        padded = cv2.copyMakeBorder(
            resized,
            top,
            pad_height - top,
            left,
            pad_width - left,
            cv2.BORDER_CONSTANT,
            value=127,
        )
        tensor = (
            torch.from_numpy(np.ascontiguousarray(padded))
            .to(self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        return tensor, {
            "scale": float(scale),
            "padding_xy": [int(left), int(top)],
            "input_shape": [int(height), int(width)],
            "network_shape": [self.image_size, self.image_size],
        }

    def infer(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        confidence_threshold: float = 0.5,
        target_bbox_xyxy: list[float] | tuple[float, ...] | None = None,
    ) -> dict:
        import torch

        if isinstance(image, (str, Path)):
            image_rgb = np.asarray(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image_rgb = np.asarray(image.convert("RGB"))
        else:
            image_rgb = np.asarray(image)
        if (
            image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
            or image_rgb.dtype != np.uint8
        ):
            raise ValueError("VGGHeads expects an RGB uint8 image")
        tensor, transform = self._preprocess(image_rgb)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            boxes, scores, flame_params = self.model(tensor)
            boxes, scores, flame_params = self.utils.nms(
                boxes,
                scores,
                flame_params,
                confidence_threshold=float(confidence_threshold),
            )
            if len(boxes) == 0:
                raise RuntimeError("VGGHeads found no head above the threshold")
            _canonical, _rotation, projected = (
                self.flame_module.reproject_spatial_vertices(
                    self.flame,
                    flame_params,
                    to_2d=False,
                )
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started

        scale = float(transform["scale"])
        pad_x, pad_y = transform["padding_xy"]
        boxes_np = boxes.detach().float().cpu().numpy()
        scores_np = scores.detach().float().cpu().numpy().reshape(-1)
        vertices_np = projected.detach().float().cpu().numpy()
        vertices_np[:, :, 0] -= pad_x
        vertices_np[:, :, 1] -= pad_y
        vertices_np /= scale
        boxes_np[:, (0, 2)] -= pad_x
        boxes_np[:, (1, 3)] -= pad_y
        boxes_np /= scale

        if target_bbox_xyxy is None:
            selected_index = int(np.argmax(scores_np))
            selection_method = "highest-confidence"
            target_ious = None
        else:
            target = np.asarray(target_bbox_xyxy, dtype=np.float64)
            if target.shape != (4,) or not np.all(np.isfinite(target)):
                raise ValueError("Target bbox must contain four finite values")
            target_ious = [
                _bbox_iou(box, target)
                for box in boxes_np
            ]
            selected_index = int(
                max(
                    range(len(boxes_np)),
                    key=lambda index: (
                        target_ious[index],
                        float(scores_np[index]),
                    ),
                )
            )
            selection_method = "maximum-target-iou"

        selected_params = flame_params[selected_index : selected_index + 1]
        parsed_params = self.head_info.FlameParams.from_3dmm(
            selected_params.detach().cpu()
        )
        pose = self.utils.calculate_rpy(parsed_params)
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.startswith("cuda")
            else 0
        )
        record = {
            "provider": "vggheads",
            "confidence": float(scores_np[selected_index]),
            "bbox_xyxy": [
                float(value) for value in boxes_np[selected_index]
            ],
            "head_pose_deg": {
                "roll": float(pose.roll),
                "pitch": float(pose.pitch),
                "yaw": float(pose.yaw),
            },
            "flame_expression": flame_expression_summary(parsed_params),
            "detections": int(len(boxes_np)),
            "selection_method": selection_method,
            "target_ious": target_ious,
            "runtime_seconds": float(elapsed),
            "peak_vram_gib": float(peak_allocated / (1024**3)),
            "incremental_peak_vram_gib": float(
                max(0, peak_allocated - self.allocated_before_inference)
                / (1024**3)
            ),
            "transform": transform,
            "vertices": int(vertices_np.shape[1]),
            "faces": int(len(self.faces)),
        }
        self.calls.append(record)
        return {
            "vertices": vertices_np[selected_index].astype(np.float32),
            "faces": self.faces.copy(),
            "flame_params": selected_params.detach().float().cpu().numpy()[0],
            "metadata": record,
        }

    def provenance(self) -> dict:
        return {
            "preflight": self.preflight,
            "compatibility": self.compatibility,
            "device": self.device,
            "image_size": self.image_size,
            "calls": self.calls,
        }


def write_depth_outputs(
    image_path: str | Path,
    provider_root: str | Path,
    model_path: str | Path,
    output_depth: str | Path,
    *,
    output_metadata: str | Path | None = None,
    output_vertices: str | Path | None = None,
    output_faces: str | Path | None = None,
    device: str = "cuda",
    target_bbox_xyxy: list[float] | tuple[float, ...] | None = None,
) -> dict:
    image_path = Path(image_path)
    with Image.open(image_path) as loaded:
        width, height = loaded.size
    provider = VGGHeadsProvider(
        provider_root,
        model_path,
        device=device,
    )
    inference = provider.infer(
        image_path,
        target_bbox_xyxy=target_bbox_xyxy,
    )
    depth, raster = rasterize_projected_mesh_depth(
        inference["vertices"],
        inference["faces"],
        height=height,
        width=width,
        front_surface="minimum-z",
    )
    output_depth = Path(output_depth)
    output_depth.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_depth, depth)
    if output_vertices:
        output_vertices = Path(output_vertices)
        output_vertices.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_vertices, inference["vertices"])
    if output_faces:
        output_faces = Path(output_faces)
        output_faces.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_faces, inference["faces"])
    evidence = {
        "schema_version": 1,
        "provider": provider.provenance(),
        "image": {
            "path": str(image_path.resolve()),
            "sha256": _sha256(image_path),
            "shape": [height, width],
        },
        "inference": inference["metadata"],
        "raster": raster,
        "outputs": {
            "depth": str(output_depth.resolve()),
            "vertices": (
                str(Path(output_vertices).resolve())
                if output_vertices
                else None
            ),
            "faces": (
                str(Path(output_faces).resolve())
                if output_faces
                else None
            ),
        },
    }
    if output_metadata:
        output_metadata = Path(output_metadata)
        output_metadata.parent.mkdir(parents=True, exist_ok=True)
        output_metadata.write_text(
            json.dumps(evidence, indent=2) + "\n",
            encoding="utf-8",
        )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--provider-root", required=True)
    preflight_parser.add_argument("--model-path")

    depth_parser = subparsers.add_parser("depth")
    depth_parser.add_argument("--image", required=True)
    depth_parser.add_argument("--provider-root", required=True)
    depth_parser.add_argument("--model-path", required=True)
    depth_parser.add_argument("--output-depth", required=True)
    depth_parser.add_argument("--output-metadata")
    depth_parser.add_argument("--output-vertices")
    depth_parser.add_argument("--output-faces")
    depth_parser.add_argument("--device", default="cuda")
    depth_parser.add_argument("--target-bbox", nargs=4, type=float)

    args = parser.parse_args()
    if args.command == "preflight":
        evidence = vggheads_preflight(
            args.provider_root,
            args.model_path,
        )
        print(json.dumps(evidence, indent=2))
        if not evidence["runnable"]:
            raise SystemExit(2)
        return
    evidence = write_depth_outputs(
        args.image,
        args.provider_root,
        args.model_path,
        args.output_depth,
        output_metadata=args.output_metadata,
        output_vertices=args.output_vertices,
        output_faces=args.output_faces,
        device=args.device,
        target_bbox_xyxy=args.target_bbox,
    )
    print(json.dumps(evidence["inference"], indent=2))


if __name__ == "__main__":
    main()
