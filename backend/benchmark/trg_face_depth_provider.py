"""Run pinned TRG face reconstruction and expose camera-space geometry."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image


TRG_SOURCE_URL = "https://github.com/asw91666/TRG-Release.git"
TRG_SOURCE_REVISION = "3916650722576599126b6242730a2210f597ac71"
TRG_CHECKPOINT_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1CQeB4W2KbNVpzx4FZfuyMVgljlOvtpwU"
)
TRG_CHECKPOINT_FILENAME = "state_dict.bin"
TRG_CHECKPOINT_SIZE_BYTES = 150_684_140
TRG_CHECKPOINT_SHA256 = (
    "75d745e5a0fe96f170cdce61ded64f080dfcbf45d3514a60db8f097fc758c77b"
)
TRG_INPUT_SIZE = 192
TRG_FOCAL_LENGTH_PX = 5000.0
TRG_PHYSICAL_FACE_PRIOR_M = 0.2
TRG_REQUIRED_SOURCE_FILES = (
    "LICENSE",
    "data/arkit_subsample.pkl",
    "data/init_vtx_cam.pkl",
    "data/triangles.npy",
    "models/trg_model/__init__.py",
    "models/trg_model/trg.py",
    "models/trg_model/core/cfgs.py",
    "models/trg_model/configs/trg_face_config.yaml",
)
TRG_REQUIRED_DEPENDENCIES = (
    "cv2",
    "torch",
    "torchgeometry",
    "torchvision",
    "yacs",
)
_TRG_IMPORT_LOCK = threading.RLock()
_TRG_UPSTREAM_MODULE_PREFIXES = ("data", "models", "util")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        (
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "-C",
            str(root),
            *args,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def trg_preflight(
    provider_root: str | Path,
    checkpoint_path: str | Path | None = None,
) -> dict:
    provider_root = Path(provider_root).resolve()
    checkpoint_path = (
        Path(checkpoint_path).resolve() if checkpoint_path else None
    )
    missing_source = [
        relative
        for relative in TRG_REQUIRED_SOURCE_FILES
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

    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in TRG_REQUIRED_DEPENDENCIES
    }
    checkpoint = {
        "path": str(checkpoint_path) if checkpoint_path else None,
        "exists": bool(checkpoint_path and checkpoint_path.is_file()),
        "size_bytes": None,
        "expected_size_bytes": TRG_CHECKPOINT_SIZE_BYTES,
        "sha256": None,
        "expected_sha256": TRG_CHECKPOINT_SHA256,
        "size_pinned": False,
        "hash_pinned": False,
    }
    if checkpoint_path and checkpoint_path.is_file():
        checkpoint["size_bytes"] = int(checkpoint_path.stat().st_size)
        checkpoint["sha256"] = _sha256(checkpoint_path)
        checkpoint["size_pinned"] = (
            checkpoint["size_bytes"] == TRG_CHECKPOINT_SIZE_BYTES
        )
        checkpoint["hash_pinned"] = (
            checkpoint["sha256"] == TRG_CHECKPOINT_SHA256
        )

    checks = {
        "source_exists": provider_root.is_dir(),
        "source_revision_pinned": source_revision == TRG_SOURCE_REVISION,
        "source_clean": source_status == "",
        "source_files_complete": not missing_source,
        "checkpoint_supplied": checkpoint_path is not None,
        "checkpoint_exists": checkpoint["exists"],
        "checkpoint_size_pinned": checkpoint["size_pinned"],
        "checkpoint_hash_pinned": checkpoint["hash_pinned"],
        "dependencies_available": all(dependencies.values()),
    }
    return {
        "schema_version": 1,
        "provider": "trg-eccv-2024",
        "source": {
            "url": TRG_SOURCE_URL,
            "expected_revision": TRG_SOURCE_REVISION,
            "actual_revision": source_revision,
            "status": source_status,
            "error": source_error,
            "root": str(provider_root),
            "missing_files": missing_source,
        },
        "checkpoint": {
            "folder_url": TRG_CHECKPOINT_FOLDER_URL,
            "filename": TRG_CHECKPOINT_FILENAME,
            **checkpoint,
        },
        "dependencies": dependencies,
        "camera_contract": {
            "coordinate_space": "provider-camera-meters",
            "physical_face_prior_m": TRG_PHYSICAL_FACE_PRIOR_M,
            "focal_length_px": TRG_FOCAL_LENGTH_PX,
            "focal_length_source": "fixed-official-demo-constant",
            "metric_depth_calibrated_for_input_camera": False,
            "interpretation": "focal-conditioned camera-space geometry",
        },
        "compatibility": {
            "upstream_files_modified": False,
            "resnet18_lfs_pointer_bypassed": True,
            "checkpoint_strict_load_required": True,
            "constructor_rot6d_compatibility": (
                "exact first-two-rotation-columns representation"
            ),
        },
        "license": {
            "repository": "MIT",
            "embedded_cfg_header": "MPG proprietary notice",
            "checkpoint_terms": "not stated separately by upstream",
            "training_data": "ARKitFace research-only",
            "production_eligible": False,
            "research_only": True,
        },
        "checks": checks,
        "runnable": bool(all(checks.values())),
    }


def official_crop_bbox(
    target_bbox_xyxy: list[int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Apply the official TRG 1.5x square landmark-bbox crop."""
    left, top, right, bottom = (
        float(value) for value in target_bbox_xyxy
    )
    if not np.all(np.isfinite((left, top, right, bottom))):
        raise ValueError("Target bbox must be finite")
    if right <= left or bottom <= top:
        raise ValueError("Target bbox must have positive area")
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    size = max(right - left, bottom - top)
    return (
        int(center_x - 0.75 * size),
        int(center_y - 0.75 * size),
        int(center_x + 0.75 * size),
        int(center_y + 0.75 * size),
    )


def crop_intrinsic_matrix(
    image_height: int,
    image_width: int,
    crop_bbox_xyxy: tuple[int, int, int, int],
    *,
    focal_length_px: float = TRG_FOCAL_LENGTH_PX,
    input_size: int = TRG_INPUT_SIZE,
) -> np.ndarray:
    """Build TRG's transposed 4x3 crop-intrinsic representation."""
    image_height = int(image_height)
    image_width = int(image_width)
    left, top, right, bottom = crop_bbox_xyxy
    crop_size = float(right - left)
    if image_height <= 0 or image_width <= 0 or crop_size <= 0:
        raise ValueError("Image and crop dimensions must be positive")
    focal_length_px = float(focal_length_px)
    if not np.isfinite(focal_length_px) or focal_length_px <= 0:
        raise ValueError("Focal length must be positive and finite")
    scale = float(input_size) / crop_size
    camera = np.array(
        [
            [focal_length_px, 0.0, 0.0],
            [0.0, focal_length_px, 0.0],
            [image_width / 2.0, image_height / 2.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    intrinsic = camera.copy()
    intrinsic[0, 0] = focal_length_px * scale
    intrinsic[1, 1] = focal_length_px * scale
    intrinsic[2, 0] = (image_width / 2.0 - left) * scale
    intrinsic[2, 1] = (image_height / 2.0 - top) * scale
    return intrinsic


def project_camera_vertices(
    camera_vertices: np.ndarray,
    *,
    image_height: int,
    image_width: int,
    focal_length_px: float = TRG_FOCAL_LENGTH_PX,
) -> np.ndarray:
    vertices = np.asarray(camera_vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Camera vertices must have shape [N, 3]")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Camera vertices must be finite")
    if np.any(vertices[:, 2] <= 0):
        raise ValueError("Camera vertices must lie in front of the camera")
    focal_length_px = float(focal_length_px)
    x = focal_length_px * vertices[:, 0] / vertices[:, 2]
    y = focal_length_px * vertices[:, 1] / vertices[:, 2]
    projected = np.column_stack(
        (
            x + float(image_width) / 2.0,
            y + float(image_height) / 2.0,
            vertices[:, 2],
        )
    )
    return projected.astype(np.float32)


def bbox_iou(
    first_xyxy: list[float] | tuple[float, float, float, float],
    second_xyxy: list[float] | tuple[float, float, float, float],
) -> float:
    a = np.asarray(first_xyxy, dtype=np.float64)
    b = np.asarray(second_xyxy, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,) or not np.all(np.isfinite((a, b))):
        raise ValueError("Bboxes must each contain four finite values")
    intersection_width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection_height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


@contextmanager
def _upstream_import_context(provider_root: Path):
    def is_upstream_name(name: str) -> bool:
        return any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in _TRG_UPSTREAM_MODULE_PREFIXES
        )

    with _TRG_IMPORT_LOCK:
        previous_cwd = Path.cwd()
        root_string = str(provider_root)
        shadowed_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if is_upstream_name(name)
        }
        for name in shadowed_modules:
            sys.modules.pop(name, None)
        sys.path.insert(0, root_string)
        os.chdir(provider_root)
        try:
            yield
        finally:
            for name in tuple(sys.modules):
                if is_upstream_name(name):
                    sys.modules.pop(name, None)
            sys.modules.update(shadowed_modules)
            os.chdir(previous_cwd)
            try:
                sys.path.remove(root_string)
            except ValueError:
                pass


class TRGFaceDepthProvider:
    def __init__(
        self,
        provider_root: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
    ) -> None:
        self.provider_root = Path(provider_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device_name = str(device)
        self.preflight = trg_preflight(
            self.provider_root,
            self.checkpoint_path,
        )
        if not self.preflight["runnable"]:
            failed = [
                name
                for name, passed in self.preflight["checks"].items()
                if not passed
            ]
            raise RuntimeError("TRG preflight failed: " + ", ".join(failed))
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TRG requested CUDA but CUDA is unavailable")
        with _upstream_import_context(self.provider_root):
            import torch.utils.model_zoo as model_zoo

            original_load_url = model_zoo.load_url
            model_zoo.load_url = lambda *args, **kwargs: {}
            try:
                from models.trg_model.core.cfgs import parse_args as parse_config
                import models.trg_model.trg as trg_module
                from util.pkl import read_pkl

                config = parse_config(
                    "models/trg_model/configs/trg_face_config.yaml"
                )
                initial = read_pkl("data/init_vtx_cam.pkl")
                original_rotmat_to_rot6d = trg_module.rotmat_to_rot6d
                trg_module.rotmat_to_rot6d = lambda rotation: (
                    rotation[..., :2]
                    .contiguous()
                    .view(rotation.shape[0], rotation.shape[1], 6)
                )
                try:
                    model = trg_module.load_trg(
                        config,
                        initial["t_vtx_1220"],
                        initial["R_t"],
                    )
                finally:
                    trg_module.rotmat_to_rot6d = original_rotmat_to_rot6d
            finally:
                model_zoo.load_url = original_load_url
            state_dict = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict, strict=True)
            model = model.to(device).eval()
        self._model = model
        self._torch = torch

    def infer(
        self,
        image_path: str | Path,
        *,
        target_bbox_xyxy: list[int] | tuple[int, int, int, int],
    ) -> dict:
        self._load()
        torch = self._torch
        device = torch.device(self.device_name)
        with Image.open(image_path) as loaded:
            image = np.asarray(loaded.convert("RGB"))
        height, width = image.shape[:2]
        crop_bbox = official_crop_bbox(target_bbox_xyxy)
        left, top, right, bottom = crop_bbox
        if right <= left or bottom <= top:
            raise ValueError("TRG crop is empty")
        source_points = np.float32(
            [[left, top], [left, bottom], [right, top]]
        )
        destination_points = np.float32(
            [
                [0, 0],
                [0, TRG_INPUT_SIZE - 1],
                [TRG_INPUT_SIZE - 1, 0],
            ]
        )
        transform = cv2.getAffineTransform(source_points, destination_points)
        crop = cv2.warpAffine(
            image,
            transform,
            (TRG_INPUT_SIZE, TRG_INPUT_SIZE),
            flags=cv2.INTER_LINEAR,
        )
        normalized = crop.astype(np.float32) / 255.0
        normalized = (
            normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_tensor = torch.from_numpy(
            normalized.transpose(2, 0, 1)[None]
        ).to(device)
        intrinsic = crop_intrinsic_matrix(
            height,
            width,
            crop_bbox,
        )
        intrinsic_tensor = torch.from_numpy(intrinsic[None]).to(device)
        bbox_info = torch.tensor(
            [
                [
                    left,
                    top,
                    right,
                    bottom,
                    TRG_FOCAL_LENGTH_PX,
                    height,
                    width,
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.inference_mode():
            predictions, _features = self._model(
                image_tensor,
                intrinsic_tensor,
                bbox_info,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        runtime_seconds = time.perf_counter() - started
        output = predictions["output"][-1]
        camera_vertices = output["pred_face_cam"][0].detach().cpu().numpy()
        world_vertices = output["pred_face_world"][0].detach().cpu().numpy()
        camera_transform = output["pred_R_t"][0].detach().cpu().numpy()
        correction_parameters = (
            output["pred_cor_param"][0].detach().cpu().numpy()
        )
        projected = project_camera_vertices(
            camera_vertices,
            image_height=height,
            image_width=width,
        )
        projected_bbox = [
            float(np.min(projected[:, 0])),
            float(np.min(projected[:, 1])),
            float(np.max(projected[:, 0])),
            float(np.max(projected[:, 1])),
        ]
        faces = np.load(self.provider_root / "data/triangles.npy").astype(
            np.int32
        )
        peak_vram = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        metadata = {
            "provider": "trg-eccv-2024",
            "runtime_seconds": float(runtime_seconds),
            "peak_vram_bytes": peak_vram,
            "device": str(device),
            "input_size": TRG_INPUT_SIZE,
            "target_bbox_xyxy": [int(value) for value in target_bbox_xyxy],
            "crop_bbox_xyxy": list(crop_bbox),
            "projected_bbox_xyxy": projected_bbox,
            "projected_target_bbox_iou": bbox_iou(
                projected_bbox,
                target_bbox_xyxy,
            ),
            "camera_z_min_m": float(np.min(camera_vertices[:, 2])),
            "camera_z_max_m": float(np.max(camera_vertices[:, 2])),
            "camera_z_span_m": float(np.ptp(camera_vertices[:, 2])),
            "camera_transform_row_vector": camera_transform.tolist(),
            "correction_parameters": correction_parameters.tolist(),
            "focal_length_px": TRG_FOCAL_LENGTH_PX,
            "focal_length_source": "fixed-official-demo-constant",
            "metric_depth_calibrated_for_input_camera": False,
            "coordinate_space": "full-source-image-pixels-plus-camera-z-meters",
            "vertex_count": int(len(projected)),
            "face_count": int(len(faces)),
        }
        return {
            "vertices": projected,
            "camera_vertices": camera_vertices.astype(np.float32),
            "world_vertices": world_vertices.astype(np.float32),
            "faces": faces,
            "metadata": metadata,
        }

    def provenance(self) -> dict:
        return self.preflight
