"""Run pinned SMIRK geometry inference without its PyTorch3D renderer."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image


SMIRK_SOURCE_URL = "https://github.com/georgeretsi/smirk.git"
SMIRK_SOURCE_REVISION = "c7de404c4389f073906a6db1adabf62efcea3f35"
SMIRK_CHECKPOINT_ID = "1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE"
SMIRK_CHECKPOINT_FILENAME = "SMIRK_em1.pt"
SMIRK_CHECKPOINT_SIZE_BYTES = 140_495_895
SMIRK_CHECKPOINT_SHA256 = (
    "26b234e3cc31a5de226bcba4321bb5d7343a2ad48234c028b57a1c2f8c2d22d0"
)
SMIRK_FLAME_MODEL_SHA256 = (
    "efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b"
)
SMIRK_IMAGE_SIZE = 224
SMIRK_CROP_SCALE = 1.4
SMIRK_MEDIAPIPE_LANDMARK_COUNT = 105
SMIRK_MEDIAPIPE_EMBEDDING = (
    "assets/mediapipe_landmark_embedding/"
    "mediapipe_landmark_embedding.npz"
)
SMIRK_REQUIRED_SOURCE_FILES = (
    "LICENSE",
    "src/smirk_encoder.py",
    "src/FLAME/FLAME.py",
    "src/FLAME/lbs.py",
    "assets/landmark_embedding.npy",
    "assets/l_eyelid.npy",
    "assets/r_eyelid.npy",
    SMIRK_MEDIAPIPE_EMBEDDING,
)
SMIRK_REQUIRED_DEPENDENCIES = (
    "cv2",
    "PIL",
    "timm",
    "torch",
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


def smirk_preflight(
    provider_root: str | Path,
    checkpoint_path: str | Path,
    flame_model_path: str | Path,
) -> dict:
    provider_root = Path(provider_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    flame_model_path = Path(flame_model_path).resolve()
    missing_source = [
        relative
        for relative in SMIRK_REQUIRED_SOURCE_FILES
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
        for name in SMIRK_REQUIRED_DEPENDENCIES
    }
    checkpoint = {
        "path": str(checkpoint_path),
        "exists": checkpoint_path.is_file(),
        "size_bytes": (
            int(checkpoint_path.stat().st_size)
            if checkpoint_path.is_file()
            else None
        ),
        "sha256": (
            _sha256(checkpoint_path)
            if checkpoint_path.is_file()
            else None
        ),
    }
    flame_model = {
        "path": str(flame_model_path),
        "exists": flame_model_path.is_file(),
        "sha256": (
            _sha256(flame_model_path)
            if flame_model_path.is_file()
            else None
        ),
    }
    checks = {
        "source_exists": provider_root.is_dir(),
        "source_revision_pinned": source_revision == SMIRK_SOURCE_REVISION,
        "source_clean": source_status == "",
        "source_files_complete": not missing_source,
        "checkpoint_exists": checkpoint["exists"],
        "checkpoint_size_pinned": (
            checkpoint["size_bytes"] == SMIRK_CHECKPOINT_SIZE_BYTES
        ),
        "checkpoint_hash_pinned": (
            checkpoint["sha256"] == SMIRK_CHECKPOINT_SHA256
        ),
        "flame_model_exists": flame_model["exists"],
        "flame_model_hash_pinned": (
            flame_model["sha256"] == SMIRK_FLAME_MODEL_SHA256
        ),
        "dependencies_available": all(dependencies.values()),
    }
    return {
        "schema_version": 1,
        "provider": "smirk",
        "source": {
            "url": SMIRK_SOURCE_URL,
            "expected_revision": SMIRK_SOURCE_REVISION,
            "actual_revision": source_revision,
            "status": source_status,
            "error": source_error,
            "root": str(provider_root),
            "missing_files": missing_source,
        },
        "checkpoint": {
            "google_drive_id": SMIRK_CHECKPOINT_ID,
            "filename": SMIRK_CHECKPOINT_FILENAME,
            "expected_size_bytes": SMIRK_CHECKPOINT_SIZE_BYTES,
            "expected_sha256": SMIRK_CHECKPOINT_SHA256,
            **checkpoint,
        },
        "flame_model": {
            "expected_sha256": SMIRK_FLAME_MODEL_SHA256,
            **flame_model,
        },
        "dependencies": dependencies,
        "license": {
            "source_code": "MIT",
            "checkpoint": "Upstream Google Drive artifact; separate audit required",
            "flame_model": (
                "Research-only local asset; do not redistribute without "
                "confirming FLAME terms"
            ),
            "production_eligible": False,
            "research_only": True,
        },
        "checks": checks,
        "runnable": bool(all(checks.values())),
    }


def _install_legacy_compatibility() -> dict:
    installed_getargspec = False
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
        installed_getargspec = True
    aliases = {
        "bool_": np.bool_,
        "int_": np.int64,
        "float_": np.float64,
        "complex_": np.complex128,
        "object_": np.object_,
        "unicode_": np.str_,
        "str_": np.str_,
    }
    installed_aliases = []
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)
            installed_aliases.append(name)
    return {
        "inspect_getargspec_installed": installed_getargspec,
        "numpy_aliases_installed": installed_aliases,
    }


@contextmanager
def _provider_import_context(provider_root: Path):
    previous_cwd = Path.cwd()
    root_text = str(provider_root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    os.chdir(provider_root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if inserted and sys.path and sys.path[0] == root_text:
            sys.path.pop(0)


def _crop_transform(
    bbox_xyxy: list[float] | tuple[float, ...],
    *,
    image_width: int,
    image_height: int,
    crop_scale: float = SMIRK_CROP_SCALE,
) -> dict:
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
        raise ValueError("SMIRK target bbox must contain four finite values")
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("SMIRK target bbox is empty")
    crop_scale = float(crop_scale)
    if not np.isfinite(crop_scale) or crop_scale <= 0:
        raise ValueError("SMIRK crop scale must be positive and finite")
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    side_unrounded = (
        0.5 * ((x1 - x0) + (y1 - y0)) * crop_scale
    )
    side = float(int(side_unrounded))
    if side <= 1:
        raise ValueError("SMIRK crop is too small")
    left = center_x - 0.5 * side
    top = center_y - 0.5 * side
    pixel_scale = (SMIRK_IMAGE_SIZE - 1) / side
    matrix = np.array(
        [
            [pixel_scale, 0.0, -left * pixel_scale],
            [0.0, pixel_scale, -top * pixel_scale],
        ],
        dtype=np.float32,
    )
    return {
        "bbox_xyxy": [x0, y0, x1, y1],
        "crop_scale": crop_scale,
        "crop_box_xyxy": [left, top, left + side, top + side],
        "crop_side_pixels": side,
        "crop_side_unrounded_pixels": float(side_unrounded),
        "network_size": SMIRK_IMAGE_SIZE,
        "image_shape": [int(image_height), int(image_width)],
        "warp_matrix": matrix,
    }


def project_smirk_vertices(
    vertices: np.ndarray,
    camera: np.ndarray,
    crop: dict,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    camera = np.asarray(camera, dtype=np.float64).reshape(-1)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("SMIRK vertices must have shape [V, 3]")
    if camera.shape != (3,) or not np.all(np.isfinite(camera)):
        raise ValueError("SMIRK camera must contain scale, tx, and ty")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("SMIRK vertices contain non-finite values")
    scale, tx, ty = (float(value) for value in camera)
    if scale <= 0:
        raise ValueError("SMIRK camera scale must be positive")
    x_ndc = scale * (vertices[:, 0] + tx)
    y_ndc = -scale * (vertices[:, 1] + ty)
    z_depth = -scale * vertices[:, 2]
    crop_pixels = 0.5 * (SMIRK_IMAGE_SIZE - 1)
    x_crop = (x_ndc + 1.0) * crop_pixels
    y_crop = (y_ndc + 1.0) * crop_pixels
    left, top, right, bottom = (
        float(value) for value in crop["crop_box_xyxy"]
    )
    side_x = right - left
    side_y = bottom - top
    x_full = left + x_crop * side_x / (SMIRK_IMAGE_SIZE - 1)
    y_full = top + y_crop * side_y / (SMIRK_IMAGE_SIZE - 1)
    return np.column_stack((x_full, y_full, z_depth)).astype(
        np.float32
    )


class SMIRKProvider:
    """Pinned SMIRK encoder and FLAME geometry path."""

    def __init__(
        self,
        provider_root: str | Path,
        checkpoint_path: str | Path,
        flame_model_path: str | Path,
        *,
        device: str = "cuda",
    ):
        import torch
        import timm

        self.provider_root = Path(provider_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.flame_model_path = Path(flame_model_path).resolve()
        self.device = str(device)
        self.preflight = smirk_preflight(
            self.provider_root,
            self.checkpoint_path,
            self.flame_model_path,
        )
        if not self.preflight["runnable"]:
            failed = [
                name
                for name, passed in self.preflight["checks"].items()
                if not passed
            ]
            raise RuntimeError("SMIRK preflight failed: " + ", ".join(failed))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SMIRK requested CUDA but CUDA is unavailable")

        self.compatibility = _install_legacy_compatibility()
        with _provider_import_context(self.provider_root):
            encoder_module = importlib.import_module("src.smirk_encoder")

            def create_backbone(name, pretrained=True):
                backbone = timm.create_model(
                    name,
                    pretrained=False,
                    features_only=True,
                )
                return backbone, backbone.feature_info[-1]["num_chs"]

            encoder_module.create_backbone = create_backbone
            flame_module = importlib.import_module("src.FLAME.FLAME")
            self.encoder = encoder_module.SmirkEncoder().to(self.device)
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            encoder_state = {
                key.replace("smirk_encoder.", ""): value
                for key, value in checkpoint.items()
                if "smirk_encoder" in key
            }
            self.encoder.load_state_dict(encoder_state, strict=True)
            self.encoder.eval()
            self.flame = flame_module.FLAME(
                flame_model_path=str(self.flame_model_path),
                flame_lmk_embedding_path=str(
                    self.provider_root / "assets/landmark_embedding.npy"
                ),
            ).to(self.device)
            self.flame.eval()
        self.faces = (
            self.flame.faces_tensor.detach().long().cpu().numpy()
        )
        with np.load(
            self.provider_root / SMIRK_MEDIAPIPE_EMBEDDING
        ) as embedding:
            self.mediapipe_landmark_indices = embedding[
                "landmark_indices"
            ].astype(np.int32)
        if self.mediapipe_landmark_indices.shape != (
            SMIRK_MEDIAPIPE_LANDMARK_COUNT,
        ):
            raise ValueError(
                "SMIRK MediaPipe embedding must contain exactly "
                f"{SMIRK_MEDIAPIPE_LANDMARK_COUNT} landmark indices"
            )
        self.calls: list[dict] = []
        self.allocated_before_inference = (
            int(torch.cuda.memory_allocated(self.device))
            if self.device.startswith("cuda")
            else 0
        )
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

    def _preprocess(
        self,
        image_rgb: np.ndarray,
        target_bbox_xyxy: list[float] | tuple[float, ...],
    ):
        import torch

        height, width = image_rgb.shape[:2]
        crop = _crop_transform(
            target_bbox_xyxy,
            image_width=width,
            image_height=height,
        )
        warped = cv2.warpAffine(
            image_rgb,
            crop["warp_matrix"],
            (SMIRK_IMAGE_SIZE, SMIRK_IMAGE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        tensor = (
            torch.from_numpy(np.ascontiguousarray(warped))
            .to(self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        crop["warp_matrix"] = crop["warp_matrix"].tolist()
        return tensor, crop

    def infer(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        target_bbox_xyxy: list[float] | tuple[float, ...],
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
            raise ValueError("SMIRK expects an RGB uint8 image")
        tensor, crop = self._preprocess(image_rgb, target_bbox_xyxy)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.encoder(tensor)
            flame_output = self.flame(outputs)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        canonical_vertices = (
            flame_output["vertices"][0].detach().float().cpu().numpy()
        )
        canonical_landmarks_mp = (
            flame_output["landmarks_mp"][0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        camera = outputs["cam"][0].detach().float().cpu().numpy()
        projected_vertices = project_smirk_vertices(
            canonical_vertices,
            camera,
            crop,
        )
        projected_landmarks_mp = project_smirk_vertices(
            canonical_landmarks_mp,
            camera,
            crop,
        )
        if len(projected_landmarks_mp) != len(
            self.mediapipe_landmark_indices
        ):
            raise ValueError(
                "SMIRK projected landmark count differs from its "
                "MediaPipe embedding"
            )
        peak_allocated = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.startswith("cuda")
            else 0
        )
        metadata = {
            "provider": "smirk",
            "runtime_seconds": float(elapsed),
            "peak_vram_gib": float(peak_allocated / (1024**3)),
            "incremental_peak_vram_gib": float(
                max(0, peak_allocated - self.allocated_before_inference)
                / (1024**3)
            ),
            "device": self.device,
            "crop": crop,
            "camera": [float(value) for value in camera],
            "pose_params": [
                float(value)
                for value in outputs["pose_params"][0]
                .detach()
                .float()
                .cpu()
                .numpy()
            ],
            "expression_l2": float(
                torch.linalg.vector_norm(outputs["expression_params"][0])
                .detach()
                .cpu()
            ),
            "jaw_params": [
                float(value)
                for value in outputs["jaw_params"][0]
                .detach()
                .float()
                .cpu()
                .numpy()
            ],
            "vertices": int(len(projected_vertices)),
            "faces": int(len(self.faces)),
            "mediapipe_landmarks": int(len(projected_landmarks_mp)),
        }
        self.calls.append(metadata)
        return {
            "vertices": projected_vertices,
            "canonical_vertices": canonical_vertices.astype(np.float32),
            "faces": self.faces.copy(),
            "projected_landmarks_mp": projected_landmarks_mp,
            "mediapipe_landmark_indices": (
                self.mediapipe_landmark_indices.copy()
            ),
            "metadata": metadata,
        }

    def provenance(self) -> dict:
        return {
            "provider": "smirk",
            "preflight": self.preflight,
            "compatibility": self.compatibility,
            "calls": self.calls,
        }
