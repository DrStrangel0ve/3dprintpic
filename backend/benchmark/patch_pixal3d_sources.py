from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from backend.benchmark.pixal3d_models import (
    DEFAULT_PIXAL3D_DINOV3_MODEL,
    DEFAULT_PIXAL3D_MOGE_MODEL,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def patch_pixal3d_sources(root: Path) -> dict:
    pipeline_path = root / "pixal3d" / "pipelines" / "pixal3d_image_to_3d.py"
    inference_path = root / "inference.py"
    if not pipeline_path.is_file() or not inference_path.is_file():
        raise FileNotFoundError(f"Expected official Pixal3D sources under {root}")

    pipeline_text = pipeline_path.read_text(encoding="utf-8")
    if "import os\n" not in pipeline_text:
        import_anchor = "from typing import *\n"
        if pipeline_text.count(import_anchor) != 1:
            raise RuntimeError("Pixal3D rembg import patch target not found exactly once")
        pipeline_text = pipeline_text.replace(import_anchor, import_anchor + "import os\n", 1)
    rembg_needle = (
        "        pipeline.rembg_model = "
        "getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])"
    )
    rembg_replacement = (
        "        rembg_args = dict(args['rembg_model']['args'])\n"
        "        rembg_model_override = os.environ.get('PIXAL3D_REMBG_MODEL')\n"
        "        if rembg_model_override:\n"
        "            rembg_args['model_name'] = rembg_model_override\n"
        "        pipeline.rembg_model = "
        "getattr(rembg, args['rembg_model']['name'])(**rembg_args)"
    )
    if rembg_replacement not in pipeline_text:
        if pipeline_text.count(rembg_needle) != 1:
            raise RuntimeError("Pixal3D rembg override patch target not found exactly once")
        pipeline_text = pipeline_text.replace(rembg_needle, rembg_replacement, 1)

    inference_text = inference_path.read_text(encoding="utf-8")
    moge_needle = f'MOGE_MODEL_NAME = "{DEFAULT_PIXAL3D_MOGE_MODEL}"'
    moge_replacement = (
        'MOGE_MODEL_NAME = os.environ.get('
        f'"PIXAL3D_MOGE_MODEL_PATH", "{DEFAULT_PIXAL3D_MOGE_MODEL}")'
    )
    if moge_replacement not in inference_text:
        if inference_text.count(moge_needle) != 1:
            raise RuntimeError("Pixal3D MoGe override patch target not found exactly once")
        inference_text = inference_text.replace(moge_needle, moge_replacement, 1)

    dino_needle = f'"model_name": "{DEFAULT_PIXAL3D_DINOV3_MODEL}"'
    dino_replacement = (
        '"model_name": os.environ.get('
        f'"PIXAL3D_DINOV3_MODEL_PATH", "{DEFAULT_PIXAL3D_DINOV3_MODEL}")'
    )
    if dino_replacement not in inference_text:
        if inference_text.count(dino_needle) != 4:
            raise RuntimeError("Pixal3D DINOv3 override patch target not found exactly four times")
        inference_text = inference_text.replace(dino_needle, dino_replacement)

    pipeline_path.write_text(pipeline_text, encoding="utf-8")
    inference_path.write_text(inference_text, encoding="utf-8")
    return {
        "pipeline": str(pipeline_path),
        "pipeline_sha256": sha256_text(pipeline_text),
        "inference": str(inference_path),
        "inference_sha256": sha256_text(inference_text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch pinned Pixal3D nested model paths into the official CLI.")
    parser.add_argument("--pixal3d-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(patch_pixal3d_sources(args.pixal3d_dir.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
