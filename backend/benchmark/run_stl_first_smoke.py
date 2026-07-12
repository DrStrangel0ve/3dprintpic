from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from backend.benchmark.stl_modes import (
    STL_MODE_DEPTH_RELIEF,
    STL_MODE_MULTIVIEW_MESH,
    STL_MODE_SINGLE_IMAGE_MESH,
    STL_MODE_SOURCE_MESH_ORACLE,
)
from backend.benchmark.ingest_stl_results import (
    DEPLOYABLE_STL_MODES,
    render_markdown as render_stl_ingest_markdown,
    summarize_run as summarize_stl_run,
)
from backend.benchmark.direct_mesh import direct_mesh_bbox_uses_hidden_source
from backend.benchmark.pixal3d_models import (
    DEFAULT_PIXAL3D_DINOV3_REVISION,
    DEFAULT_PIXAL3D_MODEL,
    DEFAULT_PIXAL3D_MODEL_REVISION,
    DEFAULT_PIXAL3D_MOGE_REVISION,
    DEFAULT_PIXAL3D_REMBG_MODEL,
    DEFAULT_PIXAL3D_REMBG_REVISION,
)
from backend.benchmark.triposg_models import (
    DEFAULT_TRIPOSG_MODEL_REVISION,
    DEFAULT_TRIPOSG_REMBG_REVISION,
)
from backend.benchmark.step1x3d_models import (
    DEFAULT_STEP1X3D_GUIDANCE,
    DEFAULT_STEP1X3D_MODEL,
    DEFAULT_STEP1X3D_MODEL_REVISION,
    DEFAULT_STEP1X3D_OCTREE_RESOLUTION,
    DEFAULT_STEP1X3D_SEED,
    DEFAULT_STEP1X3D_STEPS,
    DEFAULT_STEP1X3D_SUBFOLDER,
)


DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_TRIPOSR_PYTHON = "/content/triposr-venv/bin/python"
DEFAULT_TRIPOSR_DIR = "/content/TripoSR"
DEFAULT_TRIPOSG_PYTHON = "/content/triposg-venv/bin/python"
DEFAULT_TRIPOSG_DIR = "/content/TripoSG"
DEFAULT_PIXAL3D_PYTHON = "/content/pixal3d-venv/bin/python"
DEFAULT_PIXAL3D_DIR = "/content/Pixal3D"
DEFAULT_PIXAL3D_PROVIDER_CACHE_DIR = "/content/pixal3d-provider-cache"
DEFAULT_STEP1X3D_PYTHON = "/content/step1x3d-venv/bin/python"
DEFAULT_STEP1X3D_DIR = "/content/Step1X-3D"
DEFAULT_STEP1X3D_PROVIDER_CACHE_DIR = "/content/step1x3d-provider-cache"
SOURCE_MULTIVIEW_ORACLE_NAME = "source_mesh_bundle_multiview_oracle"
VISUAL_HULL_MULTIVIEW_NAME = "visual_hull_multiview_repaired_mesh"
MESH_TARGET_BBOX_PLACEHOLDERS = {
    "source": "{source_bbox_extents}",
    "mirror": "{mirror_bbox_extents}",
    "inferred": "{inferred_bbox_extents}",
    "reference": "{reference_bbox_extents}",
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def shell_token(value: str | Path) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(value)])
    return shlex.quote(str(value))


def image_to_mesh_command(
    *,
    python: str,
    provider: str,
    provider_dir: str | None,
    provider_device: str,
    timeout: int,
    output_mesh_repair: str = "none",
    output_mesh_raw: bool = False,
    raw_output_ext: str = "obj",
    mesh_target_max_dimension: float = 0.0,
    mesh_min_bbox_dimension: float = 0.0,
    mesh_max_bbox_aspect_ratio: float = 0.0,
    mesh_target_bbox_source: str = "none",
    mesh_target_faces: int = 0,
    mesh_max_normalized_face_density_log1p: float = 0.0,
    output_extra: list[str] | None = None,
) -> str:
    command = [
        shell_token(python),
        "-m",
        "backend.benchmark.run_image_to_mesh_provider",
        "--provider",
        provider,
        "--input-image",
        '"{input_image}"',
        "--output-mesh",
        '"{output_mesh}"',
        "--output-stl",
        '"{output_stl}"',
        "--timeout",
        str(timeout),
        "--provider-device",
        provider_device,
    ]
    if provider_dir:
        command.extend(["--provider-dir", shell_token(provider_dir)])
    if output_mesh_raw:
        raw_ext = raw_output_ext.lstrip(".") or "mesh"
        command.extend(["--raw-output-mesh", f'"{{output_dir}}/output_mesh_raw.{raw_ext}"'])
    if output_mesh_repair != "none":
        command.extend(["--mesh-repair", output_mesh_repair])
    if mesh_target_max_dimension > 0:
        command.extend(["--mesh-target-max-dimension", str(mesh_target_max_dimension)])
    if mesh_min_bbox_dimension > 0:
        command.extend(["--mesh-min-bbox-dimension", str(mesh_min_bbox_dimension)])
    if mesh_max_bbox_aspect_ratio > 0:
        command.extend(["--mesh-max-bbox-aspect-ratio", str(mesh_max_bbox_aspect_ratio)])
    bbox_placeholder = MESH_TARGET_BBOX_PLACEHOLDERS.get(mesh_target_bbox_source)
    if bbox_placeholder:
        command.extend(["--mesh-target-bbox-extents", f'"{bbox_placeholder}"'])
    if mesh_target_faces > 0:
        command.extend(["--mesh-target-faces", str(mesh_target_faces)])
    if mesh_max_normalized_face_density_log1p > 0:
        command.extend(
            [
                "--mesh-max-normalized-face-density-log1p",
                str(mesh_max_normalized_face_density_log1p),
            ]
        )
    command.extend(output_extra or [])
    return " ".join(command)


def direct_mesh_name_suffix(args: argparse.Namespace) -> str:
    bbox_source = getattr(args, "mesh_target_bbox_source", "none") or "none"
    if bbox_source == "none":
        return "_direct_mesh"
    return f"_stl_{bbox_source}_bbox_direct_mesh"


def direct_mesh_reference_fields(args: argparse.Namespace) -> dict:
    bbox_source = getattr(args, "mesh_target_bbox_source", "none") or "none"
    if bbox_source == "none":
        return {}
    reference_method = "mirror" if bbox_source == "inferred" else getattr(
        args,
        "direct_mesh_reference_method",
        "mirror",
    )
    return {
        "direct_mesh_bbox_source": bbox_source,
        "direct_mesh_reference_method": reference_method,
        "oracle_diagnostic": direct_mesh_bbox_uses_hidden_source(bbox_source, reference_method),
    }


def triposr_api_command(args: argparse.Namespace, repaired: bool) -> str:
    return image_to_mesh_command(
        python=args.triposr_python,
        provider="triposr-api",
        provider_dir=args.triposr_dir,
        provider_device=args.provider_device,
        timeout=args.direct_mesh_timeout,
        output_mesh_repair=args.mesh_repair if repaired else "none",
        output_mesh_raw=repaired,
        raw_output_ext="obj",
        mesh_target_max_dimension=getattr(args, "mesh_target_max_dimension", 0.0),
        mesh_min_bbox_dimension=getattr(args, "mesh_min_bbox_dimension", 0.0),
        mesh_max_bbox_aspect_ratio=getattr(args, "mesh_max_bbox_aspect_ratio", 0.0),
        mesh_target_bbox_source=getattr(args, "mesh_target_bbox_source", "none"),
        mesh_target_faces=getattr(args, "mesh_target_faces", 0),
        mesh_max_normalized_face_density_log1p=getattr(
            args,
            "mesh_max_normalized_face_density_log1p",
            0.0,
        ),
        output_extra=["--chunk-size", str(args.chunk_size), "--mc-resolution", str(args.mc_resolution)],
    )


def hunyuan3d_command(args: argparse.Namespace, repaired: bool) -> str:
    hunyuan_extra = [
        "--num-inference-steps",
        str(args.hunyuan_num_inference_steps),
        "--guidance-scale",
        str(args.hunyuan_guidance_scale),
        "--octree-resolution",
        str(args.hunyuan_octree_resolution),
        "--num-chunks",
        str(args.hunyuan_num_chunks),
        "--disable-progress",
    ]
    if args.hunyuan_low_vram:
        hunyuan_extra.append("--low-vram")
    return image_to_mesh_command(
        python=args.provider_python,
        provider="hunyuan3d-shape",
        provider_dir=args.hunyuan3d_dir,
        provider_device=args.provider_device,
        timeout=args.direct_mesh_timeout,
        output_mesh_repair=args.mesh_repair if repaired else "none",
        output_mesh_raw=repaired,
        raw_output_ext="glb",
        mesh_target_max_dimension=getattr(args, "mesh_target_max_dimension", 0.0),
        mesh_min_bbox_dimension=getattr(args, "mesh_min_bbox_dimension", 0.0),
        mesh_max_bbox_aspect_ratio=getattr(args, "mesh_max_bbox_aspect_ratio", 0.0),
        mesh_target_bbox_source=getattr(args, "mesh_target_bbox_source", "none"),
        mesh_target_faces=getattr(args, "mesh_target_faces", 0),
        mesh_max_normalized_face_density_log1p=getattr(
            args,
            "mesh_max_normalized_face_density_log1p",
            0.0,
        ),
        output_extra=hunyuan_extra,
    )


def triposg_command(args: argparse.Namespace, repaired: bool) -> str:
    triposg_extra = [
        "--triposg-model-revision",
        getattr(args, "triposg_model_revision", DEFAULT_TRIPOSG_MODEL_REVISION),
        "--triposg-rembg-revision",
        getattr(args, "triposg_rembg_revision", DEFAULT_TRIPOSG_REMBG_REVISION),
        "--num-inference-steps",
        str(args.triposg_num_inference_steps),
        "--guidance-scale",
        str(args.triposg_guidance_scale),
    ]
    if args.triposg_seed is not None:
        triposg_extra.extend(["--seed", str(args.triposg_seed)])
    triposg_extra.extend(
        [
            "--provider-mesh-cache-dir",
            getattr(
                args,
                "triposg_provider_cache_dir",
                "/content/triposg-provider-cache",
            ),
        ]
    )
    return image_to_mesh_command(
        python=args.triposg_python,
        provider="triposg",
        provider_dir=args.triposg_dir,
        provider_device=args.provider_device,
        timeout=args.direct_mesh_timeout,
        output_mesh_repair=args.mesh_repair if repaired else "none",
        output_mesh_raw=repaired,
        raw_output_ext="glb",
        mesh_target_max_dimension=getattr(args, "mesh_target_max_dimension", 0.0),
        mesh_min_bbox_dimension=getattr(args, "mesh_min_bbox_dimension", 0.0),
        mesh_max_bbox_aspect_ratio=getattr(args, "mesh_max_bbox_aspect_ratio", 0.0),
        mesh_target_bbox_source=getattr(args, "mesh_target_bbox_source", "none"),
        mesh_target_faces=getattr(args, "mesh_target_faces", 0),
        mesh_max_normalized_face_density_log1p=getattr(
            args,
            "mesh_max_normalized_face_density_log1p",
            0.0,
        ),
        output_extra=triposg_extra,
    )


def pixal3d_command(args: argparse.Namespace, repaired: bool) -> str:
    pixal3d_extra = [
        "--pixal3d-resolution",
        str(args.pixal3d_resolution),
        "--seed",
        str(args.pixal3d_seed),
        "--pixal3d-model-path",
        args.pixal3d_model_path,
        "--pixal3d-model-revision",
        args.pixal3d_model_revision,
        "--pixal3d-moge-revision",
        args.pixal3d_moge_revision,
        "--pixal3d-dinov3-revision",
        args.pixal3d_dinov3_revision,
        "--pixal3d-rembg-model",
        args.pixal3d_rembg_model,
        "--pixal3d-rembg-revision",
        args.pixal3d_rembg_revision,
        "--provider-mesh-cache-dir",
        args.pixal3d_provider_cache_dir,
    ]
    if args.pixal3d_fov is not None:
        pixal3d_extra.extend(["--pixal3d-fov", str(args.pixal3d_fov)])
    if args.pixal3d_low_vram:
        pixal3d_extra.append("--low-vram")
    return image_to_mesh_command(
        python=args.pixal3d_python,
        provider="pixal3d",
        provider_dir=args.pixal3d_dir,
        provider_device=args.provider_device,
        timeout=args.direct_mesh_timeout,
        output_mesh_repair=args.mesh_repair if repaired else "none",
        output_mesh_raw=repaired,
        raw_output_ext="glb",
        mesh_target_max_dimension=getattr(args, "mesh_target_max_dimension", 0.0) if repaired else 0.0,
        mesh_min_bbox_dimension=getattr(args, "mesh_min_bbox_dimension", 0.0) if repaired else 0.0,
        mesh_max_bbox_aspect_ratio=getattr(args, "mesh_max_bbox_aspect_ratio", 0.0) if repaired else 0.0,
        mesh_target_bbox_source=getattr(args, "mesh_target_bbox_source", "none") if repaired else "none",
        mesh_target_faces=getattr(args, "mesh_target_faces", 0) if repaired else 0,
        mesh_max_normalized_face_density_log1p=(
            getattr(args, "mesh_max_normalized_face_density_log1p", 0.0) if repaired else 0.0
        ),
        output_extra=pixal3d_extra,
    )


def step1x3d_command(args: argparse.Namespace, repaired: bool) -> str:
    step1x3d_extra = [
        "--step1x3d-model-path",
        args.step1x3d_model_path,
        "--step1x3d-model-revision",
        args.step1x3d_model_revision,
        "--step1x3d-subfolder",
        args.step1x3d_subfolder,
        "--step1x3d-max-faces",
        str(args.step1x3d_max_faces),
        "--num-inference-steps",
        str(args.step1x3d_num_inference_steps),
        "--guidance-scale",
        str(args.step1x3d_guidance_scale),
        "--octree-resolution",
        str(args.step1x3d_octree_resolution),
        "--seed",
        str(args.step1x3d_seed),
        "--provider-mesh-cache-dir",
        args.step1x3d_provider_cache_dir,
    ]
    return image_to_mesh_command(
        python=args.step1x3d_python,
        provider="step1x3d",
        provider_dir=args.step1x3d_dir,
        provider_device=args.provider_device,
        timeout=args.direct_mesh_timeout,
        output_mesh_repair=args.mesh_repair if repaired else "none",
        output_mesh_raw=repaired,
        raw_output_ext="glb",
        mesh_target_max_dimension=(
            getattr(args, "mesh_target_max_dimension", 0.0) if repaired else 0.0
        ),
        mesh_min_bbox_dimension=(
            getattr(args, "mesh_min_bbox_dimension", 0.0) if repaired else 0.0
        ),
        mesh_max_bbox_aspect_ratio=(
            getattr(args, "mesh_max_bbox_aspect_ratio", 0.0) if repaired else 0.0
        ),
        mesh_target_bbox_source=(
            getattr(args, "mesh_target_bbox_source", "none") if repaired else "none"
        ),
        mesh_target_faces=(
            getattr(args, "mesh_target_faces", 0) if repaired else 0
        ),
        mesh_max_normalized_face_density_log1p=(
            getattr(args, "mesh_max_normalized_face_density_log1p", 0.0)
            if repaired
            else 0.0
        ),
        output_extra=step1x3d_extra,
    )


def source_multiview_oracle_command(args: argparse.Namespace) -> str:
    return image_to_mesh_command(
        python=args.provider_python,
        provider="source-mesh-bundle-oracle",
        provider_dir=None,
        provider_device=args.provider_device,
        timeout=args.direct_mesh_timeout,
        output_mesh_repair=args.mesh_repair,
        output_mesh_raw=True,
        raw_output_ext="ply",
        mesh_target_max_dimension=getattr(args, "mesh_target_max_dimension", 0.0),
        mesh_min_bbox_dimension=getattr(args, "mesh_min_bbox_dimension", 0.0),
        mesh_max_bbox_aspect_ratio=getattr(args, "mesh_max_bbox_aspect_ratio", 0.0),
        mesh_target_bbox_source=getattr(args, "mesh_target_bbox_source", "none"),
        mesh_target_faces=getattr(args, "mesh_target_faces", 0),
        mesh_max_normalized_face_density_log1p=getattr(
            args,
            "mesh_max_normalized_face_density_log1p",
            0.0,
        ),
        output_extra=["--input-bundle", '"{input_bundle}"'],
    )


def visual_hull_multiview_command(args: argparse.Namespace) -> str:
    return image_to_mesh_command(
        python=args.provider_python,
        provider="multiview-visual-hull",
        provider_dir=None,
        provider_device=args.provider_device,
        timeout=args.direct_mesh_timeout,
        output_mesh_repair=args.mesh_repair,
        output_mesh_raw=True,
        raw_output_ext="ply",
        mesh_target_max_dimension=getattr(args, "mesh_target_max_dimension", 0.0),
        mesh_min_bbox_dimension=getattr(args, "mesh_min_bbox_dimension", 0.0),
        mesh_max_bbox_aspect_ratio=getattr(args, "mesh_max_bbox_aspect_ratio", 0.0),
        mesh_target_bbox_source=getattr(args, "mesh_target_bbox_source", "none"),
        mesh_target_faces=getattr(args, "mesh_target_faces", 0),
        mesh_max_normalized_face_density_log1p=getattr(
            args,
            "mesh_max_normalized_face_density_log1p",
            0.0,
        ),
        output_extra=[
            "--input-bundle",
            '"{input_bundle}"',
            "--visual-hull-resolution",
            str(args.visual_hull_resolution),
            "--visual-hull-grid-extent",
            str(args.visual_hull_grid_extent),
            "--visual-hull-ortho-scale",
            str(args.visual_hull_ortho_scale),
            "--visual-hull-mask-dilate",
            str(args.visual_hull_mask_dilate),
        ],
    )


def build_experiments(args: argparse.Namespace) -> list[dict]:
    direct_suffix = direct_mesh_name_suffix(args)
    reference_fields = direct_mesh_reference_fields(args)
    experiments = [
        {"name": "masked", "method": "masked", "stl_mode": STL_MODE_DEPTH_RELIEF},
        {"name": "mirror", "method": "mirror", "stl_mode": STL_MODE_DEPTH_RELIEF},
        {"name": "biharmonic", "method": "biharmonic", "stl_mode": STL_MODE_DEPTH_RELIEF},
    ]
    if args.include_source_oracle:
        experiments.append(
            {
                "name": "source_mesh_oracle",
                "method": "source-mesh-oracle",
                "stl_mode": STL_MODE_SOURCE_MESH_ORACLE,
                "skip_depth": True,
                "emit_stl": True,
                "direct_mesh_input": "full",
                "direct_mesh_output_ext": "ply",
                "source_mesh_repair": args.mesh_repair,
            }
        )
    if args.include_triposr_api:
        if args.include_raw_direct_mesh:
            experiments.append(
                {
                    "name": f"triposr_api_masked{direct_suffix}",
                    "method": "external-image-to-mesh",
                    "stl_mode": STL_MODE_SINGLE_IMAGE_MESH,
                    "skip_depth": True,
                    "emit_stl": True,
                    "direct_mesh_input": "masked",
                    "direct_mesh_output_ext": "obj",
                    "direct_mesh_timeout": args.direct_mesh_timeout,
                    "direct_mesh_command": triposr_api_command(args, repaired=False),
                    **reference_fields,
                }
            )
        for direct_input in args.triposr_direct_inputs:
            input_suffix = "masked" if direct_input == "masked" else f"{direct_input}_prefill"
            experiments.append(
                {
                    "name": f"triposr_api_{input_suffix}_repaired{direct_suffix}",
                    "method": "external-image-to-mesh",
                    "stl_mode": STL_MODE_SINGLE_IMAGE_MESH,
                    "skip_depth": True,
                    "emit_stl": True,
                    "direct_mesh_input": direct_input,
                    "direct_mesh_output_ext": "obj",
                    "direct_mesh_timeout": args.direct_mesh_timeout,
                    "direct_mesh_command": triposr_api_command(args, repaired=True),
                    **reference_fields,
                }
            )
    if args.include_hunyuan3d_shape:
        experiments.append(
            {
                "name": f"hunyuan3d_shape_masked_repaired{direct_suffix}",
                "method": "external-image-to-mesh",
                "stl_mode": STL_MODE_SINGLE_IMAGE_MESH,
                "skip_depth": True,
                "emit_stl": True,
                "direct_mesh_input": "masked",
                "direct_mesh_output_ext": "glb",
                "direct_mesh_timeout": args.direct_mesh_timeout,
                "direct_mesh_command": hunyuan3d_command(args, repaired=True),
                **reference_fields,
            }
        )
    if getattr(args, "include_triposg", False):
        for direct_input in args.triposg_direct_inputs:
            input_suffix = "masked" if direct_input == "masked" else f"{direct_input}_prefill"
            experiments.append(
                {
                    "name": f"triposg_{input_suffix}_repaired{direct_suffix}",
                    "method": "external-image-to-mesh",
                    "stl_mode": STL_MODE_SINGLE_IMAGE_MESH,
                    "skip_depth": True,
                    "emit_stl": True,
                    "direct_mesh_input": direct_input,
                    "direct_mesh_output_ext": "glb",
                    "direct_mesh_timeout": args.direct_mesh_timeout,
                    "direct_mesh_command": triposg_command(args, repaired=True),
                    **reference_fields,
                }
            )
    if getattr(args, "include_pixal3d", False):
        for direct_input in args.pixal3d_direct_inputs:
            input_suffix = "masked" if direct_input == "masked" else f"{direct_input}_prefill"
            if args.pixal3d_include_raw:
                experiments.append(
                    {
                        "name": f"pixal3d_{input_suffix}_raw_direct_mesh",
                        "method": "external-image-to-mesh",
                        "stl_mode": STL_MODE_SINGLE_IMAGE_MESH,
                        "skip_depth": True,
                        "emit_stl": True,
                        "direct_mesh_input": direct_input,
                        "direct_mesh_output_ext": "glb",
                        "direct_mesh_timeout": args.direct_mesh_timeout,
                        "direct_mesh_command": pixal3d_command(args, repaired=False),
                    }
                )
            experiments.append(
                {
                    "name": f"pixal3d_{input_suffix}_repaired{direct_suffix}",
                    "method": "external-image-to-mesh",
                    "stl_mode": STL_MODE_SINGLE_IMAGE_MESH,
                    "skip_depth": True,
                    "emit_stl": True,
                    "direct_mesh_input": direct_input,
                    "direct_mesh_output_ext": "glb",
                    "direct_mesh_timeout": args.direct_mesh_timeout,
                    "direct_mesh_command": pixal3d_command(args, repaired=True),
                    **reference_fields,
                }
            )
    if getattr(args, "include_step1x3d", False):
        for direct_input in args.step1x3d_direct_inputs:
            input_suffix = "masked" if direct_input == "masked" else f"{direct_input}_prefill"
            if args.step1x3d_include_raw:
                experiments.append(
                    {
                        "name": f"step1x3d_{input_suffix}_raw_direct_mesh",
                        "method": "external-image-to-mesh",
                        "stl_mode": STL_MODE_SINGLE_IMAGE_MESH,
                        "skip_depth": True,
                        "emit_stl": True,
                        "direct_mesh_input": direct_input,
                        "direct_mesh_output_ext": "glb",
                        "direct_mesh_timeout": args.direct_mesh_timeout,
                        "direct_mesh_command": step1x3d_command(args, repaired=False),
                    }
                )
            experiments.append(
                {
                    "name": f"step1x3d_{input_suffix}_repaired{direct_suffix}",
                    "method": "external-image-to-mesh",
                    "stl_mode": STL_MODE_SINGLE_IMAGE_MESH,
                    "skip_depth": True,
                    "emit_stl": True,
                    "direct_mesh_input": direct_input,
                    "direct_mesh_output_ext": "glb",
                    "direct_mesh_timeout": args.direct_mesh_timeout,
                    "direct_mesh_command": step1x3d_command(args, repaired=True),
                    **reference_fields,
                }
            )
    if getattr(args, "include_source_multiview_oracle", False):
        experiments.append(
            {
                "name": SOURCE_MULTIVIEW_ORACLE_NAME,
                "method": "external-multiview-to-mesh",
                "stl_mode": STL_MODE_MULTIVIEW_MESH,
                "skip_depth": True,
                "emit_stl": True,
                "direct_mesh_input": args.multiview_primary_input,
                "direct_mesh_output_ext": "ply",
                "direct_mesh_timeout": args.direct_mesh_timeout,
                "direct_mesh_command": source_multiview_oracle_command(args),
                **reference_fields,
            }
        )
    if getattr(args, "include_visual_hull_multiview", False):
        experiments.append(
            {
                "name": VISUAL_HULL_MULTIVIEW_NAME,
                "method": "external-multiview-to-mesh",
                "stl_mode": STL_MODE_MULTIVIEW_MESH,
                "skip_depth": True,
                "emit_stl": True,
                "direct_mesh_input": args.multiview_primary_input,
                "direct_mesh_output_ext": "ply",
                "direct_mesh_timeout": args.direct_mesh_timeout,
                "direct_mesh_command": visual_hull_multiview_command(args),
                **reference_fields,
            }
        )
    if args.multiview_command:
        experiments.append(
            {
                "name": args.multiview_name,
                "method": "external-multiview-to-mesh",
                "stl_mode": STL_MODE_MULTIVIEW_MESH,
                "skip_depth": True,
                "emit_stl": True,
                "direct_mesh_input": args.multiview_primary_input,
                "direct_mesh_output_ext": args.multiview_output_ext,
                "direct_mesh_timeout": args.direct_mesh_timeout,
                "direct_mesh_command": args.multiview_command,
                **reference_fields,
            }
        )
    return experiments


def rows_from_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def _intish(value) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def method_failure_rows(rows: list[dict]) -> list[dict]:
    failures = []
    for row in rows:
        if (
            _intish(row.get("error_count")) > 0
            or _intish(row.get("logged_failure_count")) > 0
            or row.get("last_error_type")
            or row.get("last_error")
        ):
            failures.append(
                {
                    "method": row.get("method", ""),
                    "error_count": row.get("error_count", ""),
                    "logged_failure_count": row.get("logged_failure_count", ""),
                    "last_error_type": row.get("last_error_type", ""),
                    "last_error": row.get("last_error", ""),
                }
            )
    return failures


def run_command(command: list[str], cwd: Path, timeout: int) -> dict:
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    output = proc.stdout or ""
    return {
        "command": command,
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 2),
        "tail": output[-9000:],
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_config_only(args: argparse.Namespace) -> dict:
    config_path = Path(args.write_config).resolve()
    experiments = build_experiments(args)
    write_json(config_path, experiments)
    return {
        "config": str(config_path),
        "experiment_count": len(experiments),
        "experiments": [experiment["name"] for experiment in experiments],
    }


def write_architecture_report(experiment_dir: Path, output_dir: Path, label: str = "stl_first_smoke") -> dict:
    if not (experiment_dir / "aggregate_summary.csv").exists() and not (
        experiment_dir / "summary_metrics.csv"
    ).exists():
        return {}

    run = summarize_stl_run(
        label,
        experiment_dir,
        score_profile="stl-quality",
        score_mode="baseline-delta",
        baseline_method="masked",
        top=20,
    )
    report = {
        "generated_at": utc_now(),
        "inputs": [
            {
                "label": label,
                "source": str(experiment_dir),
                "root": str(experiment_dir),
                "extracted_to": "",
            }
        ],
        "run_count": 1,
        "score_profile": "stl-quality",
        "score_mode": "baseline-delta",
        "baseline_method": "masked",
        "deployable_stl_modes": list(DEPLOYABLE_STL_MODES),
        "runs": [run],
    }
    report_path = output_dir / "stl_first_architecture_report.json"
    markdown_path = output_dir / "stl_first_architecture_report.md"
    write_json(report_path, report)
    markdown_path.write_text(render_stl_ingest_markdown(report) + "\n", encoding="utf-8")
    return {
        "report_json": str(report_path),
        "report_markdown": str(markdown_path),
        "deployable_winner": run.get("deployable_winner", {}),
        "promotion_eligible_winner": run.get("promotion_eligible_winner", {}),
        "oracle_diagnostic_winner": run.get("oracle_diagnostic_winner", {}),
        "best_by_stl_mode": run.get("best_by_stl_mode", []),
        "architecture_replacement_decision": run.get("architecture_replacement_decision", {}),
        "ranked_methods": run.get("ranked_methods", []),
    }


def ensure_manifest(args: argparse.Namespace, repo_dir: Path, output_dir: Path) -> tuple[Path, dict | None]:
    if args.manifest:
        return Path(args.manifest), None
    dataset_dir = output_dir / "dataset"
    command = [
        sys.executable,
        "-m",
        "backend.benchmark.generate_rendered_dataset",
        "--output-dir",
        str(dataset_dir),
        "--source",
        args.dataset_source,
        "--count",
        str(args.dataset_count),
        "--size",
        str(args.size),
        "--seed",
        str(args.seed),
        "--views-per-asset",
        str(args.views_per_asset),
        "--mesh-sample-strategy",
        args.mesh_sample_strategy,
    ]
    if args.asset_root:
        command.extend(["--asset-root", args.asset_root])
    if args.asset_glob:
        command.extend(["--asset-glob", args.asset_glob])
    if args.continue_on_error:
        command.append("--continue-on-error")
    result = run_command(command, repo_dir, timeout=args.dataset_timeout)
    return dataset_dir / "manifest.jsonl", result


def build_optimize_command(args: argparse.Namespace, manifest_path: Path, experiment_dir: Path, config_path: Path, experiments: list[dict]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "backend.benchmark.optimize_completion",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(experiment_dir),
        "--config",
        str(config_path),
        "--start-index",
        str(args.start_index),
        "--limit",
        str(args.limit),
        "--depth-provider",
        args.depth_provider,
        "--depth-model",
        args.depth_model,
        "--device",
        args.device,
        "--emit-stl",
        "--stl-target-dimension",
        str(args.stl_target_dimension),
        "--score-mode",
        "baseline-delta",
        "--score-profile",
        "stl-quality",
        "--baseline-method",
        "masked",
        "--contact-sheet",
        "--contact-sheet-methods",
        ",".join(experiment["name"] for experiment in experiments),
        "--contact-sheet-max-samples",
        str(args.contact_sheet_max_samples),
    ]
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.resume:
        command.append("--resume")
    if getattr(args, "select_candidate", False):
        command.append("--select-candidate")
        if args.candidate_method:
            command.extend(["--candidate-method", args.candidate_method])
        if args.current_method:
            command.extend(["--current-method", args.current_method])
        command.extend(["--min-paired-n", str(args.min_paired_n)])
        command.extend(["--min-win-rate", str(args.min_win_rate)])
        command.extend(["--min-ci95-low", str(args.min_ci95_low)])
        command.extend(["--min-score-margin", str(args.min_score_margin)])
        command.extend(["--min-stl-watertight", str(args.min_stl_watertight)])
        command.extend(["--min-stl-is-volume", str(args.min_stl_is_volume)])
        command.extend(["--min-stl-is-manifold", str(args.min_stl_is_manifold)])
        command.extend(["--min-stl-winding-consistent", str(args.min_stl_winding_consistent)])
        command.extend(["--min-stl-positive-volume", str(args.min_stl_positive_volume)])
        command.extend(["--min-stl-single-component", str(args.min_stl_single_component)])
        command.extend(["--min-stl-bbox-has-volume", str(args.min_stl_bbox_has_volume)])
        command.extend(
            [
                "--max-stl-nonmanifold-edge-count-log1p",
                str(args.max_stl_nonmanifold_edge_count_log1p),
            ]
        )
        command.extend(["--max-stl-degenerate-face-ratio", str(args.max_stl_degenerate_face_ratio)])
        command.extend(["--max-stl-component-excess-log1p", str(args.max_stl_component_excess_log1p)])
        command.extend(["--max-stl-bbox-aspect-ratio", str(args.max_stl_bbox_aspect_ratio)])
        command.extend(
            [
                "--max-stl-faces-per-bbox-volume-log1p",
                str(args.max_stl_faces_per_bbox_volume_log1p),
            ]
        )
        max_source_chamfer_ratio = getattr(args, "max_mesh_surface_chamfer_ratio_vs_current", None)
        if max_source_chamfer_ratio is not None:
            command.extend(
                [
                    "--max-mesh-surface-chamfer-ratio-vs-current",
                    str(max_source_chamfer_ratio),
                ]
            )
        max_source_h95_ratio = getattr(args, "max_mesh_surface_hausdorff95_ratio_vs_current", None)
        if max_source_h95_ratio is not None:
            command.extend(
                [
                    "--max-mesh-surface-hausdorff95-ratio-vs-current",
                    str(max_source_h95_ratio),
                ]
            )
        command.extend(
            [
                "--max-heldout-view-silhouette-iou-degradation-ratio",
                str(getattr(args, "max_heldout_view_silhouette_iou_degradation_ratio", 1.1)),
            ]
        )
        if args.allow_missing_split_audit:
            command.append("--allow-missing-split-audit")
    return command


def run_smoke(args: argparse.Namespace) -> dict:
    repo_dir = Path(args.repo_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    experiment_dir = output_dir / "experiment"
    config_path = output_dir / "stl_first_reconstruction_config.json"
    experiments = build_experiments(args)
    write_json(config_path, experiments)
    manifest_path, dataset_result = ensure_manifest(args, repo_dir, output_dir)
    summary: dict = {
        "marker": "STL_FIRST_RECONSTRUCTION_SMOKE",
        "started_at": utc_now(),
        "repo_dir": str(repo_dir),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "config": str(config_path),
        "experiments": [experiment["name"] for experiment in experiments],
    }
    if dataset_result is not None:
        summary["dataset"] = dataset_result
    optimize_command = build_optimize_command(args, manifest_path, experiment_dir, config_path, experiments)
    summary["benchmark"] = run_command(optimize_command, repo_dir, timeout=args.benchmark_timeout)
    summary["aggregate_summary"] = rows_from_csv(experiment_dir / "aggregate_summary.csv")
    summary["method_failures"] = method_failure_rows(summary["aggregate_summary"])
    summary["ranked_experiments"] = rows_from_csv(experiment_dir / "ranked_experiments.csv")
    architecture_report = write_architecture_report(experiment_dir, output_dir)
    if architecture_report:
        summary["architecture_report"] = architecture_report
        summary["deployable_winner"] = architecture_report.get("deployable_winner", {})
        summary["promotion_eligible_winner"] = architecture_report.get("promotion_eligible_winner", {})
        summary["oracle_diagnostic_winner"] = architecture_report.get("oracle_diagnostic_winner", {})
        summary["best_by_stl_mode"] = architecture_report.get("best_by_stl_mode", [])
        summary["architecture_replacement_decision"] = architecture_report.get(
            "architecture_replacement_decision", {}
        )
    summary["selection_decision"] = read_json(experiment_dir / "selection_decision.json")
    if (experiment_dir / "selection_decision.md").exists():
        summary["selection_decision_md"] = str(experiment_dir / "selection_decision.md")
    summary["finished_at"] = utc_now()
    write_json(output_dir / "stl_first_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an STL-first reconstruction smoke: depth-relief baselines plus opt-in single-image "
            "and multiview mesh providers, all ranked by the stl-quality objective."
        )
    )
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--output-dir", default="backend/output/completion-benchmark/experiments/stl_first_reconstruction_smoke")
    parser.add_argument(
        "--write-config",
        default=None,
        help="Write the generated STL-first experiment config to this path and exit before dataset/benchmark work.",
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dataset-source", choices=("procedural", "mesh-dir"), default="procedural")
    parser.add_argument("--dataset-count", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--asset-root", default=None)
    parser.add_argument("--asset-glob", default="**/*.glb")
    parser.add_argument("--views-per-asset", type=int, default=2)
    parser.add_argument("--mesh-sample-strategy", choices=("random", "balanced"), default="balanced")
    parser.add_argument("--include-source-oracle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-source-multiview-oracle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-visual-hull-multiview", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-triposr-api", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-hunyuan3d-shape", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-raw-direct-mesh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--triposr-direct-input",
        action="append",
        choices=("masked", "full", "mirror", "biharmonic"),
        dest="triposr_direct_inputs",
        default=None,
        help=(
            "Direct image input mode for repaired TripoSR API candidates. Repeat to compare "
            "masked, full, mirror-prefill, and biharmonic-prefill variants."
        ),
    )
    parser.add_argument("--provider-python", default=sys.executable)
    parser.add_argument("--provider-device", default="cuda")
    parser.add_argument("--triposr-python", default=DEFAULT_TRIPOSR_PYTHON)
    parser.add_argument("--triposr-dir", default=DEFAULT_TRIPOSR_DIR)
    parser.add_argument("--hunyuan3d-dir", default=None)
    parser.add_argument("--hunyuan-num-inference-steps", type=int, default=30)
    parser.add_argument("--hunyuan-guidance-scale", type=float, default=5.0)
    parser.add_argument("--hunyuan-octree-resolution", type=int, default=256)
    parser.add_argument("--hunyuan-num-chunks", type=int, default=8000)
    parser.add_argument("--hunyuan-low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-triposg", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--triposg-direct-input",
        action="append",
        choices=("masked", "full", "mirror", "biharmonic"),
        dest="triposg_direct_inputs",
        default=None,
        help=(
            "Direct image input mode for repaired TripoSG candidates. Repeat to compare "
            "masked, full, mirror-prefill, and biharmonic-prefill variants."
        ),
    )
    parser.add_argument("--triposg-python", default=DEFAULT_TRIPOSG_PYTHON)
    parser.add_argument("--triposg-dir", default=DEFAULT_TRIPOSG_DIR)
    parser.add_argument("--triposg-num-inference-steps", type=int, default=50)
    parser.add_argument("--triposg-guidance-scale", type=float, default=7.0)
    parser.add_argument("--triposg-seed", type=int, default=42)
    parser.add_argument(
        "--triposg-provider-cache-dir",
        default="/content/triposg-provider-cache",
        help="Content-addressed raw mesh cache used to make seeded TripoSG comparisons resumable.",
    )
    parser.add_argument("--triposg-model-revision", default=DEFAULT_TRIPOSG_MODEL_REVISION)
    parser.add_argument("--triposg-rembg-revision", default=DEFAULT_TRIPOSG_REMBG_REVISION)
    parser.add_argument("--include-pixal3d", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--pixal3d-direct-input",
        action="append",
        choices=("masked", "full", "mirror", "biharmonic"),
        dest="pixal3d_direct_inputs",
        default=None,
        help=(
            "Direct image input mode for Pixal3D candidates. Repeat to compare masked, full, "
            "mirror-prefill, and biharmonic-prefill variants."
        ),
    )
    parser.add_argument("--pixal3d-python", default=DEFAULT_PIXAL3D_PYTHON)
    parser.add_argument("--pixal3d-dir", default=DEFAULT_PIXAL3D_DIR)
    parser.add_argument("--pixal3d-resolution", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--pixal3d-seed", type=int, default=42)
    parser.add_argument("--pixal3d-fov", type=float, default=None)
    parser.add_argument("--pixal3d-model-path", default=DEFAULT_PIXAL3D_MODEL)
    parser.add_argument("--pixal3d-model-revision", default=DEFAULT_PIXAL3D_MODEL_REVISION)
    parser.add_argument("--pixal3d-moge-revision", default=DEFAULT_PIXAL3D_MOGE_REVISION)
    parser.add_argument("--pixal3d-dinov3-revision", default=DEFAULT_PIXAL3D_DINOV3_REVISION)
    parser.add_argument("--pixal3d-rembg-model", default=DEFAULT_PIXAL3D_REMBG_MODEL)
    parser.add_argument("--pixal3d-rembg-revision", default=DEFAULT_PIXAL3D_REMBG_REVISION)
    parser.add_argument("--pixal3d-provider-cache-dir", default=DEFAULT_PIXAL3D_PROVIDER_CACHE_DIR)
    parser.add_argument("--pixal3d-low-vram", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pixal3d-include-raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-step1x3d", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--step1x3d-direct-input",
        action="append",
        choices=("masked", "full", "mirror", "biharmonic"),
        dest="step1x3d_direct_inputs",
        default=None,
        help="Direct image input mode for raw and repaired Step1X-3D geometry candidates.",
    )
    parser.add_argument("--step1x3d-python", default=DEFAULT_STEP1X3D_PYTHON)
    parser.add_argument("--step1x3d-dir", default=DEFAULT_STEP1X3D_DIR)
    parser.add_argument("--step1x3d-model-path", default=DEFAULT_STEP1X3D_MODEL)
    parser.add_argument("--step1x3d-model-revision", default=DEFAULT_STEP1X3D_MODEL_REVISION)
    parser.add_argument("--step1x3d-subfolder", default=DEFAULT_STEP1X3D_SUBFOLDER)
    parser.add_argument(
        "--step1x3d-num-inference-steps",
        type=int,
        default=DEFAULT_STEP1X3D_STEPS,
    )
    parser.add_argument(
        "--step1x3d-guidance-scale",
        type=float,
        default=DEFAULT_STEP1X3D_GUIDANCE,
    )
    parser.add_argument(
        "--step1x3d-octree-resolution",
        type=int,
        default=DEFAULT_STEP1X3D_OCTREE_RESOLUTION,
    )
    parser.add_argument("--step1x3d-max-faces", type=int, default=0)
    parser.add_argument("--step1x3d-seed", type=int, default=DEFAULT_STEP1X3D_SEED)
    parser.add_argument(
        "--step1x3d-provider-cache-dir",
        default=DEFAULT_STEP1X3D_PROVIDER_CACHE_DIR,
    )
    parser.add_argument(
        "--step1x3d-include-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--mesh-repair", choices=("basic", "convex-hull", "printable"), default="printable")
    parser.add_argument(
        "--mesh-target-max-dimension",
        type=float,
        default=0.0,
        help="If positive, scale direct provider meshes to this STL-space max bbox side before export.",
    )
    parser.add_argument(
        "--mesh-min-bbox-dimension",
        type=float,
        default=0.0,
        help="If positive, thicken direct provider mesh bbox axes below this STL-space dimension.",
    )
    parser.add_argument(
        "--mesh-max-bbox-aspect-ratio",
        type=float,
        default=0.0,
        help="If positive, thicken direct provider mesh bbox axes until max_axis/min_axis is below this ratio.",
    )
    parser.add_argument(
        "--mesh-target-bbox-source",
        choices=("none", "source", "mirror", "inferred", "reference"),
        default="none",
        help=(
            "Optionally scale direct provider meshes to bbox extents from the source mesh, the mirror "
            "depth-relief baseline, its deployable inferred alias, or --direct-mesh-reference-method. "
            "The source option is a hidden-geometry oracle and is never promotion eligible."
        ),
    )
    parser.add_argument(
        "--mesh-target-faces",
        type=int,
        default=0,
        help="If positive, attempt Trimesh quadric decimation on provider meshes before export.",
    )
    parser.add_argument(
        "--mesh-max-normalized-face-density-log1p",
        type=float,
        default=0.0,
        help=(
            "If positive, adapt each provider mesh face cap to the scale-free STL complexity metric after "
            "bbox calibration."
        ),
    )
    parser.add_argument("--direct-mesh-timeout", type=int, default=3600)
    parser.add_argument(
        "--direct-mesh-reference-method",
        default="mirror",
        help="Reference method used for {reference_bbox_extents} when --mesh-target-bbox-source=reference.",
    )
    parser.add_argument("--multiview-command", default=None)
    parser.add_argument("--multiview-name", default="external_multiview_reconstruction")
    parser.add_argument("--multiview-primary-input", choices=("masked", "full", "mirror", "biharmonic"), default="masked")
    parser.add_argument("--multiview-output-ext", default="glb")
    parser.add_argument("--visual-hull-resolution", type=int, default=32)
    parser.add_argument("--visual-hull-grid-extent", type=float, default=1.9)
    parser.add_argument("--visual-hull-ortho-scale", type=float, default=2.0)
    parser.add_argument("--visual-hull-mask-dilate", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--depth-provider", default="depth-anything-v2")
    parser.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stl-target-dimension", type=int, default=96)
    parser.add_argument("--contact-sheet-max-samples", type=int, default=2)
    parser.add_argument("--select-candidate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidate-method", default=None)
    parser.add_argument("--current-method", default="mirror")
    parser.add_argument("--min-paired-n", type=int, default=5)
    parser.add_argument("--min-win-rate", type=float, default=0.8)
    parser.add_argument("--min-ci95-low", type=float, default=0.0)
    parser.add_argument("--min-score-margin", type=float, default=0.0)
    parser.add_argument("--min-stl-watertight", type=float, default=1.0)
    parser.add_argument("--min-stl-is-volume", type=float, default=1.0)
    parser.add_argument("--min-stl-is-manifold", type=float, default=1.0)
    parser.add_argument("--min-stl-winding-consistent", type=float, default=1.0)
    parser.add_argument("--min-stl-positive-volume", type=float, default=1.0)
    parser.add_argument("--min-stl-single-component", type=float, default=1.0)
    parser.add_argument("--min-stl-bbox-has-volume", type=float, default=1.0)
    parser.add_argument("--max-stl-nonmanifold-edge-count-log1p", type=float, default=0.0)
    parser.add_argument("--max-stl-degenerate-face-ratio", type=float, default=0.0)
    parser.add_argument("--max-stl-component-excess-log1p", type=float, default=0.0)
    parser.add_argument("--max-stl-bbox-aspect-ratio", type=float, default=10.0)
    parser.add_argument("--max-stl-faces-per-bbox-volume-log1p", type=float, default=10.0)
    parser.add_argument("--max-mesh-surface-chamfer-ratio-vs-current", type=float, default=1.1)
    parser.add_argument("--max-mesh-surface-hausdorff95-ratio-vs-current", type=float, default=1.1)
    parser.add_argument("--max-heldout-view-silhouette-iou-degradation-ratio", type=float, default=1.1)
    parser.add_argument("--allow-missing-split-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-timeout", type=int, default=600)
    parser.add_argument("--benchmark-timeout", type=int, default=7200)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.triposr_direct_inputs = args.triposr_direct_inputs or ["masked"]
    args.triposg_direct_inputs = args.triposg_direct_inputs or ["masked"]
    args.pixal3d_direct_inputs = args.pixal3d_direct_inputs or ["biharmonic"]
    args.step1x3d_direct_inputs = args.step1x3d_direct_inputs or ["biharmonic"]
    return args


def main() -> None:
    args = parse_args()
    if args.write_config:
        print(json.dumps(write_config_only(args), indent=2))
        return
    summary = run_smoke(args)
    print(json.dumps(summary, indent=2))
    if (
        summary.get("dataset", {}).get("returncode")
        or summary.get("benchmark", {}).get("returncode")
        or summary.get("method_failures")
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
