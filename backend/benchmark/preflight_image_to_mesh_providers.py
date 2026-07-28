from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from backend.benchmark.run_image_to_mesh_provider import (
    CLI_PROVIDERS,
    HUNYUAN3D_2MV_PROVIDER,
    HUNYUAN3D_SHAPE_PROVIDER,
    MULTIVIEW_VISUAL_HULL_PROVIDER,
    PROVIDERS,
    SOURCE_MESH_BUNDLE_ORACLE_PROVIDER,
    STEP1X3D_PROVIDER,
    TRELLIS2_PROVIDER,
    TRIPOSR_API_PROVIDER,
    provider_dir_config_key,
    provider_git_revision,
    resolve_provider_dir,
)
from backend.benchmark.hunyuan3d_2mv_models import (
    DEFAULT_HUNYUAN3D_2MV_MODEL,
    DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION,
    DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION,
    DEFAULT_HUNYUAN3D_2MV_SUBFOLDER,
    hunyuan3d_2mv_model_specs,
)
from backend.benchmark.pixal3d_models import (
    DEFAULT_PIXAL3D_MODEL,
    DEFAULT_PIXAL3D_REMBG_MODEL,
    pixal3d_model_specs,
)
from backend.benchmark.triposg_models import triposg_model_specs
from backend.benchmark.step1x3d_models import (
    DEFAULT_STEP1X3D_MODEL,
    DEFAULT_STEP1X3D_MODEL_REVISION,
    DEFAULT_STEP1X3D_SOURCE_REVISION,
    DEFAULT_STEP1X3D_SUBFOLDER,
    step1x3d_model_specs,
    verify_step1x3d_source_integrity,
)
from backend.benchmark.trellis2_models import (
    DEFAULT_TRELLIS2_ATTENTION_BACKEND,
    DEFAULT_TRELLIS2_MODEL,
    DEFAULT_TRELLIS2_MODEL_REVISION,
    DEFAULT_TRELLIS2_RESOLUTION,
    DEFAULT_TRELLIS2_SOURCE_REVISION,
    TRELLIS2_RESOLUTIONS,
    trellis2_model_specs,
)


BUILTIN_PROVIDERS = {SOURCE_MESH_BUNDLE_ORACLE_PROVIDER, MULTIVIEW_VISUAL_HULL_PROVIDER}
TRELLIS2_ATTENTION_BACKENDS = ("xformers", "flash_attn", "flash_attn_3")
TRELLIS2_ATTENTION_MODULES = {
    "xformers": "xformers.ops",
    "flash_attn": "flash_attn",
    "flash_attn_3": "flash_attn_interface",
}
TRELLIS2_PROBE_JSON_PREFIX = "TRELLIS2_PREFLIGHT_JSON="
HUNYUAN3D_2MV_PROBE_JSON_PREFIX = "HUNYUAN3D_2MV_PREFLIGHT_JSON="
STEP1X3D_PROBE_JSON_PREFIX = "STEP1X3D_PREFLIGHT_JSON="


def command_exists(executable: str | None) -> bool:
    text = str(executable or "").strip()
    if not text:
        return False
    path = Path(text)
    has_path_separator = any(separator and separator in text for separator in (os.sep, os.altsep))
    if path.is_absolute() or has_path_separator:
        return path.exists()
    return shutil.which(text) is not None


def safe_find_spec(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def build_trellis2_python_probe_script() -> str:
    return (
        "import importlib, json, os, sys\n"
        f"accepted = {TRELLIS2_ATTENTION_BACKENDS!r}\n"
        f"module_by_backend = {TRELLIS2_ATTENTION_MODULES!r}\n"
        "requested = os.environ.get('SPARSE_ATTN_BACKEND') or os.environ.get('ATTN_BACKEND') or ''\n"
        "payload = {\n"
        "    'python_executable': sys.executable,\n"
        "    'pipelines_importable': False,\n"
        "    'pipeline_class_importable': False,\n"
        "    'attention_backend_requested': requested,\n"
        "    'attention_backend': '',\n"
        "    'attention_backend_accepted': False,\n"
        "    'attention_backend_module': '',\n"
        "    'attention_backend_importable': False,\n"
        "    'attention_backend_api_ready': False,\n"
        "    'backend_ready': False,\n"
        "    'error': '',\n"
        "}\n"
        "try:\n"
        "    pipelines = importlib.import_module('trellis2.pipelines')\n"
        "    payload['pipelines_importable'] = True\n"
        "    getattr(pipelines, 'Trellis2ImageTo3DPipeline')\n"
        "    payload['pipeline_class_importable'] = True\n"
        "    sparse_config = importlib.import_module('trellis2.modules.sparse.config')\n"
        "    active = str(sparse_config.ATTN)\n"
        "    payload['attention_backend'] = active\n"
        "    requested_accepted = not requested or requested in accepted\n"
        "    requested_matches = not requested or requested == active\n"
        "    payload['attention_backend_accepted'] = active in accepted and requested_accepted and requested_matches\n"
        "    backend_module = module_by_backend.get(active, '')\n"
        "    payload['attention_backend_module'] = backend_module\n"
        "    if backend_module:\n"
        "        backend_api = importlib.import_module(backend_module)\n"
        "        payload['attention_backend_importable'] = True\n"
        "        if active == 'xformers':\n"
        "            getattr(backend_api, 'memory_efficient_attention')\n"
        "            getattr(backend_api.fmha.BlockDiagonalMask, 'from_seqlens')\n"
        "        elif active == 'flash_attn':\n"
        "            getattr(backend_api, 'flash_attn_varlen_qkvpacked_func')\n"
        "            getattr(backend_api, 'flash_attn_varlen_kvpacked_func')\n"
        "            getattr(backend_api, 'flash_attn_varlen_func')\n"
        "        elif active == 'flash_attn_3':\n"
        "            getattr(backend_api, 'flash_attn_varlen_func')\n"
        "        payload['attention_backend_api_ready'] = True\n"
        "except Exception as exc:\n"
        "    payload['error'] = f'{type(exc).__name__}: {exc}'\n"
        "payload['backend_ready'] = bool(\n"
        "    payload['pipelines_importable']\n"
        "    and payload['pipeline_class_importable']\n"
        "    and payload['attention_backend_accepted']\n"
        "    and payload['attention_backend_importable']\n"
        "    and payload['attention_backend_api_ready']\n"
        ")\n"
        f"print({TRELLIS2_PROBE_JSON_PREFIX!r} + json.dumps(payload, sort_keys=True))\n"
        "raise SystemExit(0 if payload['backend_ready'] else 1)\n"
    )


def probe_trellis2_provider_python(
    provider_python: str,
    provider_dir: Path,
    *,
    attention_backend: str = DEFAULT_TRELLIS2_ATTENTION_BACKEND,
    timeout_seconds: int = 60,
) -> dict:
    env = os.environ.copy()
    env["ATTN_BACKEND"] = attention_backend
    env["SPARSE_ATTN_BACKEND"] = attention_backend
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(provider_dir), current_pythonpath) if part
    )
    command = [provider_python, "-c", build_trellis2_python_probe_script()]
    try:
        completed = subprocess.run(
            command,
            cwd=provider_dir,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "python_executable": provider_python,
            "returncode": None,
            "pipelines_importable": False,
            "pipeline_class_importable": False,
            "attention_backend_requested": (
                env.get("SPARSE_ATTN_BACKEND") or env.get("ATTN_BACKEND") or ""
            ),
            "attention_backend": "",
            "attention_backend_accepted": False,
            "attention_backend_module": "",
            "attention_backend_importable": False,
            "attention_backend_api_ready": False,
            "backend_ready": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    payload = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(TRELLIS2_PROBE_JSON_PREFIX):
            try:
                payload = json.loads(line[len(TRELLIS2_PROBE_JSON_PREFIX) :])
            except json.JSONDecodeError:
                payload = None
            break
    if not isinstance(payload, dict):
        payload = {
            "python_executable": provider_python,
            "pipelines_importable": False,
            "pipeline_class_importable": False,
            "attention_backend_requested": (
                env.get("SPARSE_ATTN_BACKEND") or env.get("ATTN_BACKEND") or ""
            ),
            "attention_backend": "",
            "attention_backend_accepted": False,
            "attention_backend_module": "",
            "attention_backend_importable": False,
            "attention_backend_api_ready": False,
            "backend_ready": False,
            "error": "Provider Python did not emit a TRELLIS.2 preflight payload.",
        }
    payload["returncode"] = completed.returncode
    if completed.returncode and not payload.get("error"):
        stderr = completed.stderr.strip()
        payload["error"] = stderr[-2000:] or f"Provider Python exited with {completed.returncode}."
    payload["backend_ready"] = bool(
        payload.get("backend_ready") and completed.returncode == 0
    )
    return payload


def build_hunyuan3d_2mv_python_probe_script() -> str:
    return (
        "import importlib, json, sys\n"
        "payload = {\n"
        "    'python_executable': sys.executable,\n"
        "    'shapegen_importable': False,\n"
        "    'pipeline_class_importable': False,\n"
        "    'backend_ready': False,\n"
        "    'error': '',\n"
        "}\n"
        "try:\n"
        "    shapegen = importlib.import_module('hy3dgen.shapegen')\n"
        "    payload['shapegen_importable'] = True\n"
        "    getattr(shapegen, 'Hunyuan3DDiTFlowMatchingPipeline')\n"
        "    payload['pipeline_class_importable'] = True\n"
        "except Exception as exc:\n"
        "    payload['error'] = f'{type(exc).__name__}: {exc}'\n"
        "payload['backend_ready'] = bool(\n"
        "    payload['shapegen_importable'] and payload['pipeline_class_importable']\n"
        ")\n"
        f"print({HUNYUAN3D_2MV_PROBE_JSON_PREFIX!r} + json.dumps(payload, sort_keys=True))\n"
        "raise SystemExit(0 if payload['backend_ready'] else 1)\n"
    )


def probe_hunyuan3d_2mv_provider_python(
    provider_python: str,
    provider_dir: Path,
    *,
    timeout_seconds: int = 60,
) -> dict:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(provider_dir), current_pythonpath) if part
    )
    try:
        completed = subprocess.run(
            [provider_python, "-c", build_hunyuan3d_2mv_python_probe_script()],
            cwd=provider_dir,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "python_executable": provider_python,
            "returncode": None,
            "shapegen_importable": False,
            "pipeline_class_importable": False,
            "backend_ready": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        if not line.startswith(HUNYUAN3D_2MV_PROBE_JSON_PREFIX):
            continue
        try:
            payload = json.loads(line[len(HUNYUAN3D_2MV_PROBE_JSON_PREFIX) :])
        except json.JSONDecodeError:
            payload = None
        break
    if not isinstance(payload, dict):
        payload = {
            "python_executable": provider_python,
            "shapegen_importable": False,
            "pipeline_class_importable": False,
            "backend_ready": False,
            "error": "Provider Python did not emit a Hunyuan3D-2mv preflight payload.",
        }
    payload["returncode"] = completed.returncode
    if completed.returncode and not payload.get("error"):
        stderr = completed.stderr.strip()
        payload["error"] = stderr[-2000:] or f"Provider Python exited with {completed.returncode}."
    payload["backend_ready"] = bool(
        payload.get("backend_ready") and completed.returncode == 0
    )
    return payload


def build_step1x3d_python_probe_script() -> str:
    return (
        "import json, os, sys\n"
        "payload = {\n"
        "    'python_executable': sys.executable,\n"
        "    'pipeline_module_importable': False,\n"
        "    'pipeline_class_importable': False,\n"
        "    'torch_cuda_available': False,\n"
        "    'torch_cuda_capability': [],\n"
        "    'backend_ready': False,\n"
        "    'error': '',\n"
        "}\n"
        "try:\n"
        "    os.environ['USE_SAGEATTN'] = '0'\n"
        "    import torch\n"
        "    payload['torch_cuda_available'] = bool(torch.cuda.is_available())\n"
        "    if torch.cuda.is_available():\n"
        "        payload['torch_cuda_capability'] = list(torch.cuda.get_device_capability())\n"
        "    from step1x3d_geometry.models.pipelines import pipeline as pipeline_module\n"
        "    payload['pipeline_module_importable'] = True\n"
        "    getattr(pipeline_module, 'Step1X3DGeometryPipeline')\n"
        "    payload['pipeline_class_importable'] = True\n"
        "except Exception as exc:\n"
        "    payload['error'] = f'{type(exc).__name__}: {exc}'\n"
        "payload['backend_ready'] = bool(\n"
        "    payload['pipeline_module_importable']\n"
        "    and payload['pipeline_class_importable']\n"
        "    and payload['torch_cuda_available']\n"
        ")\n"
        f"print({STEP1X3D_PROBE_JSON_PREFIX!r} + json.dumps(payload, sort_keys=True))\n"
        "raise SystemExit(0 if payload['backend_ready'] else 1)\n"
    )


def probe_step1x3d_provider_python(
    provider_python: str,
    provider_dir: Path,
    *,
    timeout_seconds: int = 60,
) -> dict:
    env = os.environ.copy()
    env["USE_SAGEATTN"] = "0"
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(provider_dir), current_pythonpath) if part
    )
    try:
        completed = subprocess.run(
            [provider_python, "-c", build_step1x3d_python_probe_script()],
            cwd=provider_dir,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "python_executable": provider_python,
            "returncode": None,
            "pipeline_module_importable": False,
            "pipeline_class_importable": False,
            "torch_cuda_available": False,
            "torch_cuda_capability": [],
            "backend_ready": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        if not line.startswith(STEP1X3D_PROBE_JSON_PREFIX):
            continue
        try:
            payload = json.loads(line[len(STEP1X3D_PROBE_JSON_PREFIX) :])
        except json.JSONDecodeError:
            payload = None
        break
    if not isinstance(payload, dict):
        payload = {
            "python_executable": provider_python,
            "pipeline_module_importable": False,
            "pipeline_class_importable": False,
            "torch_cuda_available": False,
            "torch_cuda_capability": [],
            "backend_ready": False,
            "error": "Provider Python did not emit a Step1X-3D preflight payload.",
        }
    payload["returncode"] = completed.returncode
    if completed.returncode and not payload.get("error"):
        stderr = completed.stderr.strip()
        payload["error"] = stderr[-2000:] or f"Provider Python exited with {completed.returncode}."
    payload["backend_ready"] = bool(
        payload.get("backend_ready") and completed.returncode == 0
    )
    return payload


def flag_value(tokens: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return shlex.split(command, posix=False)


def parse_provider_command(command: str) -> dict | None:
    tokens = split_command(command)
    if "backend.benchmark.run_image_to_mesh_provider" not in tokens:
        return None
    provider = flag_value(tokens, "--provider")
    if provider not in PROVIDERS:
        return None
    parsed = {
        "command": command,
        "tokens": tokens,
        "provider": provider,
        "provider_dir": flag_value(tokens, "--provider-dir"),
        "provider_python": flag_value(tokens, "--provider-python"),
        "input_bundle": flag_value(tokens, "--input-bundle"),
        "wrapper_python": tokens[0] if tokens else "",
    }
    if provider == HUNYUAN3D_2MV_PROVIDER:
        specs = hunyuan3d_2mv_model_specs(
            model_repo=(
                flag_value(tokens, "--hunyuan3d-2mv-model-path")
                or DEFAULT_HUNYUAN3D_2MV_MODEL
            ),
            model_revision=(
                flag_value(tokens, "--hunyuan3d-2mv-model-revision")
                or DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION
            ),
            subfolder=(
                flag_value(tokens, "--hunyuan3d-2mv-subfolder")
                or DEFAULT_HUNYUAN3D_2MV_SUBFOLDER
            ),
        )
        parsed["provider_models"] = {
            name: {
                "repo_id": str(spec["repo_id"]),
                "revision": str(spec["revision"]),
                "subfolder": str(spec["subfolder"]),
            }
            for name, spec in specs.items()
        }
        parsed["provider_source_revision"] = DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION
    elif provider == "pixal3d":
        specs = pixal3d_model_specs(
            model_repo=flag_value(tokens, "--pixal3d-model-path") or DEFAULT_PIXAL3D_MODEL,
            model_revision=flag_value(tokens, "--pixal3d-model-revision") or "",
            moge_revision=flag_value(tokens, "--pixal3d-moge-revision") or "",
            dinov3_revision=flag_value(tokens, "--pixal3d-dinov3-revision") or "",
            rembg_repo=flag_value(tokens, "--pixal3d-rembg-model") or DEFAULT_PIXAL3D_REMBG_MODEL,
            rembg_revision=flag_value(tokens, "--pixal3d-rembg-revision") or "",
        )
        parsed["provider_models"] = {
            name: {"repo_id": spec["repo_id"], "revision": spec["revision"]}
            for name, spec in specs.items()
        }
    elif provider == "triposg":
        parsed["provider_models"] = triposg_model_specs(
            model_revision=flag_value(tokens, "--triposg-model-revision") or "",
            rembg_revision=flag_value(tokens, "--triposg-rembg-revision") or "",
        )
    elif provider == STEP1X3D_PROVIDER:
        specs = step1x3d_model_specs(
            model_repo=(
                flag_value(tokens, "--step1x3d-model-path")
                or DEFAULT_STEP1X3D_MODEL
            ),
            model_revision=(
                flag_value(tokens, "--step1x3d-model-revision")
                or DEFAULT_STEP1X3D_MODEL_REVISION
            ),
            subfolder=(
                flag_value(tokens, "--step1x3d-subfolder")
                or DEFAULT_STEP1X3D_SUBFOLDER
            ),
        )
        parsed["provider_models"] = {
            name: {
                "repo_id": str(spec["repo_id"]),
                "revision": str(spec["revision"]),
                "subfolder": str(spec["subfolder"]),
            }
            for name, spec in specs.items()
        }
        parsed["provider_source_revision"] = DEFAULT_STEP1X3D_SOURCE_REVISION
    elif provider == TRELLIS2_PROVIDER:
        specs = trellis2_model_specs(
            model_repo=(
                flag_value(tokens, "--trellis2-model-path")
                or DEFAULT_TRELLIS2_MODEL
            ),
            model_revision=(
                flag_value(tokens, "--trellis2-model-revision")
                or DEFAULT_TRELLIS2_MODEL_REVISION
            ),
        )
        parsed["provider_models"] = {
            name: {"repo_id": spec["repo_id"], "revision": spec["revision"]}
            for name, spec in specs.items()
        }
        parsed["provider_source_revision"] = DEFAULT_TRELLIS2_SOURCE_REVISION
        parsed["trellis2_resolution"] = (
            flag_value(tokens, "--trellis2-resolution")
            or str(DEFAULT_TRELLIS2_RESOLUTION)
        )
    return parsed


def provider_dir_env_names(provider: str) -> list[str]:
    prefix = provider.upper().replace("-", "_")
    names = [f"{prefix}_DIR"]
    config = CLI_PROVIDERS.get(provider_dir_config_key(provider), {})
    canonical = config.get("env")
    if canonical and canonical not in names:
        names.append(str(canonical))
    if provider == HUNYUAN3D_SHAPE_PROVIDER and "HUNYUAN3D_DIR" not in names:
        names.append("HUNYUAN3D_DIR")
    return names


def provider_entrypoint(provider: str) -> Path | None:
    if provider in CLI_PROVIDERS:
        runner = CLI_PROVIDERS[provider].get("runner")
        if runner == "hunyuan3d-2mv-wrapper":
            return Path("hy3dgen/shapegen/pipelines.py")
        if runner == "triposg-module":
            return Path("scripts/inference_triposg.py")
        if runner == "pixal3d-inference":
            return Path("inference.py")
        if runner == "trellis2-wrapper":
            return Path("trellis2/pipelines/trellis2_image_to_3d.py")
        if runner == "step1x3d-wrapper":
            return Path("step1x3d_geometry/models/pipelines/pipeline.py")
        return Path("run.py")
    if provider == TRIPOSR_API_PROVIDER:
        return Path("tsr/system.py")
    return None


def provider_preflight_row(parsed: dict, experiment_names: list[str] | None = None) -> dict:
    provider = parsed["provider"]
    wrapper_python = parsed.get("wrapper_python") or ""
    provider_python = parsed.get("provider_python") or wrapper_python
    setup_errors: list[str] = []
    checks: dict[str, object] = {}

    wrapper_python_found = command_exists(wrapper_python)
    provider_python_found = command_exists(provider_python)
    checks["wrapper_python_found"] = wrapper_python_found
    checks["provider_python_found"] = provider_python_found
    if not wrapper_python_found:
        setup_errors.append("Wrapper Python executable is unavailable.")
    if not provider_python_found:
        setup_errors.append("Provider Python executable is unavailable.")

    if provider == HUNYUAN3D_2MV_PROVIDER:
        model_spec = (parsed.get("provider_models") or {}).get("hunyuan3d_2mv") or {}
        model_pinned = (
            model_spec.get("repo_id") == DEFAULT_HUNYUAN3D_2MV_MODEL
            and model_spec.get("revision") == DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION
            and model_spec.get("subfolder") == DEFAULT_HUNYUAN3D_2MV_SUBFOLDER
        )
        checks["model_revision_pinned"] = model_pinned
        checks["model_revisions_complete"] = bool(model_spec.get("revision"))
        checks["model_revisions_pinned"] = model_pinned
        checks["input_bundle_configured"] = bool(parsed.get("input_bundle"))
        if not model_pinned:
            setup_errors.append(
                "Hunyuan3D-2mv requires model "
                f"{DEFAULT_HUNYUAN3D_2MV_MODEL}@{DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION} "
                f"subfolder {DEFAULT_HUNYUAN3D_2MV_SUBFOLDER}."
            )
        if not parsed.get("input_bundle"):
            setup_errors.append("Hunyuan3D-2mv requires --input-bundle.")
    elif provider == "pixal3d":
        revisions = [
            str(spec.get("revision") or "").strip()
            for spec in (parsed.get("provider_models") or {}).values()
        ]
        revisions_present = sum(bool(revision) for revision in revisions)
        revisions_complete = revisions_present in {0, 4}
        checks["model_revisions_complete"] = revisions_complete
        checks["model_revisions_pinned"] = revisions_present == 4
        if not revisions_complete:
            setup_errors.append("Pixal3D model revisions must be supplied together for all four snapshots.")
    elif provider == "triposg":
        revisions = [
            str(spec.get("revision") or "").strip()
            for spec in (parsed.get("provider_models") or {}).values()
        ]
        revisions_present = sum(bool(revision) for revision in revisions)
        revisions_complete = revisions_present in {0, 2}
        checks["model_revisions_complete"] = revisions_complete
        checks["model_revisions_pinned"] = revisions_present == 2
        if not revisions_complete:
            setup_errors.append("TripoSG model revisions must be supplied together for both snapshots.")
    elif provider == STEP1X3D_PROVIDER:
        model_spec = (parsed.get("provider_models") or {}).get("step1x3d") or {}
        model_pinned = (
            model_spec.get("repo_id") == DEFAULT_STEP1X3D_MODEL
            and model_spec.get("revision") == DEFAULT_STEP1X3D_MODEL_REVISION
            and model_spec.get("subfolder") == DEFAULT_STEP1X3D_SUBFOLDER
        )
        checks["model_revision_pinned"] = model_pinned
        checks["model_revisions_complete"] = bool(model_spec.get("revision"))
        checks["model_revisions_pinned"] = model_pinned
        if not model_pinned:
            setup_errors.append(
                "Step1X-3D requires model "
                f"{DEFAULT_STEP1X3D_MODEL}@{DEFAULT_STEP1X3D_MODEL_REVISION} "
                f"subfolder {DEFAULT_STEP1X3D_SUBFOLDER}."
            )
    elif provider == TRELLIS2_PROVIDER:
        model_spec = (parsed.get("provider_models") or {}).get("trellis2") or {}
        model_pinned = (
            model_spec.get("repo_id") == DEFAULT_TRELLIS2_MODEL
            and model_spec.get("revision") == DEFAULT_TRELLIS2_MODEL_REVISION
        )
        checks["model_revision_pinned"] = model_pinned
        checks["model_revisions_complete"] = bool(model_spec.get("revision"))
        checks["model_revisions_pinned"] = model_pinned
        if not model_pinned:
            setup_errors.append(
                "TRELLIS.2 requires model "
                f"{DEFAULT_TRELLIS2_MODEL}@{DEFAULT_TRELLIS2_MODEL_REVISION}."
            )
        try:
            resolution = int(parsed.get("trellis2_resolution"))
        except (TypeError, ValueError):
            resolution = None
        resolution_supported = resolution in TRELLIS2_RESOLUTIONS
        checks["trellis2_resolution"] = resolution
        checks["trellis2_resolution_supported"] = resolution_supported
        if not resolution_supported:
            supported = ", ".join(str(value) for value in TRELLIS2_RESOLUTIONS)
            setup_errors.append(f"TRELLIS.2 resolution must be one of: {supported}.")

    provider_dir = None
    provider_dir_resolved = False
    if provider in BUILTIN_PROVIDERS:
        provider_dir_resolved = True
    elif provider in CLI_PROVIDERS or provider == TRIPOSR_API_PROVIDER:
        try:
            provider_dir = resolve_provider_dir(provider, parsed.get("provider_dir"))
            provider_dir_resolved = True
        except FileNotFoundError:
            setup_errors.append(f"Provider repo is missing. Configure one of: {', '.join(provider_dir_env_names(provider))}.")
        entrypoint = provider_entrypoint(provider)
        if provider_dir and entrypoint:
            entrypoint_found = (provider_dir / entrypoint).exists()
            checks["entrypoint"] = entrypoint.as_posix()
            checks["entrypoint_found"] = entrypoint_found
            if not entrypoint_found:
                setup_errors.append(f"Provider repo is missing expected entrypoint: {entrypoint.as_posix()}.")
        if provider == HUNYUAN3D_2MV_PROVIDER and provider_dir:
            source_revision = provider_git_revision(provider_dir)
            source_revision_pinned = (
                source_revision == DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION
            )
            checks["provider_source_revision"] = source_revision
            checks["provider_source_revision_expected"] = (
                DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION
            )
            checks["provider_source_revision_pinned"] = source_revision_pinned
            if not source_revision_pinned:
                actual = source_revision or "<unknown>"
                setup_errors.append(
                    "Hunyuan3D-2mv provider source must be checked out at "
                    f"{DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION}; found {actual}."
                )
            package_source = provider_dir / "hy3dgen" / "shapegen" / "__init__.py"
            package_source_found = package_source.is_file()
            checks["hunyuan3d_2mv_package_source_found"] = package_source_found
            if not package_source_found:
                setup_errors.append(
                    "Hunyuan3D-2mv provider repo is missing hy3dgen/shapegen/__init__.py."
                )
            if package_source_found and provider_python_found:
                python_probe = probe_hunyuan3d_2mv_provider_python(
                    provider_python,
                    provider_dir,
                )
                checks["hunyuan3d_2mv_python_probe"] = python_probe
                checks["hunyuan3d_2mv_shapegen_importable"] = bool(
                    python_probe.get("shapegen_importable")
                )
                checks["hunyuan3d_2mv_pipeline_class_importable"] = bool(
                    python_probe.get("pipeline_class_importable")
                )
                checks["hunyuan3d_2mv_backend_ready"] = bool(
                    python_probe.get("backend_ready")
                )
                if not python_probe.get("backend_ready"):
                    detail = python_probe.get("error") or "unknown provider Python error"
                    setup_errors.append(
                        f"Hunyuan3D-2mv provider Python preflight failed: {detail}"
                    )
        if provider == STEP1X3D_PROVIDER and provider_dir:
            source_revision = provider_git_revision(provider_dir)
            source_revision_pinned = (
                source_revision == DEFAULT_STEP1X3D_SOURCE_REVISION
            )
            checks["provider_source_revision"] = source_revision
            checks["provider_source_revision_expected"] = (
                DEFAULT_STEP1X3D_SOURCE_REVISION
            )
            checks["provider_source_revision_pinned"] = source_revision_pinned
            if not source_revision_pinned:
                actual = source_revision or "<unknown>"
                setup_errors.append(
                    "Step1X-3D provider source must be checked out at "
                    f"{DEFAULT_STEP1X3D_SOURCE_REVISION}; found {actual}."
                )
            try:
                source_integrity = verify_step1x3d_source_integrity(provider_dir)
            except (FileNotFoundError, OSError, subprocess.CalledProcessError, ValueError) as exc:
                source_integrity = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                setup_errors.append(
                    f"Step1X-3D provider source integrity failed: {exc}"
                )
            else:
                source_integrity = {"ok": True, **source_integrity}
            checks["step1x3d_source_integrity"] = source_integrity
            checks["step1x3d_source_integrity_ok"] = bool(
                source_integrity.get("ok")
            )
            package_source = provider_dir / "step1x3d_geometry" / "__init__.py"
            package_source_found = package_source.is_file()
            checks["step1x3d_package_source_found"] = package_source_found
            if not package_source_found:
                setup_errors.append(
                    "Step1X-3D provider repo is missing step1x3d_geometry/__init__.py."
                )
            if package_source_found and provider_python_found:
                python_probe = probe_step1x3d_provider_python(
                    provider_python,
                    provider_dir,
                )
                checks["step1x3d_python_probe"] = python_probe
                checks["step1x3d_pipeline_module_importable"] = bool(
                    python_probe.get("pipeline_module_importable")
                )
                checks["step1x3d_pipeline_class_importable"] = bool(
                    python_probe.get("pipeline_class_importable")
                )
                checks["step1x3d_torch_cuda_available"] = bool(
                    python_probe.get("torch_cuda_available")
                )
                checks["step1x3d_torch_cuda_capability"] = (
                    python_probe.get("torch_cuda_capability") or []
                )
                checks["step1x3d_backend_ready"] = bool(
                    python_probe.get("backend_ready")
                )
                if not python_probe.get("backend_ready"):
                    detail = python_probe.get("error") or "unknown provider Python error"
                    setup_errors.append(
                        f"Step1X-3D provider Python preflight failed: {detail}"
                    )
        if provider == TRELLIS2_PROVIDER and provider_dir:
            source_revision = provider_git_revision(provider_dir)
            source_revision_pinned = (
                source_revision == DEFAULT_TRELLIS2_SOURCE_REVISION
            )
            checks["provider_source_revision"] = source_revision
            checks["provider_source_revision_expected"] = (
                DEFAULT_TRELLIS2_SOURCE_REVISION
            )
            checks["provider_source_revision_pinned"] = source_revision_pinned
            if not source_revision_pinned:
                actual = source_revision or "<unknown>"
                setup_errors.append(
                    "TRELLIS.2 provider source must be checked out at "
                    f"{DEFAULT_TRELLIS2_SOURCE_REVISION}; found {actual}."
                )
            package_source = provider_dir / "trellis2" / "__init__.py"
            package_source_found = package_source.is_file()
            checks["trellis2_package_source_found"] = package_source_found
            if (provider_dir / ".git").exists() and not package_source_found:
                setup_errors.append(
                    "TRELLIS.2 provider repo is missing trellis2/__init__.py."
                )
            if package_source_found and provider_python_found:
                python_probe = probe_trellis2_provider_python(
                    provider_python,
                    provider_dir,
                )
                checks["trellis2_python_probe"] = python_probe
                checks["trellis2_pipelines_importable"] = bool(
                    python_probe.get("pipelines_importable")
                )
                checks["trellis2_pipeline_class_importable"] = bool(
                    python_probe.get("pipeline_class_importable")
                )
                checks["trellis2_attention_backend_requested"] = (
                    python_probe.get("attention_backend_requested") or ""
                )
                checks["trellis2_attention_backend"] = (
                    python_probe.get("attention_backend") or ""
                )
                checks["trellis2_attention_backend_accepted"] = bool(
                    python_probe.get("attention_backend_accepted")
                )
                checks["trellis2_attention_backend_module"] = (
                    python_probe.get("attention_backend_module") or ""
                )
                checks["trellis2_attention_backend_importable"] = bool(
                    python_probe.get("attention_backend_importable")
                )
                checks["trellis2_attention_backend_api_ready"] = bool(
                    python_probe.get(
                        "attention_backend_api_ready",
                        python_probe.get("attention_backend_importable"),
                    )
                )
                checks["trellis2_attention_backend_ready"] = bool(
                    python_probe.get("backend_ready")
                )
                if not python_probe.get("pipelines_importable"):
                    setup_errors.append(
                        "TRELLIS.2 provider Python cannot import trellis2.pipelines."
                    )
                elif not python_probe.get("pipeline_class_importable"):
                    setup_errors.append(
                        "TRELLIS.2 provider Python cannot import Trellis2ImageTo3DPipeline."
                    )
                if not python_probe.get("attention_backend_accepted"):
                    accepted = ", ".join(TRELLIS2_ATTENTION_BACKENDS)
                    requested = (
                        python_probe.get("attention_backend_requested") or "<unset>"
                    )
                    active = python_probe.get("attention_backend") or "<unknown>"
                    setup_errors.append(
                        "TRELLIS.2 sparse attention backend must be one of "
                        f"{accepted}; requested {requested}, active {active}."
                    )
                elif not python_probe.get("attention_backend_importable"):
                    backend_module = (
                        python_probe.get("attention_backend_module") or "<unknown>"
                    )
                    setup_errors.append(
                        "TRELLIS.2 attention backend dependency is not importable: "
                        f"{backend_module}."
                    )
                elif not python_probe.get(
                    "attention_backend_api_ready",
                    python_probe.get("attention_backend_importable"),
                ):
                    backend_module = (
                        python_probe.get("attention_backend_module") or "<unknown>"
                    )
                    setup_errors.append(
                        "TRELLIS.2 attention backend is missing its required sparse "
                        f"attention API: {backend_module}."
                    )
                elif not python_probe.get("backend_ready"):
                    detail = python_probe.get("error") or "unknown provider Python error"
                    setup_errors.append(
                        f"TRELLIS.2 provider Python preflight failed: {detail}"
                    )
    elif provider == HUNYUAN3D_SHAPE_PROVIDER:
        provider_dir_value = parsed.get("provider_dir") or os.getenv("HUNYUAN3D_DIR")
        source_available = False
        if provider_dir_value:
            provider_dir = Path(str(provider_dir_value))
            provider_dir_resolved = provider_dir.exists()
            source_available = (provider_dir / "hy3dshape").exists()
        importable = safe_find_spec("hy3dshape")
        checks["hy3dshape_source_found"] = source_available
        checks["hy3dshape_importable"] = importable
        if not (source_available or importable):
            setup_errors.append("Hunyuan3D Shape is not importable. Configure HUNYUAN3D_DIR or install hy3dshape.")

    checks["provider_dir_resolved"] = provider_dir_resolved
    runnable = not setup_errors
    return {
        "provider": provider,
        "readiness": "ready" if runnable else "missing",
        "runnable": runnable,
        "experiment_names": experiment_names or [],
        "wrapper_python": wrapper_python,
        "provider_python": provider_python,
        "provider_dir_configured": bool(parsed.get("provider_dir")),
        "provider_dir_env_names": provider_dir_env_names(provider),
        "provider_models": parsed.get("provider_models", {}),
        "setup_errors": setup_errors,
        "checks": checks,
    }


def experiment_provider_commands(experiments: list[dict]) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for experiment in experiments:
        command = experiment.get("direct_mesh_command")
        if not command:
            continue
        parsed = parse_provider_command(str(command))
        if parsed is None:
            continue
        key = json.dumps(
            {
                "provider": parsed["provider"],
                "provider_dir": parsed.get("provider_dir") or "",
                "provider_python": parsed.get("provider_python") or "",
                "wrapper_python": parsed.get("wrapper_python") or "",
                "input_bundle": parsed.get("input_bundle") or "",
                "provider_models": parsed.get("provider_models") or {},
                "provider_source_revision": parsed.get("provider_source_revision") or "",
                "trellis2_resolution": parsed.get("trellis2_resolution"),
            },
            sort_keys=True,
        )
        commands.setdefault(key, []).append(experiment.get("name") or experiment.get("method") or parsed["provider"])
    return commands


def preflight_experiments(experiments: list[dict]) -> list[dict]:
    rows = []
    for key, names in sorted(experiment_provider_commands(experiments).items()):
        parsed = json.loads(key)
        rows.append(provider_preflight_row(parsed, experiment_names=names))
    return rows


def write_preflight_report(path: Path, rows: list[dict], require_runnable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "require_runnable": require_runnable,
        "rows": rows,
    }
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight image-to-mesh provider commands without loading models.")
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--require-runnable", action="store_true")
    args = parser.parse_args()

    rows = []
    for command in args.command:
        parsed = parse_provider_command(command)
        if parsed is None:
            rows.append({"command": command, "readiness": "unsupported-command", "runnable": False})
        else:
            rows.append(provider_preflight_row(parsed))
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    if args.output:
        write_preflight_report(Path(args.output), rows, require_runnable=args.require_runnable)
    if args.require_runnable:
        missing = [row for row in rows if not row.get("runnable")]
        if missing:
            details = "; ".join(f"{row.get('provider', '<unknown>')}: {', '.join(row.get('setup_errors') or [])}" for row in missing)
            raise SystemExit(f"Image-to-mesh provider preflight failed: {details}")


if __name__ == "__main__":
    main()
