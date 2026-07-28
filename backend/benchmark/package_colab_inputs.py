from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import re
import shlex
import tarfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from backend.benchmark.rank_methods import SCORE_PROFILES
from backend.benchmark.hunyuan3d_2mv_models import (
    DEFAULT_HUNYUAN3D_2MV_MODEL,
    DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION,
    DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION,
    DEFAULT_HUNYUAN3D_2MV_SUBFOLDER,
)
from backend.benchmark.pixal3d_models import (
    DEFAULT_PIXAL3D_DINOV3_MODEL,
    DEFAULT_PIXAL3D_DINOV3_REVISION,
    DEFAULT_PIXAL3D_MODEL,
    DEFAULT_PIXAL3D_MODEL_REVISION,
    DEFAULT_PIXAL3D_MOGE_MODEL,
    DEFAULT_PIXAL3D_MOGE_REVISION,
    DEFAULT_PIXAL3D_REMBG_MODEL,
    DEFAULT_PIXAL3D_REMBG_REVISION,
    pixal3d_model_specs,
)
from backend.benchmark.triposg_models import (
    DEFAULT_TRIPOSG_MODEL,
    DEFAULT_TRIPOSG_MODEL_REVISION,
    DEFAULT_TRIPOSG_REMBG_MODEL,
    DEFAULT_TRIPOSG_REMBG_REVISION,
    triposg_model_specs,
)
from backend.benchmark.step1x3d_models import (
    DEFAULT_STEP1X3D_MODEL,
    DEFAULT_STEP1X3D_MODEL_REVISION,
    DEFAULT_STEP1X3D_SOURCE_REVISION,
    DEFAULT_STEP1X3D_SUBFOLDER,
    step1x3d_model_specs,
)
from backend.benchmark.trellis2_models import (
    DEFAULT_TRELLIS2_MODEL,
    DEFAULT_TRELLIS2_MODEL_REVISION,
    DEFAULT_TRELLIS2_RESOLUTION,
    DEFAULT_TRELLIS2_SOURCE_REVISION,
    trellis2_model_specs,
)


DEFAULT_PATH_FIELDS = (
    "full_image",
    "masked_image",
    "mask",
    "gt_depth",
    "gt_silhouette",
    "silhouette",
    "mesh",
    "asset_path",
    "multiview_images",
    "multiview_masks",
    "video_path",
    "frames_dir",
)
DEFAULT_EXTRACT_ROOT = "/content/3dprintpic_colab_inputs"
DEFAULT_REPO_REMOTE = "https://github.com/DrStrangel0ve/3dprintpic.git"
DEFAULT_REPO_REF = "codex/3d-completion-benchmark-g4"
DEFAULT_INLINE_B64_CHUNK_SIZE = 76_000
DETERMINISTIC_TAR_MTIME = 0
COMPACT_RESULT_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".log", ".md"})
DEFAULT_TRELLIS2_COLAB_PYTHON = "/content/trellis2-venv/bin/python"
DEFAULT_STEP1X3D_COLAB_PYTHON = "/content/step1x3d-venv/bin/python"
TRELLIS2_XFORMERS_VERSION = "0.0.35"
TRELLIS2_UTILS3D_REVISION = "9a4eb15e4021b67b12c460c7057d642626897ec8"
TRELLIS2_CUMESH_REVISION = "12289e1062f0603f2f0d0771b02e1395d247f26f"
TRELLIS2_FLEXGEMM_REVISION = "6dd94a859c26ee8246888502eada3dd8ad85532e"
TRELLIS2_NVDIFFRAST_REVISION = "253ac4fcea7de5f396371124af597e6cc957bfae"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_utf8_lf(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def deterministic_tar_info(archive_name: str, *, size: int, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(archive_name)
    info.size = size
    info.mode = mode
    info.mtime = DETERMINISTIC_TAR_MTIME
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


@contextmanager
def open_deterministic_tar_gz(output: Path) -> Iterator[tarfile.TarFile]:
    with output.open("wb") as raw_file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=DETERMINISTIC_TAR_MTIME) as gzip_file:
            with tarfile.open(fileobj=gzip_file, mode="w") as tar:
                yield tar


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(rows: list[dict], start_index: int, limit: int | None) -> list[dict]:
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    end_index = None if limit is None else start_index + limit
    if limit is not None and limit < 0:
        raise ValueError("--limit must be non-negative")
    return rows[start_index:end_index]


def clean_parts(path: Path) -> list[str]:
    parts = []
    for part in path.parts:
        cleaned = part.replace(":", "")
        if cleaned in {"", ".", "..", "\\", "/"}:
            continue
        parts.append(cleaned)
    return parts


def archive_path_for(source: Path, root: Path, prefix: str) -> Path:
    try:
        relative = source.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(*clean_parts(source))
    return Path(prefix, *clean_parts(relative))


def resolve_input_path(value: str, manifest_dir: Path, root: Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [root / raw, manifest_dir / raw, raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def add_file_once(tar: tarfile.TarFile, source: Path, archive_name: Path, added: set[Path]) -> None:
    resolved = source.resolve()
    if resolved in added:
        return
    if not source.exists():
        raise FileNotFoundError(f"Referenced file does not exist: {source}")
    stat = source.stat()
    mode = 0o755 if stat.st_mode & 0o111 else 0o644
    info = deterministic_tar_info(archive_name.as_posix(), size=stat.st_size, mode=mode)
    with source.open("rb") as file:
        tar.addfile(info, fileobj=file)
    added.add(resolved)


def build_compact_results_archive(
    *,
    output_root: Path,
    archive_path: Path,
    extra_files: Iterable[tuple[Path, str | Path]] = (),
) -> dict:
    """Archive metrics and provenance while excluding generated mesh/render binaries."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    added: set[Path] = set()
    archive_names: list[str] = []
    with open_deterministic_tar_gz(archive_path) as tar:
        for source, archive_name_value in extra_files:
            if not source.is_file():
                continue
            archive_name = Path(archive_name_value)
            before = len(added)
            add_file_once(tar, source, archive_name, added)
            if len(added) > before:
                archive_names.append(archive_name.as_posix())

        if output_root.exists():
            for result_path in sorted(path for path in output_root.rglob("*") if path.is_file()):
                if (
                    result_path.suffix.lower() not in COMPACT_RESULT_SUFFIXES
                    and result_path.name != "artifact_contact_sheet.png"
                ):
                    continue
                relative_path = result_path.relative_to(output_root)
                archive_name = Path("output", output_root.name, *relative_path.parts)
                before = len(added)
                add_file_once(tar, result_path, archive_name, added)
                if len(added) > before:
                    archive_names.append(archive_name.as_posix())

    return {
        "archive": str(archive_path),
        "archive_files": len(archive_names),
        "archive_names": archive_names,
        "archive_size": archive_path.stat().st_size,
        "archive_sha256": file_sha256(archive_path),
    }


def colab_path(extract_root: str, archive_name: Path) -> str:
    return f"{extract_root.rstrip('/')}/{archive_name.as_posix()}"


def shell_join(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def parse_colab_env(items: Iterable[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items or ():
        key, separator, value = str(item).partition("=")
        if not separator:
            raise ValueError(f"--colab-env must be KEY=VALUE, got: {item}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"--colab-env key is not a valid environment variable name: {key}")
        env[key] = value
    return env


def build_colab_gpu_preflight_script() -> str:
    return (
        "python - <<'PY' | tee \"$GPU_PREFLIGHT_PATH\"\n"
        "import json, os, re, shutil, subprocess\n"
        "\n"
        "required_name = os.environ.get('COLAB_REQUIRE_GPU_NAME_REGEX', '').strip()\n"
        "min_memory_raw = os.environ.get('COLAB_MIN_GPU_MEMORY_GB', '').strip()\n"
        "min_memory_gb = None\n"
        "errors = []\n"
        "if min_memory_raw:\n"
        "    try:\n"
        "        min_memory_gb = float(min_memory_raw)\n"
        "    except ValueError:\n"
        "        errors.append(f'invalid COLAB_MIN_GPU_MEMORY_GB={min_memory_raw!r}')\n"
        "    else:\n"
        "        if min_memory_gb < 0:\n"
        "            errors.append(f'COLAB_MIN_GPU_MEMORY_GB must be non-negative, got {min_memory_gb}')\n"
        "\n"
        "query_error = ''\n"
        "gpus = []\n"
        "if shutil.which('nvidia-smi'):\n"
        "    try:\n"
        "        output = subprocess.check_output(\n"
        "            [\n"
        "                'nvidia-smi',\n"
        "                '--query-gpu=name,memory.total',\n"
        "                '--format=csv,noheader,nounits',\n"
        "            ],\n"
        "            text=True,\n"
        "            stderr=subprocess.STDOUT,\n"
        "        )\n"
        "    except subprocess.CalledProcessError as exc:\n"
        "        query_error = exc.output.strip() or f'nvidia-smi exited with {exc.returncode}'\n"
        "    else:\n"
        "        for line in output.splitlines():\n"
        "            line = line.strip()\n"
        "            if not line:\n"
        "                continue\n"
        "            name, _, memory_text = line.partition(',')\n"
        "            name = name.strip()\n"
        "            memory_text = memory_text.strip()\n"
        "            memory_mib = None\n"
        "            if memory_text:\n"
        "                try:\n"
        "                    memory_mib = float(memory_text)\n"
        "                except ValueError:\n"
        "                    pass\n"
        "            gpu = {'index': len(gpus), 'name': name, 'memory_total_mib': memory_mib}\n"
        "            if memory_mib is not None:\n"
        "                gpu['memory_total_gb'] = memory_mib / 1024.0\n"
        "            gpus.append(gpu)\n"
        "else:\n"
        "    query_error = 'nvidia-smi not found'\n"
        "\n"
        "eligible = list(gpus)\n"
        "if required_name:\n"
        "    try:\n"
        "        pattern = re.compile(required_name, re.IGNORECASE)\n"
        "    except re.error as exc:\n"
        "        errors.append(f'invalid COLAB_REQUIRE_GPU_NAME_REGEX={required_name!r}: {exc}')\n"
        "        eligible = []\n"
        "    else:\n"
        "        eligible = [gpu for gpu in eligible if pattern.search(gpu.get('name') or '')]\n"
        "if min_memory_gb is not None:\n"
        "    eligible = [gpu for gpu in eligible if (gpu.get('memory_total_gb') or 0.0) >= min_memory_gb]\n"
        "\n"
        "guard_requested = bool(required_name or min_memory_raw)\n"
        "if guard_requested:\n"
        "    if query_error:\n"
        "        errors.append(query_error)\n"
        "    elif not gpus:\n"
        "        errors.append('no NVIDIA GPUs reported by nvidia-smi')\n"
        "    elif not eligible:\n"
        "        seen = ', '.join(\n"
        "            f\"{gpu.get('name') or 'unknown'} ({gpu.get('memory_total_gb', 0.0):.1f} GB)\"\n"
        "            for gpu in gpus\n"
        "        )\n"
        "        errors.append(\n"
        "            'no GPU satisfied COLAB_REQUIRE_GPU_NAME_REGEX='\n"
        "            f'{required_name!r} and COLAB_MIN_GPU_MEMORY_GB={min_memory_raw!r}; saw {seen}'\n"
        "        )\n"
        "\n"
        "payload = {\n"
        "    'ok': not errors,\n"
        "    'guard_requested': guard_requested,\n"
        "    'requirements': {\n"
        "        'name_regex': required_name,\n"
        "        'min_memory_gb': min_memory_gb,\n"
        "    },\n"
        "    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),\n"
        "    'nvidia_smi_available': bool(shutil.which('nvidia-smi')),\n"
        "    'nvidia_smi_error': query_error,\n"
        "    'gpus': gpus,\n"
        "    'eligible_gpus': eligible,\n"
        "    'errors': errors,\n"
        "}\n"
        "print(json.dumps(payload, indent=2, sort_keys=True))\n"
        "if errors:\n"
        "    raise SystemExit('Colab GPU preflight failed: ' + '; '.join(errors))\n"
        "PY\n"
    )


def rewrite_manifest_rows(
    rows: Iterable[dict],
    *,
    manifest_dir: Path,
    root: Path,
    path_fields: Iterable[str],
    extract_root: str,
) -> tuple[list[dict], dict[Path, Path]]:
    source_to_archive: dict[Path, Path] = {}
    rewritten_rows = []
    path_field_set = set(path_fields)
    for row in rows:
        rewritten = dict(row)
        for field in path_field_set:
            value = row.get(field)
            if not value:
                continue

            def rewrite_one(item) -> str:
                source = resolve_input_path(str(item), manifest_dir, root)
                archive_name = source_to_archive.get(source.resolve())
                if archive_name is None:
                    archive_name = archive_path_for(source, root, "inputs/files")
                    source_to_archive[source.resolve()] = archive_name
                return colab_path(extract_root, archive_name)

            if isinstance(value, list):
                rewritten[field] = [rewrite_one(item) for item in value if item]
            else:
                rewritten[field] = rewrite_one(value)
        rewritten_rows.append(rewritten)
    return rewritten_rows, source_to_archive


def write_rewritten_manifest(tar: tarfile.TarFile, rows: list[dict]) -> Path:
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    manifest_bytes = manifest_text.encode("utf-8")
    manifest_info = deterministic_tar_info("inputs/manifest.jsonl", size=len(manifest_bytes))
    tar.addfile(manifest_info, fileobj=io.BytesIO(manifest_bytes))
    return Path("inputs/manifest.jsonl")


def add_text_file(tar: tarfile.TarFile, archive_name: str, text: str, mode: int = 0o644) -> None:
    encoded = text.encode("utf-8")
    info = deterministic_tar_info(archive_name, size=len(encoded), mode=mode)
    tar.addfile(info, fileobj=io.BytesIO(encoded))


def add_lora(tar: tarfile.TarFile, lora_path: Path, root: Path, added: set[Path]) -> Path | None:
    if not lora_path:
        return None
    if not lora_path.exists():
        raise FileNotFoundError(f"--lora-weights does not exist: {lora_path}")
    lora_base = Path("lora") / lora_path.stem if lora_path.is_file() else Path("lora") / lora_path.name
    if lora_path.is_file():
        archive_name = lora_base / lora_path.name
        add_file_once(tar, lora_path, archive_name, added)
        report_path = lora_path.with_name("training_report.json")
        if report_path.exists():
            add_file_once(tar, report_path, lora_base / "training_report.json", added)
        return archive_name

    for child in sorted(path for path in lora_path.rglob("*") if path.is_file()):
        archive_name = lora_base / child.relative_to(lora_path)
        add_file_once(tar, child, archive_name, added)
    return lora_base


def build_triposr_setup_prelude() -> str:
    return (
        "TRIPOSR_DIR=\"${TRIPOSR_DIR:-/content/TripoSR}\"\n"
        "TRIPOSR_VENV=\"${TRIPOSR_VENV:-/content/triposr-venv}\"\n"
        "export TRIPOSR_DIR TRIPOSR_VENV\n"
        "if [[ ! -d \"$TRIPOSR_DIR/.git\" ]]; then\n"
        "  rm -rf \"$TRIPOSR_DIR\"\n"
        "  git clone --filter=blob:none https://github.com/VAST-AI-Research/TripoSR \"$TRIPOSR_DIR\"\n"
        "fi\n"
        "python -m pip install -q virtualenv\n"
        "if [[ ! -x \"$TRIPOSR_VENV/bin/python\" ]]; then\n"
        "  python -m virtualenv --system-site-packages \"$TRIPOSR_VENV\"\n"
        "fi\n"
        "export PYTHONPATH=\"$TRIPOSR_DIR:${PYTHONPATH:-}\"\n"
        "if ! \"$TRIPOSR_VENV/bin/python\" - <<'PY'\n"
        "import importlib\n"
        "for name in ('torch', 'transformers', 'trimesh', 'fast_simplification', 'tsr'):\n"
        "    importlib.import_module(name)\n"
        "import torchmcubes\n"
        "PY\n"
        "then\n"
        "  \"$TRIPOSR_VENV/bin/python\" -m pip install -U pip setuptools wheel\n"
        "  \"$TRIPOSR_VENV/bin/python\" -m pip install numpy==2.0.2 omegaconf==2.3.0 Pillow==10.1.0 einops==0.7.0 transformers==4.35.0 trimesh==4.12.2 fast-simplification huggingface-hub imageio git+https://github.com/tatsy/torchmcubes.git\n"
        "fi\n"
    )


def build_triposg_setup_prelude() -> str:
    return (
        "TRIPOSG_DIR=\"${TRIPOSG_DIR:-/content/TripoSG}\"\n"
        "TRIPOSG_VENV=\"${TRIPOSG_VENV:-/content/triposg-venv}\"\n"
        "TRIPOSG_REF=\"${TRIPOSG_REF:-fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c}\"\n"
        f"TRIPOSG_MODEL_REVISION=\"${{TRIPOSG_MODEL_REVISION:-{DEFAULT_TRIPOSG_MODEL_REVISION}}}\"\n"
        f"TRIPOSG_REMBG_REVISION=\"${{TRIPOSG_REMBG_REVISION:-{DEFAULT_TRIPOSG_REMBG_REVISION}}}\"\n"
        "export TRIPOSG_DIR TRIPOSG_VENV TRIPOSG_REF TRIPOSG_MODEL_REVISION TRIPOSG_REMBG_REVISION\n"
        "if [[ ! -d \"$TRIPOSG_DIR/.git\" ]]; then\n"
        "  rm -rf \"$TRIPOSG_DIR\"\n"
        "  git clone --filter=blob:none https://github.com/VAST-AI-Research/TripoSG \"$TRIPOSG_DIR\"\n"
        "  git -C \"$TRIPOSG_DIR\" checkout \"$TRIPOSG_REF\"\n"
        "else\n"
        "  git -C \"$TRIPOSG_DIR\" fetch --filter=blob:none origin \"$TRIPOSG_REF\" || git -C \"$TRIPOSG_DIR\" fetch --filter=blob:none origin main\n"
        "  git -C \"$TRIPOSG_DIR\" checkout \"$TRIPOSG_REF\"\n"
        "fi\n"
        "git -C \"$TRIPOSG_DIR\" reset --hard \"$TRIPOSG_REF\"\n"
        "python - <<'PY'\n"
        "import os\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['TRIPOSG_DIR'])\n"
        "inference_utils = root / 'triposg' / 'inference_utils.py'\n"
        "text = inference_utils.read_text()\n"
        "text = text.replace('from diso import DiffDMC\\n', '')\n"
        "needle = '        dmc = DiffDMC(dtype=torch.float32).to(grid_logits.device)\\n'\n"
        "replacement = '        from diso import DiffDMC\\n        dmc = DiffDMC(dtype=torch.float32).to(grid_logits.device)\\n'\n"
        "if replacement not in text:\n"
        "    text = text.replace(needle, replacement)\n"
        "inference_utils.write_text(text)\n"
        "script = root / 'scripts' / 'inference_triposg.py'\n"
        "text = script.read_text()\n"
        "needle = '        guidance_scale=guidance_scale,\\n    ).samples[0]'\n"
        "replacement = '        guidance_scale=guidance_scale,\\n        use_flash_decoder=os.environ.get(\"TRIPOSG_USE_FLASH_DECODER\", \"0\") == \"1\",\\n    ).samples[0]'\n"
        "if 'TRIPOSG_USE_FLASH_DECODER' not in text:\n"
        "    text = text.replace(needle, replacement)\n"
        "script.write_text(text)\n"
        "print('TripoSG setup checkpoint: non-flash decoder patch applied')\n"
        "PY\n"
        "python -m backend.benchmark.patch_triposg_sources --triposg-dir \"$TRIPOSG_DIR\"\n"
        "python -m pip install -q virtualenv\n"
        "if [[ ! -x \"$TRIPOSG_VENV/bin/python\" ]]; then\n"
        "  python -m virtualenv --system-site-packages \"$TRIPOSG_VENV\"\n"
        "fi\n"
        "export PYTHONPATH=\"$TRIPOSG_DIR:${PYTHONPATH:-}\"\n"
        "if ! \"$TRIPOSG_VENV/bin/python\" - <<'PY'\n"
        "import os\n"
        "import importlib\n"
        "missing = []\n"
        "required = ['torch', 'diffusers', 'transformers', 'trimesh', 'fast_simplification', 'triposg.pipelines.pipeline_triposg']\n"
        "if os.environ.get('TRIPOSG_INSTALL_DISO') == '1' or os.environ.get('TRIPOSG_USE_FLASH_DECODER') == '1':\n"
        "    required.append('diso')\n"
        "for name in required:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        missing.append(f'{name} ({type(exc).__name__}: {exc})')\n"
        "if missing:\n"
        "    print('missing TripoSG Python deps: ' + ', '.join(missing))\n"
        "    raise SystemExit(1)\n"
        "PY\n"
        "then\n"
        "  \"$TRIPOSG_VENV/bin/python\" -m pip install -U pip setuptools wheel\n"
        "  python - <<'PY'\n"
        "import os\n"
        "from pathlib import Path\n"
        "source = Path(os.environ['TRIPOSG_DIR']) / 'requirements.txt'\n"
        "target = Path('/tmp/triposg_requirements_colab.txt')\n"
        "skip_prefixes = ('numpy', 'torch', 'torchvision', 'torchaudio')\n"
        "lines = []\n"
        "if source.exists():\n"
        "    for raw in source.read_text().splitlines():\n"
        "        line = raw.strip()\n"
        "        if not line or line.startswith('#'):\n"
        "            continue\n"
        "        lowered = line.lower()\n"
        "        if os.environ.get('TRIPOSG_INSTALL_DISO') != '1' and (lowered == 'diso' or lowered.startswith('diso' + ' ')):\n"
        "            continue\n"
        "        if any(lowered == prefix or lowered.startswith(prefix + spec) for prefix in skip_prefixes for spec in ('=', '<', '>', '~', ' ')):\n"
        "            continue\n"
        "        lines.append(line)\n"
        "target.write_text('\\n'.join(lines) + ('\\n' if lines else ''))\n"
        "print(target)\n"
        "PY\n"
        "  \"$TRIPOSG_VENV/bin/python\" -m pip install numpy==2.0.2 trimesh==4.12.2 fast-simplification Pillow==10.1.0 huggingface-hub imageio packaging ninja\n"
        "  if [[ \"${TRIPOSG_INSTALL_DISO:-0}\" == \"1\" || \"${TRIPOSG_USE_FLASH_DECODER:-0}\" == \"1\" ]]; then\n"
        "    export CUDA_HOME=\"${CUDA_HOME:-/usr/local/cuda}\"\n"
        "    export PATH=\"$CUDA_HOME/bin:$PATH\"\n"
        "    export LD_LIBRARY_PATH=\"$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}\"\n"
        "    export FORCE_CUDA=\"${FORCE_CUDA:-1}\"\n"
        "    export TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST:-12.0}\"\n"
        "    export MAX_JOBS=\"${MAX_JOBS:-1}\"\n"
        "    export NVCC_FLAGS=\"${NVCC_FLAGS:--O3 --threads 1}\"\n"
        "    \"$TRIPOSG_VENV/bin/python\" -m pip install -v --no-cache-dir --no-build-isolation --no-binary=:all: diso==0.1.4\n"
        "  fi\n"
        "  if [[ -s /tmp/triposg_requirements_colab.txt ]]; then\n"
        "    \"$TRIPOSG_VENV/bin/python\" -m pip install -r /tmp/triposg_requirements_colab.txt\n"
        "  fi\n"
        "  \"$TRIPOSG_VENV/bin/python\" - <<'PY'\n"
        "import os\n"
        "import importlib\n"
        "required = ['torch', 'diffusers', 'transformers', 'trimesh', 'fast_simplification', 'triposg.pipelines.pipeline_triposg']\n"
        "if os.environ.get('TRIPOSG_INSTALL_DISO') == '1' or os.environ.get('TRIPOSG_USE_FLASH_DECODER') == '1':\n"
        "    required.append('diso')\n"
        "for name in required:\n"
        "    importlib.import_module(name)\n"
        "PY\n"
        "fi\n"
        "echo \"TripoSG setup checkpoint: Python deps importable\"\n"
        "(cd \"$TRIPOSG_DIR\" && PYTHONPATH=\"$TRIPOSG_DIR${PYTHONPATH:+:$PYTHONPATH}\" \"$TRIPOSG_VENV/bin/python\" \"$TRIPOSG_DIR/scripts/inference_triposg.py\" --help >/tmp/triposg_inference_help.txt)\n"
        "echo \"TripoSG setup checkpoint: CLI imports ok\"\n"
        "if [[ \"${TRIPOSG_PREFETCH:-0}\" == \"1\" ]]; then\n"
        "  \"$TRIPOSG_VENV/bin/python\" - <<'PY'\n"
        "import os\n"
        "from pathlib import Path\n"
        "import torch\n"
        "from huggingface_hub import snapshot_download\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('TripoSG prefetch requires CUDA, but torch.cuda.is_available() is false')\n"
        "props = torch.cuda.get_device_properties(0)\n"
        "total_gb = props.total_memory / (1024 ** 3)\n"
        "print(f'TripoSG CUDA device: {props.name}, VRAM={total_gb:.1f} GB')\n"
        "if total_gb < 8:\n"
        "    raise SystemExit(f'TripoSG needs at least about 8 GB VRAM, got {total_gb:.1f} GB')\n"
        "root = Path(os.environ['TRIPOSG_DIR'])\n"
        "specs = (\n"
        f"    ('{DEFAULT_TRIPOSG_MODEL}', os.environ['TRIPOSG_MODEL_REVISION'], root / 'pretrained_weights' / 'TripoSG'),\n"
        f"    ('{DEFAULT_TRIPOSG_REMBG_MODEL}', os.environ['TRIPOSG_REMBG_REVISION'], root / 'pretrained_weights' / 'RMBG-1.4'),\n"
        ")\n"
        "for repo_id, revision, local_dir in specs:\n"
        "    path = snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir)\n"
        "    print(f'TripoSG prefetched {repo_id}@{revision}: {path}')\n"
        "PY\n"
        "  export TRIPOSG_HF_LOCAL_ONLY=1\n"
        "  echo \"TripoSG setup checkpoint: weights prefetched\"\n"
        "fi\n"
    )


def build_trellis2_setup_prelude() -> str:
    return (
        "TRELLIS2_DIR=\"${TRELLIS2_DIR:-/content/TRELLIS.2}\"\n"
        "TRELLIS2_VENV=\"${TRELLIS2_VENV:-/content/trellis2-venv}\"\n"
        "TRELLIS2_DEPS_SRC=\"${TRELLIS2_DEPS_SRC:-/content/trellis2-deps-src}\"\n"
        f"TRELLIS2_REF=\"${{TRELLIS2_REF:-{DEFAULT_TRELLIS2_SOURCE_REVISION}}}\"\n"
        f"TRELLIS2_MODEL_ID=\"${{TRELLIS2_MODEL_ID:-{DEFAULT_TRELLIS2_MODEL}}}\"\n"
        f"TRELLIS2_MODEL_REVISION=\"${{TRELLIS2_MODEL_REVISION:-{DEFAULT_TRELLIS2_MODEL_REVISION}}}\"\n"
        f"TRELLIS2_RESOLUTION=\"${{TRELLIS2_RESOLUTION:-{DEFAULT_TRELLIS2_RESOLUTION}}}\"\n"
        "TRELLIS2_ATTN_BACKEND=\"${TRELLIS2_ATTN_BACKEND:-xformers}\"\n"
        f"TRELLIS2_XFORMERS_VERSION=\"${{TRELLIS2_XFORMERS_VERSION:-{TRELLIS2_XFORMERS_VERSION}}}\"\n"
        "TRELLIS2_CUMESH_DIR=\"${TRELLIS2_CUMESH_DIR:-$TRELLIS2_DEPS_SRC/CuMesh}\"\n"
        f"TRELLIS2_CUMESH_REF=\"${{TRELLIS2_CUMESH_REF:-{TRELLIS2_CUMESH_REVISION}}}\"\n"
        "TRELLIS2_FLEXGEMM_DIR=\"${TRELLIS2_FLEXGEMM_DIR:-$TRELLIS2_DEPS_SRC/FlexGEMM}\"\n"
        f"TRELLIS2_FLEXGEMM_REF=\"${{TRELLIS2_FLEXGEMM_REF:-{TRELLIS2_FLEXGEMM_REVISION}}}\"\n"
        "TRELLIS2_NVDIFFRAST_DIR=\"${TRELLIS2_NVDIFFRAST_DIR:-$TRELLIS2_DEPS_SRC/nvdiffrast}\"\n"
        f"TRELLIS2_NVDIFFRAST_REF=\"${{TRELLIS2_NVDIFFRAST_REF:-{TRELLIS2_NVDIFFRAST_REVISION}}}\"\n"
        "if [[ \"$TRELLIS2_ATTN_BACKEND\" != \"xformers\" ]]; then\n"
        "  echo \"The reproducible TRELLIS.2 Colab setup installs xformers; got TRELLIS2_ATTN_BACKEND=$TRELLIS2_ATTN_BACKEND\" >&2\n"
        "  exit 2\n"
        "fi\n"
        "TRELLIS2_HAD_ATTN_BACKEND=\"${ATTN_BACKEND+x}\"\n"
        "TRELLIS2_PREVIOUS_ATTN_BACKEND=\"${ATTN_BACKEND-}\"\n"
        "TRELLIS2_HAD_SPARSE_ATTN_BACKEND=\"${SPARSE_ATTN_BACKEND+x}\"\n"
        "TRELLIS2_PREVIOUS_SPARSE_ATTN_BACKEND=\"${SPARSE_ATTN_BACKEND-}\"\n"
        "export TRELLIS2_DIR TRELLIS2_VENV TRELLIS2_DEPS_SRC TRELLIS2_REF TRELLIS2_MODEL_ID TRELLIS2_MODEL_REVISION TRELLIS2_RESOLUTION TRELLIS2_ATTN_BACKEND TRELLIS2_XFORMERS_VERSION\n"
        "export TRELLIS2_CUMESH_DIR TRELLIS2_CUMESH_REF TRELLIS2_FLEXGEMM_DIR TRELLIS2_FLEXGEMM_REF TRELLIS2_NVDIFFRAST_DIR TRELLIS2_NVDIFFRAST_REF\n"
        "export ATTN_BACKEND=\"$TRELLIS2_ATTN_BACKEND\"\n"
        "export SPARSE_ATTN_BACKEND=\"$TRELLIS2_ATTN_BACKEND\"\n"
        "export CUDA_HOME=\"${CUDA_HOME:-/usr/local/cuda}\"\n"
        "export PATH=\"$CUDA_HOME/bin:$PATH\"\n"
        "export LD_LIBRARY_PATH=\"$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}\"\n"
        "export MAX_JOBS=\"${MAX_JOBS:-4}\"\n"
        "mkdir -p \"$TRELLIS2_DEPS_SRC\"\n"
        "if [[ ! -d \"$TRELLIS2_DIR/.git\" ]]; then\n"
        "  rm -rf \"$TRELLIS2_DIR\"\n"
        "  git clone --filter=blob:none --recurse-submodules https://github.com/microsoft/TRELLIS.2 \"$TRELLIS2_DIR\"\n"
        "fi\n"
        "git -C \"$TRELLIS2_DIR\" fetch --filter=blob:none origin \"$TRELLIS2_REF\" || git -C \"$TRELLIS2_DIR\" fetch --filter=blob:none origin main\n"
        "git -C \"$TRELLIS2_DIR\" checkout --detach \"$TRELLIS2_REF\"\n"
        "git -C \"$TRELLIS2_DIR\" reset --hard \"$TRELLIS2_REF\"\n"
        "git -C \"$TRELLIS2_DIR\" submodule update --init --recursive\n"
        "test \"$(git -C \"$TRELLIS2_DIR\" rev-parse HEAD)\" = \"$TRELLIS2_REF\"\n"
        "if [[ ! -d \"$TRELLIS2_CUMESH_DIR/.git\" ]]; then\n"
        "  rm -rf \"$TRELLIS2_CUMESH_DIR\"\n"
        "  git clone --filter=blob:none --recurse-submodules https://github.com/JeffreyXiang/CuMesh.git \"$TRELLIS2_CUMESH_DIR\"\n"
        "fi\n"
        "git -C \"$TRELLIS2_CUMESH_DIR\" fetch --filter=blob:none origin \"$TRELLIS2_CUMESH_REF\" || git -C \"$TRELLIS2_CUMESH_DIR\" fetch --filter=blob:none origin main\n"
        "git -C \"$TRELLIS2_CUMESH_DIR\" checkout --detach \"$TRELLIS2_CUMESH_REF\"\n"
        "git -C \"$TRELLIS2_CUMESH_DIR\" reset --hard \"$TRELLIS2_CUMESH_REF\"\n"
        "git -C \"$TRELLIS2_CUMESH_DIR\" submodule update --init --recursive\n"
        "test \"$(git -C \"$TRELLIS2_CUMESH_DIR\" rev-parse HEAD)\" = \"$TRELLIS2_CUMESH_REF\"\n"
        "if [[ ! -d \"$TRELLIS2_FLEXGEMM_DIR/.git\" ]]; then\n"
        "  rm -rf \"$TRELLIS2_FLEXGEMM_DIR\"\n"
        "  git clone --filter=blob:none --recurse-submodules https://github.com/JeffreyXiang/FlexGEMM.git \"$TRELLIS2_FLEXGEMM_DIR\"\n"
        "fi\n"
        "git -C \"$TRELLIS2_FLEXGEMM_DIR\" fetch --filter=blob:none origin \"$TRELLIS2_FLEXGEMM_REF\" || git -C \"$TRELLIS2_FLEXGEMM_DIR\" fetch --filter=blob:none origin main\n"
        "git -C \"$TRELLIS2_FLEXGEMM_DIR\" checkout --detach \"$TRELLIS2_FLEXGEMM_REF\"\n"
        "git -C \"$TRELLIS2_FLEXGEMM_DIR\" reset --hard \"$TRELLIS2_FLEXGEMM_REF\"\n"
        "git -C \"$TRELLIS2_FLEXGEMM_DIR\" submodule update --init --recursive\n"
        "test \"$(git -C \"$TRELLIS2_FLEXGEMM_DIR\" rev-parse HEAD)\" = \"$TRELLIS2_FLEXGEMM_REF\"\n"
        "if [[ ! -d \"$TRELLIS2_NVDIFFRAST_DIR/.git\" ]]; then\n"
        "  rm -rf \"$TRELLIS2_NVDIFFRAST_DIR\"\n"
        "  git clone --filter=blob:none --branch v0.4.0 https://github.com/NVlabs/nvdiffrast.git \"$TRELLIS2_NVDIFFRAST_DIR\"\n"
        "fi\n"
        "git -C \"$TRELLIS2_NVDIFFRAST_DIR\" fetch --filter=blob:none origin \"$TRELLIS2_NVDIFFRAST_REF\" || git -C \"$TRELLIS2_NVDIFFRAST_DIR\" fetch --filter=blob:none origin v0.4.0\n"
        "git -C \"$TRELLIS2_NVDIFFRAST_DIR\" checkout --detach \"$TRELLIS2_NVDIFFRAST_REF\"\n"
        "git -C \"$TRELLIS2_NVDIFFRAST_DIR\" reset --hard \"$TRELLIS2_NVDIFFRAST_REF\"\n"
        "test \"$(git -C \"$TRELLIS2_NVDIFFRAST_DIR\" rev-parse HEAD)\" = \"$TRELLIS2_NVDIFFRAST_REF\"\n"
        "python -m pip install -q virtualenv\n"
        "if [[ ! -x \"$TRELLIS2_VENV/bin/python\" ]]; then\n"
        "  python -m virtualenv --system-site-packages \"$TRELLIS2_VENV\"\n"
        "fi\n"
        "TRELLIS2_PYTHON=\"$TRELLIS2_VENV/bin/python\"\n"
        "export TRELLIS2_PYTHON\n"
        "export PYTHONPATH=\"$TRELLIS2_DIR:${PYTHONPATH:-}\"\n"
        "if ! command -v nvcc >/dev/null 2>&1; then\n"
        "  echo \"TRELLIS.2 CUDA extension build requires nvcc\" >&2\n"
        "  exit 2\n"
        "fi\n"
        "if ! nvcc --version | grep -Eq 'release 12\\.8([, ]|$)'; then\n"
        "  echo \"TRELLIS.2 G4 setup requires a CUDA 12.8 toolkit to match Torch cu128\" >&2\n"
        "  exit 2\n"
        "fi\n"
        "TRELLIS2_TORCH_CUDA_ARCH=\"$(\"$TRELLIS2_PYTHON\" - <<'PY'\n"
        "import torch\n"
        "if torch.__version__.split('+', 1)[0] != '2.11.0':\n"
        "    raise SystemExit(f'TRELLIS.2 G4 setup requires Torch 2.11.0, got {torch.__version__}')\n"
        "if torch.version.cuda != '12.8':\n"
        "    raise SystemExit(f'TRELLIS.2 G4 setup requires Torch cu128, got CUDA {torch.version.cuda}')\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('TRELLIS.2 setup requires CUDA')\n"
        "major, minor = torch.cuda.get_device_capability(0)\n"
        "print(f'{major}.{minor}')\n"
        "PY\n"
        ")\"\n"
        "export TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST:-$TRELLIS2_TORCH_CUDA_ARCH}\"\n"
        "export CMAKE_CUDA_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES:-${TRELLIS2_TORCH_CUDA_ARCH/./}}\"\n"
        "if ! \"$TRELLIS2_PYTHON\" - <<'PY'\n"
        "import importlib\n"
        "from importlib.metadata import version\n"
        "for name in ('xformers.ops', 'o_voxel', 'cumesh', 'flex_gemm', 'nvdiffrast.torch'):\n"
        "    importlib.import_module(name)\n"
        "if version('xformers') != __import__('os').environ['TRELLIS2_XFORMERS_VERSION']:\n"
        "    raise SystemExit(1)\n"
        "from trellis2.pipelines import Trellis2ImageTo3DPipeline\n"
        "PY\n"
        "then\n"
        "  \"$TRELLIS2_PYTHON\" -m pip install -U pip setuptools wheel ninja cmake packaging\n"
        "  \"$TRELLIS2_PYTHON\" -m pip install --no-deps \"xformers==$TRELLIS2_XFORMERS_VERSION\" --index-url https://download.pytorch.org/whl/cu128\n"
        "  \"$TRELLIS2_PYTHON\" -m pip install imageio==2.37.0 imageio-ffmpeg==0.6.0 tqdm==4.67.1 easydict==1.13 ninja==1.11.1.3 trimesh==4.12.2 lpips==0.1.4 zstandard==0.23.0 kornia==0.8.1 timm==1.0.22\n"
        f"  \"$TRELLIS2_PYTHON\" -m pip install --no-deps \"git+https://github.com/EasternJournalist/utils3d.git@{TRELLIS2_UTILS3D_REVISION}\"\n"
        "  \"$TRELLIS2_PYTHON\" -m pip install -v --no-deps --no-build-isolation \"$TRELLIS2_NVDIFFRAST_DIR\"\n"
        "  \"$TRELLIS2_PYTHON\" -m pip install -v --no-deps --no-build-isolation \"$TRELLIS2_CUMESH_DIR\"\n"
        "  \"$TRELLIS2_PYTHON\" -m pip install -v --no-deps --no-build-isolation \"$TRELLIS2_FLEXGEMM_DIR\"\n"
        "  \"$TRELLIS2_PYTHON\" -m pip install -v --no-deps --no-build-isolation \"$TRELLIS2_DIR/o-voxel\"\n"
        "fi\n"
        "\"$TRELLIS2_PYTHON\" - <<'PY'\n"
        "import json\n"
        "import os\n"
        "from importlib.metadata import version\n"
        "import torch\n"
        "import xformers.ops as xops\n"
        "import trellis2.pipelines\n"
        "from trellis2.modules.sparse import config as sparse_config\n"
        "from trellis2.pipelines import Trellis2ImageTo3DPipeline\n"
        "accepted = ('xformers', 'flash_attn', 'flash_attn_3')\n"
        "requested = os.environ.get('SPARSE_ATTN_BACKEND') or os.environ.get('ATTN_BACKEND') or ''\n"
        "if requested not in accepted or sparse_config.ATTN != requested:\n"
        "    raise SystemExit(f'Invalid TRELLIS.2 sparse attention backend: requested={requested!r}, active={sparse_config.ATTN!r}')\n"
        "if version('xformers') != os.environ['TRELLIS2_XFORMERS_VERSION']:\n"
        "    raise SystemExit(f'Unexpected xformers version: {version(\"xformers\")}')\n"
        "query = torch.randn((1, 32, 1, 64), device='cuda', dtype=torch.float16)\n"
        "mask = xops.fmha.BlockDiagonalMask.from_seqlens([32])\n"
        "output = xops.memory_efficient_attention(query, query, query, mask)\n"
        "torch.cuda.synchronize()\n"
        "props = torch.cuda.get_device_properties(0)\n"
        "print(json.dumps({\n"
        "    'event': 'trellis2_backend_ready',\n"
        "    'python': __import__('sys').executable,\n"
        "    'pipelines_importable': True,\n"
        "    'pipeline_class': Trellis2ImageTo3DPipeline.__name__,\n"
        "    'attention_backend': sparse_config.ATTN,\n"
        "    'attention_smoke_shape': list(output.shape),\n"
        "    'xformers_version': version('xformers'),\n"
        "    'torch_version': torch.__version__,\n"
        "    'gpu': props.name,\n"
        "}, sort_keys=True))\n"
        "PY\n"
        "echo \"TRELLIS.2 setup checkpoint: provider imports and xformers CUDA attention ready\"\n"
        "TRELLIS2_PREFLIGHT_COMMAND=\"$TRELLIS2_PYTHON -m backend.benchmark.run_image_to_mesh_provider --provider trellis2 --provider-dir $TRELLIS2_DIR --trellis2-model-path $TRELLIS2_MODEL_ID --trellis2-model-revision $TRELLIS2_MODEL_REVISION --trellis2-resolution $TRELLIS2_RESOLUTION --seed 42\"\n"
        "\"$TRELLIS2_PYTHON\" -m backend.benchmark.preflight_image_to_mesh_providers --command \"$TRELLIS2_PREFLIGHT_COMMAND\" --output \"${TRELLIS2_PREFLIGHT_PATH:-/tmp/trellis2_provider_preflight.json}\" --require-runnable\n"
        "echo \"TRELLIS.2 setup checkpoint: provider preflight passed\"\n"
        "if [[ \"${TRELLIS2_PREFETCH:-1}\" == \"1\" ]]; then\n"
        "  \"$TRELLIS2_PYTHON\" -m backend.benchmark.run_image_to_mesh_provider --provider trellis2 --provider-dir \"$TRELLIS2_DIR\" --trellis2-model-path \"$TRELLIS2_MODEL_ID\" --trellis2-model-revision \"$TRELLIS2_MODEL_REVISION\" --trellis2-resolution \"$TRELLIS2_RESOLUTION\" --seed 42 --timeout 3600 --prefetch-only\n"
        "  echo \"TRELLIS.2 setup checkpoint: pinned model prefetched\"\n"
        "fi\n"
        "if [[ \"$TRELLIS2_HAD_ATTN_BACKEND\" == \"x\" ]]; then\n"
        "  export ATTN_BACKEND=\"$TRELLIS2_PREVIOUS_ATTN_BACKEND\"\n"
        "else\n"
        "  unset ATTN_BACKEND\n"
        "fi\n"
        "if [[ \"$TRELLIS2_HAD_SPARSE_ATTN_BACKEND\" == \"x\" ]]; then\n"
        "  export SPARSE_ATTN_BACKEND=\"$TRELLIS2_PREVIOUS_SPARSE_ATTN_BACKEND\"\n"
        "else\n"
        "  unset SPARSE_ATTN_BACKEND\n"
        "fi\n"
        "echo \"TRELLIS.2 setup checkpoint: caller attention environment restored\"\n"
    )


def build_pixal3d_setup_prelude() -> str:
    return (
        "PIXAL3D_DIR=\"${PIXAL3D_DIR:-/content/Pixal3D}\"\n"
        "PIXAL3D_VENV=\"${PIXAL3D_VENV:-/content/pixal3d-venv}\"\n"
        "PIXAL3D_DEPS_SRC=\"${PIXAL3D_DEPS_SRC:-/content/pixal3d-deps-src}\"\n"
        "PIXAL3D_REF=\"${PIXAL3D_REF:-cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af}\"\n"
        "MOGE_REF=\"${MOGE_REF:-07444410f1e33f402353b99d6ccd26bd31e469e8}\"\n"
        f"PIXAL3D_MODEL_ID=\"${{PIXAL3D_MODEL_ID:-{DEFAULT_PIXAL3D_MODEL}}}\"\n"
        f"PIXAL3D_MODEL_REVISION=\"${{PIXAL3D_MODEL_REVISION:-{DEFAULT_PIXAL3D_MODEL_REVISION}}}\"\n"
        f"PIXAL3D_MOGE_MODEL_ID=\"${{PIXAL3D_MOGE_MODEL_ID:-{DEFAULT_PIXAL3D_MOGE_MODEL}}}\"\n"
        f"PIXAL3D_MOGE_REVISION=\"${{PIXAL3D_MOGE_REVISION:-{DEFAULT_PIXAL3D_MOGE_REVISION}}}\"\n"
        f"PIXAL3D_DINOV3_MODEL_ID=\"${{PIXAL3D_DINOV3_MODEL_ID:-{DEFAULT_PIXAL3D_DINOV3_MODEL}}}\"\n"
        f"PIXAL3D_DINOV3_REVISION=\"${{PIXAL3D_DINOV3_REVISION:-{DEFAULT_PIXAL3D_DINOV3_REVISION}}}\"\n"
        f"PIXAL3D_REMBG_MODEL_ID=\"${{PIXAL3D_REMBG_MODEL_ID:-{DEFAULT_PIXAL3D_REMBG_MODEL}}}\"\n"
        f"PIXAL3D_REMBG_REVISION=\"${{PIXAL3D_REMBG_REVISION:-{DEFAULT_PIXAL3D_REMBG_REVISION}}}\"\n"
        "TRELLIS2_DIR=\"${TRELLIS2_DIR:-/content/TRELLIS.2}\"\n"
        "TRELLIS2_REF=\"${TRELLIS2_REF:-75fbf0183001ed9876c8dbb35de6b68552ee08bd}\"\n"
        "CUMESH_DIR=\"${CUMESH_DIR:-$PIXAL3D_DEPS_SRC/CuMesh}\"\n"
        "CUMESH_REF=\"${CUMESH_REF:-12289e1062f0603f2f0d0771b02e1395d247f26f}\"\n"
        "FLEXGEMM_DIR=\"${FLEXGEMM_DIR:-$PIXAL3D_DEPS_SRC/FlexGEMM}\"\n"
        "FLEXGEMM_REF=\"${FLEXGEMM_REF:-6dd94a859c26ee8246888502eada3dd8ad85532e}\"\n"
        "NVDIFFRAST_DIR=\"${NVDIFFRAST_DIR:-$PIXAL3D_DEPS_SRC/nvdiffrast}\"\n"
        "NVDIFFRAST_REF=\"${NVDIFFRAST_REF:-253ac4fcea7de5f396371124af597e6cc957bfae}\"\n"
        "export PIXAL3D_DIR PIXAL3D_VENV PIXAL3D_DEPS_SRC PIXAL3D_REF MOGE_REF TRELLIS2_DIR TRELLIS2_REF CUMESH_DIR CUMESH_REF FLEXGEMM_DIR FLEXGEMM_REF NVDIFFRAST_DIR NVDIFFRAST_REF\n"
        "export PIXAL3D_MODEL_ID PIXAL3D_MODEL_REVISION PIXAL3D_MOGE_MODEL_ID PIXAL3D_MOGE_REVISION PIXAL3D_DINOV3_MODEL_ID PIXAL3D_DINOV3_REVISION PIXAL3D_REMBG_MODEL_ID PIXAL3D_REMBG_REVISION\n"
        "export ATTN_BACKEND=\"${ATTN_BACKEND:-sdpa}\"\n"
        "export CUDA_HOME=\"${CUDA_HOME:-/usr/local/cuda}\"\n"
        "export PATH=\"$CUDA_HOME/bin:$PATH\"\n"
        "export LD_LIBRARY_PATH=\"$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}\"\n"
        "export MAX_JOBS=\"${MAX_JOBS:-4}\"\n"
        "mkdir -p \"$PIXAL3D_DEPS_SRC\"\n"
        "if [[ ! -d \"$PIXAL3D_DIR/.git\" ]]; then\n"
        "  rm -rf \"$PIXAL3D_DIR\"\n"
        "  git clone --filter=blob:none https://github.com/TencentARC/Pixal3D \"$PIXAL3D_DIR\"\n"
        "fi\n"
        "git -C \"$PIXAL3D_DIR\" fetch --filter=blob:none origin \"$PIXAL3D_REF\" || git -C \"$PIXAL3D_DIR\" fetch --filter=blob:none origin master\n"
        "git -C \"$PIXAL3D_DIR\" checkout --detach \"$PIXAL3D_REF\"\n"
        "git -C \"$PIXAL3D_DIR\" reset --hard \"$PIXAL3D_REF\"\n"
        "test \"$(git -C \"$PIXAL3D_DIR\" rev-parse HEAD)\" = \"$PIXAL3D_REF\"\n"
        "python -m backend.benchmark.patch_pixal3d_sources --pixal3d-dir \"$PIXAL3D_DIR\"\n"
        "if [[ ! -d \"$TRELLIS2_DIR/.git\" ]]; then\n"
        "  rm -rf \"$TRELLIS2_DIR\"\n"
        "  git clone --filter=blob:none --recurse-submodules https://github.com/microsoft/TRELLIS.2 \"$TRELLIS2_DIR\"\n"
        "fi\n"
        "git -C \"$TRELLIS2_DIR\" fetch --filter=blob:none origin \"$TRELLIS2_REF\" || git -C \"$TRELLIS2_DIR\" fetch --filter=blob:none origin main\n"
        "git -C \"$TRELLIS2_DIR\" checkout --detach \"$TRELLIS2_REF\"\n"
        "git -C \"$TRELLIS2_DIR\" reset --hard \"$TRELLIS2_REF\"\n"
        "test \"$(git -C \"$TRELLIS2_DIR\" rev-parse HEAD)\" = \"$TRELLIS2_REF\"\n"
        "git -C \"$TRELLIS2_DIR\" submodule update --init --recursive\n"
        "if [[ ! -d \"$CUMESH_DIR/.git\" ]]; then\n"
        "  rm -rf \"$CUMESH_DIR\"\n"
        "  git clone --filter=blob:none --recurse-submodules https://github.com/JeffreyXiang/CuMesh.git \"$CUMESH_DIR\"\n"
        "fi\n"
        "git -C \"$CUMESH_DIR\" fetch --filter=blob:none origin \"$CUMESH_REF\" || git -C \"$CUMESH_DIR\" fetch --filter=blob:none origin main\n"
        "git -C \"$CUMESH_DIR\" checkout --detach \"$CUMESH_REF\"\n"
        "git -C \"$CUMESH_DIR\" reset --hard \"$CUMESH_REF\"\n"
        "test \"$(git -C \"$CUMESH_DIR\" rev-parse HEAD)\" = \"$CUMESH_REF\"\n"
        "git -C \"$CUMESH_DIR\" submodule update --init --recursive\n"
        "if [[ ! -d \"$FLEXGEMM_DIR/.git\" ]]; then\n"
        "  rm -rf \"$FLEXGEMM_DIR\"\n"
        "  git clone --filter=blob:none --recurse-submodules https://github.com/JeffreyXiang/FlexGEMM.git \"$FLEXGEMM_DIR\"\n"
        "fi\n"
        "git -C \"$FLEXGEMM_DIR\" fetch --filter=blob:none origin \"$FLEXGEMM_REF\" || git -C \"$FLEXGEMM_DIR\" fetch --filter=blob:none origin main\n"
        "git -C \"$FLEXGEMM_DIR\" checkout --detach \"$FLEXGEMM_REF\"\n"
        "git -C \"$FLEXGEMM_DIR\" reset --hard \"$FLEXGEMM_REF\"\n"
        "test \"$(git -C \"$FLEXGEMM_DIR\" rev-parse HEAD)\" = \"$FLEXGEMM_REF\"\n"
        "git -C \"$FLEXGEMM_DIR\" submodule update --init --recursive\n"
        "if [[ ! -d \"$NVDIFFRAST_DIR/.git\" ]]; then\n"
        "  rm -rf \"$NVDIFFRAST_DIR\"\n"
        "  git clone --filter=blob:none --branch v0.4.0 https://github.com/NVlabs/nvdiffrast.git \"$NVDIFFRAST_DIR\"\n"
        "fi\n"
        "git -C \"$NVDIFFRAST_DIR\" fetch --filter=blob:none origin \"$NVDIFFRAST_REF\" || git -C \"$NVDIFFRAST_DIR\" fetch --filter=blob:none origin v0.4.0\n"
        "git -C \"$NVDIFFRAST_DIR\" checkout --detach \"$NVDIFFRAST_REF\"\n"
        "git -C \"$NVDIFFRAST_DIR\" reset --hard \"$NVDIFFRAST_REF\"\n"
        "test \"$(git -C \"$NVDIFFRAST_DIR\" rev-parse HEAD)\" = \"$NVDIFFRAST_REF\"\n"
        "python -m pip install -q virtualenv\n"
        "if [[ ! -x \"$PIXAL3D_VENV/bin/python\" ]]; then\n"
        "  python -m virtualenv --system-site-packages \"$PIXAL3D_VENV\"\n"
        "fi\n"
        "PIXAL3D_PYTHON=\"$PIXAL3D_VENV/bin/python\"\n"
        "export PIXAL3D_PYTHON\n"
        "export PYTHONPATH=\"$PIXAL3D_DIR:${PYTHONPATH:-}\"\n"
        "if ! command -v nvcc >/dev/null 2>&1; then\n"
        "  echo \"Pixal3D CUDA extension build requires nvcc\" >&2\n"
        "  exit 2\n"
        "fi\n"
        "nvcc --version\n"
        "if ! nvcc --version | grep -Eq 'release 12\\.8([, ]|$)'; then\n"
        "  echo \"Pixal3D G4 setup requires a CUDA 12.8 toolkit to match Torch cu128\" >&2\n"
        "  exit 2\n"
        "fi\n"
        "PIXAL3D_TORCH_CUDA_ARCH=\"$(\"$PIXAL3D_PYTHON\" - <<'PY'\n"
        "import torch\n"
        "if not torch.__version__.split('+', 1)[0].startswith('2.11.'):\n"
        "    raise SystemExit(f'Pixal3D pinned NATTEN wheel requires Torch 2.11.x, got {torch.__version__}')\n"
        "if torch.version.cuda != '12.8':\n"
        "    raise SystemExit(f'Pixal3D G4 setup requires Torch cu128, got CUDA {torch.version.cuda}')\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('Pixal3D setup requires CUDA')\n"
        "major, minor = torch.cuda.get_device_capability(0)\n"
        "print(f'{major}.{minor}')\n"
        "PY\n"
        ")\"\n"
        "export TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST:-$PIXAL3D_TORCH_CUDA_ARCH}\"\n"
        "PIXAL3D_CMAKE_CUDA_ARCH=\"${PIXAL3D_TORCH_CUDA_ARCH/./}\"\n"
        "export CMAKE_CUDA_ARCHITECTURES=\"${CMAKE_CUDA_ARCHITECTURES:-$PIXAL3D_CMAKE_CUDA_ARCH}\"\n"
        "echo \"Pixal3D setup checkpoint: Torch cu128, nvcc 12.8, CUDA arch $TORCH_CUDA_ARCH_LIST\"\n"
        "if ! \"$PIXAL3D_PYTHON\" - <<'PY'\n"
        "import importlib\n"
        "required = (\n"
        "    'torch', 'torchvision', 'cv2', 'trimesh', 'fast_simplification', 'natten',\n"
        "    'moge.model.v2', 'o_voxel', 'cumesh', 'flex_gemm', 'nvdiffrast.torch',\n"
        "    'pixal3d.pipelines',\n"
        ")\n"
        "for name in required:\n"
        "    importlib.import_module(name)\n"
        "from pixal3d.pipelines import Pixal3DImageTo3DPipeline\n"
        "PY\n"
        "then\n"
        "  \"$PIXAL3D_PYTHON\" -m pip install -U pip setuptools wheel ninja cmake packaging\n"
        "  \"$PIXAL3D_PYTHON\" - <<'PY'\n"
        "import os\n"
        "from pathlib import Path\n"
        "source = Path(os.environ['PIXAL3D_DIR']) / 'requirements.txt'\n"
        "target = Path('/tmp/pixal3d_requirements_colab.txt')\n"
        "lines = [\n"
        "    raw for raw in source.read_text(encoding='utf-8').splitlines()\n"
        "    if 'github.com/microsoft/MoGe' not in raw\n"
        "]\n"
        "target.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')\n"
        "print(f'Pixal3D requirements without unpinned MoGe: {target}')\n"
        "PY\n"
        "  \"$PIXAL3D_PYTHON\" -m pip install -r /tmp/pixal3d_requirements_colab.txt\n"
        "  \"$PIXAL3D_PYTHON\" -m pip install --no-deps \"git+https://github.com/microsoft/MoGe.git@$MOGE_REF\"\n"
        "  \"$PIXAL3D_PYTHON\" -m pip install fast-simplification huggingface-hub\n"
        "  \"$PIXAL3D_PYTHON\" -m pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl\n"
        "  \"$PIXAL3D_PYTHON\" -m pip install --no-deps 'natten==0.21.6+torch2110cu128' -f https://whl.natten.org\n"
        "  \"$PIXAL3D_PYTHON\" -m pip install -v --no-build-isolation \"$NVDIFFRAST_DIR\"\n"
        "  \"$PIXAL3D_PYTHON\" -m pip install -v --no-build-isolation \"$CUMESH_DIR\"\n"
        "  \"$PIXAL3D_PYTHON\" -m pip install -v --no-build-isolation \"$FLEXGEMM_DIR\"\n"
        "  \"$PIXAL3D_PYTHON\" -m pip install -v --no-build-isolation \"$TRELLIS2_DIR/o-voxel\"\n"
        "fi\n"
        "\"$PIXAL3D_PYTHON\" - <<'PY'\n"
        "import importlib\n"
        "import torch\n"
        "required = (\n"
        "    'torchvision', 'cv2', 'trimesh', 'fast_simplification', 'natten',\n"
        "    'moge.model.v2', 'o_voxel', 'cumesh', 'flex_gemm', 'nvdiffrast.torch',\n"
        "    'pixal3d.pipelines',\n"
        ")\n"
        "for name in required:\n"
        "    importlib.import_module(name)\n"
        "from pixal3d.pipelines import Pixal3DImageTo3DPipeline\n"
        "if not torch.__version__.split('+', 1)[0].startswith('2.11.') or torch.version.cuda != '12.8':\n"
        "    raise SystemExit(f'Pixal3D dependency install changed the pinned Torch/cu128 stack: {torch.__version__}, CUDA {torch.version.cuda}')\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('Pixal3D setup requires CUDA, but torch.cuda.is_available() is false')\n"
        "props = torch.cuda.get_device_properties(0)\n"
        "total_gb = props.total_memory / (1024 ** 3)\n"
        "print(f'Pixal3D CUDA device: {props.name}, VRAM={total_gb:.1f} GB, capability={torch.cuda.get_device_capability(0)}')\n"
        "if total_gb < 10:\n"
        "    raise SystemExit(f'Pixal3D needs at least about 10 GB VRAM in low-VRAM mode, got {total_gb:.1f} GB')\n"
        "print(f'Pixal3D torch={torch.__version__}, attention_backend={__import__(\"os\").environ.get(\"ATTN_BACKEND\")}')\n"
        "PY\n"
        "echo \"Pixal3D setup checkpoint: Python and CUDA deps importable\"\n"
        "(cd \"$PIXAL3D_DIR\" && \"$PIXAL3D_PYTHON\" inference.py --help >/tmp/pixal3d_inference_help.txt)\n"
        "echo \"Pixal3D setup checkpoint: official CLI imports ok\"\n"
        "if [[ \"${PIXAL3D_PREFETCH:-1}\" == \"1\" ]]; then\n"
        "  \"$PIXAL3D_PYTHON\" - <<'PY'\n"
        "import os\n"
        "import shlex\n"
        "from pathlib import Path\n"
        "from huggingface_hub import snapshot_download\n"
        "specs = (\n"
        "    ('pixal3d', os.environ['PIXAL3D_MODEL_ID'], os.environ['PIXAL3D_MODEL_REVISION'], 'pipeline.json'),\n"
        "    ('moge', os.environ['PIXAL3D_MOGE_MODEL_ID'], os.environ['PIXAL3D_MOGE_REVISION'], 'model.pt'),\n"
        "    ('dinov3', os.environ['PIXAL3D_DINOV3_MODEL_ID'], os.environ['PIXAL3D_DINOV3_REVISION'], 'config.json'),\n"
        "    ('rembg', os.environ['PIXAL3D_REMBG_MODEL_ID'], os.environ['PIXAL3D_REMBG_REVISION'], 'config.json'),\n"
        ")\n"
        "resolved = {}\n"
        "for name, repo_id, revision, required_file in specs:\n"
        "    path = Path(snapshot_download(repo_id=repo_id, revision=revision)).resolve()\n"
        "    required_path = path / required_file\n"
        "    if not required_path.is_file():\n"
        "        raise SystemExit(f'Pixal3D {name} snapshot is missing {required_file}: {required_path}')\n"
        "    resolved[name] = path\n"
        "    print(f'Pixal3D prefetched {repo_id}@{revision}: {path}')\n"
        "runtime_exports = {\n"
        "    'PIXAL3D_MODEL_PATH': resolved['pixal3d'],\n"
        "    'PIXAL3D_MOGE_MODEL_PATH': resolved['moge'] / 'model.pt',\n"
        "    'PIXAL3D_DINOV3_MODEL_PATH': resolved['dinov3'],\n"
        "    'PIXAL3D_REMBG_MODEL': resolved['rembg'],\n"
        "}\n"
        "offline_exports = {\n"
        "    'HF_HUB_OFFLINE': '1',\n"
        "    'TRANSFORMERS_OFFLINE': '1',\n"
        "}\n"
        "os.environ.update({name: str(value) for name, value in {**runtime_exports, **offline_exports}.items()})\n"
        "Path('/tmp/pixal3d_model_paths.sh').write_text(\n"
        "    '\\n'.join(f'export {name}={shlex.quote(str(value))}' for name, value in runtime_exports.items()) + '\\n',\n"
        "    encoding='utf-8',\n"
        ")\n"
        "from pixal3d.pipelines.rembg import BiRefNet\n"
        "model_name = os.environ['PIXAL3D_REMBG_MODEL']\n"
        "rembg = BiRefNet(model_name=model_name)\n"
        "del rembg\n"
        "print(f'Pixal3D rembg model load verified: {model_name}')\n"
        "PY\n"
        "  source /tmp/pixal3d_model_paths.sh\n"
        "  if [[ \"${PIXAL3D_PREFETCH_NAF:-1}\" == \"1\" ]]; then\n"
        "    \"$PIXAL3D_PYTHON\" - <<'PY'\n"
        "import torch\n"
        "model = torch.hub.load('valeoai/NAF', 'naf', pretrained=True, device='cpu', trust_repo=True)\n"
        "del model\n"
        "print('Pixal3D prefetched valeoai/NAF')\n"
        "PY\n"
        "  fi\n"
        "  echo \"Pixal3D setup checkpoint: weights prefetched\"\n"
        "fi\n"
    )


def build_step1x3d_setup_prelude() -> str:
    geometry_requirements = (
        "accelerate==1.5.2",
        "beautifulsoup4==4.12.3",
        "diffusers==0.32.2",
        "einops==0.8.0",
        "huggingface-hub==0.26.2",
        "jaxtyping==0.2.28",
        "omegaconf==2.3.0",
        "onnxruntime==1.21.0",
        "opencv-python-headless==4.10.0.84",
        "pymeshlab==2025.7",
        "pytorch-lightning==2.2.4",
        "rembg==2.0.65",
        "safetensors==0.4.3",
        "scikit-image==0.23.2",
        "timm==0.9.16",
        "transformers==4.48.0",
        "trimesh==4.3.2",
        "typeguard==2.13.3",
    )
    requirements = " ".join(shell_join([requirement]) for requirement in geometry_requirements)
    return (
        "\n"
        "STEP1X3D_DIR=\"${STEP1X3D_DIR:-/content/Step1X-3D}\"\n"
        "STEP1X3D_VENV=\"${STEP1X3D_VENV:-/content/step1x3d-venv}\"\n"
        "STEP1X3D_PYTHON=\"$STEP1X3D_VENV/bin/python\"\n"
        f"STEP1X3D_SOURCE_REVISION=\"${{STEP1X3D_SOURCE_REVISION:-{DEFAULT_STEP1X3D_SOURCE_REVISION}}}\"\n"
        f"STEP1X3D_MODEL_ID=\"${{STEP1X3D_MODEL_ID:-{DEFAULT_STEP1X3D_MODEL}}}\"\n"
        f"STEP1X3D_MODEL_REVISION=\"${{STEP1X3D_MODEL_REVISION:-{DEFAULT_STEP1X3D_MODEL_REVISION}}}\"\n"
        f"STEP1X3D_SUBFOLDER=\"${{STEP1X3D_SUBFOLDER:-{DEFAULT_STEP1X3D_SUBFOLDER}}}\"\n"
        "export STEP1X3D_DIR STEP1X3D_VENV STEP1X3D_PYTHON STEP1X3D_SOURCE_REVISION STEP1X3D_MODEL_ID STEP1X3D_MODEL_REVISION STEP1X3D_SUBFOLDER\n"
        "export USE_SAGEATTN=0\n"
        "if [[ ! -d \"$STEP1X3D_DIR/.git\" ]]; then\n"
        "  git clone --filter=blob:none --no-checkout https://github.com/stepfun-ai/Step1X-3D.git \"$STEP1X3D_DIR\"\n"
        "fi\n"
        "git -C \"$STEP1X3D_DIR\" fetch --depth 1 origin \"$STEP1X3D_SOURCE_REVISION\"\n"
        "git -C \"$STEP1X3D_DIR\" checkout --detach --force \"$STEP1X3D_SOURCE_REVISION\"\n"
        "STEP1X3D_RESOLVED_SOURCE=\"$(git -C \"$STEP1X3D_DIR\" rev-parse HEAD)\"\n"
        "if [[ \"$STEP1X3D_RESOLVED_SOURCE\" != \"$STEP1X3D_SOURCE_REVISION\" ]]; then\n"
        "  echo \"Step1X-3D source revision mismatch: expected $STEP1X3D_SOURCE_REVISION got $STEP1X3D_RESOLVED_SOURCE\" >&2\n"
        "  exit 2\n"
        "fi\n"
        "python -m backend.benchmark.patch_step1x3d_sources --step1x3d-dir \"$STEP1X3D_DIR\"\n"
        "if [[ ! -x \"$STEP1X3D_PYTHON\" ]]; then\n"
        "  python -m venv --system-site-packages \"$STEP1X3D_VENV\"\n"
        "fi\n"
        "\"$STEP1X3D_PYTHON\" -m pip install -U pip setuptools==69.5.1 wheel\n"
        f"\"$STEP1X3D_PYTHON\" -m pip install {requirements}\n"
        "\"$STEP1X3D_PYTHON\" -m pip check > /tmp/step1x3d_pip_check.txt || { cat /tmp/step1x3d_pip_check.txt >&2; exit 2; }\n"
        "PYTHONPATH=\"$STEP1X3D_DIR${PYTHONPATH:+:$PYTHONPATH}\" \"$STEP1X3D_PYTHON\" - <<'PY'\n"
        "import os, torch\n"
        "assert torch.__version__.startswith('2.11.'), torch.__version__\n"
        "assert torch.version.cuda == '12.8', torch.version.cuda\n"
        "assert torch.cuda.is_available()\n"
        "assert torch.cuda.get_device_capability(0) == (12, 0), torch.cuda.get_device_capability(0)\n"
        "assert 'sm_120' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()\n"
        "os.environ['USE_SAGEATTN'] = '0'\n"
        "from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline\n"
        "assert Step1X3DGeometryPipeline is not None\n"
        "PY\n"
        "if [[ \"${STEP1X3D_PREFETCH:-1}\" == \"1\" ]]; then\n"
        "  \"$STEP1X3D_PYTHON\" -m backend.benchmark.run_image_to_mesh_provider --provider step1x3d --provider-dir \"$STEP1X3D_DIR\" --step1x3d-model-path \"$STEP1X3D_MODEL_ID\" --step1x3d-model-revision \"$STEP1X3D_MODEL_REVISION\" --step1x3d-subfolder \"$STEP1X3D_SUBFOLDER\" --timeout 3600 --prefetch-only\n"
        "fi\n"
        "STEP1X3D_PREFLIGHT_COMMAND=\"$STEP1X3D_PYTHON -m backend.benchmark.run_image_to_mesh_provider --provider step1x3d --provider-dir $STEP1X3D_DIR --step1x3d-model-path $STEP1X3D_MODEL_ID --step1x3d-model-revision $STEP1X3D_MODEL_REVISION --step1x3d-subfolder $STEP1X3D_SUBFOLDER --input-image '{input_image}' --output-mesh '{output_mesh}' --provider-device cuda\"\n"
        "\"$STEP1X3D_PYTHON\" -m backend.benchmark.preflight_image_to_mesh_providers --command \"$STEP1X3D_PREFLIGHT_COMMAND\" --output \"${STEP1X3D_PREFLIGHT_PATH:-/tmp/step1x3d_provider_preflight.json}\" --require-runnable\n"
        "echo \"Step1X-3D setup checkpoint: pinned geometry provider ready\"\n"
    )


def build_hunyuan3d_2mv_setup_prelude() -> str:
    return (
        "HUNYUAN3D_2MV_DIR=\"${HUNYUAN3D_2MV_DIR:-/content/Hunyuan3D-2}\"\n"
        "HUNYUAN3D_2MV_VENV=\"${HUNYUAN3D_2MV_VENV:-/content/hunyuan3d-2mv-venv}\"\n"
        f"HUNYUAN3D_2MV_REF=\"${{HUNYUAN3D_2MV_REF:-{DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION}}}\"\n"
        f"HUNYUAN3D_2MV_MODEL_ID=\"${{HUNYUAN3D_2MV_MODEL_ID:-{DEFAULT_HUNYUAN3D_2MV_MODEL}}}\"\n"
        f"HUNYUAN3D_2MV_MODEL_REVISION=\"${{HUNYUAN3D_2MV_MODEL_REVISION:-{DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION}}}\"\n"
        f"HUNYUAN3D_2MV_SUBFOLDER=\"${{HUNYUAN3D_2MV_SUBFOLDER:-{DEFAULT_HUNYUAN3D_2MV_SUBFOLDER}}}\"\n"
        "export HUNYUAN3D_2MV_DIR HUNYUAN3D_2MV_VENV HUNYUAN3D_2MV_REF HUNYUAN3D_2MV_MODEL_ID HUNYUAN3D_2MV_MODEL_REVISION HUNYUAN3D_2MV_SUBFOLDER\n"
        "if [[ ! -d \"$HUNYUAN3D_2MV_DIR/.git\" ]]; then\n"
        "  rm -rf \"$HUNYUAN3D_2MV_DIR\"\n"
        "  git clone --filter=blob:none https://github.com/Tencent-Hunyuan/Hunyuan3D-2 \"$HUNYUAN3D_2MV_DIR\"\n"
        "fi\n"
        "git -C \"$HUNYUAN3D_2MV_DIR\" fetch --filter=blob:none origin \"$HUNYUAN3D_2MV_REF\" || git -C \"$HUNYUAN3D_2MV_DIR\" fetch --filter=blob:none origin main\n"
        "git -C \"$HUNYUAN3D_2MV_DIR\" checkout --detach \"$HUNYUAN3D_2MV_REF\"\n"
        "git -C \"$HUNYUAN3D_2MV_DIR\" reset --hard \"$HUNYUAN3D_2MV_REF\"\n"
        "test \"$(git -C \"$HUNYUAN3D_2MV_DIR\" rev-parse HEAD)\" = \"$HUNYUAN3D_2MV_REF\"\n"
        "python -m pip install -q virtualenv\n"
        "if [[ ! -x \"$HUNYUAN3D_2MV_VENV/bin/python\" ]]; then\n"
        "  python -m virtualenv --system-site-packages \"$HUNYUAN3D_2MV_VENV\"\n"
        "fi\n"
        "HUNYUAN3D_2MV_PYTHON=\"$HUNYUAN3D_2MV_VENV/bin/python\"\n"
        "export HUNYUAN3D_2MV_PYTHON\n"
        "export PYTHONPATH=\"$HUNYUAN3D_2MV_DIR:${PYTHONPATH:-}\"\n"
        "if ! \"$HUNYUAN3D_2MV_PYTHON\" - <<'PY'\n"
        "import importlib\n"
        "from importlib.metadata import version\n"
        "for name in ('torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'einops', 'cv2', 'omegaconf', 'trimesh', 'pymeshlab', 'skimage', 'safetensors', 'hy3dgen.shapegen'):\n"
        "    importlib.import_module(name)\n"
        "if not version('transformers').startswith('4.49.'):\n"
        "    raise SystemExit(1)\n"
        "PY\n"
        "then\n"
        "  \"$HUNYUAN3D_2MV_PYTHON\" -m pip install -U pip setuptools wheel\n"
        "  \"$HUNYUAN3D_2MV_PYTHON\" -m pip install transformers==4.49.0 diffusers==0.32.2 accelerate==1.4.0 huggingface-hub==0.30.2 safetensors==0.5.3 einops==0.8.1 omegaconf==2.3.0 opencv-python-headless==4.11.0.86 tqdm==4.67.1 PyYAML==6.0.2 trimesh==4.12.2 pymeshlab==2025.7.post1 scikit-image==0.25.2\n"
        "fi\n"
        "\"$HUNYUAN3D_2MV_PYTHON\" - <<'PY'\n"
        "import json\n"
        "import shutil\n"
        "import torch\n"
        "import torchvision\n"
        "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('Hunyuan3D-2mv setup requires CUDA')\n"
        "capability = torch.cuda.get_device_capability(0)\n"
        "architectures = torch.cuda.get_arch_list()\n"
        "expected_arch = f'sm_{capability[0]}{capability[1]}'\n"
        "if expected_arch not in architectures:\n"
        "    raise SystemExit(f'Torch wheel lacks {expected_arch}: {architectures}')\n"
        "value = (torch.ones((256, 256), device='cuda') @ torch.ones((256, 256), device='cuda')).sum().item()\n"
        "free_gib = shutil.disk_usage('/content').free / (1024 ** 3)\n"
        "if free_gib < 8:\n"
        "    raise SystemExit(f'Hunyuan3D-2mv setup needs at least 8 GiB free disk, got {free_gib:.1f}')\n"
        "props = torch.cuda.get_device_properties(0)\n"
        "print(json.dumps({\n"
        "    'event': 'hunyuan3d_2mv_backend_ready',\n"
        "    'pipeline_class': Hunyuan3DDiTFlowMatchingPipeline.__name__,\n"
        "    'torch_version': torch.__version__,\n"
        "    'torchvision_version': torchvision.__version__,\n"
        "    'cuda_version': torch.version.cuda,\n"
        "    'gpu': props.name,\n"
        "    'capability': list(capability),\n"
        "    'cuda_architectures': architectures,\n"
        "    'cuda_smoke_value': value,\n"
        "    'free_disk_gib': free_gib,\n"
        "}, sort_keys=True))\n"
        "PY\n"
        "echo \"Hunyuan3D-2mv setup checkpoint: provider imports and CUDA kernel ready\"\n"
        "HUNYUAN3D_2MV_PREFLIGHT_COMMAND=\"$HUNYUAN3D_2MV_PYTHON -m backend.benchmark.run_image_to_mesh_provider --provider hunyuan3d-2mv --provider-dir $HUNYUAN3D_2MV_DIR --provider-python $HUNYUAN3D_2MV_PYTHON --input-image '{input_image}' --input-bundle '{input_bundle}' --output-mesh '{output_mesh}' --hunyuan3d-2mv-model-path $HUNYUAN3D_2MV_MODEL_ID --hunyuan3d-2mv-model-revision $HUNYUAN3D_2MV_MODEL_REVISION --hunyuan3d-2mv-subfolder $HUNYUAN3D_2MV_SUBFOLDER\"\n"
        "\"$HUNYUAN3D_2MV_PYTHON\" -m backend.benchmark.preflight_image_to_mesh_providers --command \"$HUNYUAN3D_2MV_PREFLIGHT_COMMAND\" --output \"${HUNYUAN3D_2MV_PREFLIGHT_PATH:-/tmp/hunyuan3d_2mv_provider_preflight.json}\" --require-runnable\n"
        "echo \"Hunyuan3D-2mv setup checkpoint: provider preflight passed\"\n"
        "if [[ \"${HUNYUAN3D_2MV_PREFETCH:-1}\" == \"1\" ]]; then\n"
        "  \"$HUNYUAN3D_2MV_PYTHON\" -m backend.benchmark.run_image_to_mesh_provider --provider hunyuan3d-2mv --provider-dir \"$HUNYUAN3D_2MV_DIR\" --hunyuan3d-2mv-model-path \"$HUNYUAN3D_2MV_MODEL_ID\" --hunyuan3d-2mv-model-revision \"$HUNYUAN3D_2MV_MODEL_REVISION\" --hunyuan3d-2mv-subfolder \"$HUNYUAN3D_2MV_SUBFOLDER\" --timeout 3600 --prefetch-only\n"
        "  echo \"Hunyuan3D-2mv setup checkpoint: pinned model prefetched\"\n"
        "fi\n"
    )


def build_hunyuan3d_setup_prelude() -> str:
    return (
        "HUNYUAN3D_DIR=\"${HUNYUAN3D_DIR:-/content/Hunyuan3D-2.1}\"\n"
        "HUNYUAN3D_VENV=\"${HUNYUAN3D_VENV:-/content/hunyuan3d-venv}\"\n"
        "HUNYUAN3D_DEPS=\"${HUNYUAN3D_DEPS:-/content/hunyuan3d-deps}\"\n"
        "HUNYUAN3D_VENV_BACKEND=\"${HUNYUAN3D_VENV_BACKEND:-target}\"\n"
        "HUNYUAN3D_REF=\"${HUNYUAN3D_REF:-82920d643c0dc2f7bfd7255f45f62d386edfe60c}\"\n"
        "export HUNYUAN3D_DIR HUNYUAN3D_VENV HUNYUAN3D_DEPS HUNYUAN3D_VENV_BACKEND HUNYUAN3D_REF\n"
        "if [[ ! -d \"$HUNYUAN3D_DIR/.git\" ]]; then\n"
        "  rm -rf \"$HUNYUAN3D_DIR\"\n"
        "  git clone --filter=blob:none --sparse https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 \"$HUNYUAN3D_DIR\"\n"
        "  git -C \"$HUNYUAN3D_DIR\" checkout \"$HUNYUAN3D_REF\"\n"
        "  git -C \"$HUNYUAN3D_DIR\" sparse-checkout set hy3dshape/hy3dshape hy3dshape/configs\n"
        "else\n"
        "  git -C \"$HUNYUAN3D_DIR\" fetch origin \"$HUNYUAN3D_REF\" || true\n"
        "  git -C \"$HUNYUAN3D_DIR\" checkout \"$HUNYUAN3D_REF\"\n"
        "fi\n"
        "if [[ ! -d \"$HUNYUAN3D_DIR/hy3dshape/hy3dshape\" ]]; then\n"
        "  git -C \"$HUNYUAN3D_DIR\" sparse-checkout set hy3dshape/hy3dshape hy3dshape/configs\n"
        "fi\n"
        "if [[ \"$HUNYUAN3D_VENV_BACKEND\" == \"target\" ]]; then\n"
        "  echo \"Creating Hunyuan3D Python env wrapper with target deps\"\n"
        "  mkdir -p \"$HUNYUAN3D_VENV/bin\" \"$HUNYUAN3D_DEPS\"\n"
        "  HUNYUAN3D_WRAPPER_TMP=\"$HUNYUAN3D_VENV/bin/python.target.$$\"\n"
        "  cat > \"$HUNYUAN3D_WRAPPER_TMP\" <<'SH'\n"
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "HUNYUAN3D_DIR=\"${HUNYUAN3D_DIR:-/content/Hunyuan3D-2.1}\"\n"
        "HUNYUAN3D_DEPS=\"${HUNYUAN3D_DEPS:-/content/hunyuan3d-deps}\"\n"
        "export PYTHONPATH=\"$HUNYUAN3D_DEPS:$HUNYUAN3D_DIR/hy3dshape:$HUNYUAN3D_DIR:${PYTHONPATH:-}\"\n"
        "exec python \"$@\"\n"
        "SH\n"
        "  chmod +x \"$HUNYUAN3D_WRAPPER_TMP\"\n"
        "  mv -f \"$HUNYUAN3D_WRAPPER_TMP\" \"$HUNYUAN3D_VENV/bin/python\"\n"
        "  echo \"Hunyuan3D setup checkpoint: target wrapper ready\"\n"
        "elif [[ ! -x \"$HUNYUAN3D_VENV/bin/python\" ]]; then\n"
        "  if [[ \"$HUNYUAN3D_VENV_BACKEND\" == \"virtualenv\" ]]; then\n"
        "    echo \"Creating Hunyuan3D Python env with virtualenv\"\n"
        "    python -m pip install -q virtualenv\n"
        "    python -m virtualenv --system-site-packages \"$HUNYUAN3D_VENV\"\n"
        "  else\n"
        "    echo \"Creating Hunyuan3D Python env with stdlib venv\"\n"
        "    python -m venv --system-site-packages \"$HUNYUAN3D_VENV\"\n"
        "  fi\n"
        "fi\n"
        "HUNYUAN3D_PYTHON=\"$HUNYUAN3D_VENV/bin/python\"\n"
        "if [[ \"$HUNYUAN3D_VENV_BACKEND\" == \"target\" ]]; then\n"
        "  HUNYUAN3D_PIP_INSTALL=(python -m pip install --upgrade --target \"$HUNYUAN3D_DEPS\" --no-deps)\n"
        "else\n"
        "  HUNYUAN3D_PIP_INSTALL=(\"$HUNYUAN3D_PYTHON\" -m pip install)\n"
        "fi\n"
        "export PYTHONPATH=\"$HUNYUAN3D_DIR/hy3dshape:$HUNYUAN3D_DIR:${PYTHONPATH:-}\"\n"
        "if ! \"$HUNYUAN3D_PYTHON\" - <<'PY'\n"
        "import importlib.util\n"
        "missing = []\n"
        "for name in ('torch', 'diffusers', 'transformers', 'accelerate', 'trimesh', 'fast_simplification', 'pymeshlab', 'hy3dshape.pipelines'):\n"
        "    try:\n"
        "        found = importlib.util.find_spec(name) is not None\n"
        "    except ModuleNotFoundError:\n"
        "        found = False\n"
        "    if not found:\n"
        "        missing.append(name)\n"
        "if missing:\n"
        "    print('missing Hunyuan3D Python deps: ' + ', '.join(missing))\n"
        "    raise SystemExit(1)\n"
        "PY\n"
        "then\n"
        "  if [[ \"$HUNYUAN3D_VENV_BACKEND\" != \"target\" ]]; then\n"
        "    \"$HUNYUAN3D_PYTHON\" -m pip install -U pip setuptools wheel\n"
        "  fi\n"
        "  echo \"Installing Hunyuan3D Python deps: core diffusers stack\"\n"
        "  \"${HUNYUAN3D_PIP_INSTALL[@]}\" diffusers==0.30.0 transformers==4.46.0 accelerate==1.1.1 huggingface-hub==0.30.2 safetensors==0.4.4\n"
        "  echo \"Installing Hunyuan3D Python deps: geometry and image stack\"\n"
        "  \"${HUNYUAN3D_PIP_INSTALL[@]}\" einops==0.8.0 omegaconf==2.3.0 pyyaml==6.0.2 tqdm==4.66.5 opencv-python==4.10.0.84 scikit-image==0.24.0 trimesh==4.12.2 fast-simplification pygltflib==1.16.3 xatlas==0.0.9\n"
        "  echo \"Installing Hunyuan3D Python deps: model helpers\"\n"
        "  \"${HUNYUAN3D_PIP_INSTALL[@]}\" timm torchdiffeq\n"
        "  echo \"Installing Hunyuan3D Python deps: pymeshlab\"\n"
        "  \"${HUNYUAN3D_PIP_INSTALL[@]}\" pymeshlab==2023.12.post3\n"
        "  \"$HUNYUAN3D_PYTHON\" - <<'PY'\n"
        "import importlib\n"
        "for name in ('torch', 'diffusers', 'transformers', 'accelerate', 'trimesh', 'fast_simplification', 'pymeshlab', 'hy3dshape.pipelines'):\n"
        "    importlib.import_module(name)\n"
        "PY\n"
        "fi\n"
        "echo \"Hunyuan3D setup checkpoint: Python deps importable\"\n"
        "if [[ \"${HUNYUAN3D_PREFETCH:-1}\" == \"1\" ]]; then\n"
        "  \"$HUNYUAN3D_PYTHON\" -m backend.benchmark.run_image_to_mesh_provider --provider hunyuan3d-shape --provider-dir \"$HUNYUAN3D_DIR\" --model-name \"${HUNYUAN3D_MODEL:-tencent/Hunyuan3D-2.1}\" --prefetch-only\n"
        "  echo \"Hunyuan3D setup checkpoint: shape weights prefetched\"\n"
        "fi\n"
    )


def build_colab_run_script(
    *,
    archive_filename: str,
    extract_root: str,
    manifest_path: str,
    lora_path: str | None,
    colab_archive_path: str | None,
    colab_repo_dir: str,
    repo_remote: str,
    repo_ref: str,
    run_name: str,
    modern_config: str | None,
    cache_providers: Iterable[str],
    skip_cache: bool,
    cache_full: bool,
    eval_starts: Iterable[int],
    eval_limit: int,
    eval_steps: int | None,
    eval_guidance: float | None,
    eval_inpaint_max_dimension: int | None,
    depth_provider: str | None,
    depth_model: str | None,
    stl_target_dimension: int | None,
    score_profile: str,
    candidate_method: str | None,
    current_method: str | None,
    train_steps: int,
    require_modern_cache: bool,
    require_image_to_mesh_providers: bool,
    cache_download_mode: str,
    cache_max_workers: int,
    max_method_failures: int,
    min_paired_n: int,
    allow_missing_split_audit: bool,
    contact_sheet_methods: str | None,
    contact_sheet_max_samples: int | None,
    max_mesh_surface_chamfer_ratio_vs_current: float = 1.1,
    max_mesh_surface_hausdorff95_ratio_vs_current: float = 1.1,
    colab_require_gpu_name_regex: str | None = None,
    colab_min_gpu_memory_gb: float | None = None,
    include_triposr_setup: bool = False,
    include_triposg_setup: bool = False,
    include_pixal3d_setup: bool = False,
    include_trellis2_setup: bool = False,
    include_step1x3d_setup: bool = False,
    include_hunyuan3d_2mv_setup: bool = False,
    include_hunyuan3d_setup: bool = False,
) -> str:
    archive_default = colab_archive_path or f"/content/{archive_filename}"
    command = [
        "python",
        "-u",
        "-m",
        "backend.benchmark.colab_g4_orchestrator",
        "--use-current-repo",
        "--run-name",
        run_name,
        "--stage",
        "cache",
        "--stage",
        "eval",
        "--stage",
        "combine",
        "--manifest",
        manifest_path,
        "--eval-limit",
        str(eval_limit),
        "--score-profile",
        score_profile,
        "--cache-download-mode",
        cache_download_mode,
        "--cache-max-workers",
        str(cache_max_workers),
        "--min-paired-n",
        str(min_paired_n),
        "--max-mesh-surface-chamfer-ratio-vs-current",
        str(max_mesh_surface_chamfer_ratio_vs_current),
        "--max-mesh-surface-hausdorff95-ratio-vs-current",
        str(max_mesh_surface_hausdorff95_ratio_vs_current),
    ]
    if max_method_failures:
        command.extend(["--max-method-failures", str(max_method_failures)])
    if lora_path:
        command.extend(["--existing-lora-weights", lora_path, "--train-steps", str(train_steps)])
    if modern_config:
        command.extend(["--modern-config", modern_config])
    for provider in cache_providers:
        command.extend(["--cache-provider", provider])
    if skip_cache:
        command.append("--skip-cache")
    if cache_full:
        command.append("--cache-full")
    if eval_steps is not None:
        command.extend(["--eval-steps", str(eval_steps)])
    if eval_guidance is not None:
        command.extend(["--eval-guidance", str(eval_guidance)])
    if eval_inpaint_max_dimension is not None:
        command.extend(["--eval-inpaint-max-dimension", str(eval_inpaint_max_dimension)])
    if depth_provider:
        command.extend(["--depth-provider", depth_provider])
    if depth_model:
        command.extend(["--depth-model", depth_model])
    if stl_target_dimension is not None:
        command.extend(["--stl-target-dimension", str(stl_target_dimension)])
    if candidate_method:
        command.extend(["--candidate-method", candidate_method])
    if current_method:
        command.extend(["--current-method", current_method])
    for start in eval_starts:
        command.extend(["--eval-start", str(start)])
    if require_modern_cache:
        command.append("--require-modern-cache")
    if require_image_to_mesh_providers:
        command.append("--require-image-to-mesh-providers")
    if allow_missing_split_audit:
        command.append("--allow-missing-split-audit")
    if contact_sheet_methods:
        command.extend(["--contact-sheet-methods", contact_sheet_methods])
    if contact_sheet_max_samples is not None:
        command.extend(["--contact-sheet-max-samples", str(contact_sheet_max_samples)])

    provider_setup = ""
    if include_triposr_setup:
        provider_setup += build_triposr_setup_prelude()
    if include_triposg_setup:
        provider_setup += build_triposg_setup_prelude()
    if include_pixal3d_setup:
        provider_setup += build_pixal3d_setup_prelude()
    if include_trellis2_setup:
        provider_setup += build_trellis2_setup_prelude()
    if include_step1x3d_setup:
        provider_setup += build_step1x3d_setup_prelude()
    if include_hunyuan3d_2mv_setup:
        provider_setup += build_hunyuan3d_2mv_setup_prelude()
    if include_hunyuan3d_setup:
        provider_setup += build_hunyuan3d_setup_prelude()

    lora_adapter_path = colab_path(lora_path, Path("pytorch_lora_weights.safetensors")) if lora_path else ""
    lora_report_path = colab_path(lora_path, Path("training_report.json")) if lora_path else ""
    gpu_preflight_path = colab_path(extract_root, Path("gpu_preflight.json"))
    preflight_path = colab_path(extract_root, Path("launch_preflight.json"))
    trellis2_preflight_path = colab_path(
        extract_root, Path("trellis2_provider_preflight.json")
    )
    step1x3d_preflight_path = colab_path(
        extract_root, Path("step1x3d_provider_preflight.json")
    )
    hunyuan3d_2mv_preflight_path = colab_path(
        extract_root, Path("hunyuan3d_2mv_provider_preflight.json")
    )
    results_summary_path = colab_path(extract_root, Path("results_summary.json"))
    run_log_path = colab_path(extract_root, Path("run_colab_eval.log"))
    results_archive_path = f"/content/{run_name}_results.tar.gz"
    results_compact_archive_path = f"/content/{run_name}_results_compact.tar.gz"
    stl_ingest_dir = f"/content/{run_name}_stl_first_ingest"
    stl_ingest_json = f"{stl_ingest_dir}/stl_first_ingest_report.json"
    stl_ingest_md = f"{stl_ingest_dir}/stl_first_ingest_report.md"
    gpu_name_default = colab_require_gpu_name_regex or ""
    gpu_memory_default = "" if colab_min_gpu_memory_gb is None else str(colab_min_gpu_memory_gb)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"ARCHIVE_PATH=\"${{1:-{archive_default}}}\"\n"
        f"EXTRACT_ROOT=\"${{EXTRACT_ROOT:-{extract_root}}}\"\n"
        f"REPO_DIR=\"${{REPO_DIR:-{colab_repo_dir}}}\"\n"
        f"REPO_REMOTE=\"${{REPO_REMOTE:-{repo_remote}}}\"\n"
        f"REPO_REF=\"${{REPO_REF:-{repo_ref}}}\"\n\n"
        f"RUN_NAME={shell_join([run_name])}\n"
        f"RUN_LOG=\"${{RUN_LOG:-{run_log_path}}}\"\n"
        "OUTPUT_ROOT=\"${OUTPUT_ROOT:-$REPO_DIR/backend/output/completion-benchmark/colab_g4/$RUN_NAME}\"\n"
        f"RESULTS_SUMMARY=\"${{RESULTS_SUMMARY:-{results_summary_path}}}\"\n"
        f"RESULTS_ARCHIVE=\"${{RESULTS_ARCHIVE:-{results_archive_path}}}\"\n"
        f"RESULTS_COMPACT_ARCHIVE=\"${{RESULTS_COMPACT_ARCHIVE:-{results_compact_archive_path}}}\"\n"
        f"STL_INGEST_DIR=\"${{STL_INGEST_DIR:-{stl_ingest_dir}}}\"\n"
        f"STL_INGEST_JSON=\"${{STL_INGEST_JSON:-{stl_ingest_json}}}\"\n"
        f"STL_INGEST_MD=\"${{STL_INGEST_MD:-{stl_ingest_md}}}\"\n"
        f"MANIFEST_PATH={shell_join([manifest_path])}\n"
        f"LORA_PATH={shell_join([lora_path or ''])}\n"
        f"LORA_ADAPTER_PATH={shell_join([lora_adapter_path])}\n"
        f"LORA_REPORT_PATH={shell_join([lora_report_path])}\n"
        f"GPU_PREFLIGHT_PATH={shell_join([gpu_preflight_path])}\n"
        f"PREFLIGHT_PATH={shell_join([preflight_path])}\n"
        f"TRELLIS2_PREFLIGHT_PATH={shell_join([trellis2_preflight_path])}\n"
        f"STEP1X3D_PREFLIGHT_PATH={shell_join([step1x3d_preflight_path])}\n"
        f"HUNYUAN3D_2MV_PREFLIGHT_PATH={shell_join([hunyuan3d_2mv_preflight_path])}\n"
        "COLAB_REQUIRE_GPU_NAME_REGEX=\"${COLAB_REQUIRE_GPU_NAME_REGEX:-}\"\n"
        "if [[ -z \"$COLAB_REQUIRE_GPU_NAME_REGEX\" ]]; then\n"
        f"  COLAB_REQUIRE_GPU_NAME_REGEX={shell_join([gpu_name_default])}\n"
        "fi\n"
        "COLAB_MIN_GPU_MEMORY_GB=\"${COLAB_MIN_GPU_MEMORY_GB:-}\"\n"
        "if [[ -z \"$COLAB_MIN_GPU_MEMORY_GB\" ]]; then\n"
        f"  COLAB_MIN_GPU_MEMORY_GB={shell_join([gpu_memory_default])}\n"
        "fi\n"
        "export ARCHIVE_PATH EXTRACT_ROOT REPO_DIR REPO_REMOTE REPO_REF RUN_NAME RUN_LOG OUTPUT_ROOT RESULTS_SUMMARY RESULTS_ARCHIVE RESULTS_COMPACT_ARCHIVE STL_INGEST_DIR STL_INGEST_JSON STL_INGEST_MD MANIFEST_PATH LORA_PATH LORA_ADAPTER_PATH LORA_REPORT_PATH GPU_PREFLIGHT_PATH PREFLIGHT_PATH TRELLIS2_PREFLIGHT_PATH STEP1X3D_PREFLIGHT_PATH HUNYUAN3D_2MV_PREFLIGHT_PATH COLAB_REQUIRE_GPU_NAME_REGEX COLAB_MIN_GPU_MEMORY_GB\n"
        "mkdir -p \"$EXTRACT_ROOT\" \"$(dirname \"$RUN_LOG\")\" \"$(dirname \"$RESULTS_SUMMARY\")\" \"$(dirname \"$RESULTS_ARCHIVE\")\" \"$(dirname \"$RESULTS_COMPACT_ARCHIVE\")\" \"$STL_INGEST_DIR\"\n"
        "set +e\n"
        "(\n"
        "set -euo pipefail\n"
        "export PYTHONUNBUFFERED=\"${PYTHONUNBUFFERED:-1}\"\n"
        "export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}\"\n"
        "export CUDA_MODULE_LOADING=\"${CUDA_MODULE_LOADING:-LAZY}\"\n"
        "export MALLOC_ARENA_MAX=\"${MALLOC_ARENA_MAX:-2}\"\n"
        f"{build_colab_gpu_preflight_script()}"
        "if [[ -n \"${EXPECTED_SHA256:-}\" ]]; then\n"
        "  actual_sha=\"$(sha256sum \"$ARCHIVE_PATH\" | awk '{print $1}')\"\n"
        "  if [[ \"$actual_sha\" != \"$EXPECTED_SHA256\" ]]; then\n"
        "    echo \"archive sha256 mismatch: expected $EXPECTED_SHA256 got $actual_sha\" >&2\n"
        "    exit 2\n"
        "  fi\n"
        "fi\n"
        "tar -xzf \"$ARCHIVE_PATH\" -C \"$EXTRACT_ROOT\"\n"
        "test -s \"$MANIFEST_PATH\"\n"
        "if [[ -n \"$LORA_PATH\" ]]; then\n"
        "  test -s \"$LORA_ADAPTER_PATH\"\n"
        "  test -s \"$LORA_REPORT_PATH\"\n"
        "fi\n"
        "manifest_rows=\"$(python - <<'PY'\n"
        "import os, pathlib\n"
        "manifest = pathlib.Path(os.environ['MANIFEST_PATH'])\n"
        "print(sum(1 for line in manifest.read_text().splitlines() if line.strip()))\n"
        "PY\n"
        ")\"\n"
        "if [[ \"$manifest_rows\" -lt 1 ]]; then\n"
        "  echo \"rewritten manifest has no rows\" >&2\n"
        "  exit 2\n"
        "fi\n"
        "if [[ ! -d \"$REPO_DIR/.git\" ]]; then\n"
        "  rm -rf \"$REPO_DIR\"\n"
        "  git clone --filter=blob:none \"$REPO_REMOTE\" \"$REPO_DIR\"\n"
        "fi\n"
        "cd \"$REPO_DIR\"\n"
        "git fetch \"$REPO_REMOTE\" \"$REPO_REF\"\n"
        "git reset --hard FETCH_HEAD\n"
        "resolved_commit=\"$(git rev-parse HEAD)\"\n"
        "if [[ \"${COLAB_SKIP_BACKEND_INSTALL:-0}\" == \"1\" ]]; then\n"
        "  echo \"Skipping backend pip install because COLAB_SKIP_BACKEND_INSTALL=1\"\n"
        "else\n"
        "  python -m pip install -U pip\n"
        "  if [[ -f backend/requirements-cuda.txt ]]; then\n"
        "    python -m pip install -r backend/requirements-cuda.txt\n"
        "  else\n"
        "    python -m pip install -r backend/requirements.txt\n"
        "  fi\n"
        "fi\n"
        f"{provider_setup}"
        "python - <<'PY' > \"$PREFLIGHT_PATH\"\n"
        "import json, os, pathlib, subprocess\n"
        "archive = pathlib.Path(os.environ['ARCHIVE_PATH'])\n"
        "manifest = pathlib.Path(os.environ['MANIFEST_PATH'])\n"
        "payload = {\n"
        "    'archive': str(archive),\n"
        "    'archive_sha256': subprocess.check_output(['sha256sum', str(archive)], text=True).split()[0],\n"
        "    'repo_dir': os.environ['REPO_DIR'],\n"
        "    'repo_remote': os.environ['REPO_REMOTE'],\n"
        "    'repo_ref': os.environ['REPO_REF'],\n"
        "    'resolved_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),\n"
        "    'extract_root': os.environ['EXTRACT_ROOT'],\n"
        "    'gpu_preflight': os.environ['GPU_PREFLIGHT_PATH'],\n"
        "    'trellis2_provider_preflight': os.environ['TRELLIS2_PREFLIGHT_PATH'],\n"
        "    'step1x3d_provider_preflight': os.environ['STEP1X3D_PREFLIGHT_PATH'],\n"
        "    'hunyuan3d_2mv_provider_preflight': os.environ['HUNYUAN3D_2MV_PREFLIGHT_PATH'],\n"
        "    'manifest': str(manifest),\n"
        "    'manifest_rows': sum(1 for line in manifest.read_text().splitlines() if line.strip()),\n"
        "}\n"
        "if os.environ.get('LORA_PATH'):\n"
        "    payload['lora_path'] = os.environ['LORA_PATH']\n"
        "    payload['lora_adapter'] = os.environ['LORA_ADAPTER_PATH']\n"
        "print(json.dumps(payload, indent=2, sort_keys=True))\n"
        "PY\n"
        "if [[ \"${COLAB_PROVIDER_SETUP_ONLY:-0}\" == \"1\" || \"${HUNYUAN3D_SETUP_ONLY:-0}\" == \"1\" || \"${HUNYUAN3D_2MV_SETUP_ONLY:-0}\" == \"1\" || \"${TRIPOSR_SETUP_ONLY:-0}\" == \"1\" || \"${TRIPOSG_SETUP_ONLY:-0}\" == \"1\" || \"${PIXAL3D_SETUP_ONLY:-0}\" == \"1\" || \"${TRELLIS2_SETUP_ONLY:-0}\" == \"1\" || \"${STEP1X3D_SETUP_ONLY:-0}\" == \"1\" ]]; then\n"
        "  echo \"Provider setup only requested; skipping benchmark stages\"\n"
        "  exit 0\n"
        "fi\n"
        f"{shell_join(command)}\n"
        ") 2>&1 | tee \"$RUN_LOG\"\n"
        "run_status=\"${PIPESTATUS[0]}\"\n"
        "set -e\n"
        "export RUN_STATUS=\"$run_status\"\n"
        "cd \"$REPO_DIR\"\n"
        "python - <<'PY'\n"
        "from datetime import datetime, timezone\n"
        "import csv, json, os, pathlib, subprocess, sys, tarfile\n"
        "from backend.benchmark.package_colab_inputs import build_compact_results_archive\n"
        "\n"
        "def add_if_exists(tar: tarfile.TarFile, path: pathlib.Path, arcname: str) -> None:\n"
        "    if path.exists():\n"
        "        tar.add(path, arcname=arcname)\n"
        "\n"
        "def read_csv_rows(path: pathlib.Path) -> list[dict]:\n"
        "    if not path.exists():\n"
        "        return []\n"
        "    with path.open(newline='', encoding='utf-8') as csv_file:\n"
        "        return list(csv.DictReader(csv_file))\n"
        "\n"
        "def read_json_object(path: pathlib.Path) -> dict:\n"
        "    if not path.exists():\n"
        "        return {}\n"
        "    try:\n"
        "        return json.loads(path.read_text(encoding='utf-8'))\n"
        "    except Exception as exc:\n"
        "        return {'read_error': f'{type(exc).__name__}: {exc}'}\n"
        "\n"
        "def summarize_gpu_preflight(path: pathlib.Path) -> dict:\n"
        "    data = read_json_object(path)\n"
        "    summary = {'exists': path.exists(), 'path': str(path)}\n"
        "    if not data:\n"
        "        return summary\n"
        "    if data.get('read_error'):\n"
        "        summary['read_error'] = data.get('read_error')\n"
        "        return summary\n"
        "    summary.update(\n"
        "        {\n"
        "            'ok': data.get('ok'),\n"
        "            'guard_requested': data.get('guard_requested'),\n"
        "            'requirements': data.get('requirements') or {},\n"
        "            'nvidia_smi_available': data.get('nvidia_smi_available'),\n"
        "            'nvidia_smi_error': data.get('nvidia_smi_error', ''),\n"
        "            'errors': data.get('errors') or [],\n"
        "            'gpus': data.get('gpus') or [],\n"
        "            'eligible_gpus': data.get('eligible_gpus') or [],\n"
        "        }\n"
        "    )\n"
        "    return summary\n"
        "\n"
        "def compact_methods(rows: list[dict]) -> list[dict]:\n"
        "    keys = [\n"
        "        'method', 'base_method', 'stl_mode', 'n', 'attempted_n', 'success_rate',\n"
        "        'error_count', 'logged_failure_count', 'rank_score',\n"
        "        'mesh_surface_chamfer_l1_median', 'mesh_surface_hausdorff95_median',\n"
        "        'provider_inference_runtime_seconds_median', 'provider_invocation_runtime_seconds_median',\n"
        "        'direct_mesh_command_runtime_seconds_median', 'repair_runtime_seconds_median',\n"
        "        'mesh_postprocess_runtime_seconds_median', 'provider_peak_cuda_vram_gib_median',\n"
        "        'repair_total_runtime_seconds_median', 'repair_diagnostic_runtime_seconds_median',\n"
        "        'repair_component_filter_input_components_mean',\n"
        "        'repair_component_filter_output_components_mean',\n"
        "        'repair_component_filter_input_faces_median',\n"
        "        'repair_component_filter_output_faces_median',\n"
        "        'repair_component_filter_removed_area_ratio_median',\n"
        "        'repair_bounded_hole_fill_boundary_loops_median',\n"
        "        'repair_bounded_hole_fill_eligible_loops_median',\n"
        "        'repair_bounded_hole_fill_estimated_faces_median',\n"
        "        'repair_bounded_hole_fill_face_budget_median',\n"
        "        'repair_bounded_hole_fill_faces_added_median',\n"
        "        'repair_bounded_hole_fill_watertight_mean',\n"
        "        'repair_simplification_requested_target_faces_median',\n"
        "        'repair_simplification_reserved_hole_faces_median',\n"
        "        'repair_simplification_target_faces_median',\n"
        "        'repair_simplification_applied_mean',\n"
        "        'repair_simplification_audit_available_mean',\n"
        "        'repair_simplification_audit_runtime_seconds_median',\n"
        "        'repair_simplification_volume_relative_change_abs_median',\n"
        "        'repair_simplification_bbox_extent_relative_change_max_median',\n"
        "        'repair_simplification_bounds_center_shift_normalized_median',\n"
        "        'repair_simplification_surface_chamfer_l1_normalized_median',\n"
        "        'repair_simplification_surface_hausdorff95_normalized_median',\n"
        "        'repair_preclean_printable_mean', 'repair_preclean_retriangulation_attempted_mean',\n"
        "        'repair_preclean_retriangulation_accepted_mean',\n"
        "        'repair_preclean_retriangulated_printable_mean',\n"
        "        'repair_preclean_retriangulated_geometry_preserved_mean',\n"
        "        'repair_preclean_retriangulated_face_count_relative_change_abs_median',\n"
        "        'repair_preclean_retriangulated_vertex_count_relative_change_abs_median',\n"
        "        'repair_preclean_retriangulated_volume_relative_change_abs_median',\n"
        "        'repair_preclean_retriangulated_bbox_extent_relative_change_max_median',\n"
        "        'repair_preclean_retriangulated_bounds_center_shift_normalized_median',\n"
        "        'repair_preclean_retriangulated_vertex_displacement_max_normalized_median',\n"
        "        'repair_preclean_retriangulated_surface_chamfer_l1_normalized_median',\n"
        "        'repair_preclean_retriangulated_surface_hausdorff95_normalized_median',\n"
        "        'repair_cleaning_skipped_mean',\n"
        "        'repair_cleaned_printable_mean',\n"
        "        'heldout_view_silhouette_iou_mean_median', 'heldout_view_silhouette_iou_min_median',\n"
        "        'raw_mesh_volume_fill_ratio_median', 'raw_mesh_volume_fill_ratio_reliable_mean',\n"
        "        'raw_mesh_volume_fill_ratio_topology_assessed_mean',\n"
        "        'raw_mesh_volume_fill_ratio_self_intersection_assessed_mean',\n"
        "        'raw_mesh_surface_fill_ratio_median', 'stl_volume_fill_ratio_median',\n"
        "        'raw_mesh_surface_fill_ratio_supported_mean',\n"
        "        'stl_surface_fill_ratio_median',\n"
        "        'repair_volume_fill_ratio_relative_change_median',\n"
        "        'repair_volume_fill_ratio_relative_change_abs_median',\n"
        "        'repair_fill_ratio_relative_change_median',\n"
        "        'repair_fill_ratio_relative_change_abs_median',\n"
        "        'repair_fill_ratio_surface_proxy_used_mean',\n"
        "        'repair_fill_ratio_supported_mean',\n"
        "        'repair_convex_hull_used_mean',\n"
        "        'silhouette_iou_masked_median', 'stl_exists_median', 'stl_is_watertight_median',\n"
        "        'stl_is_volume_median', 'stl_is_manifold_median', 'stl_positive_volume_median',\n"
        "        'stl_single_component_median', 'stl_nonmanifold_edge_count_log1p_median',\n"
        "        'stl_degenerate_face_ratio_median', 'stl_component_excess_log1p_median',\n"
        "        'stl_bbox_aspect_ratio_median', 'stl_faces_per_normalized_bbox_volume_log1p_median',\n"
        "        'stl_faces_per_bbox_volume_log1p_median',\n"
        "    ]\n"
        "    return [{key: row.get(key, '') for key in keys if key in row} for row in rows]\n"
        "\n"
        "def summarize_benchmark_dir(path: pathlib.Path) -> dict:\n"
        "    aggregate = path / 'aggregate_summary.csv'\n"
        "    ranked = path / 'ranked_experiments.csv'\n"
        "    selection = path / 'selection_decision.json'\n"
        "    contact_sheet = path / 'artifact_contact_sheet.png'\n"
        "    image_to_mesh_preflight = path / 'image_to_mesh_provider_preflight.json'\n"
        "    aggregate_rows = read_csv_rows(aggregate)\n"
        "    ranked_rows = read_csv_rows(ranked)\n"
        "    selection_json = read_json_object(selection)\n"
        "    image_to_mesh_preflight_json = read_json_object(image_to_mesh_preflight)\n"
        "    digest = {\n"
        "        'path': str(path),\n"
        "        'aggregate_summary_exists': aggregate.exists(),\n"
        "        'ranked_experiments_exists': ranked.exists(),\n"
        "        'selection_decision_exists': selection.exists(),\n"
        "        'contact_sheet_exists': contact_sheet.exists(),\n"
        "        'image_to_mesh_provider_preflight_exists': image_to_mesh_preflight.exists(),\n"
        "        'methods': compact_methods(ranked_rows or aggregate_rows),\n"
        "    }\n"
        "    if image_to_mesh_preflight_json:\n"
        "        rows = image_to_mesh_preflight_json.get('rows') or []\n"
        "        digest['image_to_mesh_provider_readiness'] = [\n"
        "            {\n"
        "                'provider': row.get('provider', ''),\n"
        "                'readiness': row.get('readiness', ''),\n"
        "                'runnable': row.get('runnable', False),\n"
        "                'experiment_names': row.get('experiment_names', []),\n"
        "                'setup_errors': row.get('setup_errors', []),\n"
        "            }\n"
        "            for row in rows\n"
        "        ]\n"
        "    if ranked_rows:\n"
        "        top = ranked_rows[0]\n"
        "        digest['top_method'] = top.get('method', '')\n"
        "        digest['top_rank_score'] = top.get('rank_score', '')\n"
        "    if selection_json:\n"
        "        digest['decision'] = selection_json.get('decision', '')\n"
        "        digest['candidate_method'] = selection_json.get('candidate_method', '')\n"
        "        digest['failed_checks'] = [check.get('name', '') for check in selection_json.get('failed_checks', [])]\n"
        "    return digest\n"
        "\n"
        "def run_stl_ingest_report(output_root: pathlib.Path, ingest_dir: pathlib.Path, json_path: pathlib.Path, md_path: pathlib.Path, gpu_preflight_summary: dict) -> dict:\n"
        "    log_path = ingest_dir / 'stl_first_ingest.log'\n"
        "    result = {\n"
        "        'status': 'skipped',\n"
        "        'output_dir': str(ingest_dir),\n"
        "        'json': str(json_path),\n"
        "        'markdown': str(md_path),\n"
        "        'log': str(log_path),\n"
        "    }\n"
        "    if os.environ.get('COLAB_PROVIDER_SETUP_ONLY') == '1' or os.environ.get('HUNYUAN3D_SETUP_ONLY') == '1' or os.environ.get('HUNYUAN3D_2MV_SETUP_ONLY') == '1' or os.environ.get('TRIPOSR_SETUP_ONLY') == '1' or os.environ.get('TRIPOSG_SETUP_ONLY') == '1' or os.environ.get('PIXAL3D_SETUP_ONLY') == '1' or os.environ.get('TRELLIS2_SETUP_ONLY') == '1' or os.environ.get('STEP1X3D_SETUP_ONLY') == '1':\n"
        "        result['reason'] = 'provider_setup_only'\n"
        "        return result\n"
        "    if not output_root.exists():\n"
        "        result['reason'] = 'gpu_preflight_failed' if gpu_preflight_summary.get('ok') is False else 'output_root_missing'\n"
        "        if gpu_preflight_summary.get('errors'):\n"
        "            result['gpu_preflight_errors'] = gpu_preflight_summary.get('errors')\n"
        "        return result\n"
        "    ingest_dir.mkdir(parents=True, exist_ok=True)\n"
        "    cmd = [\n"
        "        sys.executable,\n"
        "        '-m',\n"
        "        'backend.benchmark.ingest_stl_results',\n"
        "        f\"{os.environ['RUN_NAME']}={output_root}\",\n"
        "        '--output-dir',\n"
        "        str(ingest_dir),\n"
        "        '--output-json',\n"
        "        str(json_path),\n"
        "        '--output-md',\n"
        "        str(md_path),\n"
        "        '--score-profile',\n"
        "        'stl-quality',\n"
        "        '--score-mode',\n"
        "        'baseline-delta',\n"
        "        '--baseline-method',\n"
        "        'masked',\n"
        "    ]\n"
        "    with log_path.open('w', encoding='utf-8') as log_file:\n"
        "        completed = subprocess.run(cmd, cwd=os.environ['REPO_DIR'], stdout=log_file, stderr=subprocess.STDOUT, text=True, check=False)\n"
        "    result['status'] = 'ok' if completed.returncode == 0 else 'failed'\n"
        "    result['returncode'] = completed.returncode\n"
        "    result['json_exists'] = json_path.exists()\n"
        "    result['markdown_exists'] = md_path.exists()\n"
        "    if json_path.exists():\n"
        "        try:\n"
        "            report = json.loads(json_path.read_text(encoding='utf-8'))\n"
        "            runs = report.get('runs') or []\n"
        "            if runs:\n"
        "                decision = runs[0].get('architecture_replacement_decision') or {}\n"
        "                result['architecture_replacement_decision'] = {\n"
        "                    'decision': decision.get('decision', ''),\n"
        "                    'recommended_method': decision.get('recommended_method', ''),\n"
        "                    'recommended_stl_mode': decision.get('recommended_stl_mode', ''),\n"
        "                    'score_delta_vs_depth_relief': decision.get('score_delta_vs_depth_relief'),\n"
        "                    'depth_relief_baseline_method': (decision.get('depth_relief_baseline') or {}).get('method', ''),\n"
        "                    'promotion_eligible_challenger_method': (decision.get('promotion_eligible_challenger') or {}).get('method', ''),\n"
        "                    'score_leading_challenger_method': (decision.get('score_leading_challenger') or {}).get('method', ''),\n"
        "                }\n"
        "                result['run_count'] = len(runs)\n"
        "        except Exception as exc:\n"
        "            result['report_read_error'] = f'{type(exc).__name__}: {exc}'\n"
        "    return result\n"
        "\n"
        "output_root = pathlib.Path(os.environ['OUTPUT_ROOT'])\n"
        "run_log = pathlib.Path(os.environ['RUN_LOG'])\n"
        "gpu_preflight = pathlib.Path(os.environ['GPU_PREFLIGHT_PATH'])\n"
        "preflight = pathlib.Path(os.environ['PREFLIGHT_PATH'])\n"
        "trellis2_preflight = pathlib.Path(os.environ['TRELLIS2_PREFLIGHT_PATH'])\n"
        "step1x3d_preflight = pathlib.Path(os.environ['STEP1X3D_PREFLIGHT_PATH'])\n"
        "hunyuan3d_2mv_preflight = pathlib.Path(os.environ['HUNYUAN3D_2MV_PREFLIGHT_PATH'])\n"
        "summary_path = pathlib.Path(os.environ['RESULTS_SUMMARY'])\n"
        "archive_path = pathlib.Path(os.environ['RESULTS_ARCHIVE'])\n"
        "compact_archive_path = pathlib.Path(os.environ['RESULTS_COMPACT_ARCHIVE'])\n"
        "stl_ingest_dir = pathlib.Path(os.environ['STL_INGEST_DIR'])\n"
        "stl_ingest_json = pathlib.Path(os.environ['STL_INGEST_JSON'])\n"
        "stl_ingest_md = pathlib.Path(os.environ['STL_INGEST_MD'])\n"
        "run_status = int(os.environ['RUN_STATUS'])\n"
        "gpu_preflight_summary = summarize_gpu_preflight(gpu_preflight)\n"
        "trellis2_preflight_summary = read_json_object(trellis2_preflight)\n"
        "trellis2_preflight_failed = any(not row.get('runnable') for row in trellis2_preflight_summary.get('rows', []))\n"
        f"step1x3d_preflight_expected = {include_step1x3d_setup!r}\n"
        "step1x3d_preflight_summary = read_json_object(step1x3d_preflight)\n"
        "step1x3d_preflight_rows = step1x3d_preflight_summary.get('rows', [])\n"
        "step1x3d_preflight_failed = step1x3d_preflight_expected and (not step1x3d_preflight_rows or any(not row.get('runnable') for row in step1x3d_preflight_rows))\n"
        f"hunyuan3d_2mv_preflight_expected = {include_hunyuan3d_2mv_setup!r}\n"
        "hunyuan3d_2mv_preflight_summary = read_json_object(hunyuan3d_2mv_preflight)\n"
        "hunyuan3d_2mv_preflight_rows = hunyuan3d_2mv_preflight_summary.get('rows', [])\n"
        "hunyuan3d_2mv_preflight_failed = hunyuan3d_2mv_preflight_expected and (not hunyuan3d_2mv_preflight_rows or any(not row.get('runnable') for row in hunyuan3d_2mv_preflight_rows))\n"
        "orchestrator_result = output_root / 'orchestrator_result.json'\n"
        "selection_decisions = sorted(output_root.glob('combined/*/selection_decision.json')) if output_root.exists() else []\n"
        "eval_summaries = [summarize_benchmark_dir(path) for path in sorted(output_root.glob('experiments/*'))] if output_root.exists() else []\n"
        "combined_summaries = [summarize_benchmark_dir(path) for path in sorted(output_root.glob('combined/*'))] if output_root.exists() else []\n"
        "stl_ingest_report = run_stl_ingest_report(output_root, stl_ingest_dir, stl_ingest_json, stl_ingest_md, gpu_preflight_summary)\n"
        "summary = {\n"
        "    'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),\n"
        "    'run_name': os.environ['RUN_NAME'],\n"
        "    'run_status': run_status,\n"
        "    'failure_stage': 'gpu_preflight' if run_status and gpu_preflight_summary.get('ok') is False else ('step1x3d_preflight' if run_status and step1x3d_preflight_failed else ('hunyuan3d_2mv_preflight' if run_status and hunyuan3d_2mv_preflight_failed else ('trellis2_preflight' if run_status and trellis2_preflight_failed else ''))),\n"
        "    'provider_setup_only': os.environ.get('COLAB_PROVIDER_SETUP_ONLY') == '1' or os.environ.get('HUNYUAN3D_SETUP_ONLY') == '1' or os.environ.get('HUNYUAN3D_2MV_SETUP_ONLY') == '1' or os.environ.get('TRIPOSR_SETUP_ONLY') == '1' or os.environ.get('TRIPOSG_SETUP_ONLY') == '1' or os.environ.get('PIXAL3D_SETUP_ONLY') == '1' or os.environ.get('TRELLIS2_SETUP_ONLY') == '1' or os.environ.get('STEP1X3D_SETUP_ONLY') == '1',\n"
        "    'output_root': str(output_root),\n"
        "    'output_root_exists': output_root.exists(),\n"
        "    'run_log': str(run_log),\n"
        "    'gpu_preflight': str(gpu_preflight),\n"
        "    'gpu_preflight_summary': gpu_preflight_summary,\n"
        "    'preflight': str(preflight),\n"
        "    'trellis2_provider_preflight': str(trellis2_preflight),\n"
        "    'trellis2_provider_preflight_summary': trellis2_preflight_summary,\n"
        "    'step1x3d_provider_preflight': str(step1x3d_preflight),\n"
        "    'step1x3d_provider_preflight_expected': step1x3d_preflight_expected,\n"
        "    'step1x3d_provider_preflight_summary': step1x3d_preflight_summary,\n"
        "    'hunyuan3d_2mv_provider_preflight': str(hunyuan3d_2mv_preflight),\n"
        "    'hunyuan3d_2mv_provider_preflight_expected': hunyuan3d_2mv_preflight_expected,\n"
        "    'hunyuan3d_2mv_provider_preflight_summary': hunyuan3d_2mv_preflight_summary,\n"
        "    'results_archive': str(archive_path),\n"
        "    'results_compact_archive': str(compact_archive_path),\n"
        "    'stl_first_ingest_report': stl_ingest_report,\n"
        "    'orchestrator_result': str(orchestrator_result) if orchestrator_result.exists() else '',\n"
        "    'selection_decisions': [str(path) for path in selection_decisions],\n"
        "    'eval_summaries': eval_summaries,\n"
        "    'combined_summaries': combined_summaries,\n"
        "}\n"
        "summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "build_compact_results_archive(\n"
        "    output_root=output_root,\n"
        "    archive_path=compact_archive_path,\n"
        "    extra_files=[\n"
        "        (gpu_preflight, 'gpu_preflight.json'),\n"
        "        (preflight, 'launch_preflight.json'),\n"
        "        (trellis2_preflight, 'trellis2_provider_preflight.json'),\n"
        "        (step1x3d_preflight, 'step1x3d_provider_preflight.json'),\n"
        "        (hunyuan3d_2mv_preflight, 'hunyuan3d_2mv_provider_preflight.json'),\n"
        "        (run_log, 'run_colab_eval.log'),\n"
        "        (summary_path, 'results_summary.json'),\n"
        "        (stl_ingest_json, 'stl_first_ingest_report.json'),\n"
        "        (stl_ingest_md, 'stl_first_ingest_report.md'),\n"
        "        (stl_ingest_dir / 'stl_first_ingest.log', 'stl_first_ingest.log'),\n"
        "    ],\n"
        ")\n"
        "archive_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "with tarfile.open(archive_path, 'w:gz') as tar:\n"
        "    add_if_exists(tar, gpu_preflight, 'gpu_preflight.json')\n"
        "    add_if_exists(tar, preflight, 'launch_preflight.json')\n"
        "    add_if_exists(tar, trellis2_preflight, 'trellis2_provider_preflight.json')\n"
        "    add_if_exists(tar, step1x3d_preflight, 'step1x3d_provider_preflight.json')\n"
        "    add_if_exists(tar, hunyuan3d_2mv_preflight, 'hunyuan3d_2mv_provider_preflight.json')\n"
        "    add_if_exists(tar, run_log, 'run_colab_eval.log')\n"
        "    add_if_exists(tar, summary_path, 'results_summary.json')\n"
        "    add_if_exists(tar, stl_ingest_json, 'stl_first_ingest_report.json')\n"
        "    add_if_exists(tar, stl_ingest_md, 'stl_first_ingest_report.md')\n"
        "    add_if_exists(tar, stl_ingest_dir / 'stl_first_ingest.log', 'stl_first_ingest.log')\n"
        "    if output_root.exists():\n"
        "        tar.add(output_root, arcname=f\"output/{output_root.name}\")\n"
        "print(json.dumps(summary, indent=2, sort_keys=True))\n"
        "PY\n"
        "echo \"Compact results archive: $RESULTS_COMPACT_ARCHIVE\"\n"
        "echo \"Results archive: $RESULTS_ARCHIVE\"\n"
        "exit \"$run_status\"\n"
    )


def build_inline_colab_launcher(
    *,
    archive_path: Path,
    colab_archive_path: str,
    extract_root: str,
    expected_sha256: str,
    expected_size: int,
    chunk_size: int = DEFAULT_INLINE_B64_CHUNK_SIZE,
    colab_env: dict[str, str] | None = None,
) -> tuple[str, dict]:
    if chunk_size <= 0:
        raise ValueError("--inline-colab-chunk-size must be positive")
    launch_env = dict(colab_env or {})
    encoded = base64.b64encode(archive_path.read_bytes()).decode("ascii")
    chunks = [encoded[index : index + chunk_size] for index in range(0, len(encoded), chunk_size)]
    chunks_text = ",\n".join(f"    {json.dumps(chunk)}" for chunk in chunks)
    result_summary_snippet = build_colab_result_summary_snippet()
    script = (
        "# Paste this into one Colab Python cell to upload and launch the packaged benchmark.\n"
        "import base64\n"
        "import hashlib\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import subprocess\n"
        "import tarfile\n"
        "\n"
        f"ARCHIVE_PATH = pathlib.Path({json.dumps(colab_archive_path)})\n"
        f"EXTRACT_ROOT = pathlib.Path({json.dumps(extract_root)})\n"
        f"EXPECTED_SHA256 = {json.dumps(expected_sha256)}\n"
        f"EXPECTED_SIZE = {expected_size}\n"
        f"LAUNCH_ENV = {json.dumps(launch_env, sort_keys=True)}\n"
        "ARCHIVE_B64_CHUNKS = [\n"
        f"{chunks_text}\n"
        "]\n"
        "\n"
        "ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
        "payload = base64.b64decode(''.join(ARCHIVE_B64_CHUNKS).encode('ascii'))\n"
        "ARCHIVE_PATH.write_bytes(payload)\n"
        "actual_sha256 = hashlib.sha256(payload).hexdigest()\n"
        "actual_size = ARCHIVE_PATH.stat().st_size\n"
        "print({'archive': str(ARCHIVE_PATH), 'bytes': actual_size, 'sha256': actual_sha256})\n"
        "if actual_size != EXPECTED_SIZE:\n"
        "    raise SystemExit(f'archive size mismatch: expected {EXPECTED_SIZE} got {actual_size}')\n"
        "if actual_sha256 != EXPECTED_SHA256:\n"
        "    raise SystemExit(f'archive sha256 mismatch: expected {EXPECTED_SHA256} got {actual_sha256}')\n"
        "\n"
        "EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)\n"
        "run_script = EXTRACT_ROOT / 'run_colab_eval.sh'\n"
        "with tarfile.open(ARCHIVE_PATH, 'r:gz') as tar:\n"
        "    try:\n"
        "        member = tar.getmember('run_colab_eval.sh')\n"
        "    except KeyError as exc:\n"
        "        raise SystemExit('archive does not contain run_colab_eval.sh; regenerate with --include-run-script') from exc\n"
        "    extracted = tar.extractfile(member)\n"
        "    if extracted is None:\n"
        "        raise SystemExit('could not read run_colab_eval.sh from archive')\n"
        "    run_script.write_bytes(extracted.read())\n"
        "run_script.chmod(0o755)\n"
        "\n"
        "env = os.environ.copy()\n"
        "env['EXTRACT_ROOT'] = str(EXTRACT_ROOT)\n"
        "env['EXPECTED_SHA256'] = EXPECTED_SHA256\n"
        "for key, value in LAUNCH_ENV.items():\n"
        "    env[key] = value\n"
        "if LAUNCH_ENV:\n"
        "    print({'launch_env': LAUNCH_ENV})\n"
        "print(f'Launching {run_script} with {ARCHIVE_PATH}')\n"
        "completed = subprocess.run(['bash', str(run_script), str(ARCHIVE_PATH)], check=False, env=env)\n"
        f"{result_summary_snippet}"
        "completed.check_returncode()\n"
    )
    return script, {
        "inline_colab_chunk_count": len(chunks),
        "inline_colab_chunk_size": chunk_size,
        "inline_colab_encoded_bytes": len(encoded),
        "inline_colab_expected_size": expected_size,
    }


def build_fetch_colab_launcher(
    *,
    payload_url: str,
    colab_archive_path: str,
    extract_root: str,
    expected_sha256: str,
    expected_size: int,
    colab_env: dict[str, str] | None = None,
) -> str:
    if not payload_url:
        raise ValueError("--fetch-colab-payload-url is required when writing --fetch-colab-launcher")
    launch_env = dict(colab_env or {})
    result_summary_snippet = build_colab_result_summary_snippet()
    return (
        "# Paste this into one Colab Python cell to download and launch the packaged benchmark.\n"
        "import hashlib\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import subprocess\n"
        "import tarfile\n"
        "import urllib.request\n"
        "\n"
        f"PAYLOAD_URL = {json.dumps(payload_url)}\n"
        f"ARCHIVE_PATH = pathlib.Path({json.dumps(colab_archive_path)})\n"
        f"EXTRACT_ROOT = pathlib.Path({json.dumps(extract_root)})\n"
        f"EXPECTED_SHA256 = {json.dumps(expected_sha256)}\n"
        f"EXPECTED_SIZE = {expected_size}\n"
        f"LAUNCH_ENV = {json.dumps(launch_env, sort_keys=True)}\n"
        "\n"
        "ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
        "print(f'Downloading {PAYLOAD_URL}')\n"
        "with urllib.request.urlopen(PAYLOAD_URL) as response:\n"
        "    payload = response.read()\n"
        "ARCHIVE_PATH.write_bytes(payload)\n"
        "\n"
        "actual_size = ARCHIVE_PATH.stat().st_size\n"
        "actual_sha256 = hashlib.sha256(payload).hexdigest()\n"
        "print({'archive': str(ARCHIVE_PATH), 'bytes': actual_size, 'sha256': actual_sha256})\n"
        "if actual_size != EXPECTED_SIZE:\n"
        "    raise SystemExit(f'archive size mismatch: expected {EXPECTED_SIZE} got {actual_size}')\n"
        "if actual_sha256 != EXPECTED_SHA256:\n"
        "    raise SystemExit(f'archive sha256 mismatch: expected {EXPECTED_SHA256} got {actual_sha256}')\n"
        "\n"
        "EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)\n"
        "run_script = EXTRACT_ROOT / 'run_colab_eval.sh'\n"
        "with tarfile.open(ARCHIVE_PATH, 'r:gz') as tar:\n"
        "    try:\n"
        "        member = tar.getmember('run_colab_eval.sh')\n"
        "    except KeyError as exc:\n"
        "        raise SystemExit('archive does not contain run_colab_eval.sh; regenerate with --include-run-script') from exc\n"
        "    extracted = tar.extractfile(member)\n"
        "    if extracted is None:\n"
        "        raise SystemExit('could not read run_colab_eval.sh from archive')\n"
        "    run_script.write_bytes(extracted.read())\n"
        "run_script.chmod(0o755)\n"
        "\n"
        "env = os.environ.copy()\n"
        "env['EXTRACT_ROOT'] = str(EXTRACT_ROOT)\n"
        "env['EXPECTED_SHA256'] = EXPECTED_SHA256\n"
        "for key, value in LAUNCH_ENV.items():\n"
        "    env[key] = value\n"
        "if LAUNCH_ENV:\n"
        "    print({'launch_env': LAUNCH_ENV})\n"
        "print(f'Launching {run_script} with {ARCHIVE_PATH}')\n"
        "completed = subprocess.run(['bash', str(run_script), str(ARCHIVE_PATH)], check=False, env=env)\n"
        f"{result_summary_snippet}"
        "completed.check_returncode()\n"
    )


def build_colab_result_summary_snippet() -> str:
    return (
        "\n"
        "def _colab_sha256_file(path):\n"
        "    digest = hashlib.sha256()\n"
        "    with pathlib.Path(path).open('rb') as handle:\n"
        "        for chunk in iter(lambda: handle.read(1024 * 1024), b''):\n"
        "            digest.update(chunk)\n"
        "    return digest.hexdigest()\n"
        "\n"
        "def _print_colab_result_summary():\n"
        "    summary_path = EXTRACT_ROOT / 'results_summary.json'\n"
        "    summary = {}\n"
        "    if summary_path.exists():\n"
        "        try:\n"
        "            summary = json.loads(summary_path.read_text(encoding='utf-8'))\n"
        "            print('---RESULTS_SUMMARY_JSON---')\n"
        "            print(json.dumps(summary, indent=2, sort_keys=True))\n"
        "        except Exception as exc:\n"
        "            print({'results_summary_read_error': f'{type(exc).__name__}: {exc}', 'path': str(summary_path)})\n"
        "    else:\n"
        "        print({'results_summary_missing': str(summary_path)})\n"
        "    archive_candidates = []\n"
        "    for archive_key in ('results_compact_archive', 'results_archive', 'result_archive'):\n"
        "        archive_value = summary.get(archive_key)\n"
        "        if archive_value:\n"
        "            archive_candidates.append(pathlib.Path(archive_value))\n"
        "    if not archive_candidates:\n"
        "        archive_candidates.extend(sorted(pathlib.Path('/content').glob('*results*.tar.gz')))\n"
        "    seen = set()\n"
        "    archives = []\n"
        "    for archive in archive_candidates:\n"
        "        archive = pathlib.Path(archive)\n"
        "        key = str(archive)\n"
        "        if key in seen or not archive.exists():\n"
        "            continue\n"
        "        seen.add(key)\n"
        "        archives.append({'path': key, 'bytes': archive.stat().st_size, 'sha256': _colab_sha256_file(archive)})\n"
        "    print('---RESULT_ARCHIVES_JSON---')\n"
        "    print(json.dumps(archives, indent=2, sort_keys=True))\n"
        "\n"
        "_print_colab_result_summary()\n"
    )


def build_colab_notebook_launcher(*, launcher_text: str, title: str) -> str:
    if not launcher_text.strip():
        raise ValueError("launcher_text must not be empty")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
            },
            "colab": {
                "provenance": [],
            },
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    "Run the code cell below in Colab to download, verify, and launch the packaged benchmark.\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": launcher_text.splitlines(keepends=True),
            },
        ],
    }
    return json.dumps(notebook, indent=2) + "\n"


def package_inputs(
    *,
    manifest: Path,
    output: Path,
    extract_root: str,
    root: Path,
    start_index: int = 0,
    limit: int | None = None,
    path_fields: Iterable[str] = DEFAULT_PATH_FIELDS,
    lora_weights: Path | None = None,
    run_script_path: Path | None = None,
    include_run_script: bool = False,
    colab_archive_path: str | None = None,
    colab_repo_dir: str = "/content/3dprintpic",
    repo_remote: str = DEFAULT_REPO_REMOTE,
    repo_ref: str = DEFAULT_REPO_REF,
    run_name: str = "g4_modelnet10_weighted_surface_eval_s20",
    modern_config: str | None = None,
    cache_providers: Iterable[str] = (),
    skip_cache: bool = False,
    cache_full: bool = False,
    eval_starts: Iterable[int] = (40, 50),
    eval_limit: int = 10,
    eval_steps: int | None = None,
    eval_guidance: float | None = None,
    eval_inpaint_max_dimension: int | None = None,
    depth_provider: str | None = None,
    depth_model: str | None = None,
    stl_target_dimension: int | None = None,
    score_profile: str = "object-surface",
    candidate_method: str | None = None,
    current_method: str | None = "mirror",
    train_steps: int = 20,
    require_modern_cache: bool = True,
    require_image_to_mesh_providers: bool = False,
    cache_download_mode: str = "snapshot",
    cache_max_workers: int = 8,
    max_method_failures: int = 2,
    min_paired_n: int = 5,
    max_mesh_surface_chamfer_ratio_vs_current: float = 1.1,
    max_mesh_surface_hausdorff95_ratio_vs_current: float = 1.1,
    allow_missing_split_audit: bool = False,
    contact_sheet_methods: str | None = None,
    contact_sheet_max_samples: int | None = None,
    colab_require_gpu_name_regex: str | None = None,
    colab_min_gpu_memory_gb: float | None = None,
    include_triposr_setup: bool = False,
    include_triposg_setup: bool = False,
    include_pixal3d_setup: bool = False,
    include_trellis2_setup: bool = False,
    include_step1x3d_setup: bool = False,
    include_hunyuan3d_2mv_setup: bool = False,
    include_hunyuan3d_setup: bool = False,
    colab_env: dict[str, str] | None = None,
    report_path: Path | None = None,
    inline_colab_launcher_path: Path | None = None,
    inline_colab_chunk_size: int = DEFAULT_INLINE_B64_CHUNK_SIZE,
    fetch_colab_launcher_path: Path | None = None,
    fetch_colab_notebook_path: Path | None = None,
    fetch_colab_payload_url: str | None = None,
) -> dict:
    if not manifest.exists():
        raise FileNotFoundError(f"--manifest does not exist: {manifest}")
    if max_method_failures < 0:
        raise ValueError("--max-method-failures must be non-negative")
    if inline_colab_launcher_path and not include_run_script:
        raise ValueError("--inline-colab-launcher requires --include-run-script")
    if fetch_colab_launcher_path and not include_run_script:
        raise ValueError("--fetch-colab-launcher requires --include-run-script")
    if fetch_colab_notebook_path and not include_run_script:
        raise ValueError("--fetch-colab-notebook requires --include-run-script")
    if (fetch_colab_launcher_path or fetch_colab_notebook_path) and not fetch_colab_payload_url:
        raise ValueError("--fetch-colab-payload-url is required when writing a fetch Colab launcher")
    if inline_colab_chunk_size <= 0:
        raise ValueError("--inline-colab-chunk-size must be positive")
    if colab_min_gpu_memory_gb is not None and colab_min_gpu_memory_gb < 0:
        raise ValueError("--colab-min-gpu-memory-gb must be non-negative")
    rows = load_jsonl(manifest)
    selected = select_rows(rows, start_index, limit)
    cache_provider_list = list(cache_providers)
    colab_env = dict(colab_env or {})
    rewritten_rows, source_to_archive = rewrite_manifest_rows(
        selected,
        manifest_dir=manifest.parent,
        root=root,
        path_fields=path_fields,
        extract_root=extract_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    added: set[Path] = set()
    with open_deterministic_tar_gz(output) as tar:
        manifest_archive_path = write_rewritten_manifest(tar, rewritten_rows)
        for source_resolved, archive_name in sorted(source_to_archive.items(), key=lambda item: item[1].as_posix()):
            add_file_once(tar, source_resolved, archive_name, added)
        lora_archive_path = add_lora(tar, lora_weights, root, added) if lora_weights else None
        run_script_text = ""
        if include_run_script:
            run_script_text = build_colab_run_script(
                archive_filename=output.name,
                extract_root=extract_root,
                manifest_path=colab_path(extract_root, manifest_archive_path),
                lora_path=colab_path(extract_root, lora_archive_path) if lora_archive_path else None,
                colab_archive_path=colab_archive_path,
                colab_repo_dir=colab_repo_dir,
                repo_remote=repo_remote,
                repo_ref=repo_ref,
                run_name=run_name,
                modern_config=modern_config,
                cache_providers=cache_provider_list,
                skip_cache=skip_cache,
                cache_full=cache_full,
                eval_starts=eval_starts,
                eval_limit=eval_limit,
                eval_steps=eval_steps,
                eval_guidance=eval_guidance,
                eval_inpaint_max_dimension=eval_inpaint_max_dimension,
                depth_provider=depth_provider,
                depth_model=depth_model,
                stl_target_dimension=stl_target_dimension,
                score_profile=score_profile,
                candidate_method=candidate_method,
                current_method=current_method,
                train_steps=train_steps,
                require_modern_cache=require_modern_cache,
                require_image_to_mesh_providers=require_image_to_mesh_providers,
                cache_download_mode=cache_download_mode,
                cache_max_workers=cache_max_workers,
                max_method_failures=max_method_failures,
                min_paired_n=min_paired_n,
                max_mesh_surface_chamfer_ratio_vs_current=(
                    max_mesh_surface_chamfer_ratio_vs_current
                ),
                max_mesh_surface_hausdorff95_ratio_vs_current=(
                    max_mesh_surface_hausdorff95_ratio_vs_current
                ),
                allow_missing_split_audit=allow_missing_split_audit,
                contact_sheet_methods=contact_sheet_methods,
                contact_sheet_max_samples=contact_sheet_max_samples,
                colab_require_gpu_name_regex=colab_require_gpu_name_regex,
                colab_min_gpu_memory_gb=colab_min_gpu_memory_gb,
                include_triposr_setup=include_triposr_setup,
                include_triposg_setup=include_triposg_setup,
                include_pixal3d_setup=include_pixal3d_setup,
                include_trellis2_setup=include_trellis2_setup,
                include_step1x3d_setup=include_step1x3d_setup,
                include_hunyuan3d_2mv_setup=include_hunyuan3d_2mv_setup,
                include_hunyuan3d_setup=include_hunyuan3d_setup,
            )
            add_text_file(tar, "run_colab_eval.sh", run_script_text, mode=0o755)

    output_sha256 = file_sha256(output)
    output_size = output.stat().st_size
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "output": str(output),
        "output_sha256": output_sha256,
        "output_size": output_size,
        "extract_root": extract_root,
        "rewritten_manifest": colab_path(extract_root, manifest_archive_path),
        "input_rows": len(rows),
        "packaged_rows": len(rewritten_rows),
        "start_index": start_index,
        "limit": limit,
        "referenced_files": len(source_to_archive),
        "archive_files": len(added),
        "path_fields": list(path_fields),
        "lora_weights": str(lora_weights) if lora_weights else "",
        "rewritten_lora_weights": colab_path(extract_root, lora_archive_path) if lora_archive_path else "",
        "modern_config": modern_config or "",
        "cache_providers": cache_provider_list,
        "skip_cache": skip_cache,
        "cache_full": cache_full,
        "require_image_to_mesh_providers": require_image_to_mesh_providers,
        "cache_download_mode": cache_download_mode,
        "cache_max_workers": cache_max_workers,
        "max_method_failures": max_method_failures,
        "eval_steps": eval_steps,
        "eval_guidance": eval_guidance,
        "eval_inpaint_max_dimension": eval_inpaint_max_dimension,
        "depth_provider": depth_provider or "",
        "depth_model": depth_model or "",
        "stl_target_dimension": stl_target_dimension,
        "candidate_method": candidate_method or "",
        "current_method": current_method or "",
        "max_mesh_surface_chamfer_ratio_vs_current": (
            max_mesh_surface_chamfer_ratio_vs_current
        ),
        "max_mesh_surface_hausdorff95_ratio_vs_current": (
            max_mesh_surface_hausdorff95_ratio_vs_current
        ),
        "allow_missing_split_audit": allow_missing_split_audit,
        "contact_sheet_methods": contact_sheet_methods or "",
        "contact_sheet_max_samples": contact_sheet_max_samples,
        "colab_require_gpu_name_regex": colab_require_gpu_name_regex or "",
        "colab_min_gpu_memory_gb": colab_min_gpu_memory_gb,
        "include_triposr_setup": include_triposr_setup,
        "include_triposg_setup": include_triposg_setup,
        "triposg_model_snapshots": (
            triposg_model_specs() if include_triposg_setup else {}
        ),
        "include_pixal3d_setup": include_pixal3d_setup,
        "pixal3d_model_snapshots": (
            {
                name: {"repo_id": spec["repo_id"], "revision": spec["revision"]}
                for name, spec in pixal3d_model_specs().items()
            }
            if include_pixal3d_setup
            else {}
        ),
        "include_trellis2_setup": include_trellis2_setup,
        "trellis2_source_revision": (
            DEFAULT_TRELLIS2_SOURCE_REVISION if include_trellis2_setup else ""
        ),
        "trellis2_attention_backend": "xformers" if include_trellis2_setup else "",
        "trellis2_python": (
            DEFAULT_TRELLIS2_COLAB_PYTHON if include_trellis2_setup else ""
        ),
        "trellis2_model_snapshots": (
            {
                name: {"repo_id": spec["repo_id"], "revision": spec["revision"]}
                for name, spec in trellis2_model_specs().items()
            }
            if include_trellis2_setup
            else {}
        ),
        "include_step1x3d_setup": include_step1x3d_setup,
        "step1x3d_source_revision": (
            DEFAULT_STEP1X3D_SOURCE_REVISION if include_step1x3d_setup else ""
        ),
        "step1x3d_python": (
            DEFAULT_STEP1X3D_COLAB_PYTHON if include_step1x3d_setup else ""
        ),
        "step1x3d_model_snapshot": (
            {
                "repo_id": DEFAULT_STEP1X3D_MODEL,
                "revision": DEFAULT_STEP1X3D_MODEL_REVISION,
                "subfolder": DEFAULT_STEP1X3D_SUBFOLDER,
            }
            if include_step1x3d_setup
            else {}
        ),
        "include_hunyuan3d_2mv_setup": include_hunyuan3d_2mv_setup,
        "hunyuan3d_2mv_source_revision": (
            DEFAULT_HUNYUAN3D_2MV_SOURCE_REVISION
            if include_hunyuan3d_2mv_setup
            else ""
        ),
        "hunyuan3d_2mv_model_snapshot": (
            {
                "repo_id": DEFAULT_HUNYUAN3D_2MV_MODEL,
                "revision": DEFAULT_HUNYUAN3D_2MV_MODEL_REVISION,
                "subfolder": DEFAULT_HUNYUAN3D_2MV_SUBFOLDER,
            }
            if include_hunyuan3d_2mv_setup
            else {}
        ),
        "include_hunyuan3d_setup": include_hunyuan3d_setup,
        "colab_env": colab_env,
        "run_script_in_archive": "run_colab_eval.sh" if include_run_script else "",
    }
    report_path = report_path or output.with_name(output.name + ".report.json")
    report["report"] = str(report_path)
    if run_script_path and include_run_script:
        write_utf8_lf(run_script_path, run_script_text)
        report["run_script"] = str(run_script_path)
    if inline_colab_launcher_path:
        launcher_text, launcher_meta = build_inline_colab_launcher(
            archive_path=output,
            colab_archive_path=colab_archive_path or f"/content/{output.name}",
            extract_root=extract_root,
            expected_sha256=output_sha256,
            expected_size=output_size,
            chunk_size=inline_colab_chunk_size,
            colab_env=colab_env,
        )
        inline_colab_launcher_path.parent.mkdir(parents=True, exist_ok=True)
        write_utf8_lf(inline_colab_launcher_path, launcher_text)
        report["inline_colab_launcher"] = str(inline_colab_launcher_path)
        report["inline_colab_launcher_sha256"] = file_sha256(inline_colab_launcher_path)
        report.update(launcher_meta)
    if fetch_colab_launcher_path or fetch_colab_notebook_path:
        fetch_launcher_text = build_fetch_colab_launcher(
            payload_url=fetch_colab_payload_url or "",
            colab_archive_path=colab_archive_path or f"/content/{output.name}",
            extract_root=extract_root,
            expected_sha256=output_sha256,
            expected_size=output_size,
            colab_env=colab_env,
        )
    if fetch_colab_launcher_path:
        fetch_colab_launcher_path.parent.mkdir(parents=True, exist_ok=True)
        write_utf8_lf(fetch_colab_launcher_path, fetch_launcher_text)
        report["fetch_colab_launcher"] = str(fetch_colab_launcher_path)
        report["fetch_colab_launcher_sha256"] = file_sha256(fetch_colab_launcher_path)
        report["fetch_colab_payload_url"] = fetch_colab_payload_url or ""
        report["fetch_colab_expected_size"] = output_size
    if fetch_colab_notebook_path:
        notebook_text = build_colab_notebook_launcher(
            launcher_text=fetch_launcher_text,
            title=f"{run_name} Colab Launcher",
        )
        fetch_colab_notebook_path.parent.mkdir(parents=True, exist_ok=True)
        write_utf8_lf(fetch_colab_notebook_path, notebook_text)
        report["fetch_colab_notebook"] = str(fetch_colab_notebook_path)
        report["fetch_colab_notebook_sha256"] = file_sha256(fetch_colab_notebook_path)
    write_utf8_lf(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package benchmark manifest assets and optional LoRA weights for Colab.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--path-field", action="append", default=None)
    parser.add_argument("--lora-weights", default=None)
    parser.add_argument("--include-run-script", action="store_true")
    parser.add_argument("--run-script", default=None)
    parser.add_argument("--colab-archive-path", default=None)
    parser.add_argument("--colab-repo-dir", default="/content/3dprintpic")
    parser.add_argument("--repo-remote", default=DEFAULT_REPO_REMOTE)
    parser.add_argument("--repo-ref", default=DEFAULT_REPO_REF)
    parser.add_argument("--run-name", default="g4_modelnet10_weighted_surface_eval_s20")
    parser.add_argument("--modern-config", default=None)
    parser.add_argument("--cache-provider", action="append", default=None)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--cache-full", action="store_true")
    parser.add_argument("--eval-start", type=int, action="append", default=None)
    parser.add_argument("--eval-limit", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--eval-guidance", type=float, default=None)
    parser.add_argument("--eval-inpaint-max-dimension", type=int, default=None)
    parser.add_argument("--depth-provider", default=None)
    parser.add_argument("--depth-model", default=None)
    parser.add_argument("--stl-target-dimension", type=int, default=None)
    parser.add_argument("--score-profile", choices=tuple(SCORE_PROFILES), default="object-surface")
    parser.add_argument("--candidate-method", default=None)
    parser.add_argument("--current-method", default="mirror")
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--no-require-modern-cache", action="store_true")
    parser.add_argument(
        "--require-image-to-mesh-providers",
        action="store_true",
        help="Pass --require-image-to-mesh-providers to the Colab eval orchestrator.",
    )
    parser.add_argument("--cache-download-mode", choices=("files", "snapshot"), default="snapshot")
    parser.add_argument("--cache-max-workers", type=int, default=8)
    parser.add_argument("--max-method-failures", type=int, default=2)
    parser.add_argument("--min-paired-n", type=int, default=5)
    parser.add_argument("--max-mesh-surface-chamfer-ratio-vs-current", type=float, default=1.1)
    parser.add_argument("--max-mesh-surface-hausdorff95-ratio-vs-current", type=float, default=1.1)
    parser.add_argument("--allow-missing-split-audit", action="store_true")
    parser.add_argument("--contact-sheet-methods", default=None)
    parser.add_argument("--contact-sheet-max-samples", type=int, default=None)
    parser.add_argument(
        "--colab-require-gpu-name-regex",
        default=None,
        help="Embed a generated-runner GPU name regex guard, for example 'RTX PRO 6000|Blackwell|G4'.",
    )
    parser.add_argument(
        "--colab-min-gpu-memory-gb",
        type=float,
        default=None,
        help="Embed a generated-runner minimum GPU memory guard before provider setup begins.",
    )
    parser.add_argument(
        "--include-triposr-setup",
        action="store_true",
        help="Embed a Colab setup prelude for /content/TripoSR and /content/triposr-venv before running eval.",
    )
    parser.add_argument(
        "--include-triposg-setup",
        action="store_true",
        help="Embed a Colab setup prelude for /content/TripoSG and /content/triposg-venv before running eval.",
    )
    parser.add_argument(
        "--include-pixal3d-setup",
        action="store_true",
        help="Embed a pinned Colab setup prelude for /content/Pixal3D and /content/pixal3d-venv before running eval.",
    )
    parser.add_argument(
        "--include-trellis2-setup",
        action="store_true",
        help="Embed a pinned Colab setup and preflight for /content/TRELLIS.2 and /content/trellis2-venv before running eval.",
    )
    parser.add_argument(
        "--include-step1x3d-setup",
        action="store_true",
        help="Embed a pinned geometry-only Colab setup and preflight for /content/Step1X-3D.",
    )
    parser.add_argument(
        "--include-hunyuan3d-setup",
        action="store_true",
        help="Embed a Colab setup prelude for /content/Hunyuan3D-2.1 and /content/hunyuan3d-venv before running eval.",
    )
    parser.add_argument(
        "--include-hunyuan3d-2mv-setup",
        action="store_true",
        help=(
            "Embed a pinned shape-only Colab setup and preflight for /content/Hunyuan3D-2 "
            "and /content/hunyuan3d-2mv-venv before running eval."
        ),
    )
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--inline-colab-launcher",
        default=None,
        help="Write a pasteable Colab Python cell that embeds the archive, verifies it, extracts run_colab_eval.sh, and launches it.",
    )
    parser.add_argument("--inline-colab-chunk-size", type=int, default=DEFAULT_INLINE_B64_CHUNK_SIZE)
    parser.add_argument(
        "--fetch-colab-launcher",
        default=None,
        help="Write a pasteable Colab Python cell that downloads the archive from --fetch-colab-payload-url, verifies it, extracts run_colab_eval.sh, and launches it.",
    )
    parser.add_argument(
        "--fetch-colab-notebook",
        default=None,
        help="Write a Colab .ipynb with the fetch launcher preloaded as a code cell.",
    )
    parser.add_argument("--fetch-colab-payload-url", default=None)
    parser.add_argument(
        "--colab-env",
        action="append",
        default=None,
        help="Embed a KEY=VALUE environment override in generated Colab launchers, for example TRIPOSG_SETUP_ONLY=1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.repo_root) if args.repo_root else repo_root()
    report = package_inputs(
        manifest=Path(args.manifest),
        output=Path(args.output),
        extract_root=args.extract_root,
        root=root,
        start_index=args.start_index,
        limit=args.limit,
        path_fields=args.path_field or DEFAULT_PATH_FIELDS,
        lora_weights=Path(args.lora_weights) if args.lora_weights else None,
        run_script_path=Path(args.run_script) if args.run_script else None,
        include_run_script=args.include_run_script,
        colab_archive_path=args.colab_archive_path,
        colab_repo_dir=args.colab_repo_dir,
        repo_remote=args.repo_remote,
        repo_ref=args.repo_ref,
        run_name=args.run_name,
        modern_config=args.modern_config,
        cache_providers=args.cache_provider or [],
        skip_cache=args.skip_cache,
        cache_full=args.cache_full,
        eval_starts=args.eval_start or [40, 50],
        eval_limit=args.eval_limit,
        eval_steps=args.eval_steps,
        eval_guidance=args.eval_guidance,
        eval_inpaint_max_dimension=args.eval_inpaint_max_dimension,
        depth_provider=args.depth_provider,
        depth_model=args.depth_model,
        stl_target_dimension=args.stl_target_dimension,
        score_profile=args.score_profile,
        candidate_method=args.candidate_method,
        current_method=args.current_method,
        train_steps=args.train_steps,
        require_modern_cache=not args.no_require_modern_cache,
        require_image_to_mesh_providers=args.require_image_to_mesh_providers,
        cache_download_mode=args.cache_download_mode,
        cache_max_workers=args.cache_max_workers,
        max_method_failures=args.max_method_failures,
        min_paired_n=args.min_paired_n,
        max_mesh_surface_chamfer_ratio_vs_current=(
            args.max_mesh_surface_chamfer_ratio_vs_current
        ),
        max_mesh_surface_hausdorff95_ratio_vs_current=(
            args.max_mesh_surface_hausdorff95_ratio_vs_current
        ),
        allow_missing_split_audit=args.allow_missing_split_audit,
        contact_sheet_methods=args.contact_sheet_methods,
        contact_sheet_max_samples=args.contact_sheet_max_samples,
        colab_require_gpu_name_regex=args.colab_require_gpu_name_regex,
        colab_min_gpu_memory_gb=args.colab_min_gpu_memory_gb,
        include_triposr_setup=args.include_triposr_setup,
        include_triposg_setup=args.include_triposg_setup,
        include_pixal3d_setup=args.include_pixal3d_setup,
        include_trellis2_setup=args.include_trellis2_setup,
        include_step1x3d_setup=args.include_step1x3d_setup,
        include_hunyuan3d_2mv_setup=args.include_hunyuan3d_2mv_setup,
        include_hunyuan3d_setup=args.include_hunyuan3d_setup,
        colab_env=parse_colab_env(args.colab_env),
        report_path=Path(args.report) if args.report else None,
        inline_colab_launcher_path=Path(args.inline_colab_launcher) if args.inline_colab_launcher else None,
        inline_colab_chunk_size=args.inline_colab_chunk_size,
        fetch_colab_launcher_path=Path(args.fetch_colab_launcher) if args.fetch_colab_launcher else None,
        fetch_colab_notebook_path=Path(args.fetch_colab_notebook) if args.fetch_colab_notebook else None,
        fetch_colab_payload_url=args.fetch_colab_payload_url,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
