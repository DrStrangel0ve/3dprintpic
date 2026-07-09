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
        "for name in ('torch', 'transformers', 'trimesh', 'tsr'):\n"
        "    importlib.import_module(name)\n"
        "import torchmcubes\n"
        "PY\n"
        "then\n"
        "  \"$TRIPOSR_VENV/bin/python\" -m pip install -U pip setuptools wheel\n"
        "  \"$TRIPOSR_VENV/bin/python\" -m pip install numpy==2.0.2 omegaconf==2.3.0 Pillow==10.1.0 einops==0.7.0 transformers==4.35.0 trimesh==4.12.2 huggingface-hub imageio git+https://github.com/tatsy/torchmcubes.git\n"
        "fi\n"
    )


def build_triposg_setup_prelude() -> str:
    return (
        "TRIPOSG_DIR=\"${TRIPOSG_DIR:-/content/TripoSG}\"\n"
        "TRIPOSG_VENV=\"${TRIPOSG_VENV:-/content/triposg-venv}\"\n"
        "TRIPOSG_REF=\"${TRIPOSG_REF:-fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c}\"\n"
        "export TRIPOSG_DIR TRIPOSG_VENV TRIPOSG_REF\n"
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
        "python -m pip install -q virtualenv\n"
        "if [[ ! -x \"$TRIPOSG_VENV/bin/python\" ]]; then\n"
        "  python -m virtualenv --system-site-packages \"$TRIPOSG_VENV\"\n"
        "fi\n"
        "export PYTHONPATH=\"$TRIPOSG_DIR:${PYTHONPATH:-}\"\n"
        "if ! \"$TRIPOSG_VENV/bin/python\" - <<'PY'\n"
        "import os\n"
        "import importlib\n"
        "missing = []\n"
        "required = ['torch', 'diffusers', 'transformers', 'trimesh', 'triposg.pipelines.pipeline_triposg']\n"
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
        "  \"$TRIPOSG_VENV/bin/python\" -m pip install numpy==2.0.2 trimesh==4.12.2 Pillow==10.1.0 huggingface-hub imageio packaging ninja\n"
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
        "required = ['torch', 'diffusers', 'transformers', 'trimesh', 'triposg.pipelines.pipeline_triposg']\n"
        "if os.environ.get('TRIPOSG_INSTALL_DISO') == '1' or os.environ.get('TRIPOSG_USE_FLASH_DECODER') == '1':\n"
        "    required.append('diso')\n"
        "for name in required:\n"
        "    importlib.import_module(name)\n"
        "PY\n"
        "fi\n"
        "echo \"TripoSG setup checkpoint: Python deps importable\"\n"
        "(cd \"$TRIPOSG_DIR\" && \"$TRIPOSG_VENV/bin/python\" -m scripts.inference_triposg --help >/tmp/triposg_inference_help.txt)\n"
        "echo \"TripoSG setup checkpoint: CLI imports ok\"\n"
        "if [[ \"${TRIPOSG_PREFETCH:-0}\" == \"1\" ]]; then\n"
        "  \"$TRIPOSG_VENV/bin/python\" - <<'PY'\n"
        "import torch\n"
        "from huggingface_hub import snapshot_download\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('TripoSG prefetch requires CUDA, but torch.cuda.is_available() is false')\n"
        "props = torch.cuda.get_device_properties(0)\n"
        "total_gb = props.total_memory / (1024 ** 3)\n"
        "print(f'TripoSG CUDA device: {props.name}, VRAM={total_gb:.1f} GB')\n"
        "if total_gb < 8:\n"
        "    raise SystemExit(f'TripoSG needs at least about 8 GB VRAM, got {total_gb:.1f} GB')\n"
        "for repo_id in ('VAST-AI/TripoSG', 'briaai/RMBG-1.4'):\n"
        "    path = snapshot_download(repo_id=repo_id)\n"
        "    print(f'TripoSG prefetched {repo_id}: {path}')\n"
        "PY\n"
        "  echo \"TripoSG setup checkpoint: weights prefetched\"\n"
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
        "for name in ('torch', 'diffusers', 'transformers', 'accelerate', 'trimesh', 'pymeshlab', 'hy3dshape.pipelines'):\n"
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
        "  \"${HUNYUAN3D_PIP_INSTALL[@]}\" einops==0.8.0 omegaconf==2.3.0 pyyaml==6.0.2 tqdm==4.66.5 opencv-python==4.10.0.84 scikit-image==0.24.0 trimesh==4.12.2 pygltflib==1.16.3 xatlas==0.0.9\n"
        "  echo \"Installing Hunyuan3D Python deps: model helpers\"\n"
        "  \"${HUNYUAN3D_PIP_INSTALL[@]}\" timm torchdiffeq\n"
        "  echo \"Installing Hunyuan3D Python deps: pymeshlab\"\n"
        "  \"${HUNYUAN3D_PIP_INSTALL[@]}\" pymeshlab==2023.12.post3\n"
        "  \"$HUNYUAN3D_PYTHON\" - <<'PY'\n"
        "import importlib\n"
        "for name in ('torch', 'diffusers', 'transformers', 'accelerate', 'trimesh', 'pymeshlab', 'hy3dshape.pipelines'):\n"
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
    cache_download_mode: str,
    cache_max_workers: int,
    max_method_failures: int,
    min_paired_n: int,
    allow_missing_split_audit: bool,
    contact_sheet_methods: str | None,
    contact_sheet_max_samples: int | None,
    include_triposr_setup: bool = False,
    include_triposg_setup: bool = False,
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
    if include_hunyuan3d_setup:
        provider_setup += build_hunyuan3d_setup_prelude()

    lora_adapter_path = colab_path(lora_path, Path("pytorch_lora_weights.safetensors")) if lora_path else ""
    lora_report_path = colab_path(lora_path, Path("training_report.json")) if lora_path else ""
    preflight_path = colab_path(extract_root, Path("launch_preflight.json"))
    results_summary_path = colab_path(extract_root, Path("results_summary.json"))
    run_log_path = colab_path(extract_root, Path("run_colab_eval.log"))
    results_archive_path = f"/content/{run_name}_results.tar.gz"
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
        f"MANIFEST_PATH={shell_join([manifest_path])}\n"
        f"LORA_PATH={shell_join([lora_path or ''])}\n"
        f"LORA_ADAPTER_PATH={shell_join([lora_adapter_path])}\n"
        f"LORA_REPORT_PATH={shell_join([lora_report_path])}\n"
        f"PREFLIGHT_PATH={shell_join([preflight_path])}\n"
        "export ARCHIVE_PATH EXTRACT_ROOT REPO_DIR REPO_REMOTE REPO_REF RUN_NAME RUN_LOG OUTPUT_ROOT RESULTS_SUMMARY RESULTS_ARCHIVE MANIFEST_PATH LORA_PATH LORA_ADAPTER_PATH LORA_REPORT_PATH PREFLIGHT_PATH\n"
        "mkdir -p \"$EXTRACT_ROOT\" \"$(dirname \"$RUN_LOG\")\" \"$(dirname \"$RESULTS_SUMMARY\")\" \"$(dirname \"$RESULTS_ARCHIVE\")\"\n"
        "set +e\n"
        "(\n"
        "set -euo pipefail\n"
        "export PYTHONUNBUFFERED=\"${PYTHONUNBUFFERED:-1}\"\n"
        "export PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}\"\n"
        "export CUDA_MODULE_LOADING=\"${CUDA_MODULE_LOADING:-LAZY}\"\n"
        "export MALLOC_ARENA_MAX=\"${MALLOC_ARENA_MAX:-2}\"\n"
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
        "    'manifest': str(manifest),\n"
        "    'manifest_rows': sum(1 for line in manifest.read_text().splitlines() if line.strip()),\n"
        "}\n"
        "if os.environ.get('LORA_PATH'):\n"
        "    payload['lora_path'] = os.environ['LORA_PATH']\n"
        "    payload['lora_adapter'] = os.environ['LORA_ADAPTER_PATH']\n"
        "print(json.dumps(payload, indent=2, sort_keys=True))\n"
        "PY\n"
        "if [[ \"${COLAB_PROVIDER_SETUP_ONLY:-0}\" == \"1\" || \"${HUNYUAN3D_SETUP_ONLY:-0}\" == \"1\" || \"${TRIPOSR_SETUP_ONLY:-0}\" == \"1\" || \"${TRIPOSG_SETUP_ONLY:-0}\" == \"1\" ]]; then\n"
        "  echo \"Provider setup only requested; skipping benchmark stages\"\n"
        "  exit 0\n"
        "fi\n"
        f"{shell_join(command)}\n"
        ") 2>&1 | tee \"$RUN_LOG\"\n"
        "run_status=\"${PIPESTATUS[0]}\"\n"
        "set -e\n"
        "export RUN_STATUS=\"$run_status\"\n"
        "python - <<'PY'\n"
        "from datetime import datetime, timezone\n"
        "import csv, json, os, pathlib, tarfile\n"
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
        "def compact_methods(rows: list[dict]) -> list[dict]:\n"
        "    keys = [\n"
        "        'method', 'base_method', 'stl_mode', 'n', 'attempted_n', 'success_rate',\n"
        "        'error_count', 'logged_failure_count', 'rank_score',\n"
        "        'mesh_surface_chamfer_l1_median', 'mesh_surface_hausdorff95_median',\n"
        "        'silhouette_iou_masked_median', 'stl_exists_median', 'stl_is_watertight_median',\n"
        "        'stl_is_volume_median', 'stl_is_manifold_median', 'stl_positive_volume_median',\n"
        "        'stl_single_component_median', 'stl_nonmanifold_edge_count_log1p_median',\n"
        "        'stl_degenerate_face_ratio_median', 'stl_component_excess_log1p_median',\n"
        "        'stl_bbox_aspect_ratio_median', 'stl_faces_per_bbox_volume_log1p_median',\n"
        "    ]\n"
        "    return [{key: row.get(key, '') for key in keys if key in row} for row in rows]\n"
        "\n"
        "def summarize_benchmark_dir(path: pathlib.Path) -> dict:\n"
        "    aggregate = path / 'aggregate_summary.csv'\n"
        "    ranked = path / 'ranked_experiments.csv'\n"
        "    selection = path / 'selection_decision.json'\n"
        "    contact_sheet = path / 'artifact_contact_sheet.png'\n"
        "    aggregate_rows = read_csv_rows(aggregate)\n"
        "    ranked_rows = read_csv_rows(ranked)\n"
        "    selection_json = read_json_object(selection)\n"
        "    digest = {\n"
        "        'path': str(path),\n"
        "        'aggregate_summary_exists': aggregate.exists(),\n"
        "        'ranked_experiments_exists': ranked.exists(),\n"
        "        'selection_decision_exists': selection.exists(),\n"
        "        'contact_sheet_exists': contact_sheet.exists(),\n"
        "        'methods': compact_methods(ranked_rows or aggregate_rows),\n"
        "    }\n"
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
        "output_root = pathlib.Path(os.environ['OUTPUT_ROOT'])\n"
        "run_log = pathlib.Path(os.environ['RUN_LOG'])\n"
        "preflight = pathlib.Path(os.environ['PREFLIGHT_PATH'])\n"
        "summary_path = pathlib.Path(os.environ['RESULTS_SUMMARY'])\n"
        "archive_path = pathlib.Path(os.environ['RESULTS_ARCHIVE'])\n"
        "orchestrator_result = output_root / 'orchestrator_result.json'\n"
        "selection_decisions = sorted(output_root.glob('combined/*/selection_decision.json')) if output_root.exists() else []\n"
        "eval_summaries = [summarize_benchmark_dir(path) for path in sorted(output_root.glob('experiments/*'))] if output_root.exists() else []\n"
        "combined_summaries = [summarize_benchmark_dir(path) for path in sorted(output_root.glob('combined/*'))] if output_root.exists() else []\n"
        "summary = {\n"
        "    'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),\n"
        "    'run_name': os.environ['RUN_NAME'],\n"
        "    'run_status': int(os.environ['RUN_STATUS']),\n"
        "    'provider_setup_only': os.environ.get('COLAB_PROVIDER_SETUP_ONLY') == '1' or os.environ.get('HUNYUAN3D_SETUP_ONLY') == '1' or os.environ.get('TRIPOSR_SETUP_ONLY') == '1' or os.environ.get('TRIPOSG_SETUP_ONLY') == '1',\n"
        "    'output_root': str(output_root),\n"
        "    'output_root_exists': output_root.exists(),\n"
        "    'run_log': str(run_log),\n"
        "    'preflight': str(preflight),\n"
        "    'results_archive': str(archive_path),\n"
        "    'orchestrator_result': str(orchestrator_result) if orchestrator_result.exists() else '',\n"
        "    'selection_decisions': [str(path) for path in selection_decisions],\n"
        "    'eval_summaries': eval_summaries,\n"
        "    'combined_summaries': combined_summaries,\n"
        "}\n"
        "summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "archive_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "with tarfile.open(archive_path, 'w:gz') as tar:\n"
        "    add_if_exists(tar, preflight, 'launch_preflight.json')\n"
        "    add_if_exists(tar, run_log, 'run_colab_eval.log')\n"
        "    add_if_exists(tar, summary_path, 'results_summary.json')\n"
        "    if output_root.exists():\n"
        "        tar.add(output_root, arcname=f\"output/{output_root.name}\")\n"
        "print(json.dumps(summary, indent=2, sort_keys=True))\n"
        "PY\n"
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
    script = (
        "# Paste this into one Colab Python cell to upload and launch the packaged benchmark.\n"
        "import base64\n"
        "import hashlib\n"
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
        "subprocess.run(['bash', str(run_script), str(ARCHIVE_PATH)], check=True, env=env)\n"
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
    return (
        "# Paste this into one Colab Python cell to download and launch the packaged benchmark.\n"
        "import hashlib\n"
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
        "subprocess.run(['bash', str(run_script), str(ARCHIVE_PATH)], check=True, env=env)\n"
    )


def build_colab_notebook_launcher(*, launcher_text: str, title: str) -> str:
    if not launcher_text.strip():
        raise ValueError("launcher_text must not be empty")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
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
    cache_download_mode: str = "snapshot",
    cache_max_workers: int = 8,
    max_method_failures: int = 2,
    min_paired_n: int = 5,
    allow_missing_split_audit: bool = False,
    contact_sheet_methods: str | None = None,
    contact_sheet_max_samples: int | None = None,
    include_triposr_setup: bool = False,
    include_triposg_setup: bool = False,
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
                cache_download_mode=cache_download_mode,
                cache_max_workers=cache_max_workers,
                max_method_failures=max_method_failures,
                min_paired_n=min_paired_n,
                allow_missing_split_audit=allow_missing_split_audit,
                contact_sheet_methods=contact_sheet_methods,
                contact_sheet_max_samples=contact_sheet_max_samples,
                include_triposr_setup=include_triposr_setup,
                include_triposg_setup=include_triposg_setup,
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
        "allow_missing_split_audit": allow_missing_split_audit,
        "contact_sheet_methods": contact_sheet_methods or "",
        "contact_sheet_max_samples": contact_sheet_max_samples,
        "include_triposr_setup": include_triposr_setup,
        "include_triposg_setup": include_triposg_setup,
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
    parser.add_argument("--cache-download-mode", choices=("files", "snapshot"), default="snapshot")
    parser.add_argument("--cache-max-workers", type=int, default=8)
    parser.add_argument("--max-method-failures", type=int, default=2)
    parser.add_argument("--min-paired-n", type=int, default=5)
    parser.add_argument("--allow-missing-split-audit", action="store_true")
    parser.add_argument("--contact-sheet-methods", default=None)
    parser.add_argument("--contact-sheet-max-samples", type=int, default=None)
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
        "--include-hunyuan3d-setup",
        action="store_true",
        help="Embed a Colab setup prelude for /content/Hunyuan3D-2.1 and /content/hunyuan3d-venv before running eval.",
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
        cache_download_mode=args.cache_download_mode,
        cache_max_workers=args.cache_max_workers,
        max_method_failures=args.max_method_failures,
        min_paired_n=args.min_paired_n,
        allow_missing_split_audit=args.allow_missing_split_audit,
        contact_sheet_methods=args.contact_sheet_methods,
        contact_sheet_max_samples=args.contact_sheet_max_samples,
        include_triposr_setup=args.include_triposr_setup,
        include_triposg_setup=args.include_triposg_setup,
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
