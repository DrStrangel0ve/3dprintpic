from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

from backend.pic_to_3d import MODERN_INPAINT_MODELS


SDXL_FP16_ALLOW_PATTERNS = [
    "*.json",
    "**/*.json",
    "*.txt",
    "**/*.txt",
    "*.model",
    "**/*.model",
    "*.fp16.safetensors",
    "**/*.fp16.safetensors",
]

SD15_FP16_ALLOW_PATTERNS = [
    "*.json",
    "**/*.json",
    "*.txt",
    "**/*.txt",
    "*.model",
    "**/*.model",
    "merges.txt",
    "**/merges.txt",
    "vocab.json",
    "**/vocab.json",
    "tokenizer/**",
    "scheduler/**",
    "text_encoder/model.fp16.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]

AMUSED_FP16_ALLOW_PATTERNS = [
    "*.json",
    "**/*.json",
    "*.txt",
    "**/*.txt",
    "*.model",
    "**/*.model",
    "tokenizer/**",
    "scheduler/**",
    "text_encoder/model.fp16.safetensors",
    "transformer/diffusion_pytorch_model.fp16.safetensors",
    "vqvae/diffusion_pytorch_model.fp16.safetensors",
]


def provider_allow_patterns(provider: str, full: bool) -> list[str] | None:
    if full:
        return None
    if provider == "sdxl-inpaint":
        return SDXL_FP16_ALLOW_PATTERNS
    if provider == "dreamshaper-inpaint":
        return SD15_FP16_ALLOW_PATTERNS
    if provider == "amused-inpaint":
        return AMUSED_FP16_ALLOW_PATTERNS
    return None


def pattern_matches(path: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def cached_file_path(model_id: str, filename: str, revision: str | None, local_dir: str | None) -> str | None:
    if local_dir:
        path = Path(local_dir) / filename
        return str(path) if path.exists() else None

    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.file_download import _CACHED_NO_EXIST

    cached = try_to_load_from_cache(model_id, filename, revision=revision)
    if cached in (None, _CACHED_NO_EXIST):
        return None
    path = Path(str(cached))
    return str(path) if path.exists() else None


def attach_cache_status(plan: dict[str, Any], revision: str | None, local_dir: str | None) -> dict[str, Any]:
    cached_files = []
    missing_files = []
    cached_size = 0
    missing_size = 0
    for file_info in plan["selected_files"]:
        path = cached_file_path(plan["model"], file_info["path"], revision=revision, local_dir=local_dir)
        if path:
            cached_size += int(file_info["size"])
            cached_files.append({**file_info, "cached_path": path})
        else:
            missing_size += int(file_info["size"])
            missing_files.append(file_info)

    plan["cache_status"] = {
        "revision": revision or "main",
        "local_dir": local_dir or "",
        "cached_file_count": len(cached_files),
        "cached_size_gb": round(cached_size / 1024**3, 3),
        "missing_file_count": len(missing_files),
        "missing_size_gb": round(missing_size / 1024**3, 3),
        "complete": len(missing_files) == 0,
        "cached_files": cached_files,
        "missing_files": missing_files,
    }
    return plan


def provider_plan(
    provider: str,
    full: bool,
    revision: str | None,
    local_dir: str | None,
    model_name: str | None = None,
) -> dict:
    from huggingface_hub import HfApi

    if provider not in MODERN_INPAINT_MODELS:
        raise ValueError(f"Unsupported provider: {provider}")

    provider_config = MODERN_INPAINT_MODELS[provider]
    model_id = model_name or provider_config["model"]
    patterns = provider_allow_patterns(provider, full=full)
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    siblings = info.siblings or []
    selected = [sibling for sibling in siblings if pattern_matches(sibling.rfilename, patterns)]
    total_size = sum((sibling.size or 0) for sibling in siblings)
    selected_size = sum((sibling.size or 0) for sibling in selected)
    plan = {
        "provider": provider,
        "model": model_id,
        "pipeline_class": provider_config.get("pipeline_class", ""),
        "full": full,
        "allow_patterns": patterns,
        "repo_file_count": len(siblings),
        "repo_size_gb": round(total_size / 1024**3, 3),
        "selected_file_count": len(selected),
        "selected_size_gb": round(selected_size / 1024**3, 3),
        "selected_files": [
            {"path": sibling.rfilename, "size": sibling.size or 0}
            for sibling in sorted(selected, key=lambda item: item.rfilename)
        ],
    }
    return attach_cache_status(plan, revision=revision, local_dir=local_dir)


def download_provider_snapshot(plan: dict, revision: str | None, local_dir: str | None, max_workers: int) -> str:
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": plan["model"],
        "allow_patterns": plan["allow_patterns"],
        "revision": revision,
        "max_workers": max_workers,
    }
    if local_dir:
        kwargs["local_dir"] = local_dir
    return snapshot_download(**kwargs)


def download_provider_files(
    plan: dict,
    revision: str | None,
    local_dir: str | None,
    max_files: int | None,
    max_gb: float | None,
) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    downloaded = []
    budget_bytes = None if max_gb is None else int(max_gb * 1024**3)
    used_budget = 0
    missing = sorted(
        plan.get("cache_status", {}).get("missing_files", []),
        key=lambda item: (int(item["size"]), item["path"]),
    )
    for file_info in missing:
        if max_files is not None and len(downloaded) >= max_files:
            break
        size = int(file_info["size"])
        if budget_bytes is not None and used_budget + size > budget_bytes:
            break

        filename = file_info["path"]
        print(f"Downloading {filename} ({size / 1024**3:.3f} GB)", file=sys.stderr)
        kwargs = {
            "repo_id": plan["model"],
            "filename": filename,
            "revision": revision,
        }
        if local_dir:
            kwargs["local_dir"] = local_dir
        path = hf_hub_download(**kwargs)
        downloaded.append({**file_info, "cached_path": path})
        used_budget += size
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or download cached weights for a modern completion provider."
    )
    parser.add_argument("provider", choices=sorted(MODERN_INPAINT_MODELS))
    parser.add_argument("--full", action="store_true", help="Download every file instead of the provider-specific light plan.")
    parser.add_argument("--download", action="store_true", help="Actually download the selected files. Omit for dry-run planning.")
    parser.add_argument(
        "--download-mode",
        choices=("files", "snapshot"),
        default="files",
        help="Use sequential per-file downloads by default for clearer resumability.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="For --download-mode files, download at most this many missing files.")
    parser.add_argument("--max-gb", type=float, default=None, help="For --download-mode files, stop before exceeding this missing-file budget.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-dir", default=None, help="Optional destination directory. Defaults to the Hugging Face cache.")
    parser.add_argument("--model-name", default=None, help="Optional Hugging Face model id override for this provider.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel download workers for snapshot_download.")
    parser.add_argument("--output", default=None, help="Optional JSON plan output path.")
    args = parser.parse_args()

    plan = provider_plan(
        args.provider,
        full=args.full,
        revision=args.revision,
        local_dir=args.local_dir,
        model_name=args.model_name,
    )
    if args.download:
        if args.download_mode == "snapshot":
            plan["snapshot_path"] = download_provider_snapshot(
                plan,
                revision=args.revision,
                local_dir=args.local_dir,
                max_workers=max(args.max_workers, 1),
            )
        else:
            plan["downloaded_files"] = download_provider_files(
                plan,
                revision=args.revision,
                local_dir=args.local_dir,
                max_files=args.max_files,
                max_gb=args.max_gb,
            )
        plan = attach_cache_status(plan, revision=args.revision, local_dir=args.local_dir)

    text = json.dumps(plan, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
