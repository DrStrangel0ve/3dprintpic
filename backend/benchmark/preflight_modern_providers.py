from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.pic_to_3d import MODERN_INPAINT_MODELS


def model_index_class(model_id: str) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_id, "model_index.json")
    with open(path, encoding="utf-8") as file:
        return str(json.load(file).get("_class_name", ""))


def safe_model_index_class(model_id: str) -> tuple[str, str]:
    try:
        return model_index_class(model_id), ""
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def provider_rows(providers: list[str]) -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    rows = []
    for provider in providers:
        if provider not in MODERN_INPAINT_MODELS:
            raise ValueError(f"Unsupported provider: {provider}")
        provider_config = MODERN_INPAINT_MODELS[provider]
        model_id = provider_config["model"]
        row = {
            "provider": provider,
            "model": model_id,
            "intended_pipeline_class": provider_config.get("pipeline_class", ""),
        }
        try:
            info = api.model_info(model_id, files_metadata=True)
            total_size = sum((sibling.size or 0) for sibling in info.siblings or [])
            tags = info.tags or []
            diffusers_tags = [tag.split("diffusers:", 1)[1] for tag in tags if tag.startswith("diffusers:")]
            advertised_class, model_index_error = safe_model_index_class(model_id)
            advertised_classes = diffusers_tags + ([advertised_class] if advertised_class else [])
            row.update(
                {
                    "ok": True,
                    "private": bool(info.private),
                    "gated": getattr(info, "gated", None),
                    "disabled": getattr(info, "disabled", None),
                    "file_count": len(info.siblings or []),
                    "size_gb": round(total_size / 1024**3, 3),
                    "advertised_pipeline_classes": diffusers_tags,
                    "model_index_class": advertised_class,
                    "model_index_error": model_index_error,
                    "class_matches_advertised": row["intended_pipeline_class"] in set(advertised_classes),
                    "tags": tags,
                    "license": next((tag.split("license:", 1)[1] for tag in tags if tag.startswith("license:")), ""),
                }
            )
        except Exception as exc:
            row.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check access and approximate size for modern inpainting providers.")
    parser.add_argument("--providers", default="sdxl-inpaint,dreamshaper-inpaint,amused-inpaint,flux-fill,qwen-image-inpaint,qwen-image-edit")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    providers = [provider.strip() for provider in args.providers.split(",") if provider.strip()]
    rows = provider_rows(providers)
    for row in rows:
        print(json.dumps(row))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
