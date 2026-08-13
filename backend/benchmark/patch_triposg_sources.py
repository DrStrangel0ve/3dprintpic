from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from backend.benchmark.triposg_models import DEFAULT_TRIPOSG_MODEL, DEFAULT_TRIPOSG_REMBG_MODEL


def patch_triposg_sources(root: Path) -> dict:
    inference_path = root / "scripts" / "inference_triposg.py"
    vae_path = root / "triposg" / "models" / "autoencoders" / "autoencoder_kl_triposg.py"
    if not inference_path.is_file():
        raise FileNotFoundError(f"Expected official TripoSG inference source under {root}")
    if not vae_path.is_file():
        raise FileNotFoundError(f"Expected official TripoSG VAE source under {root}")

    text = inference_path.read_text(encoding="utf-8")
    snapshot_replacements = (
        (
            f'snapshot_download(repo_id="{DEFAULT_TRIPOSG_MODEL}", local_dir=triposg_weights_dir)',
            (
                f'snapshot_download(repo_id="{DEFAULT_TRIPOSG_MODEL}", '
                "revision=os.environ['TRIPOSG_MODEL_REVISION'], local_dir=triposg_weights_dir, "
                "local_files_only=os.environ.get('TRIPOSG_HF_LOCAL_ONLY') == '1')"
            ),
            DEFAULT_TRIPOSG_MODEL,
            "TRIPOSG_MODEL_REVISION",
        ),
        (
            f'snapshot_download(repo_id="{DEFAULT_TRIPOSG_REMBG_MODEL}", local_dir=rmbg_weights_dir)',
            (
                f'snapshot_download(repo_id="{DEFAULT_TRIPOSG_REMBG_MODEL}", '
                "revision=os.environ['TRIPOSG_REMBG_REVISION'], local_dir=rmbg_weights_dir, "
                "local_files_only=os.environ.get('TRIPOSG_HF_LOCAL_ONLY') == '1')"
            ),
            DEFAULT_TRIPOSG_REMBG_MODEL,
            "TRIPOSG_REMBG_REVISION",
        ),
    )

    def snapshot_is_pinned(repo_id: str, revision_env: str) -> bool:
        marker = f'repo_id="{repo_id}"'
        offset = text.find(marker)
        if offset < 0:
            return False
        call_window = text[offset : offset + 700]
        return (
            f"revision=os.environ['{revision_env}']" in call_window
            and "local_files_only=os.environ.get('TRIPOSG_HF_LOCAL_ONLY') == '1'"
            in call_window
        )

    for needle, replacement, repo_id, revision_env in snapshot_replacements:
        if replacement in text or snapshot_is_pinned(repo_id, revision_env):
            continue
        if text.count(needle) != 1:
            raise RuntimeError(f"TripoSG snapshot patch target not found exactly once: {needle}")
        text = text.replace(needle, replacement, 1)

    dtype_needle = (
        "pipe: TripoSGPipeline = "
        "TripoSGPipeline.from_pretrained(triposg_weights_dir).to(device, dtype)"
    )
    dtype_replacement = (
        "pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained("
        "triposg_weights_dir, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)"
    )
    if dtype_replacement not in text and not (
        "torch_dtype=dtype" in text and "low_cpu_mem_usage=True" in text
    ):
        if text.count(dtype_needle) != 1:
            raise RuntimeError("TripoSG load-time dtype patch target not found exactly once")
        text = text.replace(dtype_needle, dtype_replacement, 1)

    inference_path.write_text(text, encoding="utf-8")
    vae_text = vae_path.read_text(encoding="utf-8")
    vae_needle = "            q = self.proj_query(q)"
    vae_replacement = (
        "            q = self.proj_query("
        "q.to(dtype=self.proj_query.weight.dtype))"
    )
    if vae_replacement not in vae_text:
        if vae_text.count(vae_needle) != 1:
            raise RuntimeError("TripoSG VAE query dtype patch target not found exactly once")
        vae_text = vae_text.replace(vae_needle, vae_replacement, 1)
    vae_path.write_text(vae_text, encoding="utf-8")
    return {
        "inference": str(inference_path),
        "inference_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "vae": str(vae_path),
        "vae_sha256": hashlib.sha256(vae_text.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch immutable model revisions into the official TripoSG CLI.")
    parser.add_argument("--triposg-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(patch_triposg_sources(args.triposg_dir.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
