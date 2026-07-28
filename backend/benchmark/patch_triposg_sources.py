from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from backend.benchmark.triposg_models import DEFAULT_TRIPOSG_MODEL, DEFAULT_TRIPOSG_REMBG_MODEL


def patch_triposg_sources(root: Path) -> dict:
    inference_path = root / "scripts" / "inference_triposg.py"
    if not inference_path.is_file():
        raise FileNotFoundError(f"Expected official TripoSG inference source under {root}")

    text = inference_path.read_text(encoding="utf-8")
    replacements = (
        (
            f'snapshot_download(repo_id="{DEFAULT_TRIPOSG_MODEL}", local_dir=triposg_weights_dir)',
            (
                f'snapshot_download(repo_id="{DEFAULT_TRIPOSG_MODEL}", '
                "revision=os.environ['TRIPOSG_MODEL_REVISION'], local_dir=triposg_weights_dir, "
                "local_files_only=os.environ.get('TRIPOSG_HF_LOCAL_ONLY') == '1')"
            ),
        ),
        (
            f'snapshot_download(repo_id="{DEFAULT_TRIPOSG_REMBG_MODEL}", local_dir=rmbg_weights_dir)',
            (
                f'snapshot_download(repo_id="{DEFAULT_TRIPOSG_REMBG_MODEL}", '
                "revision=os.environ['TRIPOSG_REMBG_REVISION'], local_dir=rmbg_weights_dir, "
                "local_files_only=os.environ.get('TRIPOSG_HF_LOCAL_ONLY') == '1')"
            ),
        ),
    )
    for needle, replacement in replacements:
        if replacement in text:
            continue
        if text.count(needle) != 1:
            raise RuntimeError(f"TripoSG snapshot patch target not found exactly once: {needle}")
        text = text.replace(needle, replacement, 1)

    inference_path.write_text(text, encoding="utf-8")
    return {
        "inference": str(inference_path),
        "inference_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch immutable model revisions into the official TripoSG CLI.")
    parser.add_argument("--triposg-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(patch_triposg_sources(args.triposg_dir.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
