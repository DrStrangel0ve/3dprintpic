from __future__ import annotations


DEFAULT_TRIPOSG_MODEL = "VAST-AI/TripoSG"
DEFAULT_TRIPOSG_MODEL_REVISION = "2c1c516d22d58db486a058d98d31bb6177344e06"
DEFAULT_TRIPOSG_REMBG_MODEL = "briaai/RMBG-1.4"
DEFAULT_TRIPOSG_REMBG_REVISION = "2ceba5a5efaec153162aedea169f76caf9b46cf8"


def triposg_model_specs(
    *,
    model_revision: str = DEFAULT_TRIPOSG_MODEL_REVISION,
    rembg_revision: str = DEFAULT_TRIPOSG_REMBG_REVISION,
) -> dict[str, dict[str, str]]:
    return {
        "triposg": {
            "repo_id": DEFAULT_TRIPOSG_MODEL,
            "revision": model_revision,
        },
        "rembg": {
            "repo_id": DEFAULT_TRIPOSG_REMBG_MODEL,
            "revision": rembg_revision,
        },
    }
