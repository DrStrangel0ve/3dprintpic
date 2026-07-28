from __future__ import annotations


DEFAULT_PIXAL3D_MODEL = "TencentARC/Pixal3D"
DEFAULT_PIXAL3D_MODEL_REVISION = "0b31f9160aa400719af409098bff7936a932f726"
DEFAULT_PIXAL3D_MOGE_MODEL = "Ruicheng/moge-2-vitl"
DEFAULT_PIXAL3D_MOGE_REVISION = "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"
DEFAULT_PIXAL3D_DINOV3_MODEL = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_PIXAL3D_DINOV3_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
DEFAULT_PIXAL3D_REMBG_MODEL = "ZhengPeng7/BiRefNet"
DEFAULT_PIXAL3D_REMBG_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"


def pixal3d_model_specs(
    *,
    model_repo: str = DEFAULT_PIXAL3D_MODEL,
    model_revision: str = DEFAULT_PIXAL3D_MODEL_REVISION,
    moge_revision: str = DEFAULT_PIXAL3D_MOGE_REVISION,
    dinov3_revision: str = DEFAULT_PIXAL3D_DINOV3_REVISION,
    rembg_repo: str = DEFAULT_PIXAL3D_REMBG_MODEL,
    rembg_revision: str = DEFAULT_PIXAL3D_REMBG_REVISION,
) -> dict[str, dict[str, str]]:
    return {
        "pixal3d": {
            "repo_id": model_repo,
            "revision": model_revision,
            "required_file": "pipeline.json",
        },
        "moge": {
            "repo_id": DEFAULT_PIXAL3D_MOGE_MODEL,
            "revision": moge_revision,
            "required_file": "model.pt",
        },
        "dinov3": {
            "repo_id": DEFAULT_PIXAL3D_DINOV3_MODEL,
            "revision": dinov3_revision,
            "required_file": "config.json",
        },
        "rembg": {
            "repo_id": rembg_repo,
            "revision": rembg_revision,
            "required_file": "config.json",
        },
    }
