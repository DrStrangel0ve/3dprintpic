"""Launch one checksum-pinned private live API relief request.

The launcher deliberately omits the two background form fields so the endpoint
defaults are exercised. Its response and request record must remain ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx

from backend.benchmark.run_private_background_photo_detail_replay import (
    _require_ignored,
    _sha256,
    _validate_scene,
)


FORM_FIELDS = {
    "selection_job_id": None,
    "target_dimension": "512",
    "z_scale": "30",
    "max_xy_size": "128",
    "sigma": "0.35",
}
OMITTED_BACKGROUND_FIELDS = (
    "background_photo_detail_mm",
    "selection_background_depth_ratio",
)


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return payload


def _response_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run(
    scene_config: str | Path,
    clean_repository: str | Path,
    output_dir: str | Path,
    *,
    base_url: str = "http://127.0.0.1:8005",
    scene_label: str = "scene-01",
    timeout_seconds: float = 900.0,
) -> dict:
    repository = Path(__file__).resolve().parents[2]
    clean_repository = Path(clean_repository).resolve()
    output_dir = Path(output_dir).resolve()
    config_path = Path(scene_config).resolve()
    _require_ignored(config_path, repository)
    output_dir.mkdir(parents=True, exist_ok=True)
    _require_ignored(output_dir, clean_repository)

    config = _load_json(config_path)
    matches = [
        scene
        for scene in config.get("scenes", [])
        if scene.get("label") == scene_label
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one private scene labeled {scene_label}")
    scene = _validate_scene(matches[0], repository)
    selection_metadata = _load_json(Path(scene["selection_metadata"]))
    selection_id = selection_metadata.get("job_id")
    if not isinstance(selection_id, str):
        raise ValueError("Private selection metadata has no job identifier")
    clean_selection = clean_repository / "backend" / "output" / "selection" / selection_id
    source_path = clean_selection / "source.png"
    mask_path = clean_selection / "selection_mask.png"
    if (
        not source_path.is_file()
        or not mask_path.is_file()
        or _sha256(source_path) != _sha256(Path(scene["source_image"]))
        or _sha256(mask_path) != _sha256(Path(scene["selection_mask"]))
    ):
        raise ValueError("Clean API selection bundle does not match pinned private input")

    form = dict(FORM_FIELDS)
    form["selection_job_id"] = selection_id
    if any(field in form for field in OMITTED_BACKGROUND_FIELDS):
        raise AssertionError("Launcher must omit background fields")
    endpoint = base_url.rstrip("/") + "/process_image"
    with source_path.open("rb") as source_handle:
        response = httpx.post(
            endpoint,
            files={"file": ("source.png", source_handle, "image/png")},
            data=form,
            timeout=float(timeout_seconds),
        )
    response_path = output_dir / "response.json"
    response_path.write_bytes(response.content)
    response_payload = _load_json(response_path)
    response_job_id = response_payload.get("job_id")
    if response.status_code != 200 or not isinstance(response_job_id, str):
        raise RuntimeError(
            f"Live API request failed with HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    request_record = {
        "schema_version": 1,
        "scene_label": scene_label,
        "endpoint": "/process_image",
        "posted_form_fields": sorted(form),
        "omitted_background_fields": list(OMITTED_BACKGROUND_FIELDS),
        "http_status": int(response.status_code),
        "response_sha256": _response_sha256(response.content),
        "response_job_id": response_job_id,
    }
    record_path = output_dir / "request_record.json"
    record_path.write_text(
        json.dumps(request_record, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "http_status": int(response.status_code),
        "scene_label": scene_label,
        "posted_form_fields": sorted(form),
        "omitted_background_fields": list(OMITTED_BACKGROUND_FIELDS),
        "response_path": str(response_path),
        "request_record_path": str(record_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--scene-label", default="scene-01")
    parser.add_argument("--clean-repository", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    result = run(
        args.scene_config,
        args.clean_repository,
        args.output_dir,
        base_url=args.base_url,
        scene_label=args.scene_label,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({key: result[key] for key in ("http_status", "scene_label")}, indent=2))


if __name__ == "__main__":
    main()
