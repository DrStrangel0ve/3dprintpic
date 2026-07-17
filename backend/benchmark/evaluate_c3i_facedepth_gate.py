"""Evaluate the published FaceDepth checkpoint on the pinned C3I demo rows."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.benchmark.evaluate_c3i_synface_demo_gate import (
    _compact_quality,
    gate_decision,
    summarize_rows,
)
from backend.benchmark.c3i_synface_corpus import (
    DEMO_ASSET_LICENSE,
    verified_corpus_asset,
)
from backend.benchmark.train_face_depth_head import (
    MODEL_ID,
    MODEL_REVISION,
    _padded_box,
)
from backend.benchmark.train_face_surface_adapter import _sha256
from backend.benchmark.train_face_surface_fusion_adapter import (
    _detect_production_region,
    _infer_depth_pair,
    _production_baseline,
    selected_image_from_exact_mask,
)
from backend.benchmark.run_cc0_live_face_variation_matrix import (
    FACE_PART_NAMES,
    _exact_face_depth_quality,
)


METHOD = "c3i-synface-demo-facedepth-checkpoint-gate"
PROVIDER_REPOSITORY = "https://github.com/khan9048/Facial_depth_estimation"
PROVIDER_REVISION = "dc8adfffbfd38818b72d0ad3776eb66724fafe95"
PROVIDER_CHECKPOINT_URL = (
    "https://nuigalwayie-my.sharepoint.com/:u:/g/personal/"
    "f_khan4_nuigalway_ie/"
    "EepkuVajAhdIjZoQm5Weyx4BjXcEZy-uw5OWxxMXq1WJPA?e=rv3aSY"
)
MIRROR_REPOSITORY = "https://github.com/Yimin-zhou/2DImage-Relighting"
MIRROR_REVISION = "bdf553705d4c996fb25d1747a0a005d80fa8b06d"
MIRROR_CHECKPOINT_BLOB = "61ab3564afb9f2fe01aee84d6810aca38c6ad624"
EXPECTED_CHECKPOINT_BYTES = 57_906_985
EXPECTED_CHECKPOINT_SHA256 = (
    "95ac733d761459e7f2072ba2862467c4d605b4cf4ae90491a757d93553123b17"
)
INPUT_WIDTH = 640
INPUT_HEIGHT = 480
COLOR_ORDERS = ("rgb", "bgr")
INPUT_MODES = ("official-stretch", "native-aspect")


def checkpoint_preflight(checkpoint_path: str | Path) -> dict:
    checkpoint_path = Path(checkpoint_path).resolve()
    exists = checkpoint_path.is_file()
    size = checkpoint_path.stat().st_size if exists else None
    digest = _sha256(checkpoint_path) if exists else None
    checks = {
        "checkpoint_present": exists,
        "checkpoint_size_exact": size == EXPECTED_CHECKPOINT_BYTES,
        "checkpoint_sha256_exact": digest == EXPECTED_CHECKPOINT_SHA256,
    }
    return {
        "checkpoint_bytes": size,
        "checkpoint_sha256": digest,
        "expected_checkpoint_bytes": EXPECTED_CHECKPOINT_BYTES,
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checks": checks,
        "ready": bool(all(checks.values())),
    }


def _build_facedepth_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torchvision.models import mobilenet_v2

    class UpSample(nn.Module):
        def __init__(self, skip_input: int, output_features: int):
            super().__init__()
            self.convA = nn.Conv2d(
                skip_input,
                output_features,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.leakyreluA = nn.LeakyReLU(0.2)
            self.convB = nn.Conv2d(
                output_features,
                output_features,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.leakyreluB = nn.LeakyReLU(0.2)

        def forward(self, values, skip):
            upsampled = functional.interpolate(
                values,
                size=[skip.size(2), skip.size(3)],
                mode="bilinear",
                align_corners=True,
            )
            joined = torch.cat([upsampled, skip], dim=1)
            return self.leakyreluB(
                self.convB(self.leakyreluA(self.convA(joined)))
            )

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.original_model = mobilenet_v2(weights=None)

        def forward(self, values):
            features = [values]
            for layer in self.original_model.features._modules.values():
                features.append(layer(features[-1]))
            return features

    class Decoder(nn.Module):
        def __init__(self, num_features: int = 1280, decoder_width: float = 0.6):
            super().__init__()
            features = int(num_features * decoder_width)
            self.conv2 = nn.Conv2d(
                num_features,
                features,
                kernel_size=1,
                stride=1,
                padding=1,
            )
            self.up0 = UpSample(features + 320, features // 2)
            self.up1 = UpSample(features // 2 + 160, features // 2)
            self.up2 = UpSample(features // 2 + 64, features // 4)
            self.up3 = UpSample(features // 4 + 32, features // 8)
            self.up4 = UpSample(features // 8 + 24, features // 8)
            self.up5 = UpSample(features // 8 + 16, features // 16)
            self.conv3 = nn.Conv2d(
                features // 16,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        def forward(self, features):
            blocks = (
                features[2],
                features[4],
                features[6],
                features[9],
                features[15],
                features[18],
                features[19],
            )
            values = self.conv2(blocks[6])
            values = self.up0(values, blocks[5])
            values = self.up1(values, blocks[4])
            values = self.up2(values, blocks[3])
            values = self.up3(values, blocks[2])
            values = self.up4(values, blocks[1])
            values = self.up5(values, blocks[0])
            return self.conv3(values)

    class FaceDepthModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()

        def forward(self, values):
            return self.decoder(self.encoder(values))

    return FaceDepthModel()


def load_facedepth_model(checkpoint_path: str | Path, *, device: str):
    import torch

    preflight = checkpoint_preflight(checkpoint_path)
    if not preflight["ready"]:
        failed = [name for name, passed in preflight["checks"].items() if not passed]
        raise RuntimeError(
            "FaceDepth checkpoint preflight failed: " + ", ".join(failed)
        )
    model = _build_facedepth_model()
    state = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, preflight


def infer_facedepth(
    model,
    image: Image.Image,
    *,
    device: str,
    color_order: str,
    input_mode: str = "native-aspect",
) -> np.ndarray:
    import torch

    if color_order not in COLOR_ORDERS:
        raise ValueError(f"Unsupported FaceDepth color order: {color_order}")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unsupported FaceDepth input mode: {input_mode}")
    values = np.asarray(image.convert("RGB"), dtype=np.float32)
    if color_order == "bgr":
        values = values[..., ::-1].copy()
    if input_mode == "official-stretch":
        network_width, network_height = INPUT_WIDTH, INPUT_HEIGHT
    else:
        scale = INPUT_HEIGHT / float(max(image.width, image.height))
        network_width = max(
            32,
            int(round(image.width * scale / 32.0)) * 32,
        )
        network_height = max(
            32,
            int(round(image.height * scale / 32.0)) * 32,
        )
    resized = cv2.resize(
        values,
        (network_width, network_height),
        interpolation=cv2.INTER_CUBIC,
    )
    tensor = torch.from_numpy(
        resized.transpose(2, 0, 1)[None] / 255.0
    ).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        prediction = model(tensor)
        prediction = torch.nn.functional.interpolate(
            prediction,
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
    depth = prediction.float().cpu().numpy()
    finite = np.isfinite(depth)
    if not np.any(finite):
        raise RuntimeError("FaceDepth emitted no finite values")
    low = float(np.min(depth[finite]))
    high = float(np.max(depth[finite]))
    if high - low <= 1e-8:
        raise RuntimeError("FaceDepth emitted no usable depth span")
    return np.clip((depth - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _embed_crop(
    local_depth: np.ndarray,
    reference: np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    if x1 <= x0 or y1 <= y0:
        raise ValueError("FaceDepth crop box is empty")
    embedded = np.asarray(reference, dtype=np.float32).copy()
    embedded[y0:y1, x0:x1] = cv2.resize(
        np.asarray(local_depth, dtype=np.float32),
        (x1 - x0, y1 - y0),
        interpolation=cv2.INTER_CUBIC,
    )
    return embedded


def _measure_quality(
    candidate: np.ndarray,
    *,
    exact_path: Path,
    mask_path: Path,
    part_paths: dict[str, Path],
) -> dict:
    with tempfile.TemporaryDirectory(prefix="c3i-facedepth-quality-") as temporary:
        candidate_path = Path(temporary) / "candidate.npy"
        np.save(candidate_path, np.asarray(candidate, dtype=np.float32))
        return _compact_quality(
            _exact_face_depth_quality(
                candidate_path,
                exact_path,
                mask_path,
                expected_scale_sign=-1.0,
                part_mask_paths=part_paths,
            )
        )


def _candidate_decision(
    rows: list[dict],
    candidate_key: str,
    current_summary: dict,
    *,
    expected_row_count: int | None = None,
) -> dict:
    candidate_summary = summarize_rows(rows, candidate_key)
    comparison_rows = [
        {
            "row_id": row["row_id"],
            "global_depth": row["current_refined"],
            "current_refined": row[candidate_key],
        }
        for row in rows
    ]
    decision = gate_decision(
        current_summary,
        candidate_summary,
        rows=comparison_rows,
        expected_row_count=expected_row_count,
    )
    return {
        "summary": candidate_summary,
        "decision": decision,
    }


def promotion_eligible_candidate(candidate_key: str) -> bool:
    return candidate_key.endswith("_fused")


def evaluate_c3i_facedepth(
    corpus_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    limit: int | None = None,
) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    corpus_root = Path(corpus_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_root / "summary.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if corpus.get("provider") != "c3i-synface-official-demo-depth":
        raise ValueError("FaceDepth gate received an unsupported corpus provider")
    if corpus.get("demo_asset_license") != DEMO_ASSET_LICENSE:
        raise ValueError(
            "FaceDepth gate requires explicit demo-license provenance"
        )
    rows = list(corpus["rows"])
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("FaceDepth gate selected no rows")

    facedepth, checkpoint = load_facedepth_model(
        checkpoint_path,
        device=device,
    )
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    processor = AutoImageProcessor.from_pretrained(
        snapshot,
        local_files_only=True,
        use_fast=False,
    )
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    global_model = AutoModelForDepthEstimation.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    global_model.eval()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    measured = []
    for index, row in enumerate(rows, start=1):
        source_path = verified_corpus_asset(corpus_root, row["source"])
        mask_path = verified_corpus_asset(corpus_root, row["selection_mask"])
        exact_path = verified_corpus_asset(corpus_root, row["exact_depth"])
        part_paths = {
            name: verified_corpus_asset(
                corpus_root,
                row["exact_face_parts"][name],
            )
            for name in FACE_PART_NAMES
        }
        source = Image.open(source_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        selected = selected_image_from_exact_mask(source, mask)
        selection = np.asarray(mask) >= 128
        region, detection = _detect_production_region(
            np.asarray(selected),
            selection,
        )
        crop_box = _padded_box(
            region["bbox"],
            selected.width,
            selected.height,
            0.35,
        )
        crop = selected.crop(crop_box)
        global_depth, local_depth = _infer_depth_pair(
            processor,
            global_model,
            selected,
            crop,
            device=device,
            dtype=dtype,
        )
        current_depth, current_metadata = _production_baseline(
            selected,
            global_depth,
            local_depth,
            region,
        )
        candidates = {}
        for input_mode in INPUT_MODES:
            for color_order in COLOR_ORDERS:
                provider_local = infer_facedepth(
                    facedepth,
                    crop,
                    device=device,
                    color_order=color_order,
                    input_mode=input_mode,
                )
                raw = _embed_crop(provider_local, global_depth, crop_box)
                fused, metadata = _production_baseline(
                    selected,
                    global_depth,
                    provider_local,
                    region,
                )
                base_key = (
                    f"facedepth_{input_mode.replace('-', '_')}_{color_order}"
                )
                face_metadata = (
                    metadata.get("faces", [{}])[0]
                    if metadata.get("faces")
                    else {}
                )
                common_metadata = {
                    "color_order": color_order,
                    "input_mode": input_mode,
                    "local_shape": list(provider_local.shape),
                    "refined_faces": metadata.get("refined_faces"),
                    "alignment_scale": face_metadata.get("scale"),
                    "alignment_offset": face_metadata.get("offset"),
                    "alignment_anchor_pixels": face_metadata.get(
                        "anchor_pixels"
                    ),
                    "max_abs_correction": face_metadata.get(
                        "max_abs_correction"
                    ),
                }
                candidates[f"{base_key}_raw"] = {
                    "quality": _measure_quality(
                        raw,
                        exact_path=exact_path,
                        mask_path=mask_path,
                        part_paths=part_paths,
                    ),
                    "metadata": {
                        **common_metadata,
                        "integration": "raw-crop-diagnostic",
                    },
                }
                candidates[f"{base_key}_fused"] = {
                    "quality": _measure_quality(
                        fused,
                        exact_path=exact_path,
                        mask_path=mask_path,
                        part_paths=part_paths,
                    ),
                    "metadata": {
                        **common_metadata,
                        "integration": "production-native-fusion",
                    },
                }
        global_quality = _measure_quality(
            global_depth,
            exact_path=exact_path,
            mask_path=mask_path,
            part_paths=part_paths,
        )
        current_quality = _measure_quality(
            current_depth,
            exact_path=exact_path,
            mask_path=mask_path,
            part_paths=part_paths,
        )
        measured.append(
            {
                "row_id": row["row_id"],
                "face_height_pixels": row["render"]["face_bbox_height_pixels"],
                "detection": detection,
                "current_metadata": {
                    "refined_faces": current_metadata.get("refined_faces"),
                },
                "global_depth": global_quality,
                "current_refined": current_quality,
                **{
                    key: value["quality"]
                    for key, value in candidates.items()
                },
                "provider_metadata": {
                    key: value["metadata"]
                    for key, value in candidates.items()
                },
            }
        )
        print(f"measured {index}/{len(rows)} rows", flush=True)

    global_summary = summarize_rows(measured, "global_depth")
    current_summary = summarize_rows(measured, "current_refined")
    candidate_keys = sorted(measured[0]["provider_metadata"])
    provider_results = {
        key: {
            **_candidate_decision(
                measured,
                key,
                current_summary,
                expected_row_count=int(corpus["row_count"]),
            ),
            "promotion_eligible": promotion_eligible_candidate(key),
        }
        for key in candidate_keys
    }
    advancing = [
        key
        for key, value in provider_results.items()
        if value["promotion_eligible"]
        and value["decision"]["status"] == "advance-current-path"
    ]
    peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if str(device).startswith("cuda")
        else 0.0
    )
    results = {
        "schema_version": 1,
        "method": METHOD,
        "status": "advance-provider" if advancing else "hold",
        "source_geometry_training_and_evaluation_only": True,
        "provider": {
            "repository": PROVIDER_REPOSITORY,
            "revision": PROVIDER_REVISION,
            "published_checkpoint_url": PROVIDER_CHECKPOINT_URL,
            "published_checkpoint_url_status": "404-observed-2026-07-18",
            "mirror_repository": MIRROR_REPOSITORY,
            "mirror_revision": MIRROR_REVISION,
            "mirror_checkpoint_blob": MIRROR_CHECKPOINT_BLOB,
            "verification": (
                "Research mirror only; exact bytes are pinned, but no publisher "
                "checksum is available."
            ),
            "checkpoint": checkpoint,
            "input_modes": {
                "official-stretch": [INPUT_HEIGHT, INPUT_WIDTH],
                "native-aspect": (
                    "long side 480 pixels; both dimensions rounded to "
                    "multiples of 32"
                ),
            },
            "depth_semantics": "inverse camera depth; higher is nearer",
        },
        "corpus": {
            "summary_sha256": _sha256(corpus_path),
            "provider": corpus["provider"],
            "source_revision": corpus["source_revision"],
            "demo_asset_license": corpus["demo_asset_license"],
            "raw_dataset_reference": corpus["raw_dataset_reference"],
            "row_count": len(rows),
        },
        "global_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "model_sha256": _sha256(snapshot / "model.safetensors"),
        },
        "runtime": {
            "device": str(device),
            "seconds": time.perf_counter() - started,
            "peak_torch_allocated_inference_gib": peak_vram,
        },
        "global_depth": global_summary,
        "current_refined": current_summary,
        "provider_candidates": provider_results,
        "advancing_candidates": advancing,
        "rows": measured,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    results = evaluate_c3i_facedepth(
        args.corpus_root,
        args.checkpoint,
        args.output_dir,
        device=args.device,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "status": results["status"],
                "advancing_candidates": results["advancing_candidates"],
                "current_failures": results["current_refined"][
                    "combined_part_failures"
                ],
                "provider_failures": {
                    key: value["summary"]["combined_part_failures"]
                    for key, value in results["provider_candidates"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
