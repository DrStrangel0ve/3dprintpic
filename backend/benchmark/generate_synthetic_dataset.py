import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def draw_sample(index: int, size: int, output_dir: Path) -> dict:
    rng = np.random.default_rng(1309 + index)
    bg = tuple(int(x) for x in rng.integers(210, 246, size=3))
    image = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(image)

    floor_y = int(size * rng.uniform(0.72, 0.82))
    draw.rectangle((0, floor_y, size, size), fill=tuple(int(x) for x in rng.integers(145, 205, size=3)))

    cx = int(size * rng.uniform(0.42, 0.58))
    head_r = int(size * rng.uniform(0.085, 0.13))
    body_w = int(size * rng.uniform(0.18, 0.27))
    body_h = int(size * rng.uniform(0.28, 0.38))
    head_cy = int(size * rng.uniform(0.22, 0.31))
    body_top = head_cy + head_r
    skin = tuple(int(x) for x in rng.integers([190, 130, 95], [246, 205, 170]))
    shirt = tuple(int(x) for x in rng.integers(55, 180, size=3))

    draw.ellipse((cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r), fill=skin)
    draw.rounded_rectangle(
        (cx - body_w // 2, body_top, cx + body_w // 2, body_top + body_h),
        radius=max(4, body_w // 8),
        fill=shirt,
    )

    arm_w = max(7, body_w // 4)
    arm_h = int(body_h * rng.uniform(0.62, 0.9))
    draw.rounded_rectangle((cx - body_w // 2 - arm_w, body_top + 8, cx - body_w // 2, body_top + arm_h), radius=6, fill=skin)
    draw.rounded_rectangle((cx + body_w // 2, body_top + 8, cx + body_w // 2 + arm_w, body_top + arm_h), radius=6, fill=skin)

    leg_w = max(8, body_w // 3)
    leg_h = int(size * rng.uniform(0.18, 0.25))
    leg_top = body_top + body_h
    pants = tuple(int(x) for x in rng.integers(40, 120, size=3))
    draw.rounded_rectangle((cx - leg_w, leg_top, cx - 2, min(size, leg_top + leg_h)), radius=5, fill=pants)
    draw.rounded_rectangle((cx + 2, leg_top, cx + leg_w, min(size, leg_top + leg_h)), radius=5, fill=pants)

    yy, xx = np.mgrid[0:size, 0:size]
    silhouette = np.zeros((size, size), dtype=bool)
    silhouette |= ((xx - cx) ** 2 + (yy - head_cy) ** 2) <= head_r**2
    silhouette |= (np.abs(xx - cx) <= body_w // 2) & (yy >= body_top) & (yy <= body_top + body_h)
    silhouette |= (xx >= cx - body_w // 2 - arm_w) & (xx <= cx - body_w // 2) & (yy >= body_top + 8) & (yy <= body_top + arm_h)
    silhouette |= (xx >= cx + body_w // 2) & (xx <= cx + body_w // 2 + arm_w) & (yy >= body_top + 8) & (yy <= body_top + arm_h)
    silhouette |= (np.abs(xx - (cx - leg_w // 2)) <= leg_w // 2) & (yy >= leg_top) & (yy <= min(size, leg_top + leg_h))
    silhouette |= (np.abs(xx - (cx + leg_w // 2)) <= leg_w // 2) & (yy >= leg_top) & (yy <= min(size, leg_top + leg_h))

    depth = np.ones((size, size), dtype=np.float32)
    body_depth = 0.45 + 0.16 * ((xx - cx) / max(1, body_w)) ** 2
    head_depth = 0.34 + 0.20 * (((xx - cx) ** 2 + (yy - head_cy) ** 2) / max(1, head_r**2))
    depth[silhouette] = np.minimum(depth[silhouette], body_depth[silhouette])
    head_mask = ((xx - cx) ** 2 + (yy - head_cy) ** 2) <= head_r**2
    depth[head_mask] = np.minimum(depth[head_mask], head_depth[head_mask])
    depth = np.clip(depth, 0.0, 1.0)

    mask_side = "right" if index % 2 == 0 else "left"
    mask = np.zeros((size, size), dtype=np.uint8)
    if mask_side == "right":
        mask[:, size // 2 :] = 255
        completion_mode = "mirror-left-to-right"
    else:
        mask[:, : size // 2] = 255
        completion_mode = "mirror-right-to-left"

    masked_pixels = np.asarray(image).copy()
    masked_pixels[mask > 127] = 255
    masked = Image.fromarray(masked_pixels)

    sample_id = f"synthetic_{index:03d}"
    full_path = output_dir / f"{sample_id}_full.png"
    masked_path = output_dir / f"{sample_id}_masked.png"
    mask_path = output_dir / f"{sample_id}_mask.png"
    silhouette_path = output_dir / f"{sample_id}_silhouette.png"
    depth_path = output_dir / f"{sample_id}_depth.npy"
    image.save(full_path)
    masked.save(masked_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    Image.fromarray(silhouette.astype(np.uint8) * 255).save(silhouette_path)
    np.save(depth_path, depth)

    return {
        "id": sample_id,
        "full_image": str(full_path),
        "masked_image": str(masked_path),
        "mask": str(mask_path),
        "gt_depth": str(depth_path),
        "gt_silhouette": str(silhouette_path),
        "silhouette": str(silhouette_path),
        "completion_mode": completion_mode,
        "source": "synthetic_parametric",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiny deterministic completion benchmark.")
    parser.add_argument("--output-dir", default="backend/output/completion-benchmark/synthetic")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for index in range(args.count):
            manifest_file.write(json.dumps(draw_sample(index, args.size, output_dir)) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
