from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from PIL import Image

from backend.benchmark.mesh_rendering import (
    RenderConfig,
    apply_mask,
    iter_mesh_paths,
    iter_mesh_paths_balanced,
    load_mesh,
    make_half_mask,
    make_procedural_mesh,
    mesh_category,
    mesh_source_split,
    render_mesh,
    sample_camera,
)


def _stable_color(index: int, seed: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed + 4400 + index * 17)
    return tuple(int(x) for x in rng.integers(70, 220, size=3))


def write_sample(
    *,
    index: int,
    mesh,
    output_dir: Path,
    config: RenderConfig,
    source: str,
    asset_id: str,
    asset_path: str | None,
    asset_category: str | None = None,
    asset_source_split: str | None = None,
    asset_key: str | None = None,
    view_index: int,
    seed: int,
    sample_id: str | None = None,
) -> dict:
    sample_id = sample_id or f"{source}_{index:04d}"
    camera = sample_camera(index + view_index, seed=seed)
    result = render_mesh(mesh, camera=camera, config=config, base_color=_stable_color(index, seed))

    full_path = output_dir / f"{sample_id}_full.png"
    masked_path = output_dir / f"{sample_id}_masked.png"
    mask_path = output_dir / f"{sample_id}_mask.png"
    silhouette_path = output_dir / f"{sample_id}_silhouette.png"
    depth_path = output_dir / f"{sample_id}_depth.npy"
    mesh_path = output_dir / f"{sample_id}.ply"

    full_pixels = np.clip(result.rgb * 255, 0, 255).astype(np.uint8)
    Image.fromarray(full_pixels).save(full_path)
    np.save(depth_path, result.depth)
    Image.fromarray(result.silhouette.astype(np.uint8) * 255).save(silhouette_path)
    mesh.export(mesh_path)

    mask_side = "right" if index % 2 == 0 else "left"
    completion_mode = "mirror-left-to-right" if mask_side == "right" else "mirror-right-to-left"
    mask = make_half_mask(config.size, mask_side)
    Image.fromarray(mask).save(mask_path)
    Image.fromarray(apply_mask(result.rgb, mask)).save(masked_path)

    return {
        "id": sample_id,
        "full_image": str(full_path),
        "masked_image": str(masked_path),
        "mask": str(mask_path),
        "gt_depth": str(depth_path),
        "gt_silhouette": str(silhouette_path),
        "silhouette": str(silhouette_path),
        "mesh": str(mesh_path),
        "completion_mode": completion_mode,
        "source": source,
        "asset_id": asset_id,
        "asset_path": asset_path,
        "asset_category": asset_category,
        "asset_source_split": asset_source_split,
        "asset_key": asset_key or asset_path or asset_id,
        "view_index": view_index,
        "camera": camera.to_dict(),
    }


def asset_key_for_path(asset_path: Path, asset_root: Path) -> str:
    try:
        return asset_path.resolve().relative_to(asset_root.resolve()).as_posix()
    except ValueError:
        return asset_path.as_posix()


def generate_dataset(
    *,
    output_dir: Path,
    source: str,
    count: int,
    size: int,
    seed: int,
    asset_root: Path | None = None,
    asset_glob: str = "**/*.glb",
    views_per_asset: int = 1,
    mesh_sample_strategy: str = "random",
    continue_on_error: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = RenderConfig(size=size)
    rows = []
    failures_path = output_dir / "render_failures.jsonl"
    if failures_path.exists():
        failures_path.unlink()

    if source == "procedural":
        for index in range(count):
            rows.append(
                write_sample(
                    index=index,
                    mesh=make_procedural_mesh(index),
                    output_dir=output_dir,
                    config=config,
                    source="procedural_mesh",
                    asset_id=f"procedural_{index:04d}",
                    asset_path=None,
                    asset_category="procedural",
                    asset_source_split="procedural",
                    asset_key=f"procedural:{index:04d}",
                    view_index=0,
                    seed=seed,
                )
            )
    elif source == "mesh-dir":
        if asset_root is None:
            raise ValueError("--asset-root is required when --source mesh-dir")
        asset_limit = max(1, int(np.ceil(count / max(1, views_per_asset))))
        if mesh_sample_strategy == "balanced":
            paths = list(iter_mesh_paths_balanced(asset_root, asset_glob, limit=asset_limit, seed=seed))
        else:
            paths = list(iter_mesh_paths(asset_root, asset_glob, limit=asset_limit, seed=seed))
        sample_index = 0
        for asset_path in paths:
            try:
                mesh = load_mesh(asset_path)
                path_hash = hashlib.sha1(str(asset_path).encode("utf-8")).hexdigest()[:10]
                category = mesh_category(asset_path, asset_root)
                source_split = mesh_source_split(asset_path)
                asset_key = asset_key_for_path(asset_path, asset_root)
                for view_index in range(views_per_asset):
                    if sample_index >= count:
                        break
                    rows.append(
                        write_sample(
                            index=sample_index,
                            mesh=mesh,
                            output_dir=output_dir,
                            config=config,
                            source="mesh_dir",
                            asset_id=asset_path.stem,
                            asset_path=str(asset_path),
                            asset_category=category,
                            asset_source_split=source_split,
                            asset_key=asset_key,
                            view_index=view_index,
                            seed=seed,
                            sample_id=f"mesh_dir_{path_hash}_v{view_index:02d}",
                        )
                    )
                    sample_index += 1
            except Exception as exc:
                failure = {
                    "asset_path": str(asset_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                with failures_path.open("a", encoding="utf-8") as failures_file:
                    failures_file.write(json.dumps(failure) + "\n")
                if not continue_on_error:
                    raise
            if sample_index >= count:
                break
    else:
        raise ValueError(f"Unsupported source: {source}")

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for row in rows:
            manifest_file.write(json.dumps(row) + "\n")
    summary_path = output_dir / "dataset_summary.json"
    categories = {}
    source_splits = {}
    for row in rows:
        category = row.get("asset_category") or "unknown"
        categories[category] = categories.get(category, 0) + 1
        source_split = row.get("asset_source_split") or "unknown"
        source_splits[source_split] = source_splits.get(source_split, 0) + 1
    summary_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "source": source,
                "count_requested": count,
                "count_rendered": len(rows),
                "size": size,
                "seed": seed,
                "asset_root": str(asset_root) if asset_root else None,
                "asset_glob": asset_glob,
                "views_per_asset": views_per_asset,
                "mesh_sample_strategy": mesh_sample_strategy if source == "mesh-dir" else None,
                "category_counts": dict(sorted(categories.items())),
                "source_split_counts": dict(sorted(source_splits.items())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rendered 3D mesh samples for the completion benchmark.")
    parser.add_argument("--output-dir", default="backend/output/completion-benchmark/rendered")
    parser.add_argument("--source", choices=("procedural", "mesh-dir"), default="procedural")
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--asset-root")
    parser.add_argument("--asset-glob", default="**/*.glb")
    parser.add_argument("--views-per-asset", type=int, default=1)
    parser.add_argument("--mesh-sample-strategy", choices=("random", "balanced"), default="random")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    manifest_path = generate_dataset(
        output_dir=Path(args.output_dir),
        source=args.source,
        count=args.count,
        size=args.size,
        seed=args.seed,
        asset_root=Path(args.asset_root) if args.asset_root else None,
        asset_glob=args.asset_glob,
        views_per_asset=args.views_per_asset,
        mesh_sample_strategy=args.mesh_sample_strategy,
        continue_on_error=args.continue_on_error,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
