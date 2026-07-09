from __future__ import annotations

import argparse
from pathlib import Path

from backend.benchmark.generate_rendered_dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate procedural 3D mesh renders for the completion benchmark.")
    parser.add_argument("--output-dir", default="backend/output/completion-benchmark/procedural_mesh")
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    manifest_path = generate_dataset(
        output_dir=Path(args.output_dir),
        source="procedural",
        count=args.count,
        size=args.size,
        seed=args.seed,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
