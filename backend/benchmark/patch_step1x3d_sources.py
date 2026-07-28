from __future__ import annotations

import argparse
from pathlib import Path


EAGER_IMPORTS = "from . import data, models, systems"
GEOMETRY_ONLY_IMPORTS = "from . import models"


def patch_step1x3d_sources(step1x3d_dir: Path) -> list[Path]:
    package_init = step1x3d_dir / "step1x3d_geometry" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(
            f"Step1X-3D package initializer does not exist: {package_init}"
        )
    text = package_init.read_text(encoding="utf-8")
    if GEOMETRY_ONLY_IMPORTS in text and EAGER_IMPORTS not in text:
        return []
    if text.count(EAGER_IMPORTS) != 1:
        raise ValueError(
            "Step1X-3D source no longer has the expected eager import marker"
        )
    package_init.write_text(
        text.replace(EAGER_IMPORTS, GEOMETRY_ONLY_IMPORTS),
        encoding="utf-8",
    )
    return [package_init]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limit pinned Step1X-3D imports to its geometry inference modules."
    )
    parser.add_argument("--step1x3d-dir", type=Path, required=True)
    args = parser.parse_args()
    changed = patch_step1x3d_sources(args.step1x3d_dir)
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
