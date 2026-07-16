"""Build a deterministic CC0 face-depth training corpus with sealed identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from backend.benchmark.makehuman_face_fixture import (
    MakeHumanProfileSpec,
    build_makehuman_face_fixture,
    load_makehuman_face_fixture,
)


CORPUS_SEED = 20260716
DEFAULT_ASSET_DIR = (
    Path(__file__).parent / "assets" / "makehuman_cc0_face_training"
)
FIXTURE_FILENAME = "makehuman_cc0_face_training.npz"
EXPRESSIONS = ("neutral", "smile", "mouth_open", "asymmetric")
SPLIT_BY_IDENTITY = {
    "mh_caucasian_female": "train",
    "mh_caucasian_male": "train",
    "mh_african_female": "train",
    "mh_african_male": "train",
    "mh_asian_female": "train",
    "mh_asian_male": "train",
    "mh_mixed_female": "validation",
    "mh_mixed_male": "sealed",
}


@dataclass(frozen=True)
class FaceTrainingSceneSpec:
    row_id: str
    split: str
    identity_group: str
    expression: str
    profile_name: str
    target_dimension: int
    camera_yaw_deg: float
    camera_elevation_deg: float
    camera_distance: float
    camera_scale: float
    horizontal_offset: float
    background_profile: str
    lighting_profile: str
    occlusion: str | None = None


def _paired_targets(directory: str, stem: str, scale: float) -> tuple[tuple[str, float], ...]:
    return (
        (f"{directory}/l-{stem}.target", scale),
        (f"{directory}/r-{stem}.target", scale),
    )


def _base_identity_specs() -> tuple[dict, ...]:
    return (
        {
            "name": "mh_caucasian_female",
            "ancestry": "caucasian",
            "sex": "female",
            "skin_tone": (0.74, 0.50, 0.39),
            "macro": (("macrodetails/caucasian-female-young.target", 1.0),),
            "shape": (
                ("head/head-oval.target", 0.32),
                ("nose/nose-scale-horiz-decr.target", 0.22),
                ("mouth/mouth-scale-horiz-incr.target", 0.16),
                *_paired_targets("cheek", "cheek-bones-incr", 0.18),
            ),
        },
        {
            "name": "mh_caucasian_male",
            "ancestry": "caucasian",
            "sex": "male",
            "skin_tone": (0.68, 0.45, 0.34),
            "macro": (("macrodetails/caucasian-male-young.target", 1.0),),
            "shape": (
                ("head/head-square.target", 0.30),
                ("nose/nose-hump-incr.target", 0.20),
                ("nose/nose-scale-depth-incr.target", 0.16),
                *_paired_targets("cheek", "cheek-volume-decr", 0.14),
            ),
        },
        {
            "name": "mh_african_female",
            "ancestry": "african",
            "sex": "female",
            "skin_tone": (0.40, 0.24, 0.18),
            "macro": (("macrodetails/african-female-young.target", 1.0),),
            "shape": (
                ("head/head-round.target", 0.30),
                ("nose/nose-nostrils-width-incr.target", 0.22),
                ("mouth/mouth-upperlip-volume-incr.target", 0.16),
                *_paired_targets("cheek", "cheek-inner-incr", 0.14),
            ),
        },
        {
            "name": "mh_african_male",
            "ancestry": "african",
            "sex": "male",
            "skin_tone": (0.30, 0.17, 0.13),
            "macro": (("macrodetails/african-male-young.target", 1.0),),
            "shape": (
                ("head/head-rectangular.target", 0.28),
                ("nose/nose-width2-incr.target", 0.20),
                ("mouth/mouth-lowerlip-volume-incr.target", 0.16),
                *_paired_targets("cheek", "cheek-bones-incr", 0.16),
            ),
        },
        {
            "name": "mh_asian_female",
            "ancestry": "asian",
            "sex": "female",
            "skin_tone": (0.66, 0.44, 0.32),
            "macro": (("macrodetails/asian-female-young.target", 1.0),),
            "shape": (
                ("head/head-diamond.target", 0.28),
                ("nose/nose-scale-depth-decr.target", 0.18),
                ("eyes/l-eye-scale-incr.target", 0.13),
                ("eyes/r-eye-scale-incr.target", 0.13),
                *_paired_targets("cheek", "cheek-volume-incr", 0.13),
            ),
        },
        {
            "name": "mh_asian_male",
            "ancestry": "asian",
            "sex": "male",
            "skin_tone": (0.58, 0.38, 0.28),
            "macro": (("macrodetails/asian-male-young.target", 1.0),),
            "shape": (
                ("head/head-invertedtriangular.target", 0.26),
                ("nose/nose-point-width-incr.target", 0.18),
                ("mouth/mouth-scale-horiz-decr.target", 0.16),
                *_paired_targets("cheek", "cheek-inner-decr", 0.13),
            ),
        },
        {
            "name": "mh_mixed_female",
            "ancestry": "caucasian",
            "sex": "female",
            "skin_tone": (0.53, 0.34, 0.25),
            "macro": (
                ("macrodetails/african-female-young.target", 0.50),
                ("macrodetails/asian-female-young.target", 0.50),
            ),
            "shape": (
                ("head/head-triangular.target", 0.25),
                ("nose/nose-curve-concave.target", 0.18),
                ("forehead/forehead-temple-incr.target", 0.15),
                *_paired_targets("cheek", "cheek-trans-up", 0.12),
            ),
        },
        {
            "name": "mh_mixed_male",
            "ancestry": "caucasian",
            "sex": "male",
            "skin_tone": (0.47, 0.29, 0.22),
            "macro": (
                ("macrodetails/caucasian-male-young.target", 0.50),
                ("macrodetails/african-male-young.target", 0.50),
            ),
            "shape": (
                ("head/head-diamond.target", 0.24),
                ("nose/nose-greek-incr.target", 0.17),
                ("forehead/forehead-nubian-incr.target", 0.14),
                *_paired_targets("cheek", "cheek-volume-decr", 0.12),
            ),
        },
    )


def _expression_targets(ancestry: str, expression: str) -> tuple[tuple[str, float], ...]:
    root = f"expression/units/{ancestry}"
    return {
        "neutral": (),
        "smile": ((f"{root}/mouth-corner-puller.target", 0.50),),
        "mouth_open": (
            (f"{root}/mouth-open.target", 0.36),
            (f"{root}/mouth-eversion.target", 0.12),
        ),
        "asymmetric": (
            (f"{root}/eyebrows-left-up.target", 0.34),
            (f"{root}/mouth-part-later.target", 0.22),
        ),
    }[expression]


def build_training_profiles() -> tuple[MakeHumanProfileSpec, ...]:
    profiles = []
    for identity in _base_identity_specs():
        for expression in EXPRESSIONS:
            profiles.append(
                MakeHumanProfileSpec(
                    name=f"{identity['name']}__{expression}",
                    skin_tone=identity["skin_tone"],
                    targets=tuple(identity["macro"])
                    + tuple(identity["shape"])
                    + _expression_targets(identity["ancestry"], expression),
                )
            )
    return tuple(profiles)


TRAINING_PROFILES = build_training_profiles()


_SCENE_CONDITIONS = (
    # Six of ten views are deliberately small and turned.
    (256, -38.0, -4.0, 7.8, 1.00, -0.17, "deep_shelves", "side_right", None),
    (384, 38.0, 3.0, 11.4, 1.00, 0.14, "layered_studio", "soft_left", None),
    (256, -30.0, 2.0, 7.2, 1.00, -0.12, "structured_room", "overhead", None),
    (384, 30.0, -3.0, 10.9, 1.00, 0.11, "deep_shelves", "overhead", "eye_band"),
    (256, -22.0, 4.0, 6.7, 1.00, -0.10, "layered_studio", "side_right", None),
    (384, 22.0, -4.0, 10.4, 1.00, 0.10, "structured_room", "soft_left", "eye_band"),
    (256, -26.0, 1.0, 4.8, 1.00, -0.08, "deep_shelves", "soft_left", None),
    (384, 26.0, -2.0, 4.4, 1.00, 0.07, "layered_studio", "side_right", None),
    (384, -13.0, 2.0, 3.35, 1.04, -0.04, "structured_room", "overhead", None),
    (256, 13.0, -1.0, 3.20, 1.05, 0.04, "deep_shelves", "soft_left", None),
)


def build_training_matrix() -> tuple[FaceTrainingSceneSpec, ...]:
    rows = []
    for profile_index, profile in enumerate(TRAINING_PROFILES):
        identity_group, expression = profile.name.split("__", maxsplit=1)
        split = SPLIT_BY_IDENTITY[identity_group]
        for condition_index, condition in enumerate(_SCENE_CONDITIONS):
            (
                dimension,
                yaw,
                elevation,
                distance,
                scale,
                offset,
                background,
                lighting,
                occlusion,
            ) = condition
            # Rotate condition assignments across profiles without changing coverage.
            shifted = (condition_index + profile_index * 3) % len(_SCENE_CONDITIONS)
            selected = _SCENE_CONDITIONS[shifted]
            (
                dimension,
                yaw,
                elevation,
                distance,
                scale,
                offset,
                background,
                lighting,
                occlusion,
            ) = selected
            rows.append(
                FaceTrainingSceneSpec(
                    row_id=f"{profile.name}_{condition_index:02d}",
                    split=split,
                    identity_group=identity_group,
                    expression=expression,
                    profile_name=profile.name,
                    target_dimension=dimension,
                    camera_yaw_deg=yaw,
                    camera_elevation_deg=elevation,
                    camera_distance=distance,
                    camera_scale=scale,
                    horizontal_offset=offset,
                    background_profile=background,
                    lighting_profile=lighting,
                    occlusion=occlusion,
                )
            )
    return tuple(rows)


TRAINING_MATRIX = build_training_matrix()


def select_training_rows(
    *,
    split: str | None = None,
    limit: int | None = None,
) -> tuple[FaceTrainingSceneSpec, ...]:
    rows = tuple(
        spec for spec in TRAINING_MATRIX if split is None or spec.split == split
    )
    if limit is None or int(limit) >= len(rows):
        return rows
    count = max(int(limit), 0)
    if count == 0:
        return ()
    indices = np.linspace(0, len(rows) - 1, num=count, dtype=np.int64)
    return tuple(rows[int(index)] for index in indices)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_training_fixture(source_root: str | Path, output_dir: str | Path) -> dict:
    return build_makehuman_face_fixture(
        source_root,
        output_dir,
        profiles=TRAINING_PROFILES,
        fixture_filename=FIXTURE_FILENAME,
    )


def render_training_slice(
    asset_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str | None = None,
    limit: int | None = None,
) -> dict:
    from backend.benchmark.run_cc0_live_face_variation_matrix import (
        FaceSceneSpec,
        OccluderSpec,
        _render_scene_arrays,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = load_makehuman_face_fixture(asset_dir)
    rows = select_training_rows(split=split, limit=limit)
    records = []
    for spec in rows:
        occluder = (
            OccluderSpec(
                0.39,
                0.49,
                0.61,
                0.62,
                rgb=(8, 13, 18),
                anchor="eye_band",
                opacity=0.58,
            )
            if spec.occlusion == "eye_band"
            else None
        )
        render_spec = FaceSceneSpec(
            row_id=spec.row_id,
            profile_name=spec.profile_name,
            target_dimension=spec.target_dimension,
            camera_yaw_deg=spec.camera_yaw_deg,
            camera_distance=spec.camera_distance,
            camera_elevation_deg=spec.camera_elevation_deg,
            camera_scale=spec.camera_scale,
            horizontal_offset=spec.horizontal_offset,
            background_profile=spec.background_profile,
            lighting_profile=spec.lighting_profile,
            occluder=occluder,
        )
        source, selection, exact, parts, geometry = _render_scene_arrays(
            render_spec,
            fixture,
        )
        row_dir = output_dir / "rows" / spec.row_id
        part_dir = row_dir / "exact_face_parts"
        part_dir.mkdir(parents=True, exist_ok=True)
        source_path = row_dir / "source.png"
        selection_path = row_dir / "selection_mask.png"
        exact_path = row_dir / "exact_depth.npy"
        Image.fromarray(source).save(source_path)
        Image.fromarray(selection).save(selection_path)
        np.save(exact_path, exact)
        part_records = {}
        for name, values in sorted((parts or {}).items()):
            path = part_dir / f"{name}.png"
            Image.fromarray(values.astype(np.uint8) * 255).save(path)
            part_records[name] = {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256(path),
            }
        records.append(
            {
                "row_id": spec.row_id,
                "split": spec.split,
                "identity_group": spec.identity_group,
                "expression": spec.expression,
                "spec": asdict(spec),
                "render": geometry,
                "source": {
                    "path": source_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(source_path),
                },
                "selection_mask": {
                    "path": selection_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(selection_path),
                },
                "exact_depth": {
                    "path": exact_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(exact_path),
                },
                "exact_face_parts": part_records,
            }
        )
    summary = {
        "schema_version": 1,
        "privacy": "CC0 MakeHuman geometry and deterministic procedural scenes only",
        "corpus_seed": CORPUS_SEED,
        "asset_manifest_sha256": _sha256(Path(asset_dir) / "asset.json"),
        "requested_split": split,
        "requested_limit": limit,
        "row_count": len(records),
        "rows": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root")
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--render-output")
    parser.add_argument("--split", choices=("train", "validation", "sealed"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.source_root:
        build_training_fixture(args.source_root, args.asset_dir)
    if args.render_output:
        summary = render_training_slice(
            args.asset_dir,
            args.render_output,
            split=args.split,
            limit=args.limit,
        )
        print(json.dumps({"row_count": summary["row_count"]}, indent=2))
    elif not args.source_root:
        parser.error("provide --source-root, --render-output, or both")


if __name__ == "__main__":
    main()
