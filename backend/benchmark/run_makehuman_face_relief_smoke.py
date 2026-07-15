"""Exercise high-relief face and background preservation on exact CC0 heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from backend.benchmark.face_part_metrics import (
    FACE_PART_AFFINE_MM_GATES,
    FACE_PART_GATES,
    FACE_PART_METRIC_SCHEMA_VERSION,
    face_part_affine_surface_error_metrics,
    face_part_cross_height_metrics,
    load_persisted_face_part_masks,
)
from backend.benchmark.makehuman_face_fixture import (
    FACE_PART_NAMES,
    load_makehuman_face_fixture,
    make_profile_vertex_colors,
)
from backend.benchmark.mesh_rendering import CameraSpec, RenderConfig, RenderResult, render_mesh
from backend.benchmark.run_canonical_face_relief_smoke import (
    _compact_appearance,
    _finite,
)
from backend.benchmark.run_makehuman_face_depth_smoke import (
    DEFAULT_ASSET_DIR,
    _make_scene,
    _save_masks,
)
from backend.benchmark.run_relief_scene_regression import (
    _git_provenance,
    _mesh_topology,
    _scene_checks,
)
from backend.benchmark.run_relief_visual_sweep import (
    CROSS_HEIGHT_FACE_GATES,
    FACE_APPEARANCE_GATES,
    _appearance_checks,
    _cross_height_face_shape_metrics,
    _stl_heightfield_agreement,
)
from backend.pic_to_3d import (
    DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
    _surface_lighting_agreement_metrics,
    compose_selection_depth_with_context,
    depth_data_to_3d_model,
)


PROVENANCE_PATHS = (
    "backend/pic_to_3d.py",
    "backend/face_depth_refinement.py",
    "backend/benchmark/face_part_metrics.py",
    "backend/benchmark/makehuman_face_fixture.py",
    "backend/benchmark/mesh_rendering.py",
    "backend/benchmark/run_canonical_face_relief_smoke.py",
    "backend/benchmark/run_makehuman_face_depth_smoke.py",
    "backend/benchmark/run_makehuman_face_relief_smoke.py",
    "backend/benchmark/run_relief_scene_regression.py",
    "backend/benchmark/run_relief_visual_sweep.py",
    "backend/benchmark/assets/makehuman_cc0_heads",
)


@dataclass(frozen=True)
class SceneSpec:
    profile_name: str
    framing: str
    phase_rad: float


DEFAULT_SCENES = (
    SceneSpec("caucasian_female_smile", "centered", 0.31),
    SceneSpec("african_male_neutral", "left_frame", 1.17),
    SceneSpec("asian_female_asymmetric", "right_frame", 2.03),
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crop_bounds(render_size: int, crop_size: int, framing: str) -> tuple[int, int, int, int]:
    render_size = int(render_size)
    crop_size = int(crop_size)
    if render_size < crop_size or crop_size < 32:
        raise ValueError("render_size must be at least crop_size >= 32")
    if framing not in {"centered", "left_frame", "right_frame"}:
        raise ValueError(f"Unsupported framing: {framing}")
    center = (render_size - crop_size) // 2
    x_start = center
    if framing == "left_frame":
        x_start += int(round(46.0 * render_size / 384.0))
    elif framing == "right_frame":
        x_start -= int(round(64.0 * render_size / 384.0))
    x_start = int(np.clip(x_start, 0, render_size - crop_size))
    y_start = center
    return y_start, y_start + crop_size, x_start, x_start + crop_size


def _crop_render(rendered: RenderResult, bounds: tuple[int, int, int, int]) -> RenderResult:
    y0, y1, x0, x1 = bounds
    part_masks = {
        name: np.asarray(mask)[y0:y1, x0:x1].copy()
        for name, mask in (rendered.part_masks or {}).items()
    }
    surface_z = None
    if rendered.surface_z is not None:
        surface_z = np.asarray(rendered.surface_z)[y0:y1, x0:x1].copy()
    return RenderResult(
        rgb=np.asarray(rendered.rgb)[y0:y1, x0:x1].copy(),
        depth=np.asarray(rendered.depth)[y0:y1, x0:x1].copy(),
        silhouette=np.asarray(rendered.silhouette)[y0:y1, x0:x1].copy(),
        surface_z=surface_z,
        part_masks=part_masks,
    )


def _render_scene(
    fixture: dict,
    spec: SceneSpec,
    *,
    render_size: int,
    crop_size: int,
) -> tuple[RenderResult, dict]:
    profile = fixture["profiles"][spec.profile_name]
    effective_render_size = crop_size if spec.framing == "centered" else render_size
    colors = make_profile_vertex_colors(
        profile["mesh"].vertices,
        profile["skin_tone"],
        fixture["part_weights"],
        fixture["surface_weights"],
    )
    full = render_mesh(
        profile["mesh"],
        CameraSpec(azimuth_deg=0.0, elevation_deg=0.0),
        RenderConfig(
            size=int(effective_render_size),
            projection="perspective",
            perspective_fov_y_deg=32.0,
            camera_distance=3.2,
            background_rgb=(0.84, 0.87, 0.91),
            ambient=0.42,
            diffuse=0.53,
            specular=0.035,
            shininess=48.0,
            light_direction=(-0.35, -0.20, 0.90),
        ),
        (180, 120, 100),
        vertex_part_weights=fixture["part_weights"],
        vertex_colors=colors,
    )
    bounds = (
        (0, crop_size, 0, crop_size)
        if spec.framing == "centered"
        else _crop_bounds(render_size, crop_size, spec.framing)
    )
    cropped = _crop_render(full, bounds)
    full_face_pixels = int(np.count_nonzero(full.silhouette))
    cropped_face_pixels = int(np.count_nonzero(cropped.silhouette))
    face_fraction = cropped_face_pixels / max(full_face_pixels, 1)
    frame_contact = bool(
        np.any(cropped.silhouette[0])
        or np.any(cropped.silhouette[-1])
        or np.any(cropped.silhouette[:, 0])
        or np.any(cropped.silhouette[:, -1])
    )
    head_rows = max(1, int(round(0.78 * crop_size)))
    head_frame_contact = bool(
        np.any(cropped.silhouette[:head_rows, 0])
        or np.any(cropped.silhouette[:head_rows, -1])
    )
    part_retention = {
        name: float(
            np.count_nonzero(cropped.part_masks[name])
            / max(np.count_nonzero(full.part_masks[name]), 1)
        )
        for name in FACE_PART_NAMES
    }
    checks = {
        "face_coverage": face_fraction >= (0.90 if spec.framing != "centered" else 0.995),
        "expected_frame_contact": frame_contact == (spec.framing != "centered"),
        "expected_head_frame_contact": head_frame_contact == (spec.framing != "centered"),
        "named_parts_complete": all(np.any(cropped.part_masks[name]) for name in FACE_PART_NAMES),
        "named_parts_retained": min(part_retention.values()) >= 0.995,
    }
    return cropped, {
        "render_size": int(effective_render_size),
        "stress_render_size": int(render_size),
        "crop_size": int(crop_size),
        "crop_bounds_yxyx": list(bounds),
        "framing": spec.framing,
        "full_face_pixels": full_face_pixels,
        "cropped_face_pixels": cropped_face_pixels,
        "visible_face_fraction": float(face_fraction),
        "frame_contact": frame_contact,
        "head_frame_contact": head_frame_contact,
        "head_frame_contact_row_limit": head_rows,
        "named_part_retention": part_retention,
        "checks": {**checks, "passed": bool(all(checks.values()))},
    }


def _emit_row(
    output_dir: Path,
    fixture: dict,
    spec: SceneSpec,
    *,
    relief_height_mm: float,
    render_size: int,
    crop_size: int,
    physical_size_mm: float,
    background_depth_ratio: float,
    provider_name: str = "oracle",
    input_depth: np.ndarray | None = None,
    value_transform: str = "linear",
    invert: bool = True,
    low_percentile: float = 0.0,
    high_percentile: float = 100.0,
    use_part_feature_weight: bool = True,
) -> tuple[dict, dict]:
    started = time.perf_counter()
    base_row_id = (
        f"{spec.profile_name}_{spec.framing}_{relief_height_mm:04.1f}mm"
        .replace(".", "p")
    )
    row_id = base_row_id if provider_name == "oracle" else f"{provider_name}_{base_row_id}"
    row_dir = output_dir / row_id
    row_dir.mkdir(parents=True, exist_ok=True)
    rendered, framing = _render_scene(
        fixture,
        spec,
        render_size=render_size,
        crop_size=crop_size,
    )
    face_mask = np.asarray(rendered.silhouette, dtype=bool)
    part_masks = {
        name: np.asarray(rendered.part_masks[name], dtype=bool) & face_mask
        for name in FACE_PART_NAMES
    }
    exact_depth, source_rgb = _make_scene(
        rendered.depth,
        face_mask,
        rendered.rgb,
        phase=spec.phase_rad,
    )
    source_path = row_dir / "source.png"
    exact_path = row_dir / "exact_depth.npy"
    Image.fromarray(np.clip(source_rgb * 255.0, 0, 255).astype(np.uint8)).save(source_path)
    np.save(exact_path, exact_depth.astype(np.float32, copy=False))
    mask_metadata_path = _save_masks(row_dir, face_mask, part_masks)

    relief_depth = exact_depth if input_depth is None else np.asarray(input_depth, dtype=np.float32)
    if relief_depth.shape != exact_depth.shape:
        raise ValueError(
            f"Provider depth shape {relief_depth.shape} does not match scene {exact_depth.shape}"
        )
    input_depth_path = None
    if input_depth is not None:
        input_depth_path = row_dir / "provider_depth.npy"
        np.save(input_depth_path, relief_depth.astype(np.float32, copy=False))

    input_pitch_mm = float(physical_size_mm) / max(int(crop_size) - 1, 1)
    composed, compose_stats = compose_selection_depth_with_context(
        relief_depth,
        face_mask,
        value_transform=value_transform,
        relief_height_mm=float(relief_height_mm),
        sample_pitch_mm=input_pitch_mm,
        max_slope_mm_per_mm=2.0,
        background_depth_ratio=float(background_depth_ratio),
        background_feather_mm=1.5,
        background_smoothing_mm=0.6,
    )
    depth_path = row_dir / "composed_depth.npy"
    surface_path = row_dir / "emitted_surface.npy"
    reference_path = row_dir / "reference_surface.npy"
    stl_path = row_dir / "relief.stl"
    np.save(depth_path, composed.astype(np.float32, copy=False))
    feature_weight = None
    if use_part_feature_weight:
        feature_weight = np.maximum.reduce(
            [np.asarray(mask, dtype=np.float32) for mask in part_masks.values()]
        )
    postprocess = depth_data_to_3d_model(
        depth_path,
        output_stl_path=str(stl_path),
        target_dimension=-1,
        z_scale=float(relief_height_mm),
        invert=bool(invert),
        sigma=0.0,
        max_xy_size=float(physical_size_mm),
        relief_gamma=1.0,
        detail_boost=0.0,
        low_percentile=float(low_percentile),
        high_percentile=float(high_percentile),
        base_border_px=1,
        value_transform=value_transform,
        minimum_feature_mm=0.8,
        max_relief_slope=2.0,
        face_region_mask=face_mask,
        selection_region_mask=face_mask,
        selection_background_depth_ratio=float(background_depth_ratio),
        source_image=source_path,
        background_photo_detail_mm=0.0,
        feature_weight_mask=feature_weight,
        printable_feature_depth_mm=0.8,
        feature_bridge_depth_mm=0.8,
        surface_output_path=surface_path,
        reference_surface_output_path=reference_path,
    )
    postprocess_path = row_dir / "postprocess.json"
    postprocess_path.write_text(json.dumps(postprocess, indent=2), encoding="utf-8")
    emitted = np.load(surface_path).astype(np.float64)
    reference = np.load(reference_path).astype(np.float64)
    transformed = load_persisted_face_part_masks(
        mask_metadata_path,
        surface_grid_transform=postprocess["surface_grid_transform"],
    )
    if len(transformed) != 1:
        raise ValueError(f"Expected one transformed face, got {len(transformed)}")
    masks = transformed[0]
    pitch_mm = float(postprocess["mesh_sample_pitch_mm"])
    appearance = _surface_lighting_agreement_metrics(
        reference,
        emitted,
        masks["face_mask"],
        sample_pitch_mm=pitch_mm,
        component_metrics=False,
    )
    named_parts = face_part_cross_height_metrics(
        reference,
        emitted,
        masks["face_mask"],
        masks["part_masks"],
        sample_pitch_mm=pitch_mm,
    )
    part_mm = face_part_affine_surface_error_metrics(
        reference,
        emitted,
        masks["face_mask"],
        masks["part_masks"],
    )
    appearance_checks = _appearance_checks(appearance, FACE_APPEARANCE_GATES)
    topology = _mesh_topology(trimesh.load_mesh(stl_path, process=True))
    shell = _stl_heightfield_agreement(
        stl_path,
        surface_path,
        expected_max_xy_size_mm=float(physical_size_mm),
    )
    scene_checks = _scene_checks(compose_stats, postprocess, topology)
    feature_stats = postprocess["printable_feature_depth"]
    selected_compression = postprocess["face_height_stabilization"].get(
        "gradient_compression_selected",
        {},
    )
    reversal_projection = selected_compression.get(
        "direction_reversal_projection",
        {"enabled": False, "reason": "no_selected_compression"},
    )
    checks = {
        "framing": bool(framing["checks"]["passed"]),
        "source_context": bool(
            scene_checks["context_composed"] and scene_checks["source_context_signal"]
        ),
        "absolute_face_appearance": bool(appearance_checks["passed"]),
        "absolute_named_parts": bool(named_parts["passed"]),
        "absolute_named_part_mm_error": bool(part_mm["passed"]),
        "background_preservation": bool(scene_checks["background_preservation"]),
        "localized_background_structure": bool(scene_checks["localized_background_structure"]),
        "physical_emission": bool(scene_checks["physical_emission"]),
        "feasible_attachment": bool(scene_checks["feasible_attachment"]),
        "printable_mesh": bool(topology["printable"]),
        "complete_shell": bool(shell["passed"]),
        "screened_face_reconstruction": bool(
            postprocess["face_boundary_alignment"].get("method")
            == "screened_gradient_domain_compression"
        ),
        "redundant_emboss_suppressed": bool(
            feature_stats.get("suppressed_after_screened_face_reconstruction", False)
            and float(postprocess["effective_printable_feature_depth_mm"]) == 0.0
        ),
        "direction_reversal_projection": bool(
            not reversal_projection.get("enabled", False)
            or reversal_projection.get("passed", False)
        ),
    }
    background = postprocess["background_depth_preservation"]
    cap = postprocess["selection_background_physical_cap"]
    artifact_paths = (
        source_path,
        exact_path,
        depth_path,
        reference_path,
        surface_path,
        stl_path,
        postprocess_path,
        mask_metadata_path,
        *sorted((row_dir / "face_parts").glob("*.png")),
        *((input_depth_path,) if input_depth_path is not None else ()),
    )
    row = {
        "row_id": row_id,
        "provider": provider_name,
        "provider_depth_semantics": {
            "value_transform": value_transform,
            "invert": bool(invert),
            "uses_oracle_depth": input_depth is None,
            "uses_oracle_part_feature_weight": bool(use_part_feature_weight),
            "normalization_percentiles": [float(low_percentile), float(high_percentile)],
        },
        "profile_name": spec.profile_name,
        "framing": framing,
        "relief_height_mm": float(relief_height_mm),
        "runtime_seconds": float(time.perf_counter() - started),
        "checks": {**checks, "passed": bool(all(checks.values()))},
        "absolute_face": {**_compact_appearance(appearance), "checks": appearance_checks},
        "absolute_named_parts": named_parts,
        "absolute_named_part_mm_error": part_mm,
        "background": {
            "passed": bool(background.get("passed", False)),
            "correlation": _finite(background.get("correlation")),
            "rms_retention": _finite(background.get("rms_retention")),
            "span_retention": _finite(background.get("span_retention")),
            "gradient_correlation": _finite(background.get("gradient_correlation")),
            "candidate_coverage_ratio": _finite(background.get("candidate_coverage_ratio")),
            "output_rms_mm": _finite(background.get("output_rms_mm")),
            "output_span_p02_p98_mm": _finite(background.get("output_span_p02_p98_mm")),
        },
        "physical_cap": {
            "emission_passed": bool(cap.get("emission_passed", False)),
            "far_background_max_mm": _finite(cap.get("far_background_max_mm")),
            "far_background_ceiling_mm": _finite(cap.get("far_background_ceiling_mm")),
            "feasible_attachment_jump_max_mm": _finite(cap.get("feasible_attachment_jump_max_mm")),
            "conflicting_attachment_pixels": int(cap.get("conflicting_attachment_pixels", 0)),
        },
        "direction_reversal_projection": {
            key: reversal_projection.get(key)
            for key in (
                "enabled",
                "reason",
                "passed",
                "iterations",
                "corrected_pixels",
                "initial_cardinal_reversals",
                "initial_diagonal_reversals",
                "final_cardinal_reversals",
                "final_diagonal_reversals",
                "correction_p95_mm",
                "correction_max_mm",
            )
        },
        "topology": topology,
        "shell": {key: shell.get(key) for key in (
            "passed",
            "complete_shell_verified",
            "facet_geometry_passed",
            "coverage_ratio",
            "max_abs_error_mm",
            "rms_error_mm",
            "expected_shell_triangle_count",
            "actual_shell_triangle_count",
            "invalid_shell_triangle_count",
        )},
        "artifacts": {
            path.relative_to(row_dir).as_posix(): {
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        },
    }
    metric_context = {
        "surface_path": surface_path,
        "face_mask": masks["face_mask"],
        "part_masks": masks["part_masks"],
        "sample_pitch_mm": pitch_mm,
        "surface_grid_transform": postprocess["surface_grid_transform"],
    }
    return row, metric_context


def _cross_height(rows: list[dict], contexts: dict[str, dict]) -> dict:
    records = []
    for profile_name in sorted({row["profile_name"] for row in rows}):
        profile_rows = sorted(
            (row for row in rows if row["profile_name"] == profile_name),
            key=lambda row: row["relief_height_mm"],
        )
        if len(profile_rows) != 2:
            continue
        low, high = profile_rows
        low_context = contexts[low["row_id"]]
        high_context = contexts[high["row_id"]]
        low_surface = np.load(low_context["surface_path"])
        high_surface = np.load(high_context["surface_path"])
        whole = _cross_height_face_shape_metrics(
            low_surface,
            high_surface,
            low_context["face_mask"],
        )
        named = face_part_cross_height_metrics(
            low_surface,
            high_surface,
            low_context["face_mask"],
            low_context["part_masks"],
            sample_pitch_mm=low_context["sample_pitch_mm"],
        )
        records.append({
            "profile_name": profile_name,
            "reference_height_mm": float(low["relief_height_mm"]),
            "candidate_height_mm": float(high["relief_height_mm"]),
            "whole_face": whole,
            "named_parts": named,
            "passed": bool(whole["passed"] and named["passed"]),
        })
    expected = len({row["profile_name"] for row in rows})
    return {
        "gates": CROSS_HEIGHT_FACE_GATES,
        "named_part_gates": FACE_PART_GATES,
        "expected_comparison_count": expected,
        "comparison_count": len(records),
        "coverage_complete": len(records) == expected,
        "quality_passed": bool(records) and all(record["passed"] for record in records),
        "records": records,
        "passed": bool(len(records) == expected and all(record["passed"] for record in records)),
    }


def run(
    output_dir: str | Path,
    *,
    asset_dir: str | Path = DEFAULT_ASSET_DIR,
    scenes: tuple[SceneSpec, ...] = DEFAULT_SCENES,
    relief_heights_mm: tuple[float, ...] = (30.0, 40.0),
    render_size: int = 384,
    crop_size: int = 256,
    physical_size_mm: float = 96.0,
    background_depth_ratio: float = DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
    allow_dirty: bool = False,
    allow_failures: bool = False,
) -> dict:
    if len(relief_heights_mm) != 2 or len(set(relief_heights_mm)) != 2:
        raise ValueError("Exactly two distinct relief heights are required")
    if len({scene.profile_name for scene in scenes}) != len(scenes):
        raise ValueError("Scene profile names must be unique")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = load_makehuman_face_fixture(asset_dir)
    missing = sorted({scene.profile_name for scene in scenes} - set(fixture["profiles"]))
    if missing:
        raise ValueError(f"Unknown MakeHuman profiles: {missing}")
    provenance = _git_provenance(PROVENANCE_PATHS)
    rows = []
    contexts = {}
    for scene in scenes:
        for height in relief_heights_mm:
            row, context = _emit_row(
                output_dir,
                fixture,
                scene,
                relief_height_mm=float(height),
                render_size=render_size,
                crop_size=crop_size,
                physical_size_mm=physical_size_mm,
                background_depth_ratio=background_depth_ratio,
            )
            if row["row_id"] in contexts:
                raise ValueError(f"Duplicate row id: {row['row_id']}")
            rows.append(row)
            contexts[row["row_id"]] = context
    expected_rows = len(scenes) * len(relief_heights_mm)
    cross_height = _cross_height(rows, contexts)
    checks = {
        "expected_rows": len(rows) == expected_rows,
        "unique_rows": len({row["row_id"] for row in rows}) == len(rows),
        "all_row_gates_passed": bool(rows) and all(row["checks"]["passed"] for row in rows),
        "cross_height_face_quality": bool(cross_height["passed"]),
        "implementation_provenance_clean": bool(
            provenance.get("available") and provenance.get("clean")
        ),
    }
    summary = {
        "schema_version": 1,
        "run_kind": "makehuman_cc0_varied_scene_high_relief_fidelity",
        "privacy": "checksum-pinned CC0 generated heads and deterministic analytic backgrounds only",
        "asset": fixture["manifest"],
        "implementation_provenance": provenance,
        "allow_dirty": bool(allow_dirty),
        "allow_failures": bool(allow_failures),
        "matrix": {
            "scenes": [scene.__dict__ for scene in scenes],
            "relief_heights_mm": [float(value) for value in relief_heights_mm],
            "render_size": int(render_size),
            "crop_size": int(crop_size),
            "physical_size_mm": float(physical_size_mm),
            "background_depth_ratio": float(background_depth_ratio),
            "expected_rows": expected_rows,
            "completed_rows": len(rows),
        },
        "face_appearance_gates": FACE_APPEARANCE_GATES,
        "named_face_part_metric_schema_version": FACE_PART_METRIC_SCHEMA_VERSION,
        "named_face_part_gates": FACE_PART_GATES,
        "named_face_part_affine_mm_gates": FACE_PART_AFFINE_MM_GATES,
        "cross_height_face_consistency": cross_height,
        "checks": {
            **checks,
            "passed": bool(
                checks["expected_rows"]
                and checks["unique_rows"]
                and checks["all_row_gates_passed"]
                and checks["cross_height_face_quality"]
                and (allow_dirty or checks["implementation_provenance_clean"])
            ),
        },
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    failures = [name for name, passed in checks.items() if not passed and name != "implementation_provenance_clean"]
    if not allow_dirty and not checks["implementation_provenance_clean"]:
        failures.append("implementation_provenance_clean")
    if failures and not allow_failures:
        raise RuntimeError("MakeHuman face relief smoke failed: " + ", ".join(failures))
    return summary


def _float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="backend/output/makehuman-face-relief-smoke",
    )
    parser.add_argument("--relief-heights-mm", type=_float_tuple, default=(30.0, 40.0))
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--physical-size-mm", type=float, default=96.0)
    parser.add_argument(
        "--background-depth-ratio",
        type=float,
        default=DEFAULT_SELECTION_BACKGROUND_DEPTH_RATIO,
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()
    run(
        args.output_dir,
        relief_heights_mm=args.relief_heights_mm,
        render_size=args.render_size,
        crop_size=args.crop_size,
        physical_size_mm=args.physical_size_mm,
        background_depth_ratio=args.background_depth_ratio,
        allow_dirty=args.allow_dirty,
        allow_failures=args.allow_failures,
    )


if __name__ == "__main__":
    main()
