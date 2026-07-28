"""Run a bounded continuous low-rank MHR representation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import scipy
from scipy.optimize import differential_evolution

from backend.benchmark.c3i_synface_corpus import verified_corpus_asset
from backend.benchmark.evaluate_face_depth_head_exact_gate import _quality
from backend.benchmark.evaluate_vggheads_small_face_exact_gate import (
    build_small_face_candidate,
)
from backend.benchmark.mhr_parametric_geometry import (
    camera_space_vertices,
    crop_mask,
    load_mhr_geometry_runtime,
    render_mhr_camera_depth,
)
from backend.benchmark.mhr_face_training_corpus import (
    MHR_EXPRESSION_DIMENSION,
    MHR_IDENTITY_DIMENSION,
    MHR_POSE_DIMENSION,
    preflight_mhr_root,
)
from backend.benchmark.train_face_depth_head import (
    _validate_training_corpus_summary,
)


METHOD = "mhr-parametric-low-rank-bounded-continuous-audit"
HARD_ROW_ID = "small_side_lit_shelves_256"
DEFAULT_ANCHOR_ROW_ID = "mhr_training_identity_007__scene_04"
SEARCH_SEED = 20260719
IDENTITY_COEFFICIENT_LIMIT = 1.8
EXPRESSION_COEFFICIENT_LIMIT = 0.6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_declared_asset(root: str | Path, record: dict, label: str) -> dict:
    root = Path(root).resolve()
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be a declared asset record")
    relative_path = record.get("path")
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{label} is missing a relative path")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{label} is missing a SHA256 declaration")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError(f"{label} is missing a size declaration")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its declared root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual_size = int(path.stat().st_size)
    if actual_size != expected_size:
        raise ValueError(
            f"{label} size mismatch: expected {expected_size}, got {actual_size}"
        )
    actual_hash = _sha256(path)
    if actual_hash != expected_hash.lower():
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return {
        "path": path,
        "relative_path": relative_path.replace("\\", "/"),
        "sha256": actual_hash,
        "size_bytes": actual_size,
        "verified": True,
    }


def observed_asset(path: str | Path, root: str | Path, label: str) -> dict:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} escapes its expected root") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return {
        "relative_path": relative_path,
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _row_by_id(rows: list[dict], row_id: str) -> dict:
    matches = [row for row in rows if row.get("row_id") == row_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for {row_id}")
    return matches[0]


def _compact_quality(quality: dict) -> dict:
    if "shape_failed_parts" in quality:
        shape_failed = quality["shape_failed_parts"]
        affine_failed = quality["affine_failed_parts"]
    else:
        shape_failed = quality["named_part_shape"]["failed_parts"]
        affine_failed = quality["named_part_affine_mm"]["failed_parts"]
    return {
        "shape_correlation": float(quality["shape_correlation"]),
        "gradient_correlation": float(quality["gradient_correlation"]),
        "normalized_rmse": float(quality["normalized_rmse"]),
        "shape_failed_parts": list(shape_failed),
        "affine_failed_parts": list(affine_failed),
        "combined_part_failures": int(len(shape_failed) + len(affine_failed)),
    }


def fit_pca(values: np.ndarray, rank: int) -> dict:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(
            "PCA supervision must be a finite matrix with at least two rows"
        )
    maximum_rank = min(matrix.shape[0] - 1, matrix.shape[1])
    if not 1 <= int(rank) <= maximum_rank:
        raise ValueError(f"PCA rank must be between 1 and {maximum_rank}")
    mean = np.mean(matrix, axis=0)
    _left, singular, components = np.linalg.svd(matrix - mean, full_matrices=False)
    scales = singular[: int(rank)] / np.sqrt(matrix.shape[0] - 1)
    if np.any(scales <= 1e-10):
        raise ValueError("PCA supervision contains a degenerate selected component")
    variance = np.square(singular)
    explained = float(np.sum(variance[: int(rank)]) / np.sum(variance))
    return {
        "mean": mean,
        "components": components[: int(rank)],
        "scales": scales,
        "rank": int(rank),
        "explained_variance_ratio": explained,
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
    }


def fit_geometry_pca(
    values: np.ndarray,
    displacement_jacobian: np.ndarray,
    vertex_area_weights: np.ndarray,
    rank: int,
    *,
    region_vertex_weights: dict[str, np.ndarray] | None = None,
) -> dict:
    matrix = np.asarray(values, dtype=np.float64)
    jacobian = np.asarray(displacement_jacobian, dtype=np.float64)
    weights = np.asarray(vertex_area_weights, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Geometry PCA supervision must be a finite matrix with at least two rows"
        )
    if jacobian.shape != (matrix.shape[1], len(weights), 3) or not np.all(
        np.isfinite(jacobian)
    ):
        raise ValueError("Geometry PCA Jacobian has the wrong finite shape")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Geometry PCA vertex weights must be finite and positive")
    maximum_rank = min(matrix.shape[0] - 1, matrix.shape[1])
    if not 1 <= int(rank) <= maximum_rank:
        raise ValueError(f"Geometry PCA rank must be between 1 and {maximum_rank}")

    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    weighted_jacobian = (jacobian * np.sqrt(weights)[None, :, None]).reshape(
        matrix.shape[1], -1
    )
    geometry_metric = weighted_jacobian @ weighted_jacobian.T
    ridge = max(
        float(np.trace(geometry_metric) / matrix.shape[1]) * 1e-10,
        1e-12,
    )
    factor = np.linalg.cholesky(
        geometry_metric + np.eye(matrix.shape[1], dtype=np.float64) * ridge
    )
    transformed = centered @ factor
    _left, singular, transformed_components = np.linalg.svd(
        transformed,
        full_matrices=False,
    )
    scales = singular[: int(rank)] / np.sqrt(matrix.shape[0] - 1)
    if np.any(scales <= 1e-10):
        raise ValueError("Geometry PCA contains a degenerate selected component")
    components = transformed_components[: int(rank)] @ np.linalg.inv(factor)
    variance = np.square(singular)
    explained = float(np.sum(variance[: int(rank)]) / np.sum(variance))
    region_retention = {}
    if region_vertex_weights:
        projected_scores = transformed @ transformed_components[: int(rank)].T
        reconstructed = projected_scores @ components
        residual = centered - reconstructed
        for name, region_values in sorted(region_vertex_weights.items()):
            region = np.asarray(region_values, dtype=np.float64).reshape(-1)
            if (
                region.shape != weights.shape
                or not np.all(np.isfinite(region))
                or np.any(region < 0.0)
                or not np.any(region > 0.0)
            ):
                raise ValueError(f"Geometry PCA region {name!r} has invalid weights")
            region_jacobian = (jacobian * np.sqrt(region)[None, :, None]).reshape(
                matrix.shape[1], -1
            )
            total_geometry = centered @ region_jacobian
            residual_geometry = residual @ region_jacobian
            total_energy = float(np.sum(np.square(total_geometry)))
            if total_energy <= 1e-12:
                raise ValueError(f"Geometry PCA region {name!r} has no variance")
            region_retention[name] = float(
                1.0 - np.sum(np.square(residual_geometry)) / total_energy
            )
    return {
        "mean": mean,
        "components": components,
        "scales": scales,
        "rank": int(rank),
        "explained_variance_ratio": explained,
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "geometry_metric_ridge": ridge,
        "weighted_vertices": int(len(weights)),
        "vertex_weight_min": float(np.min(weights)),
        "vertex_weight_max": float(np.max(weights)),
        "region_retention": region_retention,
        "minimum_region_retention": (
            float(min(region_retention.values())) if region_retention else None
        ),
    }


def mhr_topology_face_regions(neutral_vertices: np.ndarray) -> dict[str, np.ndarray]:
    camera, _record = camera_space_vertices(
        neutral_vertices,
        yaw_degrees=0.0,
        elevation_degrees=0.0,
        distance=3.0,
    )
    x_mid = 0.5 * (float(np.min(camera[:, 0])) + float(np.max(camera[:, 0])))
    y_mid = 0.5 * (float(np.min(camera[:, 1])) + float(np.max(camera[:, 1])))
    x_scale = max(float(np.ptp(camera[:, 0])) * 0.5, 1e-8)
    y_scale = max(float(np.ptp(camera[:, 1])) * 0.5, 1e-8)
    x = (camera[:, 0] - x_mid) / x_scale
    y = (camera[:, 1] - y_mid) / y_scale
    front = camera[:, 2] <= np.percentile(camera[:, 2], 55.0)

    def box(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        return front & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

    regions = {
        "left_eye": box(-0.58, -0.05, -0.28, 0.02),
        "right_eye": box(0.05, 0.58, -0.28, 0.02),
        "left_eyebrow": box(-0.62, -0.03, -0.48, -0.22),
        "right_eyebrow": box(0.03, 0.62, -0.48, -0.22),
        "nose": box(-0.24, 0.24, -0.20, 0.34),
        "mouth": box(-0.46, 0.46, 0.28, 0.58),
    }
    too_small = [name for name, mask in regions.items() if np.count_nonzero(mask) < 64]
    if too_small:
        raise ValueError(
            "MHR topology face regions are too small: " + ", ".join(too_small)
        )
    return regions


def mhr_head_displacement_jacobians(
    model,
    head_mask: np.ndarray,
    *,
    device: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict,
]:
    import torch

    mask = np.asarray(head_mask, dtype=np.float32) > 0.5
    faces = model.character_torch.mesh.faces.detach().cpu().numpy().astype(np.int64)
    selected_faces = faces[np.all(mask[faces], axis=1)]
    used = np.unique(selected_faces)
    remap = np.full(len(mask), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    compact_faces = remap[selected_faces]
    batch = 1 + 20 + MHR_EXPRESSION_DIMENSION
    identity = torch.zeros((batch, MHR_IDENTITY_DIMENSION), device=device)
    expression = torch.zeros((batch, MHR_EXPRESSION_DIMENSION), device=device)
    pose = torch.zeros((batch, MHR_POSE_DIMENSION), device=device)
    identity[1:21, 20:40] = torch.eye(20, device=device)
    expression[21:] = torch.eye(MHR_EXPRESSION_DIMENSION, device=device)
    with torch.inference_mode():
        vertices, _skeleton = model(identity, pose, expression, False)
    compact = vertices[:, torch.as_tensor(used, device=device)].detach().cpu().numpy()
    neutral = compact[0]
    identity_jacobian = compact[1:21] - neutral
    expression_jacobian = compact[21:] - neutral

    triangles = neutral[compact_faces]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    vertex_areas = np.zeros(len(used), dtype=np.float64)
    for column in range(3):
        np.add.at(vertex_areas, compact_faces[:, column], face_areas / 3.0)
    positive = vertex_areas[vertex_areas > 0.0]
    if not len(positive):
        raise ValueError("MHR compact head has no positive vertex areas")
    vertex_areas = np.maximum(vertex_areas, np.percentile(positive, 1.0))
    vertex_areas /= np.mean(vertex_areas)
    region_masks = mhr_topology_face_regions(neutral)
    region_vertex_weights = {
        name: vertex_areas * region_mask for name, region_mask in region_masks.items()
    }
    combined_vertex_weights = vertex_areas.copy()
    total_area_weight = float(np.sum(vertex_areas))
    for region_weight in region_vertex_weights.values():
        combined_vertex_weights += (
            region_weight
            * total_area_weight
            / (len(region_vertex_weights) * float(np.sum(region_weight)))
        )
    combined_vertex_weights /= np.mean(combined_vertex_weights)
    return (
        identity_jacobian.astype(np.float32),
        expression_jacobian.astype(np.float32),
        combined_vertex_weights,
        region_vertex_weights,
        {
            "method": (
                "pinned-mhr-unit-blendshape-area-and-six-region-weighted-head-jacobian"
            ),
            "head_vertices": int(len(used)),
            "head_faces": int(len(compact_faces)),
            "identity_dimensions": 20,
            "expression_dimensions": MHR_EXPRESSION_DIMENSION,
            "vertex_area_weight_min": float(np.min(combined_vertex_weights)),
            "vertex_area_weight_max": float(np.max(combined_vertex_weights)),
            "topology_region_vertex_counts": {
                name: int(np.count_nonzero(region_mask))
                for name, region_mask in region_masks.items()
            },
            "region_weighting": "global-area-plus-equal-total-weight-six-regions",
            "source_geometry_used": False,
        },
    )


def decode_pca_delta(
    anchor: np.ndarray,
    basis: dict,
    latent_delta: np.ndarray,
    *,
    coefficient_limit: float,
) -> np.ndarray:
    anchor_values = np.asarray(anchor, dtype=np.float64).reshape(-1)
    delta = np.asarray(latent_delta, dtype=np.float64).reshape(-1)
    if anchor_values.shape != (int(basis["columns"]),):
        raise ValueError("PCA anchor has the wrong coefficient width")
    if delta.shape != (int(basis["rank"]),) or not np.all(np.isfinite(delta)):
        raise ValueError("PCA latent delta has the wrong finite shape")
    displacement = (delta * basis["scales"]) @ basis["components"]
    return np.clip(
        anchor_values + displacement,
        -float(coefficient_limit),
        float(coefficient_limit),
    ).astype(np.float32)


def strictly_better(candidate: dict, reference: dict) -> bool:
    part_sets_do_not_regress = all(
        set(candidate.get(name) or []).issubset(set(reference.get(name) or []))
        for name in ("shape_failed_parts", "affine_failed_parts")
    )
    return bool(
        int(candidate["combined_part_failures"])
        < int(reference["combined_part_failures"])
        and float(candidate["shape_correlation"])
        > float(reference["shape_correlation"])
        and float(candidate["gradient_correlation"])
        > float(reference["gradient_correlation"])
        and float(candidate["normalized_rmse"]) < float(reference["normalized_rmse"])
        and part_sets_do_not_regress
    )


def search_score(candidate: dict, reference: dict) -> float:
    failure_improves = int(candidate["combined_part_failures"]) < int(
        reference["combined_part_failures"]
    )
    metrics_strictly_improve = bool(
        float(candidate["shape_correlation"]) > float(reference["shape_correlation"])
        and float(candidate["gradient_correlation"])
        > float(reference["gradient_correlation"])
        and float(candidate["normalized_rmse"]) < float(reference["normalized_rmse"])
    )
    metrics_do_not_regress = bool(
        float(candidate["shape_correlation"]) >= float(reference["shape_correlation"])
        and float(candidate["gradient_correlation"])
        >= float(reference["gradient_correlation"])
        and float(candidate["normalized_rmse"]) <= float(reference["normalized_rmse"])
    )
    part_sets_do_not_regress = all(
        set(candidate.get(name) or []).issubset(set(reference.get(name) or []))
        for name in ("shape_failed_parts", "affine_failed_parts")
    )
    if failure_improves and metrics_strictly_improve and part_sets_do_not_regress:
        pareto_tier = 0
    elif metrics_do_not_regress and part_sets_do_not_regress:
        pareto_tier = 1
    else:
        pareto_tier = 2
    failure_score = 1000.0 * float(candidate["combined_part_failures"])
    regression_penalty = 100.0 * (
        max(
            0.0,
            float(reference["shape_correlation"])
            - float(candidate["shape_correlation"]),
        )
        + max(
            0.0,
            float(reference["gradient_correlation"])
            - float(candidate["gradient_correlation"]),
        )
        + max(
            0.0,
            float(candidate["normalized_rmse"]) - float(reference["normalized_rmse"]),
        )
    )
    quality_tiebreak = (
        -float(candidate["shape_correlation"])
        - float(candidate["gradient_correlation"])
        + float(candidate["normalized_rmse"])
    )
    return float(
        pareto_tier * 1_000_000.0
        + failure_score
        + regression_penalty
        + quality_tiebreak
    )


def clamp_search_initial(value: float, lower: float, upper: float) -> float:
    value = float(value)
    lower = float(lower)
    upper = float(upper)
    if not all(np.isfinite(item) for item in (value, lower, upper)) or lower > upper:
        raise ValueError("Search initial value and bounds must be finite and ordered")
    return float(np.clip(value, lower, upper))


def clear_output_artifacts(output_dir: str | Path) -> None:
    root = Path(output_dir)
    for artifact_name in (
        "evidence.json",
        "scratch_candidate.npy",
        "best_candidate_depth.npy",
        "best_provider_depth.npy",
        "strict_incumbent_candidate_depth.npy",
        "strict_incumbent_provider_depth.npy",
    ):
        (root / artifact_name).unlink(missing_ok=True)


def validate_training_partition(
    summary_rows: list[dict],
    manifest_rows: list[dict],
) -> dict[str, dict]:
    summary_train_rows = {
        str(row["row_id"]): row for row in summary_rows if row.get("split") == "train"
    }
    manifest_row_ids = {str(row["row_id"]) for row in manifest_rows}
    if manifest_row_ids != set(summary_train_rows):
        raise ValueError(
            "MHR supervision manifest does not match the authoritative train rows"
        )
    for manifest_row in manifest_rows:
        row_id = str(manifest_row["row_id"])
        identity_group = row_id.split("__scene_", 1)[0]
        summary_row = summary_train_rows[row_id]
        if (
            manifest_row.get("split") != "train"
            or summary_row.get("identity_group") != identity_group
            or (summary_row.get("spec") or {}).get("identity_group") != identity_group
            or (summary_row.get("spec") or {}).get("split") != "train"
        ):
            raise ValueError(
                "MHR supervision identity or split disagrees with the summary"
            )
    return summary_train_rows


def _load_training_family(
    corpus_root: Path,
    anchor_row_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    summary_path = corpus_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = _validate_training_corpus_summary(summary)
    manifest_record = summary["geometry_target_contract"]["supervision_manifest"]
    manifest_path = verified_corpus_asset(corpus_root, manifest_record)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("rows") or []
    if len(rows) != 240 or any(row.get("split") != "train" for row in rows):
        raise ValueError("Low-rank MHR audit requires exactly 240 train-only rows")
    if len({row.get("row_id") for row in rows}) != len(rows):
        raise ValueError("Low-rank MHR audit row IDs must be unique")
    validate_training_partition(summary["rows"], rows)

    identities: dict[str, np.ndarray] = {}
    expressions = []
    anchor_identity = None
    anchor_expression = None
    for row in rows:
        row_id = str(row["row_id"])
        identity_group = row_id.split("__scene_", 1)[0]
        targets = row["geometry_targets"]
        identity = np.load(
            verified_corpus_asset(corpus_root, targets["identity"])
        ).astype(np.float32)[20:40]
        expression = np.load(
            verified_corpus_asset(corpus_root, targets["expression"])
        ).astype(np.float32)
        previous = identities.setdefault(identity_group, identity)
        if not np.array_equal(previous, identity):
            raise ValueError("MHR identity coefficients changed within one identity")
        expressions.append(expression)
        if row_id == anchor_row_id:
            anchor_identity = identity
            anchor_expression = expression
    if len(identities) != 30:
        raise ValueError("Low-rank MHR audit requires 30 training identities")
    if anchor_identity is None or anchor_expression is None:
        raise ValueError(f"MHR anchor row is missing: {anchor_row_id}")
    return (
        np.stack(list(identities.values())),
        np.stack(expressions),
        anchor_identity,
        anchor_expression,
        {
            "contract": contract,
            "summary_sha256": _sha256(summary_path),
            "supervision_manifest_sha256": _sha256(manifest_path),
            "supervision_rows": len(rows),
            "identity_groups": len(identities),
            "all_basis_rows_are_train_only": True,
            "manifest_matches_summary_train_partition": True,
            "anchor_row_id": anchor_row_id,
        },
    )


def _load_incumbent(results_path: Path) -> tuple[dict, dict]:
    evidence = json.loads(results_path.read_text(encoding="utf-8"))
    record = evidence["exact_hard_row"]["incumbent"]
    required = {
        "combined_part_failures",
        "shape_correlation",
        "gradient_correlation",
        "normalized_rmse",
        "shape_failed_parts",
        "affine_failed_parts",
    }
    if not required.issubset(record):
        raise ValueError("Incumbent exact metrics have an unexpected schema")
    return dict(record), {
        "path": str(results_path),
        "sha256": _sha256(results_path),
    }


def evaluate(
    run_root: str | Path,
    corpus_root: str | Path,
    mhr_root: str | Path,
    incumbent_results: str | Path,
    output_dir: str | Path,
    *,
    anchor_row_id: str = DEFAULT_ANCHOR_ROW_ID,
    basis_mode: str = "geometry-active",
    identity_rank: int = 12,
    expression_rank: int = 34,
    latent_bound: float = 3.0,
    yaw_bound_degrees: float = 10.0,
    elevation_bound_degrees: float = 6.0,
    anchor_yaw_offset_degrees: float = 12.0,
    alpha_bounds: tuple[float, float] = (0.05, 0.60),
    population_multiplier: int = 3,
    maximum_iterations: int = 4,
    device: str = "cuda",
) -> dict:
    import torch

    run_root = Path(run_root).resolve()
    corpus_root = Path(corpus_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_output_artifacts(output_dir)
    started = time.perf_counter()
    if not np.isfinite(latent_bound) or not 0.0 < float(latent_bound) <= 3.0:
        raise ValueError("MHR latent bound must be finite and in (0, 3]")
    if not 1 <= int(population_multiplier) <= 16:
        raise ValueError("MHR population multiplier must be in [1, 16]")
    if not 0 <= int(maximum_iterations) <= 50:
        raise ValueError("MHR maximum iterations must be in [0, 50]")
    alpha_low, alpha_high = (float(value) for value in alpha_bounds)
    if not 0.0 < alpha_low < alpha_high <= 1.0:
        raise ValueError("MHR alpha bounds must satisfy 0 < low < high <= 1")
    if basis_mode not in {"coefficient", "geometry-active"}:
        raise ValueError(f"Unknown MHR basis mode: {basis_mode}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for MHR audit but is unavailable")

    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = _row_by_id(summary["rows"], HARD_ROW_ID)
    verified_declared_assets = {
        "source": verify_declared_asset(run_root, row["source"], "source image"),
        "selection_mask": verify_declared_asset(
            run_root,
            row["selection_mask"],
            "selection mask",
        ),
        "exact_depth": verify_declared_asset(
            run_root,
            row["exact_depth"],
            "exact depth",
        ),
        "candidate_request": verify_declared_asset(
            run_root,
            row["variants"]["candidate"]["request_artifact"],
            "candidate request",
        ),
        "candidate_response": verify_declared_asset(
            run_root,
            row["variants"]["candidate"]["response_artifact"],
            "candidate response",
        ),
        "exact_face_parts": {
            name: verify_declared_asset(
                run_root,
                record,
                f"exact face part {name}",
            )
            for name, record in row["exact_face_part_masks"]["files"].items()
        },
    }
    candidate_response = json.loads(
        verified_declared_assets["candidate_response"]["path"].read_text(
            encoding="utf-8"
        )
    )
    candidate_job_id = str(row["variants"]["candidate"]["job_id"])
    if candidate_response.get("job_id") != candidate_job_id:
        raise ValueError("Candidate response job ID does not match the summary")
    job_dir = run_root.parent / candidate_job_id
    job_dir = job_dir.resolve()
    metadata = json.loads(
        (job_dir / "output_face_refinement_metadata.json").read_text(encoding="utf-8")
    )
    face = metadata["faces"][0]
    crop_bbox = tuple(int(value) for value in face["crop_bbox"])
    support_crop = crop_mask(
        job_dir / face["part_masks"]["face_file"],
        crop_bbox,
    )
    x0, y0, x1, y1 = crop_bbox
    source_path = verified_declared_assets["source"]["path"]
    with Image.open(source_path) as loaded:
        image_width, image_height = loaded.size
    baseline = np.load(job_dir / "output_depth_data_face_refined.npy").astype(
        np.float32
    )
    local = np.load(
        job_dir / "face_refinement/face_00_depth/output_depth_data.npy"
    ).astype(np.float32)
    selection = (
        np.asarray(
            Image.open(verified_declared_assets["selection_mask"]["path"]).convert(
                "L"
            )
        )
        >= 128
    )
    consumed_job_assets = {
        "face_refinement_metadata": observed_asset(
            job_dir / "output_face_refinement_metadata.json",
            job_dir,
            "face refinement metadata",
        ),
        "baseline_depth": observed_asset(
            job_dir / "output_depth_data_face_refined.npy",
            job_dir,
            "baseline face-refined depth",
        ),
        "local_face_depth": observed_asset(
            job_dir / "face_refinement/face_00_depth/output_depth_data.npy",
            job_dir,
            "local face depth",
        ),
        "face_support_mask": observed_asset(
            job_dir / face["part_masks"]["face_file"],
            job_dir,
            "face support mask",
        ),
    }
    support = np.zeros_like(baseline, dtype=bool)
    support[y0:y1, x0:x1] = support_crop
    scratch_path = output_dir / "scratch_candidate.npy"
    np.save(scratch_path, baseline)
    baseline_quality = _compact_quality(_quality(run_root, row, scratch_path))
    incumbent, incumbent_provenance = _load_incumbent(Path(incumbent_results))

    (
        identity_rows,
        expression_rows,
        anchor_identity,
        anchor_expression,
        corpus_provenance,
    ) = _load_training_family(corpus_root, anchor_row_id)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    mhr_preflight = preflight_mhr_root(mhr_root)
    if not mhr_preflight["runnable"]:
        failed = [
            name for name, passed in mhr_preflight["checks"].items() if not passed
        ]
        raise RuntimeError("MHR preflight failed: " + ", ".join(failed))
    model, head_mask, mhr_runtime = load_mhr_geometry_runtime(
        mhr_root,
        device=device,
    )
    mhr_runtime["source_revision"] = mhr_preflight["revision"]
    mhr_runtime["expected_source_revision"] = mhr_preflight["expected_revision"]
    mhr_runtime["license"] = mhr_preflight["license"]
    mhr_runtime["preflight_checks"] = mhr_preflight["checks"]
    geometry_basis_provenance = None
    if basis_mode == "geometry-active":
        (
            identity_jacobian,
            expression_jacobian,
            vertex_area_weights,
            region_vertex_weights,
            geometry_basis_provenance,
        ) = mhr_head_displacement_jacobians(
            model,
            head_mask,
            device=device,
        )
        identity_basis = fit_geometry_pca(
            identity_rows,
            identity_jacobian,
            vertex_area_weights,
            identity_rank,
            region_vertex_weights=region_vertex_weights,
        )
        expression_basis = fit_geometry_pca(
            expression_rows,
            expression_jacobian,
            vertex_area_weights,
            expression_rank,
            region_vertex_weights=region_vertex_weights,
        )
        minimum_region_retention = min(
            float(identity_basis["minimum_region_retention"]),
            float(expression_basis["minimum_region_retention"]),
        )
        if minimum_region_retention < 0.95:
            raise ValueError(
                "Geometry-active MHR basis retains less than 95% in a face region: "
                f"{minimum_region_retention:.6f}"
            )
    else:
        identity_basis = fit_pca(identity_rows, identity_rank)
        expression_basis = fit_pca(expression_rows, expression_rank)

    identity_slice = slice(0, identity_rank)
    expression_slice = slice(identity_rank, identity_rank + expression_rank)
    yaw_index = identity_rank + expression_rank
    elevation_index = yaw_index + 1
    alpha_index = elevation_index + 1
    dimensions = alpha_index + 1
    bounds = (
        [(-float(latent_bound), float(latent_bound))] * identity_rank
        + [(-float(latent_bound), float(latent_bound))] * expression_rank
        + [
            (-float(yaw_bound_degrees), float(yaw_bound_degrees)),
            (-float(elevation_bound_degrees), float(elevation_bound_degrees)),
            (alpha_low, alpha_high),
        ]
    )
    initial = np.zeros(dimensions, dtype=np.float64)
    initial[yaw_index] = clamp_search_initial(
        anchor_yaw_offset_degrees,
        -float(yaw_bound_degrees),
        float(yaw_bound_degrees),
    )
    initial[alpha_index] = min(max(0.5, alpha_low), alpha_high)
    known_yaw = float(row["scene"]["camera_yaw_deg"])
    known_elevation = float(row["scene"]["camera_elevation_deg"])

    state = {
        "evaluations": 0,
        "failed_evaluations": 0,
        "best": None,
        "best_score": float("inf"),
        "strict_baseline": None,
        "strict_incumbent": None,
    }

    def objective(vector: np.ndarray) -> float:
        state["evaluations"] += 1
        try:
            identity = decode_pca_delta(
                anchor_identity,
                identity_basis,
                vector[identity_slice],
                coefficient_limit=IDENTITY_COEFFICIENT_LIMIT,
            )
            expression = decode_pca_delta(
                anchor_expression,
                expression_basis,
                vector[expression_slice],
                coefficient_limit=EXPRESSION_COEFFICIENT_LIMIT,
            )
            yaw_offset = float(vector[yaw_index])
            elevation_offset = float(vector[elevation_index])
            alpha = float(vector[alpha_index])
            crop_depth, geometry = render_mhr_camera_depth(
                model,
                head_mask,
                identity,
                expression,
                support_crop,
                yaw_degrees=known_yaw + yaw_offset,
                elevation_degrees=known_elevation + elevation_offset,
                device=device,
            )
            provider = np.full(
                (image_height, image_width),
                np.nan,
                dtype=np.float32,
            )
            provider[y0:y1, x0:x1] = crop_depth
            candidate, fusion = build_small_face_candidate(
                baseline,
                local,
                provider,
                support,
                selection,
                crop_bbox,
                provider_alpha=alpha,
            )
            np.save(scratch_path, candidate)
            quality = _compact_quality(_quality(run_root, row, scratch_path))
            score = search_score(quality, incumbent)
            record = {
                "evaluation": int(state["evaluations"]),
                "score": float(score),
                "yaw_offset_degrees": yaw_offset,
                "elevation_offset_degrees": elevation_offset,
                "yaw_degrees": known_yaw + yaw_offset,
                "elevation_degrees": known_elevation + elevation_offset,
                "alpha": alpha,
                "identity_latent_delta": np.asarray(
                    vector[identity_slice], dtype=np.float64
                ).tolist(),
                "expression_latent_delta": np.asarray(
                    vector[expression_slice], dtype=np.float64
                ).tolist(),
                "identity_coefficient_min": float(np.min(identity)),
                "identity_coefficient_max": float(np.max(identity)),
                "expression_coefficient_min": float(np.min(expression)),
                "expression_coefficient_max": float(np.max(expression)),
                "geometry": geometry,
                "fusion": fusion,
                **quality,
            }
            if score < float(state["best_score"]):
                state["best_score"] = float(score)
                state["best"] = record
                np.save(output_dir / "best_candidate_depth.npy", candidate)
                np.save(output_dir / "best_provider_depth.npy", provider)
            if strictly_better(record, baseline_quality) and (
                state["strict_baseline"] is None
                or score < search_score(state["strict_baseline"], incumbent)
            ):
                state["strict_baseline"] = record
            if strictly_better(record, incumbent) and (
                state["strict_incumbent"] is None
                or score < search_score(state["strict_incumbent"], incumbent)
            ):
                state["strict_incumbent"] = record
                np.save(output_dir / "strict_incumbent_candidate_depth.npy", candidate)
                np.save(output_dir / "strict_incumbent_provider_depth.npy", provider)
            if state["evaluations"] % 32 == 0:
                print(
                    json.dumps(
                        {
                            "evaluations": state["evaluations"],
                            "best_failures": state["best"]["combined_part_failures"],
                            "best_score": state["best_score"],
                            "strict_incumbent": state["strict_incumbent"] is not None,
                        }
                    ),
                    flush=True,
                )
            return float(score)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            state["failed_evaluations"] += 1
            if state["failed_evaluations"] <= 5:
                print(
                    json.dumps(
                        {
                            "evaluation": state["evaluations"],
                            "failure": f"{type(exc).__name__}: {exc}",
                        }
                    ),
                    flush=True,
                )
            return 1e12

    optimizer_started = time.perf_counter()
    result = differential_evolution(
        objective,
        bounds,
        seed=SEARCH_SEED,
        strategy="best1bin",
        maxiter=int(maximum_iterations),
        popsize=int(population_multiplier),
        tol=0.0,
        atol=0.0,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        init="sobol",
        updating="immediate",
        workers=1,
        x0=initial,
    )
    optimizer_seconds = float(time.perf_counter() - optimizer_started)
    scratch_path.unlink(missing_ok=True)
    if state["best"] is None:
        raise RuntimeError("Low-rank MHR audit produced no valid candidate")
    best_candidate = np.load(output_dir / "best_candidate_depth.npy")
    background_bit_exact = bool(
        np.array_equal(best_candidate[~selection], baseline[~selection])
    )
    peak_vram_gb = 0.0
    if device.startswith("cuda"):
        peak_vram_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))

    checks = {
        "diagnostic_only": True,
        "pose_oracle_diagnostic": True,
        "candidate_generation_uses_ground_truth_camera_pose": True,
        "source_geometry_used_only_for_scoring": False,
        "declared_input_assets_verified": True,
        "mhr_preflight_passed": bool(mhr_preflight["runnable"]),
        "basis_is_train_only": corpus_provenance["all_basis_rows_are_train_only"],
        "background_bit_exact": background_bit_exact,
        "all_evaluations_valid": state["failed_evaluations"] == 0,
        "strictly_beats_baseline": state["strict_baseline"] is not None,
        "strictly_beats_incumbent": state["strict_incumbent"] is not None,
        "global_upper_bound_claimed": False,
    }
    evidence = {
        "schema_version": 2,
        "method": METHOD,
        "status": "diagnostic-only",
        "promotion_eligible": False,
        "source_geometry": "evaluation-and-pose-oracle-metadata",
        "row_id": HARD_ROW_ID,
        "source_summary_sha256": _sha256(summary_path),
        "implementation": {
            "path": "backend/benchmark/evaluate_mhr_parametric_low_rank.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "input_assets": {
            "declared": {
                name: (
                    {
                        part: {
                            key: value
                            for key, value in asset.items()
                            if key != "path"
                        }
                        for part, asset in record.items()
                    }
                    if name == "exact_face_parts"
                    else {key: value for key, value in record.items() if key != "path"}
                )
                for name, record in verified_declared_assets.items()
            },
            "consumed_job": consumed_job_assets,
            "candidate_job_id": candidate_job_id,
        },
        "source_image_sha256": verified_declared_assets["source"]["sha256"],
        "selection_mask_sha256": verified_declared_assets["selection_mask"][
            "sha256"
        ],
        "mhr_runtime": mhr_runtime,
        "corpus": corpus_provenance,
        "basis": {
            "mode": basis_mode,
            "geometry": geometry_basis_provenance,
            "identity": {
                "rank": identity_basis["rank"],
                "rows": identity_basis["rows"],
                "columns": identity_basis["columns"],
                "explained_variance_ratio": identity_basis["explained_variance_ratio"],
                "geometry_metric_ridge": identity_basis.get("geometry_metric_ridge"),
                "weighted_vertices": identity_basis.get("weighted_vertices"),
                "region_retention": identity_basis.get("region_retention"),
                "minimum_region_retention": identity_basis.get(
                    "minimum_region_retention"
                ),
            },
            "expression": {
                "rank": expression_basis["rank"],
                "rows": expression_basis["rows"],
                "columns": expression_basis["columns"],
                "explained_variance_ratio": expression_basis[
                    "explained_variance_ratio"
                ],
                "geometry_metric_ridge": expression_basis.get("geometry_metric_ridge"),
                "weighted_vertices": expression_basis.get("weighted_vertices"),
                "region_retention": expression_basis.get("region_retention"),
                "minimum_region_retention": expression_basis.get(
                    "minimum_region_retention"
                ),
            },
            "anchor_row_id": anchor_row_id,
            "delta_bound_standard_deviations": float(latent_bound),
            "coefficient_clipping": {
                "identity": IDENTITY_COEFFICIENT_LIMIT,
                "expression": EXPRESSION_COEFFICIENT_LIMIT,
            },
        },
        "search": {
            "scope": "bounded-local-low-rank-continuous-search",
            "optimizer": "scipy.optimize.differential_evolution",
            "scipy_version": scipy.__version__,
            "seed": SEARCH_SEED,
            "dimensions": dimensions,
            "population_multiplier": int(population_multiplier),
            "maximum_iterations": int(maximum_iterations),
            "population_size": int(len(result.population)),
            "evaluations": int(state["evaluations"]),
            "failed_evaluations": int(state["failed_evaluations"]),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "optimizer_fun": float(result.fun),
            "optimizer_nit": int(result.nit),
            "optimizer_nfev": int(result.nfev),
            "yaw_offset_bound_degrees": float(yaw_bound_degrees),
            "elevation_offset_bound_degrees": float(elevation_bound_degrees),
            "alpha_bounds": [alpha_low, alpha_high],
            "known_pose_usage": "oracle-centered-candidate-generation",
            "known_pose_available_in_production": False,
            "global_or_complete_family_coverage": False,
            "runtime_seconds": optimizer_seconds,
            "device": device,
            "peak_vram_gb": peak_vram_gb,
        },
        "incumbent_provenance": incumbent_provenance,
        "baseline": baseline_quality,
        "incumbent": incumbent,
        "best": state["best"],
        "strict_baseline": state["strict_baseline"],
        "strict_incumbent": state["strict_incumbent"],
        "checks": checks,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    return evidence


def _parse_alpha_bounds(value: str) -> tuple[float, float]:
    values = tuple(float(item.strip()) for item in value.split(","))
    if len(values) != 2 or not all(np.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("Expected two finite comma-separated values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--mhr-root", required=True)
    parser.add_argument("--incumbent-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchor-row-id", default=DEFAULT_ANCHOR_ROW_ID)
    parser.add_argument(
        "--basis-mode",
        choices=("coefficient", "geometry-active"),
        default="geometry-active",
    )
    parser.add_argument("--identity-rank", type=int, default=12)
    parser.add_argument("--expression-rank", type=int, default=34)
    parser.add_argument("--latent-bound", type=float, default=3.0)
    parser.add_argument("--yaw-bound-degrees", type=float, default=10.0)
    parser.add_argument("--elevation-bound-degrees", type=float, default=6.0)
    parser.add_argument("--anchor-yaw-offset-degrees", type=float, default=12.0)
    parser.add_argument(
        "--alpha-bounds",
        type=_parse_alpha_bounds,
        default=(0.05, 0.60),
    )
    parser.add_argument("--population-multiplier", type=int, default=3)
    parser.add_argument("--maximum-iterations", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evidence = evaluate(
        args.run_root,
        args.corpus_root,
        args.mhr_root,
        args.incumbent_results,
        args.output_dir,
        anchor_row_id=args.anchor_row_id,
        basis_mode=args.basis_mode,
        identity_rank=args.identity_rank,
        expression_rank=args.expression_rank,
        latent_bound=args.latent_bound,
        yaw_bound_degrees=args.yaw_bound_degrees,
        elevation_bound_degrees=args.elevation_bound_degrees,
        anchor_yaw_offset_degrees=args.anchor_yaw_offset_degrees,
        alpha_bounds=args.alpha_bounds,
        population_multiplier=args.population_multiplier,
        maximum_iterations=args.maximum_iterations,
        device=args.device,
    )
    print(json.dumps(evidence["checks"], indent=2))


if __name__ == "__main__":
    main()
