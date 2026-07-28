"""Train coarse-to-fine face geometry against the exact part selector."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from backend.benchmark import train_coarse_to_fine_face_geometry as coarse
from backend.benchmark.train_face_surface_fusion_adapter import (
    apply_training_residual,
)


METHOD = (
    "dinov2_pyramid448_coarse_to_fine_normal_"
    "selector_aligned_face_geometry"
)
DEFAULT_SELECTOR_MEAN_NON_REGRESSION_WEIGHT = 4.0
DEFAULT_SELECTOR_WORST_NON_REGRESSION_WEIGHT = 8.0
OWNED_PATH = "backend/benchmark/train_selector_aligned_face_geometry.py"
_BASE_GEOMETRY_LOSS = coarse.geometry_loss


def _eroded_or_original(mask, *, radius: int, minimum_samples: int):
    import torch.nn.functional as functional

    values = mask[None, None].float()
    width = 2 * int(radius) + 1
    eroded = (
        functional.avg_pool2d(
            values,
            kernel_size=width,
            stride=1,
            padding=int(radius),
        )[0, 0]
        >= 1.0 - 1e-6
    )
    original_count = int(mask.count_nonzero().item())
    eroded_count = int(eroded.count_nonzero().item())
    if (
        eroded_count >= int(minimum_samples)
        and eroded_count / max(original_count, 1) >= 0.35
    ):
        return eroded
    return mask


def _quantile_span(values):
    import torch

    if values.numel() == 0:
        return values.new_tensor(0.0)
    quantiles = torch.quantile(
        values.float(),
        torch.tensor((0.05, 0.95), device=values.device),
    )
    return (quantiles[1] - quantiles[0]).clamp_min(1e-6)


def _centered_correlation(first, second):
    if first.numel() < 2 or second.numel() < 2:
        return first.new_tensor(0.0)
    first = first.float() - first.float().mean()
    second = second.float() - second.float().mean()
    denominator = (
        first.square().sum().sqrt() * second.square().sum().sqrt()
    ).clamp_min(1e-6)
    return (first * second).sum() / denominator


def _shared_affine_expected(reference, candidate, mask):
    reference_values = reference[mask].float()
    candidate_values = candidate[mask].float()
    reference_centered = reference_values - reference_values.mean()
    candidate_centered = candidate_values - candidate_values.mean()
    scale = (
        reference_centered * candidate_centered
    ).sum() / reference_centered.square().sum().clamp_min(1e-6)
    offset = candidate_values.mean() - scale * reference_values.mean()
    return scale * reference.float() + offset


def _raw_gradient_correlation(reference, candidate, mask):
    import torch

    reference_x = reference[:, 1:] - reference[:, :-1]
    candidate_x = candidate[:, 1:] - candidate[:, :-1]
    valid_x = mask[:, 1:] & mask[:, :-1]
    reference_y = reference[1:, :] - reference[:-1, :]
    candidate_y = candidate[1:, :] - candidate[:-1, :]
    valid_y = mask[1:, :] & mask[:-1, :]
    return torch.minimum(
        _centered_correlation(reference_x[valid_x], candidate_x[valid_x]),
        _centered_correlation(reference_y[valid_y], candidate_y[valid_y]),
    )


def selector_aligned_part_non_regression(
    prediction,
    baseline,
    target,
    face_mask,
    part_masks,
):
    """Approximate exact named-part selector metrics during training."""

    import torch
    import torch.nn.functional as functional

    if not (
        prediction.shape == baseline.shape == target.shape == face_mask.shape
    ):
        raise ValueError("Selector-aligned face tensors disagree")
    if (
        part_masks.ndim != 4
        or part_masks.shape[0] != prediction.shape[0]
        or part_masks.shape[-2:] != prediction.shape[-2:]
    ):
        raise ValueError("Selector-aligned part masks disagree")
    row_means = []
    row_worsts = []
    metric_totals = {
        "affine_rmse": [],
        "affine_bias": [],
        "affine_p95": [],
        "affine_span": [],
        "shape_rmse": [],
        "shape_correlation": [],
        "raw_gradient_correlation": [],
    }
    for row in range(prediction.shape[0]):
        face = _eroded_or_original(
            face_mask[row, 0] > 0.5,
            radius=2,
            minimum_samples=64,
        )
        current = prediction[row, 0].float()
        prior = baseline[row, 0].float()
        exact = target[row, 0].float()
        current_expected = _shared_affine_expected(exact, current, face)
        prior_expected = _shared_affine_expected(exact, prior, face)

        exact_median = torch.quantile(exact[face], 0.5)
        current_median = torch.quantile(current[face], 0.5)
        prior_median = torch.quantile(prior[face], 0.5)
        exact_normalized = (exact - exact_median) / _quantile_span(exact[face])
        current_normalized = (
            current - current_median
        ) / _quantile_span(current[face])
        prior_normalized = (
            prior - prior_median
        ) / _quantile_span(prior[face])
        row_losses = []
        for part_index in range(part_masks.shape[1]):
            part = _eroded_or_original(
                (part_masks[row, part_index] > 0.5) & face,
                radius=1,
                minimum_samples=12,
            )
            if int(part.count_nonzero().item()) < 2:
                continue
            current_error = current[part] - current_expected[part]
            prior_error = prior[part] - prior_expected[part]
            current_expected_span = _quantile_span(current_expected[part])
            prior_expected_span = _quantile_span(prior_expected[part])
            current_affine = (
                current_error.square().mean().sqrt() / current_expected_span,
                current_error.mean().abs() / current_expected_span,
                torch.quantile(current_error.abs(), 0.95)
                / current_expected_span,
                (
                    _quantile_span(current[part]) / current_expected_span
                    - 1.0
                ).abs(),
            )
            prior_affine = (
                prior_error.square().mean().sqrt() / prior_expected_span,
                prior_error.mean().abs() / prior_expected_span,
                torch.quantile(prior_error.abs(), 0.95)
                / prior_expected_span,
                (
                    _quantile_span(prior[part]) / prior_expected_span - 1.0
                ).abs(),
            )
            current_shape_rmse = (
                (current_normalized[part] - exact_normalized[part])
                .square()
                .mean()
                .sqrt()
            )
            prior_shape_rmse = (
                (prior_normalized[part] - exact_normalized[part])
                .square()
                .mean()
                .sqrt()
            )
            current_shape_correlation = _centered_correlation(
                exact_normalized[part],
                current_normalized[part],
            )
            prior_shape_correlation = _centered_correlation(
                exact_normalized[part],
                prior_normalized[part],
            )
            current_gradient_correlation = _raw_gradient_correlation(
                exact_normalized,
                current_normalized,
                part,
            )
            prior_gradient_correlation = _raw_gradient_correlation(
                exact_normalized,
                prior_normalized,
                part,
            )
            losses = (
                functional.relu(current_affine[0] - prior_affine[0]),
                functional.relu(current_affine[1] - prior_affine[1]),
                functional.relu(current_affine[2] - prior_affine[2]),
                functional.relu(current_affine[3] - prior_affine[3]),
                functional.relu(current_shape_rmse - prior_shape_rmse),
                functional.relu(
                    prior_shape_correlation - current_shape_correlation
                ),
                functional.relu(
                    prior_gradient_correlation
                    - current_gradient_correlation
                ),
            )
            for name, loss in zip(metric_totals, losses, strict=True):
                metric_totals[name].append(loss)
            row_losses.extend(losses)
        if row_losses:
            stacked = torch.stack(row_losses)
            row_means.append(stacked.mean())
            row_worsts.append(stacked.max())
    if not row_means:
        zero = prediction.sum() * 0.0
        return zero, zero, {name: 0.0 for name in metric_totals}
    mean_loss = torch.stack(row_means).mean()
    worst_loss = torch.stack(row_worsts).mean()
    details = {
        name: float(torch.stack(values).mean().detach())
        if values
        else 0.0
        for name, values in metric_totals.items()
    }
    return mean_loss, worst_loss, details


def selector_geometry_loss(
    residual,
    coarse_residual,
    tensors: dict,
    *,
    coarse_loss_weight: float = coarse.DEFAULT_COARSE_LOSS_WEIGHT,
    fine_low_frequency_weight: float = coarse.DEFAULT_FINE_LOW_FREQUENCY_WEIGHT,
    selector_mean_non_regression_weight: float = (
        DEFAULT_SELECTOR_MEAN_NON_REGRESSION_WEIGHT
    ),
    selector_worst_non_regression_weight: float = (
        DEFAULT_SELECTOR_WORST_NON_REGRESSION_WEIGHT
    ),
):
    total, details = _BASE_GEOMETRY_LOSS(
        residual,
        coarse_residual,
        tensors,
        coarse_loss_weight=coarse_loss_weight,
        fine_low_frequency_weight=fine_low_frequency_weight,
    )
    prediction, _correction, _stats = apply_training_residual(
        residual, tensors
    )
    selector_mean, selector_worst, selector_details = (
        selector_aligned_part_non_regression(
            prediction,
            tensors["baseline"],
            tensors["target"],
            tensors["exact_face"],
            tensors["parts_individual"],
        )
    )
    total = (
        total
        + float(selector_mean_non_regression_weight) * selector_mean
        + float(selector_worst_non_regression_weight) * selector_worst
    )
    return total, {
        **details,
        "selector_mean_non_regression": float(selector_mean.detach()),
        "selector_worst_non_regression": float(selector_worst.detach()),
        **{
            f"selector_{name}_non_regression": value
            for name, value in selector_details.items()
        },
    }


def _code_provenance(original):
    provenance = original()
    repo = Path(__file__).resolve().parents[2]
    path = repo / OWNED_PATH
    provenance["files"][OWNED_PATH] = {
        "sha256": coarse.base._file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    safe_directory = f"safe.directory={repo.as_posix()}"
    try:
        status = subprocess.run(
            (
                "git",
                "-c",
                safe_directory,
                "status",
                "--porcelain",
                "--",
                OWNED_PATH,
                *coarse.EXECUTION_CRITICAL_PATHS,
            ),
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        status = ["git-status-unavailable"]
    provenance["owned_git_status"] = status
    return provenance


def run_experiment(args: argparse.Namespace) -> dict:
    import torch

    original_method = coarse.METHOD
    original_loss = coarse.geometry_loss
    original_provenance = coarse._code_provenance
    try:
        coarse.METHOD = METHOD
        coarse.geometry_loss = lambda residual, stage, tensors, **options: (
            selector_geometry_loss(
                residual,
                stage,
                tensors,
                **options,
                selector_mean_non_regression_weight=(
                    args.selector_mean_non_regression_weight
                ),
                selector_worst_non_regression_weight=(
                    args.selector_worst_non_regression_weight
                ),
            )
        )
        coarse._code_provenance = lambda: _code_provenance(
            original_provenance
        )
        evidence = coarse.run_experiment(args)
    finally:
        coarse.METHOD = original_method
        coarse.geometry_loss = original_loss
        coarse._code_provenance = original_provenance

    selector_options = {
        "selector_mean_non_regression_weight": float(
            args.selector_mean_non_regression_weight
        ),
        "selector_worst_non_regression_weight": float(
            args.selector_worst_non_regression_weight
        ),
    }
    output = Path(args.output_dir)
    checkpoint_path = output / "coarse_to_fine_face_geometry.pt"
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    payload["training"]["loss_options"].update(selector_options)
    torch.save(payload, checkpoint_path)
    evidence["training"]["loss_options"].update(selector_options)
    evidence["architecture"]["selector_alignment"] = {
        "shared_face_affine_fit": True,
        "part_metrics": [
            "rmse",
            "bias",
            "p95_absolute_error",
            "span_retention",
            "shape_rmse",
            "shape_correlation",
            "raw_gradient_correlation",
        ],
        **selector_options,
    }
    evidence["checkpoint"] = {
        "path": checkpoint_path.name,
        "sha256": coarse.base._file_sha256(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("overfit", "split"), default="overfit")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=coarse.DEFAULT_OVERFIT_ROWS)
    parser.add_argument("--epochs", type=int, default=coarse.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=coarse.DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=coarse.DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--maximum-loss-ratio",
        type=float,
        default=coarse.DEFAULT_OVERFIT_MAXIMUM_RATIO,
    )
    parser.add_argument(
        "--coarse-loss-weight",
        type=float,
        default=coarse.DEFAULT_COARSE_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--fine-low-frequency-weight",
        type=float,
        default=coarse.DEFAULT_FINE_LOW_FREQUENCY_WEIGHT,
    )
    parser.add_argument(
        "--selector-mean-non-regression-weight",
        type=float,
        default=DEFAULT_SELECTOR_MEAN_NON_REGRESSION_WEIGHT,
    )
    parser.add_argument(
        "--selector-worst-non-regression-weight",
        type=float,
        default=DEFAULT_SELECTOR_WORST_NON_REGRESSION_WEIGHT,
    )
    parser.add_argument(
        "--small-face-weight-reference-px",
        type=float,
        default=coarse.DEFAULT_SMALL_FACE_WEIGHT_REFERENCE_PX,
    )
    parser.add_argument(
        "--maximum-small-face-sample-weight",
        type=float,
        default=coarse.DEFAULT_MAXIMUM_SMALL_FACE_SAMPLE_WEIGHT,
    )
    parser.add_argument(
        "--yaw-sample-weight-strength",
        type=float,
        default=coarse.base.DEFAULT_YAW_SAMPLE_WEIGHT_STRENGTH,
    )
    parser.add_argument(
        "--maximum-combined-sample-weight",
        type=float,
        default=coarse.base.DEFAULT_MAXIMUM_COMBINED_SAMPLE_WEIGHT,
    )
    parser.add_argument("--expected-corpus-summary-sha256", required=True)
    parser.add_argument("--expected-cache-row-sha256", required=True)
    args = parser.parse_args()
    evidence = run_experiment(args)
    print(
        json.dumps(
            evidence.get("capacity_gate", evidence.get("decision")),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
