"""Contracts for a frozen-feature spatial face-depth pyramid."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence


FEATURE_LEVELS = 4
FEATURE_CHANNELS = (64, 64, 64, 64)
CONDITIONING_CHANNELS = 9


@dataclass(frozen=True)
class SpatialPyramidConfig:
    feature_channels: tuple[int, ...] = FEATURE_CHANNELS
    pyramid_channels: int = 24
    conditioning_channels: int = CONDITIONING_CHANNELS
    maximum_residual: float = 0.30

    def validated(self) -> "SpatialPyramidConfig":
        if len(self.feature_channels) != FEATURE_LEVELS:
            raise ValueError("Spatial face pyramid requires exactly four feature levels")
        if any(int(value) < 1 for value in self.feature_channels):
            raise ValueError("Spatial face pyramid feature widths must be positive")
        if not 8 <= int(self.pyramid_channels) <= 128:
            raise ValueError("Spatial face pyramid width must be in [8, 128]")
        if int(self.conditioning_channels) != CONDITIONING_CHANNELS:
            raise ValueError(
                f"Spatial face conditioning must contain {CONDITIONING_CHANNELS} channels"
            )
        if not math.isfinite(float(self.maximum_residual)) or not 0.01 <= float(
            self.maximum_residual
        ) <= 1.0:
            raise ValueError("Spatial face residual cap must be in [0.01, 1.0]")
        return self

    def provenance(self) -> dict:
        self.validated()
        return {
            **asdict(self),
            "feature_levels": FEATURE_LEVELS,
            "conditioning_layout": (
                "coarse_camera_z,rays_xyz,face_support,attachment_boundary,"
                "rgb_laplacian_luma_3scale"
            ),
            "output": "bounded-supported-camera-z-residual",
        }


@dataclass(frozen=True)
class SpatialPyramidLossWeights:
    camera_z: float = 0.60
    gradient: float = 0.40
    gradient_correlation: float = 0.0
    normal: float = 0.20
    laplacian: float = 0.15
    attachment: float = 0.20
    part_non_regression: float = 0.30
    residual: float = 0.02

    def validated(self) -> "SpatialPyramidLossWeights":
        values = asdict(self)
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in values.values()
        ):
            raise ValueError("Spatial face loss weights must be finite and nonnegative")
        if not any(float(value) > 0.0 for value in values.values()):
            raise ValueError("Spatial face loss must contain a positive term")
        return self


def parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def matched_deepest_control_features(features: Sequence):
    """Replace every spatial scale with the same deepest feature tensor."""

    import torch.nn.functional as functional

    if len(features) != FEATURE_LEVELS:
        raise ValueError("Matched control requires four feature maps")
    deepest = features[-1]
    return tuple(
        functional.interpolate(
            deepest,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        for feature in features
    )


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _residual_block(channels: int):
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(_group_count(channels), channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(_group_count(channels), channels),
            )
            self.activation = nn.SiLU()

        def forward(self, values):
            return self.activation(values + self.layers(values))

    return ResidualBlock()


def build_spatial_pyramid(config: SpatialPyramidConfig = SpatialPyramidConfig()):
    """Build a top-down FPN that predicts one supported camera-Z residual."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    config = config.validated()
    width = int(config.pyramid_channels)

    class SpatialFacePyramid(nn.Module):
        def __init__(self):
            super().__init__()
            self.lateral = nn.ModuleList(
                nn.Conv2d(int(channels), width, 1)
                for channels in config.feature_channels
            )
            self.refine = nn.ModuleList(
                _residual_block(width) for _ in config.feature_channels
            )
            fused_channels = width * FEATURE_LEVELS + int(
                config.conditioning_channels
            )
            self.fuse_input = nn.Sequential(
                nn.Conv2d(fused_channels, width, 3, padding=1),
                nn.GroupNorm(_group_count(width), width),
                nn.SiLU(),
            )
            self.fuse = nn.Sequential(
                _residual_block(width),
                _residual_block(width),
            )
            self.output = nn.Conv2d(width, 1, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        def forward(self, features, conditioning, support, boundary):
            if len(features) != FEATURE_LEVELS:
                raise ValueError("Spatial face pyramid received the wrong feature count")
            batch = conditioning.shape[0]
            if (
                conditioning.ndim != 4
                or conditioning.shape[1] != config.conditioning_channels
            ):
                raise ValueError("Spatial face conditioning has the wrong shape")
            if support.shape != (batch, 1, *conditioning.shape[-2:]):
                raise ValueError("Spatial face support does not match conditioning")
            if boundary.shape != support.shape:
                raise ValueError("Spatial face attachment boundary does not match support")
            if not (
                torch.isfinite(conditioning).all()
                and torch.isfinite(support).all()
                and torch.isfinite(boundary).all()
            ):
                raise ValueError("Spatial face conditioning must be finite")
            for index, (feature, channels) in enumerate(
                zip(features, config.feature_channels, strict=True)
            ):
                if (
                    feature.ndim != 4
                    or feature.shape[0] != batch
                    or feature.shape[1] != channels
                    or not torch.isfinite(feature).all()
                ):
                    raise ValueError(f"Spatial feature level {index} is invalid")

            pyramid = [None] * FEATURE_LEVELS
            top = None
            for index in reversed(range(FEATURE_LEVELS)):
                lateral = self.lateral[index](features[index])
                if top is not None:
                    top = functional.interpolate(
                        top,
                        size=lateral.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    lateral = lateral + top
                top = self.refine[index](lateral)
                pyramid[index] = top
            target_shape = conditioning.shape[-2:]
            packed = [
                functional.interpolate(
                    values,
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                )
                for values in pyramid
            ]
            packed.append(conditioning)
            hidden = self.fuse(self.fuse_input(torch.cat(packed, dim=1)))
            residual = float(config.maximum_residual) * torch.tanh(
                self.output(hidden)
            )
            supported = torch.clamp(support, 0.0, 1.0) * (
                1.0 - torch.clamp(boundary, 0.0, 1.0)
            )
            return residual * supported

    return SpatialFacePyramid()


def camera_points_from_z(camera_z, rays):
    """Back-project camera-Z through unit rays without assuming orthography."""

    import torch

    if camera_z.ndim != 4 or camera_z.shape[1] != 1:
        raise ValueError("Camera-Z must have shape [B, 1, H, W]")
    if rays.shape != (camera_z.shape[0], 3, *camera_z.shape[-2:]):
        raise ValueError("Camera rays do not match camera-Z")
    if not torch.isfinite(camera_z).all() or not torch.isfinite(rays).all():
        raise ValueError("Camera back-projection inputs must be finite")
    distance = camera_z / rays[:, 2:3].clamp_min(1e-6)
    return rays * distance


def camera_surface_normals(camera_z, rays):
    """Estimate oriented camera-space normals from adjacent back-projected points."""

    import torch
    import torch.nn.functional as functional

    points = camera_points_from_z(camera_z, rays)
    tangent_x = points[..., :-1, 1:] - points[..., :-1, :-1]
    tangent_y = points[..., 1:, :-1] - points[..., :-1, :-1]
    normals = torch.linalg.cross(tangent_x, tangent_y, dim=1)
    return functional.normalize(normals, dim=1, eps=1e-8)


def _weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _smooth_l1(prediction, target, weights):
    import torch.nn.functional as functional

    return _weighted_mean(
        functional.smooth_l1_loss(prediction, target, reduction="none", beta=0.02),
        weights,
    )


def _axis_gradient_loss(prediction, target, mask):
    valid_x = mask[..., :, 1:] * mask[..., :, :-1]
    valid_y = mask[..., 1:, :] * mask[..., :-1, :]
    error_x = (prediction[..., :, 1:] - prediction[..., :, :-1]) - (
        target[..., :, 1:] - target[..., :, :-1]
    )
    error_y = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :]
    )
    return _weighted_mean(error_x.abs(), valid_x) + _weighted_mean(
        error_y.abs(), valid_y
    )


def _masked_correlation_loss(first, second, weights):
    count = weights.sum().clamp_min(1.0)
    first_mean = (first * weights).sum() / count
    second_mean = (second * weights).sum() / count
    centered_first = first - first_mean
    centered_second = second - second_mean
    covariance = (centered_first * centered_second * weights).sum()
    first_norm = torch_sqrt((centered_first.square() * weights).sum())
    second_norm = torch_sqrt((centered_second.square() * weights).sum())
    correlation = covariance / (first_norm * second_norm).clamp_min(1e-8)
    return 1.0 - correlation.clamp(-1.0, 1.0)


def torch_sqrt(values):
    import torch

    return torch.sqrt(values)


def _axis_gradient_correlation_loss(prediction, target, mask):
    valid_x = mask[..., :, 1:] * mask[..., :, :-1]
    valid_y = mask[..., 1:, :] * mask[..., :-1, :]
    prediction_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return torch_maximum(
        _masked_correlation_loss(prediction_x, target_x, valid_x),
        _masked_correlation_loss(prediction_y, target_y, valid_y),
    )


def torch_maximum(first, second):
    import torch

    return torch.maximum(first, second)


def _equal_part_mean(function, prediction, target, parts):
    values = [
        function(prediction, target, part)
        for part in parts
        if int(part.count_nonzero()) >= 1
    ]
    if not values:
        raise ValueError("Spatial face loss needs at least one nonempty named part")
    return sum(values) / len(values)


def spatial_pyramid_loss(
    residual,
    coarse,
    target,
    support,
    parts,
    boundary,
    rays,
    *,
    weights: SpatialPyramidLossWeights = SpatialPyramidLossWeights(),
):
    """Apply equal-part camera geometry losses and a baseline non-regression hinge."""

    import torch
    import torch.nn.functional as functional

    weights = weights.validated()
    tensors = (residual, coarse, target, support, boundary)
    if any(value.ndim != 4 or value.shape[1] != 1 for value in tensors):
        raise ValueError("Spatial face loss tensors must have shape [B, 1, H, W]")
    if any(value.shape != target.shape for value in tensors):
        raise ValueError("Spatial face loss tensors must share one shape")
    if rays.shape != (target.shape[0], 3, *target.shape[-2:]):
        raise ValueError("Spatial face loss rays do not match the target")
    if not parts:
        raise ValueError("Spatial face loss requires named part masks")
    prediction = coarse + residual
    face_value = _smooth_l1(prediction, target, support)
    part_value = _equal_part_mean(_smooth_l1, prediction, target, parts)
    face_gradient = _axis_gradient_loss(prediction, target, support)
    part_gradient = _equal_part_mean(
        _axis_gradient_loss, prediction, target, parts
    )
    face_gradient_correlation = _axis_gradient_correlation_loss(
        prediction, target, support
    )
    part_gradient_correlation = _equal_part_mean(
        _axis_gradient_correlation_loss, prediction, target, parts
    )

    predicted_normals = camera_surface_normals(prediction, rays)
    target_normals = camera_surface_normals(target, rays)
    normal_support = (
        support[..., 1:, 1:]
        * support[..., 1:, :-1]
        * support[..., :-1, 1:]
        * support[..., :-1, :-1]
    )
    normal = _weighted_mean(
        1.0 - (predicted_normals * target_normals).sum(dim=1, keepdim=True),
        normal_support,
    )
    laplace_kernel = torch.tensor(
        ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0)),
        device=target.device,
        dtype=target.dtype,
    )[None, None]
    predicted_laplace = functional.conv2d(prediction, laplace_kernel, padding=1)
    target_laplace = functional.conv2d(target, laplace_kernel, padding=1)
    laplacian = _weighted_mean(
        (predicted_laplace - target_laplace).abs(), support
    )
    attachment_band = functional.max_pool2d(boundary, 5, stride=1, padding=2)
    attachment = _weighted_mean(residual.abs(), attachment_band * support)

    non_regression_terms = []
    for part in parts:
        if int(part.count_nonzero()) < 1:
            continue
        candidate_value = _smooth_l1(prediction, target, part)
        baseline_value = _smooth_l1(coarse, target, part)
        candidate_gradient = _axis_gradient_loss(prediction, target, part)
        baseline_gradient = _axis_gradient_loss(coarse, target, part)
        non_regression_terms.append(
            0.5
            * (
                functional.relu(candidate_value - baseline_value)
                + functional.relu(candidate_gradient - baseline_gradient)
            )
        )
    part_non_regression = sum(non_regression_terms) / len(non_regression_terms)
    residual_regularization = _weighted_mean(residual.abs(), support)
    camera_z = 0.5 * (face_value + part_value)
    gradient = 0.5 * (face_gradient + part_gradient)
    gradient_correlation = 0.5 * (
        face_gradient_correlation + part_gradient_correlation
    )
    total = (
        float(weights.camera_z) * camera_z
        + float(weights.gradient) * gradient
        + float(weights.gradient_correlation) * gradient_correlation
        + float(weights.normal) * normal
        + float(weights.laplacian) * laplacian
        + float(weights.attachment) * attachment
        + float(weights.part_non_regression) * part_non_regression
        + float(weights.residual) * residual_regularization
    )
    return total, {
        "camera_z": camera_z.detach(),
        "gradient": gradient.detach(),
        "gradient_correlation": gradient_correlation.detach(),
        "normal": normal.detach(),
        "laplacian": laplacian.detach(),
        "attachment": attachment.detach(),
        "part_non_regression": part_non_regression.detach(),
        "residual": residual_regularization.detach(),
    }, prediction
