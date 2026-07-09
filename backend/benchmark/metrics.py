from __future__ import annotations

import math

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_rgb(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) > 127


def load_bool_mask(path):
    return load_mask(path)


def psnr_rgb(reference, prediction, mask=None):
    ref, pred = _masked_values(reference, prediction, mask)
    if ref.size == 0:
        return math.nan
    mse = float(np.mean((ref - pred) ** 2))
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def ssim_rgb(reference, prediction, mask=None):
    min_dim = min(reference.shape[:2])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        return math.nan
    score, similarity = structural_similarity(
        reference,
        prediction,
        channel_axis=2,
        data_range=1.0,
        win_size=win_size,
        full=True,
    )
    if mask is None:
        return float(score)
    if similarity.ndim == 3:
        similarity = similarity.mean(axis=2)
    values = similarity[mask]
    if values.size == 0:
        return math.nan
    return float(np.mean(values))


def mae_rgb(reference, prediction, mask=None):
    ref, pred = _masked_values(reference, prediction, mask)
    if ref.size == 0:
        return math.nan
    return float(np.mean(np.abs(ref - pred)))


def seam_mae(reference, prediction, mask, band_px=8):
    from scipy.ndimage import binary_dilation

    band = binary_dilation(mask, iterations=band_px) & binary_dilation(~mask, iterations=band_px)
    return mae_rgb(reference, prediction, band)


def fit_scale_shift(prediction, target, fit_mask):
    x = prediction[fit_mask].reshape(-1)
    y = target[fit_mask].reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return 1.0, 0.0
    a = np.stack([x, np.ones_like(x)], axis=1)
    scale, shift = np.linalg.lstsq(a, y, rcond=None)[0]
    return float(scale), float(shift)


def align_depth(prediction, target, fit_mask):
    scale, shift = fit_scale_shift(prediction, target, fit_mask)
    return prediction * scale + shift, scale, shift


def depth_metrics(reference, prediction, eval_mask, fit_mask=None):
    if fit_mask is None:
        fit_mask = ~eval_mask
    aligned, scale, shift = align_depth(prediction, reference, fit_mask)
    ref = reference[eval_mask]
    pred = aligned[eval_mask]
    valid = np.isfinite(ref) & np.isfinite(pred)
    if not np.any(valid):
        return {"depth_scale": scale, "depth_shift": shift, "depth_mae": math.nan, "depth_rmse": math.nan, "depth_corr": math.nan}
    ref, pred = ref[valid], pred[valid]
    error = pred - ref
    corr = np.corrcoef(ref, pred)[0, 1] if len(ref) > 2 and np.std(ref) > 0 and np.std(pred) > 0 else math.nan
    return {
        "depth_scale": scale,
        "depth_shift": shift,
        "depth_mae": float(np.mean(np.abs(error))),
        "depth_rmse": float(np.sqrt(np.mean(error**2))),
        "depth_corr": float(corr),
    }


def silhouette_iou(reference_depth, prediction_depth, mask=None, threshold=0.98):
    ref = reference_depth < threshold
    pred = prediction_depth < threshold
    if mask is not None:
        ref = ref & mask
        pred = pred & mask
    union = np.logical_or(ref, pred).sum()
    if union == 0:
        return math.nan
    return float(np.logical_and(ref, pred).sum() / union)


def silhouette_iou_from_reference_mask(reference_mask, prediction_depth, mask=None, threshold=0.98):
    ref = reference_mask.astype(bool)
    pred = prediction_depth < threshold
    if mask is not None:
        ref = ref & mask
        pred = pred & mask
    union = np.logical_or(ref, pred).sum()
    if union == 0:
        return math.nan
    return float(np.logical_and(ref, pred).sum() / union)


def depth_to_point_cloud(depth, mask, max_points=4096):
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    valid = mask & np.isfinite(depth)
    if depth.ndim != 2 or mask.shape != depth.shape or not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    rows, cols = np.nonzero(valid)
    z = depth[rows, cols]
    scale = max(depth.shape[0] - 1, depth.shape[1] - 1, 1)
    x = (cols.astype(np.float32) - (depth.shape[1] - 1) / 2.0) / scale
    y = (rows.astype(np.float32) - (depth.shape[0] - 1) / 2.0) / scale
    points = np.stack([x, y, z.astype(np.float32)], axis=1)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
        points = points[indices]
    return points


def surface_distance_metrics(reference_depth, aligned_prediction_depth, eval_mask, max_points=4096):
    reference_depth = np.asarray(reference_depth, dtype=np.float32)
    aligned_prediction_depth = np.asarray(aligned_prediction_depth, dtype=np.float32)
    ref_points = depth_to_point_cloud(reference_depth, eval_mask, max_points=max_points)
    pred_points = depth_to_point_cloud(aligned_prediction_depth, eval_mask, max_points=max_points)
    if len(ref_points) == 0 or len(pred_points) == 0:
        return {
            "surface_chamfer_l1": math.nan,
            "surface_chamfer_rmse": math.nan,
            "surface_rmse": math.nan,
            "surface_hausdorff95": math.nan,
            "surface_point_count": 0,
        }

    from scipy.spatial import cKDTree

    ref_tree = cKDTree(ref_points)
    pred_tree = cKDTree(pred_points)
    pred_to_ref, _ = ref_tree.query(pred_points, k=1)
    ref_to_pred, _ = pred_tree.query(ref_points, k=1)
    combined_nearest = np.concatenate([pred_to_ref, ref_to_pred])

    common_mask = np.asarray(eval_mask, dtype=bool) & np.isfinite(reference_depth) & np.isfinite(aligned_prediction_depth)
    if np.any(common_mask):
        point_rmse = float(np.sqrt(np.mean((aligned_prediction_depth[common_mask] - reference_depth[common_mask]) ** 2)))
    else:
        point_rmse = math.nan

    return {
        "surface_chamfer_l1": float((np.mean(pred_to_ref) + np.mean(ref_to_pred)) / 2.0),
        "surface_chamfer_rmse": float(np.sqrt(np.mean(combined_nearest**2))),
        "surface_rmse": point_rmse,
        "surface_hausdorff95": float(max(np.quantile(pred_to_ref, 0.95), np.quantile(ref_to_pred, 0.95))),
        "surface_point_count": int(min(len(ref_points), len(pred_points))),
    }


def _masked_values(reference, prediction, mask):
    if mask is None:
        return reference.reshape(-1, reference.shape[-1]), prediction.reshape(-1, prediction.shape[-1])
    return reference[mask], prediction[mask]
