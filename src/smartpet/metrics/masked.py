from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


def _kernel(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    x = torch.arange(window_size, device=device, dtype=torch.float32) - (window_size - 1) / 2
    kernel = torch.exp(-(x.square()) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _filter3d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    radius = kernel.numel() // 2
    kd = kernel.view(1, 1, -1, 1, 1)
    kh = kernel.view(1, 1, 1, -1, 1)
    kw = kernel.view(1, 1, 1, 1, -1)
    x = F.conv3d(x, kd, padding=(radius, 0, 0))
    x = F.conv3d(x, kh, padding=(0, radius, 0))
    return F.conv3d(x, kw, padding=(0, 0, radius))


def masked_ssim3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    mask = mask.float()
    kernel = _kernel(window_size, sigma, prediction.device)
    support = _filter3d(mask, kernel).clamp_min(EPS)
    mu_x = _filter3d(prediction * mask, kernel) / support
    mu_y = _filter3d(target * mask, kernel) / support
    ex2 = _filter3d(prediction.square() * mask, kernel) / support
    ey2 = _filter3d(target.square() * mask, kernel) / support
    exy = _filter3d(prediction * target * mask, kernel) / support
    var_x = (ex2 - mu_x.square()).clamp_min(0)
    var_y = (ey2 - mu_y.square()).clamp_min(0)
    cov = exy - mu_x * mu_y
    selected = target[mask.bool()]
    data_range = (selected.max() - selected.min()).clamp_min(EPS)
    c1 = (0.01 * data_range).square()
    c2 = (0.03 * data_range).square()
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    denominator = (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    ).clamp_min(EPS)
    score = numerator / denominator
    return score[mask.bool()].mean()


def masked_image_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    device: torch.device,
    window_size: int = 11,
    sigma: float = 1.5,
) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) < 16:
        raise ValueError("mask contains fewer than 16 voxels")
    pred = torch.from_numpy(
        np.ascontiguousarray(prediction, dtype=np.float32)
    ).to(device)[None, None]
    true = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)).to(device)[None, None]
    m5 = torch.from_numpy(np.ascontiguousarray(mask)).to(device)[None, None]
    selected_pred = pred[m5.bool()]
    selected_true = true[m5.bool()]
    error = selected_pred - selected_true
    mae = error.abs().mean()
    mse = error.square().mean()
    rmse = mse.sqrt()
    target_rms = selected_true.square().mean().sqrt().clamp_min(EPS)
    target_abs_mean = selected_true.abs().mean().clamp_min(EPS)
    data_range = (selected_true.max() - selected_true.min()).clamp_min(EPS)
    psnr = 20 * torch.log10(data_range / rmse.clamp_min(EPS))
    ssim = masked_ssim3d(pred, true, m5, window_size=window_size, sigma=sigma)
    pred_mean = selected_pred.mean()
    target_mean = selected_true.mean()
    bias = pred_mean - target_mean
    values = {
        "mae": float(mae.item()),
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "nrmse": float((rmse / target_rms).item()),
        "nmae_pct": float((100 * mae / target_abs_mean).item()),
        "psnr_db": float(psnr.item()),
        "ssim": float(ssim.item()),
        "mean_suv": float(pred_mean.item()),
        "target_mean_suv": float(target_mean.item()),
        "mean_suv_bias": float(bias.item()),
        "mean_suv_bias_pct": float((100 * bias / target_mean.abs().clamp_min(EPS)).item()),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError(f"Non-finite metrics: {values}")
    return values
