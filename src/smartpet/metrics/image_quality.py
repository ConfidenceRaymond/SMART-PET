from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_EPS = 1e-8


def _gaussian_kernel_1d(
    window_size: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - (window_size - 1) / 2
    kernel = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def _separable_filter3d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    radius = kernel.numel() // 2
    kd = kernel.view(1, 1, -1, 1, 1).expand(channels, 1, -1, 1, 1)
    kh = kernel.view(1, 1, 1, -1, 1).expand(channels, 1, 1, -1, 1)
    kw = kernel.view(1, 1, 1, 1, -1).expand(channels, 1, 1, 1, -1)
    x = F.conv3d(x, kd, padding=(radius, 0, 0), groups=channels)
    x = F.conv3d(x, kh, padding=(0, radius, 0), groups=channels)
    return F.conv3d(x, kw, padding=(0, 0, radius), groups=channels)


def ssim3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Return mean 3D SSIM per sample using a separable Gaussian window."""

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            "Expected equal [B,C,D,H,W] tensors, got "
            f"{prediction.shape}, {target.shape}"
        )
    prediction = prediction.float()
    target = target.float()
    kernel = _gaussian_kernel_1d(
        window_size,
        sigma,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    mu_x = _separable_filter3d(prediction, kernel)
    mu_y = _separable_filter3d(target, kernel)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x2 = (_separable_filter3d(prediction.square(), kernel) - mu_x2).clamp_min(0.0)
    sigma_y2 = (_separable_filter3d(target.square(), kernel) - mu_y2).clamp_min(0.0)
    sigma_xy = _separable_filter3d(prediction * target, kernel) - mu_xy

    reduce_dims = tuple(range(1, target.ndim))
    data_range = target.amax(dim=reduce_dims) - target.amin(dim=reduce_dims)
    data_range = data_range.clamp_min(_EPS).view(-1, 1, 1, 1, 1)
    c1 = (0.01 * data_range).square()
    c2 = (0.03 * data_range).square()
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    score = numerator / denominator.clamp_min(_EPS)
    return score.mean(dim=reduce_dims)


def _snr_and_cnr(
    volume: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute deterministic whole-image SNR/CNR using reference-defined tissue masks.

    This is an image-space monitoring proxy, not the paper's atlas/ROI clinical analysis.
    SNR is foreground mean / foreground standard deviation. CNR contrasts the upper
    and lower quartiles inside the reference foreground.
    """

    positive = reference[reference > 0]
    if positive.numel() < 16:
        positive = reference.reshape(-1)
    threshold = torch.quantile(positive, 0.05)
    foreground = reference > threshold
    ref_values = reference[foreground]
    values = volume[foreground]
    if values.numel() < 16:
        values = volume.reshape(-1)
        ref_values = reference.reshape(-1)

    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(_EPS)
    snr = mean.abs() / std

    q25 = torch.quantile(ref_values, 0.25)
    q75 = torch.quantile(ref_values, 0.75)
    low = values[ref_values <= q25]
    high = values[ref_values >= q75]
    if low.numel() < 2 or high.numel() < 2:
        return snr, torch.zeros_like(snr)
    contrast = (high.mean() - low.mean()).abs()
    pooled = torch.sqrt(
        high.var(unbiased=False) + low.var(unbiased=False) + _EPS
    )
    return snr, contrast / pooled


def image_quality_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Compute batch-mean quantitative image fidelity metrics.

    PSNR, SSIM and NRMSE follow their conventional image-fidelity direction.
    ``snr_error`` and ``cnr_error`` are absolute differences from target-derived
    SNR/CNR; lower is better, matching the direction reported in SMART-PET.
    """

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            "Expected equal [B,C,D,H,W] tensors, got "
            f"{prediction.shape}, {target.shape}"
        )
    prediction = prediction.float()
    target = target.float()
    reduce_dims = tuple(range(1, target.ndim))
    error = prediction - target
    mae_per = error.abs().mean(dim=reduce_dims)
    mse_per = error.square().mean(dim=reduce_dims)
    rmse_per = torch.sqrt(mse_per)
    target_energy = torch.sqrt(target.square().mean(dim=reduce_dims)).clamp_min(_EPS)
    nrmse_per = rmse_per / target_energy
    data_range = (target.amax(dim=reduce_dims) - target.amin(dim=reduce_dims)).clamp_min(_EPS)
    psnr_per = 20.0 * torch.log10(data_range / rmse_per.clamp_min(_EPS))
    ssim_per = ssim3d(prediction, target)

    prediction_snrs: list[torch.Tensor] = []
    target_snrs: list[torch.Tensor] = []
    prediction_cnrs: list[torch.Tensor] = []
    target_cnrs: list[torch.Tensor] = []
    snr_errors: list[torch.Tensor] = []
    cnr_errors: list[torch.Tensor] = []
    for index in range(target.shape[0]):
        target_snr, target_cnr = _snr_and_cnr(target[index], target[index])
        pred_snr, pred_cnr = _snr_and_cnr(prediction[index], target[index])
        prediction_snrs.append(pred_snr)
        target_snrs.append(target_snr)
        prediction_cnrs.append(pred_cnr)
        target_cnrs.append(target_cnr)
        snr_errors.append((pred_snr - target_snr).abs())
        cnr_errors.append((pred_cnr - target_cnr).abs())

    prediction_mean = prediction.mean(dim=reduce_dims)
    target_mean = target.mean(dim=reduce_dims)
    mean_bias = prediction_mean - target_mean
    mean_bias_pct = 100.0 * mean_bias / target_mean.abs().clamp_min(_EPS)
    nmae_pct = 100.0 * mae_per / target.abs().mean(dim=reduce_dims).clamp_min(_EPS)

    return {
        "mae": float(mae_per.mean().item()),
        "mse": float(mse_per.mean().item()),
        "rmse": float(rmse_per.mean().item()),
        "nrmse": float(nrmse_per.mean().item()),
        "nmae_pct": float(nmae_pct.mean().item()),
        "psnr_db": float(psnr_per.mean().item()),
        "ssim": float(ssim_per.mean().item()),
        "mean_suv": float(prediction_mean.mean().item()),
        "target_mean_suv": float(target_mean.mean().item()),
        "mean_suv_bias": float(mean_bias.mean().item()),
        "mean_suv_bias_pct": float(mean_bias_pct.mean().item()),
        "snr": float(torch.stack(prediction_snrs).mean().item()),
        "target_snr": float(torch.stack(target_snrs).mean().item()),
        "snr_error": float(torch.stack(snr_errors).mean().item()),
        "cnr": float(torch.stack(prediction_cnrs).mean().item()),
        "target_cnr": float(torch.stack(target_cnrs).mean().item()),
        "cnr_error": float(torch.stack(cnr_errors).mean().item()),
    }


def numpy_image_quality_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    pred = torch.from_numpy(np.asarray(prediction, dtype=np.float32))[None, None].to(device)
    true = torch.from_numpy(np.asarray(target, dtype=np.float32))[None, None].to(device)
    with torch.no_grad():
        return image_quality_metrics(pred, true)


def legacy_vgg19_feature_l1(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    weights_path: str | Path,
    device: str | torch.device = "cpu",
    slice_step: int = 4,
) -> float:
    """Reproduce the historical repository's metric labelled as FID.

    This is *not* mathematical Fréchet inception distance. It is the mean L1
    distance between VGG19 feature maps of axial PET slices. The explicit name
    prevents the historical proxy from being misreported as true FID.
    """

    try:
        from torchvision.models import vgg19
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torchvision is required for the historical VGG19 FID proxy") from exc

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    model = vgg19(weights=None)
    state: Any = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("VGG19 weights must contain a state_dict mapping")
    cleaned = {str(k).removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    features = torch.nn.Sequential(*list(model.features.children())[:35]).to(device).eval()

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Expected equal 3D volumes")
    vmax = float(np.percentile(target[target > 0], 99.5)) if np.any(target > 0) else 1.0
    vmax = max(vmax, _EPS)
    indices = range(0, target.shape[2], max(1, int(slice_step)))
    losses: list[float] = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        for index in indices:
            target_slice = target[:, :, index]
            if float(target_slice.max()) <= 0:
                continue
            pred_slice = np.clip(prediction[:, :, index] / vmax, 0.0, 1.0)
            true_slice = np.clip(target_slice / vmax, 0.0, 1.0)
            pred_tensor = torch.from_numpy(pred_slice)[None, None].to(device)
            true_tensor = torch.from_numpy(true_slice)[None, None].to(device)
            pred_tensor = F.interpolate(
                pred_tensor,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            true_tensor = F.interpolate(
                true_tensor,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            pred_tensor = (pred_tensor.expand(-1, 3, -1, -1) - mean) / std
            true_tensor = (true_tensor.expand(-1, 3, -1, -1) - mean) / std
            losses.append(float(F.l1_loss(features(pred_tensor), features(true_tensor)).item()))
    if not losses:
        return math.nan
    return float(np.mean(losses))
