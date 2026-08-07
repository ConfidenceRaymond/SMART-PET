from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F


def gaussian_kernel_3d(
    *,
    window_size: int,
    sigma: float,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create the normalized separable Gaussian kernel used by the legacy code."""

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")
    coordinates = torch.arange(window_size, dtype=dtype, device=device)
    center = window_size // 2
    one_d = torch.exp(-((coordinates - center) ** 2) / (2.0 * float(sigma) ** 2))
    one_d = one_d / one_d.sum()
    kernel = torch.einsum("i,j,k->ijk", one_d, one_d, one_d)
    return kernel.reshape(1, 1, window_size, window_size, window_size).expand(
        channels, 1, -1, -1, -1
    )


def _validate_feature_tensor(x: torch.Tensor) -> None:
    if x.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W], got {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        raise TypeError("Feature tensor must have a floating-point dtype")
    if not torch.isfinite(x).all():
        raise ValueError("Feature tensor contains non-finite values")


def _local_variance(x: torch.Tensor, *, window_size: int, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel_3d(
        window_size=window_size,
        sigma=sigma,
        channels=x.shape[1],
        dtype=x.dtype,
        device=x.device,
    )
    padding = window_size // 2
    mean = F.conv3d(x, kernel, padding=padding, groups=x.shape[1])
    return F.conv3d(x * x, kernel, padding=padding, groups=x.shape[1]) - mean.square()


def legacy_self_similarity_map(
    x: torch.Tensor,
    *,
    window_size: int = 11,
    dynamic_range: float | None = None,
) -> torch.Tensor:
    """Reproduce the uploaded ``self_ssim3d`` statistic.

    The historical helper accepts ``sigma`` but its window constructor hard-codes
    sigma=1.5. It also converts the observed dynamic range to an integer before
    computing C2. Both behaviours are intentionally retained here.
    """

    _validate_feature_tensor(x)
    variance = _local_variance(x, window_size=window_size, sigma=1.5)
    observed = float((x.max() - x.min()).detach().cpu())
    value_range = float(int(observed)) if dynamic_range is None else float(dynamic_range)
    c2 = (0.03 * value_range) ** 2
    return (2.0 * variance + c2) / (2.0 * variance.square() + c2)


def paper_equation4_map(
    x: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 3.0,
    dynamic_range: float | None = None,
) -> torch.Tensor:
    """Evaluate the published variance-squared Equation 4 with explicit sigma."""

    _validate_feature_tensor(x)
    variance = _local_variance(x, window_size=window_size, sigma=sigma)
    value_range = (
        float((x.max() - x.min()).detach().cpu())
        if dynamic_range is None
        else float(dynamic_range)
    )
    c2 = (0.03 * value_range) ** 2
    return (2.0 * variance + c2) / (2.0 * variance.square() + c2)


def v030_luminance_similarity_map(
    x: torch.Tensor,
    *,
    kernel_size: int = 7,
    constant: float = 1e-4,
) -> torch.Tensor:
    """Return the luminance-like map implemented by SMART-PET v0.3.0."""

    _validate_feature_tensor(x)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    mean = F.avg_pool3d(x, kernel_size, stride=1, padding=kernel_size // 2)
    return (2.0 * x * mean + constant) / (x.square() + mean.square() + constant)


def tensor_statistics(x: torch.Tensor) -> dict[str, float]:
    values = x.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
        "finite_fraction": float(torch.isfinite(values).double().mean()),
        "near_zero_fraction": float((values.abs() <= 1e-6).double().mean()),
        "near_one_fraction": float(((values - 1.0).abs() <= 1e-6).double().mean()),
    }


def pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    x = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    y = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(x, y) / denominator)


def attention_comparison_report(
    x: torch.Tensor,
    *,
    scales: Sequence[float] = (0.1, 1.0, 10.0),
) -> dict[str, Any]:
    """Compare historical, published, and v0.3.0 similarity statistics."""

    _validate_feature_tensor(x)
    report: dict[str, Any] = {
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "scales": {},
    }
    for raw_scale in scales:
        scale = float(raw_scale)
        if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
            raise ValueError(f"scales must contain positive finite values, got {scale}")
        sample = x * scale
        legacy = legacy_self_similarity_map(sample)
        paper = paper_equation4_map(sample)
        modern = v030_luminance_similarity_map(sample)
        report["scales"][f"{scale:g}"] = {
            "legacy_code_sigma_1_5_integer_range": tensor_statistics(legacy),
            "paper_equation4_sigma_3_float_range": tensor_statistics(paper),
            "v0_3_0_luminance": tensor_statistics(modern),
            "correlations": {
                "legacy_vs_paper": pearson_correlation(legacy, paper),
                "legacy_vs_v0_3_0": pearson_correlation(legacy, modern),
                "paper_vs_v0_3_0": pearson_correlation(paper, modern),
            },
        }
    return report
