from __future__ import annotations

import numpy as np


def asinh_normalize(volume: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """RECAP-compatible reversible PET transform: asinh(max(SUV, 0) / scale)."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    array = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("volume contains NaN or infinite values")
    return np.arcsinh(np.clip(array, 0.0, None) / np.float32(scale)).astype(np.float32)


def asinh_denormalize(volume: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    array = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("normalized volume contains NaN or infinite values")
    return (np.sinh(array) * np.float32(scale)).astype(np.float32)
