from __future__ import annotations

import numpy as np

from smartpet.data.normalization import asinh_denormalize, asinh_normalize


def prepare_inference_input(
    volume: np.ndarray,
    *,
    domain: str,
    asinh_scale: float,
) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Inference input contains NaN or infinite values")
    if domain == "suv":
        return asinh_normalize(array, asinh_scale)
    if domain == "normalized":
        minimum = float(array.min())
        if minimum < -1e-5:
            raise ValueError(
                "Normalized SMART-PET input must be non-negative because the training "
                f"transform is asinh(max(SUV, 0) / scale); found minimum={minimum:.6g}"
            )
        return np.clip(array, 0.0, None).astype(np.float32)
    raise ValueError(f"Unsupported input domain: {domain}")


def prepare_shared_inference_outputs(
    prediction_normalized: np.ndarray,
    *,
    asinh_scale: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Prepare normalized and SUV outputs from one model prediction.

    The prediction is validated and clipped exactly once. Both returned arrays
    therefore share the same normalized prediction and cannot diverge because of
    separate model forward passes.
    """

    prediction = np.asarray(prediction_normalized, dtype=np.float32)
    if not np.isfinite(prediction).all():
        raise ValueError("Model prediction contains NaN or infinite values")

    negative_count = int(np.count_nonzero(prediction < 0.0))
    normalized = np.clip(prediction, 0.0, None).astype(np.float32)
    suv = asinh_denormalize(normalized, asinh_scale)

    if not np.isfinite(suv).all():
        raise ValueError(
            "Inverse asinh produced non-finite SUV values; the normalized model "
            "prediction is outside the numerically supported output range"
        )

    return normalized, np.asarray(suv, dtype=np.float32), negative_count


def prepare_inference_output(
    prediction_normalized: np.ndarray,
    *,
    domain: str,
    asinh_scale: float,
) -> tuple[np.ndarray, int]:
    normalized, suv, negative_count = prepare_shared_inference_outputs(
        prediction_normalized,
        asinh_scale=asinh_scale,
    )
    if domain == "normalized":
        return normalized, negative_count
    if domain == "suv":
        return suv, negative_count
    raise ValueError(f"Unsupported output domain: {domain}")
