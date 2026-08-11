from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from .metadata import activity_factor_to_bq_per_ml


def decay_corrected_dose_mbq(
    *,
    net_injected_dose_mbq: float,
    decay_reference: str,
    injection_datetime: datetime | None = None,
    acquisition_datetime: datetime | None = None,
    radionuclide_half_life_seconds: float | None = None,
) -> float:
    """Return dose matched to the PET image decay-reference time.

    ``ADMIN`` means the voxel values are decay-corrected to administration time,
    so the net injected activity at injection is used directly. ``START`` means
    the voxel values are decay-corrected to acquisition start; the administered
    activity is physically decayed from injection to acquisition start.
    """
    dose = float(net_injected_dose_mbq)
    if not math.isfinite(dose) or dose <= 0:
        raise ValueError("net_injected_dose_mbq must be positive and finite")

    reference = str(decay_reference).strip().upper()
    if reference == "ADMIN":
        return dose
    if reference != "START":
        raise ValueError("decay_reference must be ADMIN or START")
    if injection_datetime is None or acquisition_datetime is None:
        raise ValueError("START decay reference requires injection and acquisition datetimes")
    half_life = float(radionuclide_half_life_seconds or 0.0)
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError("START decay reference requires a positive radionuclide half-life")
    elapsed_seconds = (acquisition_datetime - injection_datetime).total_seconds()
    if elapsed_seconds < 0:
        raise ValueError("acquisition_datetime cannot precede injection_datetime")
    return dose * math.exp(-math.log(2.0) * elapsed_seconds / half_life)


def effective_suv_denominator_mbq(
    *,
    net_injected_dose_mbq: float,
    decay_reference: str,
    count_scaling: str,
    count_fraction: float,
    injection_datetime: datetime | None = None,
    acquisition_datetime: datetime | None = None,
    radionuclide_half_life_seconds: float | None = None,
) -> tuple[float, float]:
    """Return ``(dose_at_image_reference, SUV_denominator)`` in MBq.

    ``count_scaling='quantitative'`` means reconstruction preserves calibrated
    activity concentration even when fewer counts were retained; the full
    decay-matched dose remains the SUV denominator.

    ``count_scaling='count_scaled'`` means voxel values are proportional to the
    retained count fraction. The denominator is then multiplied by that
    fraction to restore comparable SUV scale. This mode matches the historical
    ULDP D10/D1 processing, but it must not be assumed for every low-count PET.
    """
    fraction = float(count_fraction)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("count_fraction must be in (0,1]")
    dose_at_reference = decay_corrected_dose_mbq(
        net_injected_dose_mbq=net_injected_dose_mbq,
        decay_reference=decay_reference,
        injection_datetime=injection_datetime,
        acquisition_datetime=acquisition_datetime,
        radionuclide_half_life_seconds=radionuclide_half_life_seconds,
    )
    mode = str(count_scaling).strip().lower()
    if mode == "quantitative":
        denominator = dose_at_reference
    elif mode == "count_scaled":
        denominator = dose_at_reference * fraction
    else:
        raise ValueError("count_scaling must be quantitative or count_scaled")
    return float(dose_at_reference), float(denominator)


def suv_from_activity_concentration(
    activity: np.ndarray,
    *,
    weight_kg: float,
    injected_dose_mbq: float,
    activity_unit: str,
) -> np.ndarray:
    """Convert calibrated PET activity concentration to body-weight SUV.

    Formula
    -------
    ``SUVbw = activity[Bq/mL] * body_weight[g] / dose_at_image_reference[Bq]``

    ``injected_dose_mbq`` must already be the correct effective denominator for
    the image. Use :func:`effective_suv_denominator_mbq` to derive it from net
    administered activity, decay-reference timing, and count-scaling metadata.
    Raw sinograms or arbitrary scanner counts are not supported.
    """
    if weight_kg <= 0 or injected_dose_mbq <= 0:
        raise ValueError("weight_kg and injected_dose_mbq must be positive")
    array = np.asarray(activity, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Activity image contains NaN or infinite values")
    activity_bq_per_ml = array * np.float32(activity_factor_to_bq_per_ml(activity_unit))
    scale = np.float32((weight_kg * 1_000.0) / (injected_dose_mbq * 1_000_000.0))
    return (activity_bq_per_ml * scale).astype(np.float32)
