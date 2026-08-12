from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from smartpet.preprocessing.suv import (
    decay_corrected_dose_mbq,
    effective_suv_denominator_mbq,
    suv_from_activity_concentration,
    uncorrected_frame_to_admin_factor,
)


def test_body_weight_suv_formula_for_kbq_per_ml() -> None:
    # 5 kBq/mL, 70 kg, 175 MBq -> SUVbw = 2.0
    activity = np.full((2, 2, 2), 5.0, dtype=np.float32)
    suv = suv_from_activity_concentration(
        activity,
        weight_kg=70.0,
        injected_dose_mbq=175.0,
        activity_unit="kBq/mL",
    )
    np.testing.assert_allclose(suv, 2.0, rtol=0.0, atol=1e-6)


def test_count_scaled_decimation_uses_fractional_denominator() -> None:
    full = np.full((1,), 5.0, dtype=np.float32)
    low = 0.1 * full
    _, target_denominator = effective_suv_denominator_mbq(
        net_injected_dose_mbq=175,
        decay_reference="ADMIN",
        count_scaling="quantitative",
        count_fraction=1.0,
    )
    _, source_denominator = effective_suv_denominator_mbq(
        net_injected_dose_mbq=175,
        decay_reference="ADMIN",
        count_scaling="count_scaled",
        count_fraction=0.1,
    )
    target_suv = suv_from_activity_concentration(
        full, weight_kg=70, injected_dose_mbq=target_denominator, activity_unit="kBq/mL"
    )
    source_suv = suv_from_activity_concentration(
        low, weight_kg=70, injected_dose_mbq=source_denominator, activity_unit="kBq/mL"
    )
    np.testing.assert_allclose(source_suv, target_suv, rtol=0.0, atol=1e-6)


def test_quantitative_low_count_image_does_not_scale_dose_by_fraction() -> None:
    dose_at_reference, denominator = effective_suv_denominator_mbq(
        net_injected_dose_mbq=175,
        decay_reference="ADMIN",
        count_scaling="quantitative",
        count_fraction=0.1,
    )
    assert dose_at_reference == pytest.approx(175)
    assert denominator == pytest.approx(175)


def test_start_decay_reference_decays_injected_activity() -> None:
    injection = datetime(2026, 1, 1, 10, 0, 0)
    acquisition = injection + timedelta(seconds=600)
    # One half-life elapsed: denominator is half the injected activity.
    corrected = decay_corrected_dose_mbq(
        net_injected_dose_mbq=200,
        decay_reference="START",
        injection_datetime=injection,
        acquisition_datetime=acquisition,
        radionuclide_half_life_seconds=600,
    )
    assert corrected == pytest.approx(100.0)


def test_acquisition_cannot_precede_injection() -> None:
    injection = datetime(2026, 1, 1, 10, 0, 0)
    with pytest.raises(ValueError, match="cannot precede"):
        decay_corrected_dose_mbq(
            net_injected_dose_mbq=200,
            decay_reference="START",
            injection_datetime=injection,
            acquisition_datetime=injection - timedelta(seconds=1),
            radionuclide_half_life_seconds=600,
        )


def test_uncorrected_frame_average_can_be_corrected_to_admin() -> None:
    injection = datetime(2026, 1, 1, 10, 0, 0)
    factor = uncorrected_frame_to_admin_factor(
        injection_datetime=injection,
        acquisition_datetime=injection,
        image_duration_seconds=600,
        radionuclide_half_life_seconds=600,
    )
    # Average decay over one half-life is (1 - 0.5) / ln(2).
    assert factor == pytest.approx(2.0 * np.log(2.0), rel=1e-12)


def test_none_decay_reference_is_not_directly_usable_as_dose_reference() -> None:
    with pytest.raises(ValueError, match="Correct the activity image to ADMIN"):
        decay_corrected_dose_mbq(
            net_injected_dose_mbq=200,
            decay_reference="NONE",
        )
