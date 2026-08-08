import pytest
import torch

from smartpet.training.trainer import (
    _masked_suv_validation_metrics,
)


def test_masked_suv_metrics_are_subject_macro_metrics():
    target = torch.ones(
        (2, 1, 2, 2, 2),
        dtype=torch.float32,
    )

    prediction = target.clone()

    # Subject 0:
    #   +10% everywhere
    # Subject 1:
    #   -20% everywhere
    prediction[0] *= 1.10
    prediction[1] *= 0.80

    mask = torch.ones(
        (1, 1, 2, 2, 2),
        dtype=torch.float32,
    )

    metrics = _masked_suv_validation_metrics(
        prediction,
        target,
        mask,
    )

    # Per-subject values:
    #
    # subject 0:
    #   NMAE = 10%
    #   bias = +10%
    #
    # subject 1:
    #   NMAE = 20%
    #   bias = -20%
    #
    # Subject-level macro averages:
    #   NMAE        = 15%
    #   signed bias = -5%
    #   |bias|      = 15%

    assert metrics[
        "val_prediction_masked_nmae_pct_suv"
    ] == pytest.approx(
        15.0,
        abs=1e-4,
    )

    assert metrics[
        "val_prediction_masked_mean_suv_bias_pct_suv"
    ] == pytest.approx(
        -5.0,
        abs=1e-4,
    )

    assert metrics[
        "val_prediction_masked_abs_mean_suv_bias_pct_suv"
    ] == pytest.approx(
        15.0,
        abs=1e-4,
    )


def test_masked_suv_metrics_ignore_values_outside_mask():
    target = torch.ones(
        (1, 1, 2, 2, 2),
        dtype=torch.float32,
    )
    prediction = target.clone()

    mask = torch.zeros(
        (1, 1, 2, 2, 2),
        dtype=torch.float32,
    )

    # Only z=0 participates.
    mask[:, :, 0] = 1.0

    # Perfect inside mask.
    prediction[:, :, 0] = target[:, :, 0]

    # Deliberately catastrophic outside-mask error.
    prediction[:, :, 1] = 1000.0

    metrics = _masked_suv_validation_metrics(
        prediction,
        target,
        mask,
    )

    assert metrics[
        "val_prediction_masked_nmae_pct_suv"
    ] == pytest.approx(0.0)

    assert metrics[
        "val_prediction_masked_mean_suv_bias_pct_suv"
    ] == pytest.approx(0.0)

    assert metrics[
        "val_prediction_masked_abs_mean_suv_bias_pct_suv"
    ] == pytest.approx(0.0)


def test_masked_suv_metric_rejects_zero_target_mass():
    target = torch.zeros(
        (1, 1, 2, 2, 2),
        dtype=torch.float32,
    )
    prediction = torch.zeros_like(target)
    mask = torch.ones_like(target)

    with pytest.raises(
        ValueError,
        match="denominator",
    ):
        _masked_suv_validation_metrics(
            prediction,
            target,
            mask,
        )


def test_mask_shape_must_match_spatial_patch():
    target = torch.ones(
        (1, 1, 2, 2, 2),
        dtype=torch.float32,
    )
    prediction = target.clone()

    wrong_mask = torch.ones(
        (1, 1, 3, 2, 2),
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="mask shape mismatch",
    ):
        _masked_suv_validation_metrics(
            prediction,
            target,
            wrong_mask,
        )
