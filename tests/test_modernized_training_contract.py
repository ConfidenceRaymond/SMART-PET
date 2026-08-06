from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from smartpet.data.dataset import PairRecord
from smartpet.metrics import image_quality_metrics
from smartpet.models import (
    SmartPETGenerator,
    initialize_gan_weights,
    initialize_identity_residual_head,
    positive_softplus_residual,
)
from smartpet.training.preview import select_preview_record


def test_positive_softplus_residual_is_nonnegative_and_identity_centred() -> None:
    source = torch.tensor([0.0, 0.1, 1.0, 3.0], dtype=torch.float32).view(1, 1, 1, 1, 4)
    residual = torch.zeros_like(source)
    actual = positive_softplus_residual(source, residual)
    assert torch.all(actual >= 0)
    torch.testing.assert_close(actual, source.clamp_min(1e-6), rtol=1e-5, atol=1e-6)


def test_modernized_generator_output_is_nonnegative() -> None:
    model = SmartPETGenerator(
        base_channels=1,
        attention_levels=(),
        output_mode="positive_softplus_residual",
    )
    model.apply(initialize_gan_weights)
    source = torch.rand(1, 1, 128, 128, 128)
    with torch.no_grad():
        prediction = model(source)
    assert prediction.shape == source.shape
    assert torch.isfinite(prediction).all()
    assert float(prediction.min()) >= 0.0


def test_zero_initialized_residual_head_starts_as_identity() -> None:
    model = SmartPETGenerator(
        base_channels=1,
        attention_levels=(),
        output_mode="positive_softplus_residual",
    )
    model.apply(initialize_gan_weights)
    initialize_identity_residual_head(model.output)
    source = torch.rand(1, 1, 128, 128, 128)
    with torch.no_grad():
        prediction = model(source)
    torch.testing.assert_close(
        prediction,
        source.clamp_min(1e-6),
        rtol=1e-5,
        atol=1e-6,
    )


def test_normal_initialization_is_explicit() -> None:
    module = nn.Sequential(
        nn.Conv3d(4, 16, 3, bias=True),
        nn.InstanceNorm3d(16, affine=True),
        nn.ConvTranspose3d(16, 4, 3, bias=True),
    )
    module.apply(initialize_gan_weights)
    conv_weights = torch.cat(
        [module[0].weight.detach().flatten(), module[2].weight.detach().flatten()]
    )
    assert abs(float(conv_weights.mean())) < 0.003
    assert 0.016 < float(conv_weights.std()) < 0.024
    assert torch.count_nonzero(module[0].bias) == 0
    assert torch.count_nonzero(module[2].bias) == 0
    assert abs(float(module[1].weight.detach().mean()) - 1.0) < 0.02
    assert torch.count_nonzero(module[1].bias) == 0


def test_image_quality_identity_metrics_have_expected_limits() -> None:
    target = torch.linspace(0, 3, 16**3).reshape(1, 1, 16, 16, 16)
    metrics = image_quality_metrics(target, target)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["nrmse"] == pytest.approx(0.0)
    assert metrics["ssim"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["snr_error"] == pytest.approx(0.0)
    assert metrics["cnr_error"] == pytest.approx(0.0)
    assert np.isfinite(metrics["psnr_db"])
    assert metrics["psnr_db"] > 100


def test_preview_selection_is_reproducible_and_supports_override() -> None:
    records = [
        PairRecord(f"s{index}", Path(f"source{index}"), Path(f"target{index}"))
        for index in range(5)
    ]
    first = select_preview_record(
        records,
        subject_id=None,
        selection="fixed_random",
        seed=5104,
        epoch=0,
    )
    later = select_preview_record(
        records,
        subject_id=None,
        selection="fixed_random",
        seed=5104,
        epoch=100,
    )
    assert first == later
    overridden = select_preview_record(
        records,
        subject_id="s3",
        selection="fixed_random",
        seed=5104,
        epoch=0,
    )
    assert overridden.subject_id == "s3"


def test_random_each_epoch_changes_selection_for_some_epochs() -> None:
    records = [
        PairRecord(f"s{index}", Path(f"source{index}"), Path(f"target{index}"))
        for index in range(20)
    ]
    selected = {
        select_preview_record(
            records,
            subject_id=None,
            selection="random_each_epoch",
            seed=5104,
            epoch=epoch,
        ).subject_id
        for epoch in range(8)
    }
    assert len(selected) > 1


def test_positive_softplus_residual_has_finite_nonzero_gradient() -> None:
    source = torch.tensor([0.0, 0.1, 1.0, 3.0], dtype=torch.float32).view(1, 1, 1, 1, 4)
    residual = torch.zeros_like(source, requires_grad=True)
    prediction = positive_softplus_residual(source, residual)
    prediction.sum().backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()
    assert torch.all(residual.grad > 0)


def test_image_quality_metrics_include_paper_and_suv_fields() -> None:
    target = torch.linspace(0, 3, 16**3).reshape(1, 1, 16, 16, 16)
    prediction = target * 0.95
    metrics = image_quality_metrics(prediction, target)
    expected = {
        "mae",
        "mse",
        "rmse",
        "nrmse",
        "nmae_pct",
        "psnr_db",
        "ssim",
        "mean_suv",
        "target_mean_suv",
        "mean_suv_bias",
        "mean_suv_bias_pct",
        "snr",
        "target_snr",
        "snr_error",
        "cnr",
        "target_cnr",
        "cnr_error",
    }
    assert expected == set(metrics)
    assert all(np.isfinite(value) for value in metrics.values())


def test_professional_preview_panel_renders(tmp_path: Path) -> None:
    from smartpet.metrics import numpy_image_quality_metrics
    from smartpet.training.preview import save_preview_panel

    shape = (24, 28, 20)
    coordinates = np.indices(shape, dtype=np.float32)
    center = np.asarray(shape, dtype=np.float32)[:, None, None, None] / 2
    radius = np.sqrt(((coordinates - center) ** 2).sum(axis=0))
    target = np.clip(3.0 * (1.0 - radius / 10.0), 0.0, None).astype(np.float32)
    lowdose = target * 0.85
    prediction = target * 0.97
    metrics = {
        "source_vs_target": numpy_image_quality_metrics(lowdose, target),
        "prediction_vs_target": numpy_image_quality_metrics(prediction, target),
    }
    output = tmp_path / "preview.png"
    save_preview_panel(
        lowdose_suv=lowdose,
        prediction_suv=prediction,
        target_suv=target,
        output_path=output,
        subject_id="synthetic",
        epoch_label="epoch 1, step 459",
        metrics=metrics,
    )
    assert output.is_file()
    assert output.stat().st_size > 10_000
