from __future__ import annotations

import pytest

from smartpet.conformance.gradients import generator_gradient_attribution


def test_historical_graph_is_effectively_weighted_l1_only() -> None:
    report = generator_gradient_attribution(seed=2023)
    historical = report["historical_executed"]
    assert historical["l1_gradient_norm"] > 0
    assert historical["gan_real_pair_gradient_norm"] == 0
    assert historical["vgg_detached_gradient_norm"] == 0
    assert historical["total_to_l1_gradient_ratio"] == pytest.approx(0.01, rel=1e-6)


def test_corrected_gan_reaches_generator_and_discriminator_step_does_not() -> None:
    report = generator_gradient_attribution(seed=2023)
    corrected = report["corrected_lsgan"]
    assert corrected["gan_fake_pair_gradient_norm"] > 0
    assert corrected["total_gradient_norm"] > corrected["l1_gradient_norm"]
    assert report["discriminator_step"]["generator_gradient_norm_with_detached_fake"] == 0
