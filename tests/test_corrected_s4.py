from __future__ import annotations

import torch
import torch.nn as nn

from smartpet.conformance.architecture import architecture_report
from smartpet.conformance.legacy_reference import paper_equation4_map
from smartpet.models import PatchDiscriminator3D, SmartPETGenerator, initialize_gan_weights
from smartpet.models.attention import (
    SSAB3D,
    ChannelSpatialAttention3D,
    SimilarityAttention3D,
)


def test_paper_exact_similarity_matches_reference_equation() -> None:
    torch.manual_seed(2023)
    sample = torch.randn(1, 1, 9, 9, 9)
    module = SimilarityAttention3D(1, mode="paper_exact")
    actual = module.similarity_map(sample)
    expected = paper_equation4_map(sample)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_scale_consistent_similarity_is_invariant_to_positive_rescaling() -> None:
    torch.manual_seed(19)
    sample = torch.randn(2, 3, 9, 9, 9)
    module = SimilarityAttention3D(3, mode="scale_consistent")
    baseline = module.similarity_map(sample)
    rescaled = module.similarity_map(sample * 7.5)
    torch.testing.assert_close(rescaled, baseline, rtol=2e-5, atol=2e-6)


def test_equation5_feature_convolution_receives_feature_map() -> None:
    torch.manual_seed(7)
    module = SimilarityAttention3D(
        1,
        mode="scale_consistent",
        window_size=3,
        gate_kernel_size=1,
    )
    nn.init.zeros_(module.conv_similarity.weight)
    nn.init.zeros_(module.conv_similarity.bias)
    nn.init.ones_(module.conv_feature.weight)
    nn.init.zeros_(module.conv_feature.bias)
    sample = torch.randn(1, 1, 5, 5, 5)
    actual = module(sample)
    expected = sample * torch.sigmoid(sample)
    torch.testing.assert_close(actual, expected)


def test_channel_spatial_projection_restores_input_branch() -> None:
    sample = torch.linspace(-1.0, 1.0, 5**3).reshape(1, 1, 5, 5, 5)
    without_projection = ChannelSpatialAttention3D(1, input_projection=False)
    with_projection = ChannelSpatialAttention3D(1, input_projection=True)
    for module in (without_projection, with_projection):
        for parameter in module.parameters():
            nn.init.zeros_(parameter)
    nn.init.ones_(with_projection.input_projection.weight)

    expected_without = sample * 0.25
    expected_with = sample * 0.5 * torch.sigmoid(sample)
    torch.testing.assert_close(without_projection(sample), expected_without)
    torch.testing.assert_close(with_projection(sample), expected_with)


class _Ones(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)


def test_s4_fusion_does_not_apply_a_second_feature_gate() -> None:
    block = SSAB3D(1, similarity_mode="scale_consistent")
    block.self_attention = _Ones()
    block.similarity = _Ones()
    block.channel_spatial = _Ones()
    nn.init.zeros_(block.fuse.weight)
    nn.init.zeros_(block.fuse.bias)
    block.fuse.weight.data[0, 0, 1, 1, 1] = 1.0
    sample = torch.full((1, 1, 5, 5, 5), 2.0)
    torch.testing.assert_close(block(sample), torch.full_like(sample, 3.0))


def test_corrected_architecture_inventory_and_spectral_norm() -> None:
    generator = SmartPETGenerator(
        base_channels=1,
        attention_levels=(2, 3),
        output_mode="positive_softplus_residual",
        similarity_mode="scale_consistent",
        encoder_convs_per_level=2,
        channel_spatial_input_projection=True,
        generator_spectral_norm=False,
    )
    discriminator = PatchDiscriminator3D(base_channels=1, spectral_norm=True)
    report = architecture_report(generator, discriminator)
    assert report["generator"]["encoder_conv3d_counts"] == [2] * 7
    assert report["generator"]["similarity_mode"] == "scale_consistent"
    assert report["generator"]["channel_spatial_input_projection"] is True
    assert report["generator"]["spectral_norm_paths"] == []
    assert len(report["discriminator"]["spectral_norm_paths"]) == 5
    assert report["discriminator"]["spectral_norm"] is True


def test_generator_spectral_norm_is_configurable_and_initializable() -> None:
    model = SmartPETGenerator(
        base_channels=1,
        attention_levels=(),
        generator_spectral_norm=True,
        encoder_convs_per_level=2,
    )
    model.apply(initialize_gan_weights)
    report = architecture_report(model, PatchDiscriminator3D(base_channels=1))
    assert len(report["generator"]["spectral_norm_paths"]) == 20
    original = model.encoders[0].net[0].weight_orig
    assert 0.01 < float(original.detach().std()) < 0.03


def test_v030_default_state_dict_contract_remains_strictly_loadable() -> None:
    source = SmartPETGenerator(base_channels=1, attention_levels=(2, 3))
    restored = SmartPETGenerator(base_channels=1, attention_levels=(2, 3))
    restored.load_state_dict(source.state_dict(), strict=True)
    assert source.similarity_mode == "v030_luminance"
    assert source.encoder_convs_per_level == 1
    assert source.channel_spatial_input_projection is False
    assert source.generator_spectral_norm is False
