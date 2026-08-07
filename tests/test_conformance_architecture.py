from __future__ import annotations

from smartpet.conformance.architecture import architecture_report
from smartpet.models.discriminator import PatchDiscriminator3D
from smartpet.models.generator import SmartPETGenerator


def test_v030_architecture_report_is_machine_readable() -> None:
    generator = SmartPETGenerator(
        base_channels=2,
        attention_levels=(2, 3),
        output_mode="positive_softplus_residual",
    )
    discriminator = PatchDiscriminator3D(base_channels=2)
    report = architecture_report(generator, discriminator)
    g = report["generator"]
    d = report["discriminator"]
    assert g["encoder_levels"] == 7
    assert g["encoder_conv3d_counts"] == [1, 1, 1, 1, 1, 1, 1]
    assert g["attention_levels"] == [2, 3]
    assert g["output_mode"] == "positive_softplus_residual"
    assert g["spectral_norm_paths"] == []
    assert d["spectral_norm_paths"] == []
    assert d["has_sigmoid"] is False
    assert d["output_contract"] == "raw_logits"
    assert g["parameters"] > 0
    assert d["parameters"] > 0
