from __future__ import annotations

import math

import torch

from smartpet.conformance.legacy_reference import (
    attention_comparison_report,
    legacy_self_similarity_map,
    paper_equation4_map,
    v030_luminance_similarity_map,
)


def test_similarity_reference_maps_are_finite_and_shape_preserving() -> None:
    torch.manual_seed(11)
    sample = torch.randn(1, 2, 12, 12, 12)
    maps = (
        legacy_self_similarity_map(sample),
        paper_equation4_map(sample),
        v030_luminance_similarity_map(sample),
    )
    for result in maps:
        assert result.shape == sample.shape
        assert torch.isfinite(result).all()
    assert not torch.allclose(maps[0], maps[1])
    assert not torch.allclose(maps[0], maps[2])


def test_attention_report_records_scale_fragility_and_low_cross_method_correlation() -> None:
    torch.manual_seed(2023)
    report = attention_comparison_report(torch.randn(1, 2, 16, 16, 16))
    one = report["scales"]["1"]
    ten = report["scales"]["10"]
    paper_std_one = one["paper_equation4_sigma_3_float_range"]["std"]
    paper_std_ten = ten["paper_equation4_sigma_3_float_range"]["std"]
    assert paper_std_ten < paper_std_one
    correlation = one["correlations"]["paper_vs_v0_3_0"]
    assert math.isfinite(correlation)
    assert abs(correlation) < 0.5
