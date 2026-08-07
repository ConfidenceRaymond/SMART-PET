from __future__ import annotations

import torch

from smartpet.models.attention import SimilarityAttention3D, _box_average_3d


def test_v030_deterministic_box_average_matches_avg_pool3d() -> None:
    torch.manual_seed(3107)
    sample = torch.randn(2, 3, 9, 10, 11, dtype=torch.float64)
    expected = torch.nn.functional.avg_pool3d(sample, 7, stride=1, padding=3)
    actual = _box_average_3d(sample, kernel_size=7, padding=3)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_v030_deterministic_smoothing_preserves_state_dict_contract() -> None:
    module = SimilarityAttention3D(2, mode="v030_luminance")
    state_keys_before = tuple(module.state_dict().keys())
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    sample = torch.randn(1, 2, 9, 9, 9, requires_grad=True)
    try:
        torch.use_deterministic_algorithms(True)
        output = module(sample).mean()
        output.backward()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
    assert sample.grad is not None
    assert tuple(module.state_dict().keys()) == state_keys_before
