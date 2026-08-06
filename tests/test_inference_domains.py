from __future__ import annotations

import numpy as np
import pytest

from smartpet.inference.domains import prepare_inference_input, prepare_inference_output


def test_suv_input_is_asinh_normalized() -> None:
    source = np.array([-2.0, 0.0, 1.0, 3.0], dtype=np.float32)
    actual = prepare_inference_input(source, domain="suv", asinh_scale=1.0)
    expected = np.arcsinh(np.array([0.0, 0.0, 1.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_normalized_input_is_not_normalized_twice() -> None:
    source = np.array([0.0, 0.5, 2.0], dtype=np.float32)
    actual = prepare_inference_input(source, domain="normalized", asinh_scale=1.0)
    np.testing.assert_array_equal(actual, source)


def test_normalized_input_rejects_material_negative_values() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        prepare_inference_input(
            np.array([-0.1, 0.0], dtype=np.float32),
            domain="normalized",
            asinh_scale=1.0,
        )


def test_prediction_is_clipped_before_suv_inverse() -> None:
    prediction = np.array([-1.0, 0.0, np.arcsinh(2.0)], dtype=np.float32)
    output, clipped = prepare_inference_output(
        prediction,
        domain="suv",
        asinh_scale=1.0,
    )
    assert clipped == 1
    np.testing.assert_allclose(output, np.array([0.0, 0.0, 2.0]), rtol=1e-6, atol=1e-6)


def test_normalized_output_remains_nonnegative() -> None:
    output, clipped = prepare_inference_output(
        np.array([-0.25, 0.5], dtype=np.float32),
        domain="normalized",
        asinh_scale=1.0,
    )
    assert clipped == 1
    np.testing.assert_array_equal(output, np.array([0.0, 0.5], dtype=np.float32))
