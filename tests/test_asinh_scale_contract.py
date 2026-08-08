import numpy as np

from smartpet.data.normalization import (
    asinh_denormalize,
    asinh_normalize,
)


def test_asinh_contract_at_nonunit_scales():
    suv = np.array(
        [-3.0, -0.25, 0.0, 0.1, 1.0, 4.0, 10.0, 35.0],
        dtype=np.float32,
    )
    nonnegative = np.clip(suv, 0.0, None)

    for scale in (0.5, 1.0, 2.0, 5.0):
        normalized = asinh_normalize(suv, scale=scale)

        # Lock the exact preprocessing contract:
        #
        # y = asinh(max(SUV, 0) / scale)
        expected_normalized = np.arcsinh(
            nonnegative / np.float32(scale)
        ).astype(np.float32)

        np.testing.assert_allclose(
            normalized,
            expected_normalized,
            rtol=2e-6,
            atol=2e-6,
        )

        restored = asinh_denormalize(
            normalized,
            scale=scale,
        )

        # Lock the exact inverse:
        #
        # SUV_nonnegative = scale * sinh(y)
        np.testing.assert_allclose(
            restored,
            nonnegative,
            rtol=2e-6,
            atol=2e-6,
        )


def test_asinh_inverse_scale_is_multiplicative():
    normalized = np.array(
        [0.0, 0.25, 1.0, 2.0],
        dtype=np.float32,
    )

    for scale in (0.5, 2.0, 5.0):
        actual = asinh_denormalize(
            normalized,
            scale=scale,
        )
        expected = (
            np.float32(scale) * np.sinh(normalized)
        ).astype(np.float32)

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2e-6,
            atol=2e-6,
        )
