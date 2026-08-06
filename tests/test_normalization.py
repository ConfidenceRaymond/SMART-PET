import numpy as np

from smartpet.data.normalization import asinh_denormalize, asinh_normalize


def test_asinh_roundtrip_and_negative_clip():
    data = np.array([-2.0, 0.0, 0.5, 2.0, 10.0], dtype=np.float32)
    normalized = asinh_normalize(data, 1.0)
    restored = asinh_denormalize(normalized, 1.0)
    np.testing.assert_allclose(restored, np.clip(data, 0, None), rtol=1e-6, atol=1e-6)
