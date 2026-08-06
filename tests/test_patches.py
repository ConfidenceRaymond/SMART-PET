import numpy as np

from smartpet.data.patches import PatchAccumulator, extract_patch, grid_starts


def test_grid_reconstructs_identity():
    rng = np.random.default_rng(7)
    volume = rng.normal(size=(25, 27, 29)).astype(np.float32)
    patch = (12, 12, 12)
    accumulator = PatchAccumulator(volume.shape, patch)
    for origin in grid_starts(volume.shape, patch, (7, 8, 9)):
        accumulator.add(extract_patch(volume, origin, patch), origin)
    np.testing.assert_allclose(accumulator.finalize(), volume, rtol=1e-5, atol=1e-5)
