from __future__ import annotations

import numpy as np
import torch

from smartpet.metrics.masked import masked_image_metrics


def test_background_outside_mask_does_not_change_metrics():
    target = np.zeros((12, 12, 12), dtype=np.float32)
    prediction = np.zeros_like(target)
    mask = np.zeros_like(target, dtype=bool)
    mask[3:9, 3:9, 3:9] = True
    target[mask] = 2.0
    prediction[mask] = 1.8
    first = masked_image_metrics(
        prediction, target, mask, device=torch.device("cpu"), window_size=3
    )
    prediction[~mask] = 1000.0
    second = masked_image_metrics(
        prediction, target, mask, device=torch.device("cpu"), window_size=3
    )
    assert first == second
