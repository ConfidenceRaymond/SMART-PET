from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from smartpet.data.patches import PatchAccumulator, extract_patch, grid_starts, pad_to_patch, unpad


@torch.no_grad()
def predict_volume(
    model: nn.Module,
    volume: np.ndarray,
    *,
    device: torch.device,
    patch_size: Sequence[int] = (128, 128, 128),
    stride: Sequence[int] = (64, 64, 64),
    amp: bool = True,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    model.eval()
    padded, pads = pad_to_patch(volume, patch_size)
    accumulator = PatchAccumulator(padded.shape, patch_size)
    for origin in grid_starts(padded.shape, patch_size, stride):
        patch = extract_patch(padded, origin, patch_size)
        tensor = torch.from_numpy(patch[None, None]).to(device=device, dtype=torch.float32)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp and device.type == "cuda",
        ):
            prediction = model(tensor)
        accumulator.add(prediction[0, 0].float().cpu().numpy(), origin)
    return unpad(accumulator.finalize(), pads)
