from __future__ import annotations

import random

import numpy as np
import torch

from smartpet.training.checkpoint import restore_rng


def test_restore_rng_coerces_cpu_state_to_byte_tensor() -> None:
    original_python = random.getstate()
    original_numpy = np.random.get_state()
    original_torch = torch.random.get_rng_state()

    desired_torch = original_torch.clone()

    state = {
        "python": original_python,
        "numpy": original_numpy,
        # Deliberately use the wrong dtype to exercise normalization.
        "torch_cpu": desired_torch.to(dtype=torch.int16),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    try:
        restore_rng(state)
        restored = torch.random.get_rng_state()

        assert restored.device.type == "cpu"
        assert restored.dtype == torch.uint8
        assert torch.equal(restored, desired_torch)
    finally:
        random.setstate(original_python)
        np.random.set_state(original_numpy)
        torch.random.set_rng_state(original_torch)
