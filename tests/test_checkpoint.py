from pathlib import Path

import torch

from smartpet.models import PatchDiscriminator3D, SmartPETGenerator
from smartpet.training.checkpoint import atomic_save, load_checkpoint


def test_full_checkpoint_uses_trusted_load(tmp_path: Path):
    g = SmartPETGenerator(base_channels=1, attention_levels=())
    d = PatchDiscriminator3D(base_channels=1)
    path = tmp_path / "checkpoint.pt"
    atomic_save(
        {
            "generator_state": g.state_dict(),
            "discriminator_state": d.state_dict(),
            "numpy_like": ("x",),
        },
        path,
    )
    result = load_checkpoint(path, generator=g, discriminator=d, device=torch.device("cpu"))
    assert "generator_state" in result
