from __future__ import annotations

from pathlib import Path

import pytest
import torch

from smartpet.models import PatchDiscriminator3D, SmartPETGenerator
from smartpet.training.checkpoint import atomic_save, load_checkpoint


class _WriteMarker:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        statement = f"open({str(self.marker)!r}, 'w', encoding='utf-8').write('executed')"
        return exec, (statement,)


def test_full_checkpoint_loads_under_weights_only_policy(tmp_path: Path) -> None:
    g = SmartPETGenerator(base_channels=1, attention_levels=())
    d = PatchDiscriminator3D(base_channels=1)
    path = tmp_path / "checkpoint.pt"
    atomic_save(
        {
            "generator_state": g.state_dict(),
            "discriminator_state": d.state_dict(),
            "primitive_tuple": ("x",),
        },
        path,
    )
    result = load_checkpoint(path, generator=g, discriminator=d, device=torch.device("cpu"))
    assert "generator_state" in result


def test_load_checkpoint_refuses_pickle_code_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    path = tmp_path / "hostile.pt"
    torch.save({"generator_state": {}, "payload": _WriteMarker(marker)}, path)
    generator = SmartPETGenerator(base_channels=1, attention_levels=())

    with pytest.raises(RuntimeError, match="weights_only=True"):
        load_checkpoint(
            path,
            generator=generator,
            discriminator=None,
            device=torch.device("cpu"),
        )

    assert not marker.exists(), "checkpoint payload executed during safe load"


def test_finetuning_requires_discriminator_state(tmp_path: Path) -> None:
    generator = SmartPETGenerator(base_channels=1, attention_levels=())
    discriminator = PatchDiscriminator3D(base_channels=1)
    path = tmp_path / "inference_only.pt"
    atomic_save({"generator_state": generator.state_dict(), "config": {}}, path)

    with pytest.raises(RuntimeError, match="inference-only"):
        load_checkpoint(
            path,
            generator=generator,
            discriminator=discriminator,
            device=torch.device("cpu"),
            require_discriminator=True,
        )
