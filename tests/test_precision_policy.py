from __future__ import annotations

from pathlib import Path

import pytest
import torch

from smartpet.checkpoint_io import safe_torch_load
from smartpet.models import PatchDiscriminator3D, SmartPETGenerator
from smartpet.training.checkpoint import load_checkpoint, save_checkpoint
from smartpet.training.distributed import Runtime
from smartpet.training.precision import (
    optimizer_max_step,
    require_optimizer_advanced,
    resolve_precision,
    select_cuda_amp_dtype,
)


def test_auto_amp_prefers_bfloat16_when_supported() -> None:
    assert select_cuda_amp_dtype("auto", bf16_supported=True) == "bf16"
    assert select_cuda_amp_dtype("auto", bf16_supported=False) == "fp16"


def test_explicit_bfloat16_rejects_unsupported_device() -> None:
    with pytest.raises(RuntimeError, match="not supported"):
        select_cuda_amp_dtype("bf16", bf16_supported=False)


def test_cpu_precision_disables_autocast_and_scaler() -> None:
    policy = resolve_precision(
        amp=True,
        amp_dtype="auto",
        device=torch.device("cpu"),
        bf16_supported=True,
    )
    assert policy.resolved == "fp32"
    assert not policy.autocast_enabled
    assert not policy.scaler_enabled


def test_optimizer_progress_tracks_only_real_adam_steps() -> None:
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    assert optimizer_max_step(optimizer) == 0

    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()

    assert optimizer_max_step(optimizer) == 1
    assert require_optimizer_advanced(optimizer, before=0, name="test") == 1


def test_require_optimizer_advanced_rejects_skipped_step() -> None:
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with pytest.raises(RuntimeError, match="did not complete"):
        require_optimizer_advanced(optimizer, before=0, name="generator")


def _complete_one_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    total = sum(parameter.square().sum() for parameter in model.parameters())
    total.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_checkpoint_records_and_validates_optimizer_updates(tmp_path: Path) -> None:
    generator = SmartPETGenerator(base_channels=1, attention_levels=())
    discriminator = PatchDiscriminator3D(base_channels=1)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    _complete_one_step(generator, g_optimizer)
    _complete_one_step(discriminator, d_optimizer)

    runtime = Runtime(torch.device("cpu"), 0, 1, 0, False)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        scaler=None,
        g_scheduler=None,
        d_scheduler=None,
        epoch=0,
        global_step=1,
        runtime=runtime,
        config={"amp_dtype": "auto"},
        best_metric=1.0,
        precision={"resolved": "fp32"},
    )

    checkpoint = safe_torch_load(path)
    assert checkpoint["format_version"] == 4
    assert checkpoint["g_optimizer_updates"] == 1
    assert checkpoint["d_optimizer_updates"] == 1
    assert checkpoint["precision"]["resolved"] == "fp32"

    restored_g = SmartPETGenerator(base_channels=1, attention_levels=())
    restored_d = PatchDiscriminator3D(base_channels=1)
    restored_g_optimizer = torch.optim.Adam(restored_g.parameters(), lr=1e-4)
    restored_d_optimizer = torch.optim.Adam(restored_d.parameters(), lr=1e-4)
    load_checkpoint(
        path,
        generator=restored_g,
        discriminator=restored_d,
        device=torch.device("cpu"),
        g_optimizer=restored_g_optimizer,
        d_optimizer=restored_d_optimizer,
        validate_optimizer_progress=True,
    )


def test_checkpoint_refuses_false_global_step(tmp_path: Path) -> None:
    generator = SmartPETGenerator(base_channels=1, attention_levels=())
    discriminator = PatchDiscriminator3D(base_channels=1)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    runtime = Runtime(torch.device("cpu"), 0, 1, 0, False)

    with pytest.raises(RuntimeError, match="inconsistent with global_step"):
        save_checkpoint(
            tmp_path / "invalid.pt",
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            scaler=None,
            g_scheduler=None,
            d_scheduler=None,
            epoch=0,
            global_step=2,
            runtime=runtime,
            config={},
            best_metric=1.0,
        )
