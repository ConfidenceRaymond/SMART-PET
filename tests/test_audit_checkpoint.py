from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from smartpet.cli.audit_checkpoint import audit_checkpoint
from smartpet.models import PatchDiscriminator3D, SmartPETGenerator
from smartpet.training.checkpoint import save_checkpoint
from smartpet.training.distributed import Runtime


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _valid_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    generator = SmartPETGenerator(base_channels=1, attention_levels=())
    discriminator = PatchDiscriminator3D(base_channels=1)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    _step(generator, g_optimizer)
    _step(discriminator, d_optimizer)
    save_checkpoint(
        run_dir / "checkpoints" / "last.pt",
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        scaler=None,
        g_scheduler=None,
        d_scheduler=None,
        epoch=0,
        global_step=1,
        runtime=Runtime(torch.device("cpu"), 0, 1, 0, False),
        config={"backend": "single"},
        best_metric=0.5,
        precision={"resolved": "fp32"},
    )
    metrics = run_dir / "metrics.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    with metrics.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "global_step",
                "g_optimizer_updates",
                "d_optimizer_updates",
                "train_g",
                "train_d",
                "val_l1",
                "precision",
            ]
        )
        writer.writerow([0, 1, 1, 1, 2.0, 0.7, 0.5, "fp32"])
    return run_dir


def test_audit_accepts_consistent_checkpoint(tmp_path: Path) -> None:
    run_dir = _valid_run(tmp_path)
    result = audit_checkpoint(
        run_dir / "checkpoints" / "last.pt",
        metrics_path=run_dir / "metrics.csv",
        expected_step=1,
        expected_world_size=1,
        expected_precision="fp32",
    )
    assert result["generator_updates"] == 1
    assert result["discriminator_updates"] == 1


def test_audit_rejects_legacy_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"format_version": 1}, path)
    with pytest.raises(RuntimeError, match="predates optimizer-integrity"):
        audit_checkpoint(path)
