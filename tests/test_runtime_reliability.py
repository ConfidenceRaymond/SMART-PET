from __future__ import annotations

import multiprocessing as mp
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from smartpet.metrics.image_quality import _quantile
from smartpet.training.distributed import (
    Runtime,
    fail_collectively,
    setup,
    validate_eval_partition,
)
from smartpet.training.precision import optimizer_max_step
from smartpet.training.trainer import (
    _backward_and_prepare_step,
    _finish_joint_optimizer_step,
    _record_scaler_skip,
)


def _collective_failure_worker(rank: int, init_method: str, queue) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    runtime = Runtime(torch.device("cpu"), rank, 2, rank, True)
    try:
        fail_collectively(runtime, rank == 0, "rank-local validation failure")
    except RuntimeError as error:
        queue.put((rank, str(error)))
    finally:
        dist.destroy_process_group()




def test_ddp_setup_uses_explicit_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29501")
    monkeypatch.setenv("SMARTPET_DIST_TIMEOUT_MIN", "7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    runtime = setup("ddp")

    assert runtime.distributed
    assert captured["backend"] == "gloo"
    assert captured["timeout"] == timedelta(minutes=7)


def test_ddp_setup_rejects_nonpositive_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("SMARTPET_DIST_TIMEOUT_MIN", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="must be a positive integer"):
        setup("ddp")


def test_large_quantile_uses_deterministic_subsample() -> None:
    values = torch.zeros((300, 300, 300), dtype=torch.float32)
    result = _quantile(values, 0.5)
    assert result.item() == 0.0


def test_validation_partition_rejects_empty_rank() -> None:
    runtime = Runtime(torch.device("cpu"), 0, 4, 0, True)
    with pytest.raises(ValueError, match="3 subjects.*world_size is 4"):
        validate_eval_partition(3, runtime)


def test_collective_failure_reaches_every_cpu_rank(tmp_path: Path) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo distributed backend is unavailable")

    rendezvous = tmp_path / "gloo-rendezvous"
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_collective_failure_worker,
            args=(rank, f"file://{rendezvous}", queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("collective failure test process hung")
        assert process.exitcode == 0

    results = sorted(queue.get(timeout=2) for _ in range(2))
    assert [rank for rank, _ in results] == [0, 1]
    assert all("Collective abort (1 rank(s))" in message for _, message in results)
    assert all("rank 1: rank-local validation failure" in message for _, message in results)


def _make_optimizer(parameter: torch.nn.Parameter) -> torch.optim.Optimizer:
    return torch.optim.Adam([parameter], lr=1e-3)


def test_fp16_scaler_skip_keeps_joint_optimizer_progress_aligned() -> None:
    d_parameter = torch.nn.Parameter(torch.tensor(1.0))
    g_parameter = torch.nn.Parameter(torch.tensor(1.0))
    d_optimizer = _make_optimizer(d_parameter)
    g_optimizer = _make_optimizer(g_parameter)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=8.0)
    runtime = Runtime(torch.device("cpu"), 0, 1, 0, False)

    d_optimizer.zero_grad(set_to_none=True)
    d_grad, d_finite = _backward_and_prepare_step(
        loss=d_parameter.square(),
        optimizer=d_optimizer,
        parameters=[d_parameter],
        scaler=scaler,
        grad_clip=1.0,
    )
    g_optimizer.zero_grad(set_to_none=True)
    g_grad, g_finite = _backward_and_prepare_step(
        loss=g_parameter * torch.tensor(float("inf")),
        optimizer=g_optimizer,
        parameters=[g_parameter],
        scaler=scaler,
        grad_clip=1.0,
    )

    initial_scale = scaler.get_scale()
    stepped = _finish_joint_optimizer_step(
        d_optimizer=d_optimizer,
        g_optimizer=g_optimizer,
        scaler=scaler,
        local_finite=d_finite and g_finite,
        runtime=runtime,
    )

    assert torch.isfinite(d_grad)
    assert not torch.isfinite(g_grad)
    assert not stepped
    assert optimizer_max_step(d_optimizer) == 0
    assert optimizer_max_step(g_optimizer) == 0
    assert scaler.get_scale() < initial_scale


def test_finite_scaler_step_advances_both_optimizers_once() -> None:
    d_parameter = torch.nn.Parameter(torch.tensor(1.0))
    g_parameter = torch.nn.Parameter(torch.tensor(1.0))
    d_optimizer = _make_optimizer(d_parameter)
    g_optimizer = _make_optimizer(g_parameter)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=8.0)
    runtime = Runtime(torch.device("cpu"), 0, 1, 0, False)

    d_optimizer.zero_grad(set_to_none=True)
    _, d_finite = _backward_and_prepare_step(
        loss=d_parameter.square(),
        optimizer=d_optimizer,
        parameters=[d_parameter],
        scaler=scaler,
        grad_clip=1.0,
    )
    g_optimizer.zero_grad(set_to_none=True)
    _, g_finite = _backward_and_prepare_step(
        loss=g_parameter.square(),
        optimizer=g_optimizer,
        parameters=[g_parameter],
        scaler=scaler,
        grad_clip=1.0,
    )

    assert _finish_joint_optimizer_step(
        d_optimizer=d_optimizer,
        g_optimizer=g_optimizer,
        scaler=scaler,
        local_finite=d_finite and g_finite,
        runtime=runtime,
    )
    assert optimizer_max_step(d_optimizer) == 1
    assert optimizer_max_step(g_optimizer) == 1


def test_repeated_scaler_skips_raise_at_configured_limit() -> None:
    consecutive, total = _record_scaler_skip(0, 0, maximum=3)
    assert (consecutive, total) == (1, 1)
    consecutive, total = _record_scaler_skip(consecutive, total, maximum=3)
    assert (consecutive, total) == (2, 2)
    with pytest.raises(RuntimeError, match="3 consecutive"):
        _record_scaler_skip(consecutive, total, maximum=3)
