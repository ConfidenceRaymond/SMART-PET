from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from smartpet import __version__
from smartpet.checkpoint_io import safe_torch_load

from .distributed import Runtime, unwrap
from .precision import optimizer_max_step

CHECKPOINT_FORMAT_VERSION = 4


def _serialize_python_rng(state: tuple[Any, ...]) -> dict[str, Any]:
    version, internal, gaussian = state
    return {
        "version": int(version),
        "state": [int(value) for value in internal],
        "gaussian": None if gaussian is None else float(gaussian),
    }


def _deserialize_python_rng(payload: object) -> tuple[Any, ...]:
    if isinstance(payload, tuple):
        return payload
    if not isinstance(payload, dict):
        raise TypeError("python RNG state must be a dictionary")
    return (
        int(payload["version"]),
        tuple(int(value) for value in payload["state"]),
        None if payload.get("gaussian") is None else float(payload["gaussian"]),
    )


def _serialize_numpy_rng(state: tuple[Any, ...]) -> dict[str, Any]:
    bit_generator, keys, position, has_gaussian, cached_gaussian = state
    keys_array = np.asarray(keys, dtype=np.uint32)
    return {
        "bit_generator": str(bit_generator),
        "keys": [int(value) for value in keys_array.tolist()],
        "position": int(position),
        "has_gaussian": int(has_gaussian),
        "cached_gaussian": float(cached_gaussian),
    }


def _deserialize_numpy_rng(payload: object) -> tuple[Any, ...]:
    if isinstance(payload, tuple):
        return payload
    if not isinstance(payload, dict):
        raise TypeError("numpy RNG state must be a dictionary")
    return (
        str(payload["bit_generator"]),
        np.asarray(payload["keys"], dtype=np.uint32),
        int(payload["position"]),
        int(payload["has_gaussian"]),
        float(payload["cached_gaussian"]),
    )


def capture_rng() -> dict[str, Any]:
    """Capture RNG state using only weights-only-safe primitive/tensor values."""

    state: dict[str, Any] = {
        "python": _serialize_python_rng(random.getstate()),
        "numpy": _serialize_numpy_rng(np.random.get_state()),
        "torch_cpu": torch.random.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    return state


def _as_cpu_byte_tensor(
    value: object,
    *,
    field_name: str,
) -> torch.Tensor:
    """Normalize a serialized RNG state to contiguous CPU uint8."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.uint8)
    else:
        tensor = torch.as_tensor(
            value,
            dtype=torch.uint8,
            device="cpu",
        )

    if tensor.ndim != 1:
        raise TypeError(
            f"{field_name} RNG state must be one-dimensional; "
            f"received shape={tuple(tensor.shape)}"
        )

    return tensor.contiguous()


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(_deserialize_python_rng(state["python"]))
    np.random.set_state(_deserialize_numpy_rng(state["numpy"]))
    torch.random.set_rng_state(
        _as_cpu_byte_tensor(
            state["torch_cpu"],
            field_name="torch_cpu",
        )
    )
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [
                _as_cpu_byte_tensor(value, field_name=f"torch_cuda[{index}]")
                for index, value in enumerate(state["torch_cuda"])
            ]
        )


def gather_rng(runtime: Runtime) -> list[dict[str, Any]]:
    local = capture_rng()
    if not runtime.distributed:
        return [local]
    gathered: list[Any] = [None] * runtime.world_size
    dist.all_gather_object(gathered, local)
    return gathered


def gather_objects(runtime: Runtime, local: Any) -> list[Any]:
    if not runtime.distributed:
        return [local]
    gathered: list[Any] = [None] * runtime.world_size
    dist.all_gather_object(gathered, local)
    return gathered


def atomic_save(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, delete=False) as handle:
        temp = Path(handle.name)
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def _validate_progress(
    *,
    global_step: int,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    g_updates = optimizer_max_step(g_optimizer)
    d_updates = optimizer_max_step(d_optimizer)
    expected = int(global_step)
    if g_updates != expected or d_updates != expected:
        raise RuntimeError(
            "Checkpoint optimizer progress is inconsistent with global_step: "
            f"global_step={expected}, generator_updates={g_updates}, "
            f"discriminator_updates={d_updates}. Refusing to save or resume a checkpoint "
            "that counted skipped optimizer steps."
        )
    return g_updates, d_updates


def save_checkpoint(
    path: str | Path,
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    g_scheduler: object | None,
    d_scheduler: object | None,
    epoch: int,
    global_step: int,
    runtime: Runtime,
    config: dict[str, Any],
    best_metric: float,
    precision: dict[str, object] | None = None,
    batch_in_epoch: int = 0,
    epoch_complete: bool = True,
    samples_seen: int = 0,
    training_state: dict[str, Any] | None = None,
) -> None:
    g_updates, d_updates = _validate_progress(
        global_step=global_step,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
    )
    rng_states = gather_rng(runtime)
    training_states = gather_objects(runtime, dict(training_state or {}))
    if not runtime.is_main:
        return
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "artifact_type": "smartpet_training_checkpoint",
        "smartpet_version": __version__,
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "epoch_complete": bool(epoch_complete),
        "global_step": int(global_step),
        "samples_seen": int(samples_seen),
        "g_optimizer_updates": g_updates,
        "d_optimizer_updates": d_updates,
        "generator_state": unwrap(generator).state_dict(),
        "discriminator_state": unwrap(discriminator).state_dict(),
        "g_optimizer_state": g_optimizer.state_dict(),
        "d_optimizer_state": d_optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "g_scheduler_state": g_scheduler.state_dict() if g_scheduler is not None else None,
        "d_scheduler_state": d_scheduler.state_dict() if d_scheduler is not None else None,
        "rng_states": rng_states,
        "world_size": runtime.world_size,
        "config": dict(config),
        "precision": dict(precision or {}),
        "best_metric": float(best_metric),
        "training_state": dict(training_states[0]),
        "training_states": training_states,
    }
    atomic_save(payload, path)


def _required(checkpoint: dict[str, Any], field: str) -> Any:
    if field not in checkpoint:
        raise RuntimeError(f"Checkpoint is missing required field {field!r}")
    return checkpoint[field]


def load_checkpoint(
    path: str | Path,
    *,
    generator: nn.Module,
    discriminator: nn.Module | None,
    device: torch.device,
    g_optimizer: torch.optim.Optimizer | None = None,
    d_optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    g_scheduler: object | None = None,
    d_scheduler: object | None = None,
    validate_optimizer_progress: bool = False,
    restore_rank_rng: int | None = None,
    require_discriminator: bool = False,
) -> dict[str, Any]:
    """Load a SMART-PET checkpoint under the weights-only security boundary."""

    del device  # State is loaded on CPU first; module/optimizer placement is managed by callers.
    checkpoint = safe_torch_load(path)
    generator_state = _required(checkpoint, "generator_state")
    unwrap(generator).load_state_dict(generator_state)

    if discriminator is not None:
        if "discriminator_state" in checkpoint:
            unwrap(discriminator).load_state_dict(checkpoint["discriminator_state"])
        elif require_discriminator:
            raise RuntimeError(
                f"{path} contains no discriminator_state. It is an inference-only artifact, "
                "not a safe adversarial fine-tuning checkpoint. Use a full training checkpoint."
            )

    if g_optimizer is not None:
        g_optimizer.load_state_dict(_required(checkpoint, "g_optimizer_state"))
    if d_optimizer is not None:
        d_optimizer.load_state_dict(_required(checkpoint, "d_optimizer_state"))
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    if g_scheduler is not None and checkpoint.get("g_scheduler_state") is not None:
        g_scheduler.load_state_dict(checkpoint["g_scheduler_state"])
    if d_scheduler is not None and checkpoint.get("d_scheduler_state") is not None:
        d_scheduler.load_state_dict(checkpoint["d_scheduler_state"])

    if validate_optimizer_progress:
        format_version = int(checkpoint.get("format_version", 0))
        if format_version < CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(
                f"Checkpoint format_version={format_version} predates the safe v0.3.1 "
                f"resume format {CHECKPOINT_FORMAT_VERSION}; convert the trusted legacy file first."
            )
        if g_optimizer is None or d_optimizer is None:
            raise ValueError("Both optimizers are required to validate checkpoint progress")
        _validate_progress(
            global_step=int(_required(checkpoint, "global_step")),
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
        )
    if restore_rank_rng is not None:
        states = checkpoint.get("rng_states", [])
        if int(restore_rank_rng) >= len(states):
            raise RuntimeError(
                f"Checkpoint has {len(states)} RNG states, cannot restore rank {restore_rank_rng}"
            )
        restore_rng(states[int(restore_rank_rng)])
    return checkpoint
