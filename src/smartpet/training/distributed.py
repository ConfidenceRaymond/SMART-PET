from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Sampler


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int
    world_size: int
    local_rank: int
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup(backend: str) -> Runtime:
    backend = backend.lower()
    if backend not in {"auto", "single", "ddp"}:
        raise ValueError(f"Unsupported backend={backend!r}; expected auto, single, or ddp")
    if backend == "auto":
        backend = "ddp" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "single"
    if backend == "ddp":
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            raise RuntimeError(
                "DDP backend requires torchrun or an equivalent launcher that defines "
                "RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, and MASTER_PORT."
            )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
        if use_cuda:
            torch.cuda.set_device(local_rank)
        timeout_minutes = int(os.environ.get("SMARTPET_DIST_TIMEOUT_MIN", "30"))
        if timeout_minutes <= 0:
            raise ValueError("SMARTPET_DIST_TIMEOUT_MIN must be a positive integer")
        init_kwargs = {
            "backend": "nccl" if use_cuda else "gloo",
            "init_method": "env://",
            "timeout": timedelta(minutes=timeout_minutes),
        }
        if use_cuda:
            try:
                dist.init_process_group(**init_kwargs, device_id=device)
            except TypeError:
                dist.init_process_group(**init_kwargs)
        else:
            dist.init_process_group(**init_kwargs)
        return Runtime(device, dist.get_rank(), dist.get_world_size(), local_rank, True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Runtime(device, 0, 1, 0, False)


def wrap(
    model: nn.Module,
    runtime: Runtime,
    *,
    broadcast_buffers: bool = False,
) -> nn.Module:
    if not runtime.distributed:
        return model
    kwargs = {
        "broadcast_buffers": bool(broadcast_buffers),
        "find_unused_parameters": False,
    }
    if runtime.device.type == "cuda":
        kwargs.update(device_ids=[runtime.local_rank], output_device=runtime.local_rank)
    return nn.parallel.DistributedDataParallel(model, **kwargs)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def barrier(runtime: Runtime) -> None:
    if runtime.distributed:
        if runtime.device.type == "cuda":
            dist.barrier(device_ids=[runtime.local_rank])
        else:
            dist.barrier()


def cleanup(runtime: Runtime) -> None:
    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()


def all_ranks_true(runtime: Runtime, local_value: bool) -> bool:
    """Return ``True`` only when every distributed rank reports ``True``."""

    if not runtime.distributed:
        return bool(local_value)
    flag = torch.tensor(
        [1 if local_value else 0],
        device=runtime.device,
        dtype=torch.int32,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def fail_collectively(runtime: Runtime, local_ok: bool, message: str) -> None:
    """Raise on every rank when one or more ranks report a local failure."""

    if not runtime.distributed:
        if not local_ok:
            raise RuntimeError(message)
        return

    flag = torch.tensor(
        [0 if local_ok else 1],
        device=runtime.device,
        dtype=torch.int32,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    failures = int(flag.item())
    if failures == 0:
        return

    local_message = None if local_ok else f"rank {runtime.rank}: {message}"
    gathered: list[str | None] = [None] * runtime.world_size
    dist.all_gather_object(gathered, local_message)
    details = "; ".join(item for item in gathered if item)
    raise RuntimeError(f"Collective abort ({failures} rank(s)): {details}")


def validate_eval_partition(dataset_size: int, runtime: Runtime) -> None:
    """Require at least one validation sample on every distributed rank."""

    if runtime.distributed and int(dataset_size) < runtime.world_size:
        raise ValueError(
            f"Validation manifest has {dataset_size} subjects but world_size is "
            f"{runtime.world_size}. Every rank must receive at least one validation "
            "subject. Reduce the DDP world size or enlarge the validation split."
        )


class DistributedEvalSampler(Sampler[int]):
    """Partition validation indices without padding or duplication."""

    def __init__(self, dataset, runtime: Runtime) -> None:
        self.length = len(dataset)
        self.rank = runtime.rank
        self.world_size = runtime.world_size

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.length:
            return 0
        return (self.length - 1 - self.rank) // self.world_size + 1
