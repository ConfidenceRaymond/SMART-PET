from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass

import torch

AMP_DTYPES = ("auto", "bf16", "fp16")


@dataclass(frozen=True)
class PrecisionPolicy:
    requested: str
    resolved: str
    autocast_enabled: bool
    autocast_dtype: torch.dtype | None
    scaler_enabled: bool

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        dtype = payload["autocast_dtype"]
        payload["autocast_dtype"] = str(dtype).replace("torch.", "") if dtype is not None else None
        return payload


def select_cuda_amp_dtype(requested: str, *, bf16_supported: bool) -> str:
    value = requested.strip().lower()
    if value not in AMP_DTYPES:
        raise ValueError(f"Unsupported AMP dtype {requested!r}; choose one of {AMP_DTYPES}")
    if value == "auto":
        return "bf16" if bf16_supported else "fp16"
    if value == "bf16" and not bf16_supported:
        raise RuntimeError(
            "bfloat16 was requested but is not supported by the visible CUDA device; "
            "use --amp-dtype fp16 or --no-amp"
        )
    return value


def resolve_precision(
    *,
    amp: bool,
    amp_dtype: str,
    device: torch.device,
    bf16_supported: bool | None = None,
) -> PrecisionPolicy:
    requested = amp_dtype.strip().lower()
    if requested not in AMP_DTYPES:
        raise ValueError(f"Unsupported AMP dtype {amp_dtype!r}; choose one of {AMP_DTYPES}")

    if not amp or device.type != "cuda":
        return PrecisionPolicy(
            requested=requested,
            resolved="fp32",
            autocast_enabled=False,
            autocast_dtype=None,
            scaler_enabled=False,
        )

    if bf16_supported is None:
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    resolved = select_cuda_amp_dtype(requested, bf16_supported=bf16_supported)
    dtype = torch.bfloat16 if resolved == "bf16" else torch.float16
    return PrecisionPolicy(
        requested=requested,
        resolved=resolved,
        autocast_enabled=True,
        autocast_dtype=dtype,
        scaler_enabled=resolved == "fp16",
    )


def autocast_context(
    policy: PrecisionPolicy,
    device: torch.device,
) -> AbstractContextManager[object]:
    if not policy.autocast_enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=policy.autocast_dtype)


def optimizer_max_step(optimizer_or_state: torch.optim.Optimizer | dict[str, object]) -> int:
    if isinstance(optimizer_or_state, torch.optim.Optimizer):
        raw_state = optimizer_or_state.state.values()
    else:
        state = optimizer_or_state.get("state", {})
        if not isinstance(state, dict):
            return 0
        raw_state = state.values()

    steps: list[int] = []
    for parameter_state in raw_state:
        if not isinstance(parameter_state, dict) or "step" not in parameter_state:
            continue
        value = parameter_state["step"]
        if isinstance(value, torch.Tensor):
            value = value.item()
        steps.append(int(value))
    return max(steps, default=0)


def require_optimizer_advanced(
    optimizer: torch.optim.Optimizer,
    *,
    before: int,
    name: str,
) -> int:
    after = optimizer_max_step(optimizer)
    if after != before + 1:
        raise RuntimeError(
            f"{name} optimizer did not complete its update: before={before}, after={after}. "
            "This usually means fp16 GradScaler skipped the step after detecting non-finite "
            "gradients. Use bfloat16 on supported GPUs or disable AMP for diagnosis."
        )
    return after
