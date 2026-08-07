from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch.nn as nn


def _has_spectral_norm(module: nn.Module) -> bool:
    if hasattr(module, "weight_orig"):
        return True
    parametrizations = getattr(module, "parametrizations", None)
    return parametrizations is not None and hasattr(parametrizations, "weight")


def _parameter_count(module: nn.Module, *, trainable_only: bool) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if (not trainable_only) or parameter.requires_grad
    )


def _module_inventory(module: nn.Module) -> dict[str, int]:
    counts = Counter(type(child).__name__ for child in module.modules())
    return dict(sorted(counts.items()))


def _spectral_norm_paths(module: nn.Module) -> list[str]:
    return [name or "<root>" for name, child in module.named_modules() if _has_spectral_norm(child)]


def architecture_report(generator: nn.Module, discriminator: nn.Module) -> dict[str, Any]:
    """Create a JSON-serializable static architecture report."""

    encoders = getattr(generator, "encoders", ())
    encoder_conv_counts = [
        sum(isinstance(child, nn.Conv3d) for child in encoder.modules()) for encoder in encoders
    ]
    attention = getattr(generator, "attention", {})
    attention_blocks = {
        str(name): type(block).__name__ for name, block in getattr(attention, "items", lambda: [])()
    }
    discriminator_has_sigmoid = any(
        isinstance(child, nn.Sigmoid) for child in discriminator.modules()
    )
    report = {
        "generator": {
            "class": type(generator).__name__,
            "parameters": _parameter_count(generator, trainable_only=False),
            "trainable_parameters": _parameter_count(generator, trainable_only=True),
            "module_counts": _module_inventory(generator),
            "spectral_norm_paths": _spectral_norm_paths(generator),
            "encoder_levels": len(encoder_conv_counts),
            "encoder_conv3d_counts": encoder_conv_counts,
            "attention_levels": list(getattr(generator, "attention_levels", ())),
            "attention_blocks": attention_blocks,
            "output_mode": getattr(generator, "output_mode", None),
            "output_head": type(getattr(generator, "output", None)).__name__,
        },
        "discriminator": {
            "class": type(discriminator).__name__,
            "parameters": _parameter_count(discriminator, trainable_only=False),
            "trainable_parameters": _parameter_count(discriminator, trainable_only=True),
            "module_counts": _module_inventory(discriminator),
            "spectral_norm_paths": _spectral_norm_paths(discriminator),
            "has_sigmoid": discriminator_has_sigmoid,
            "output_contract": "probability" if discriminator_has_sigmoid else "raw_logits",
        },
    }
    return report


def load_legacy_architecture_contract(path: str | Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ValueError(f"Unsupported legacy architecture contract: {path}")
    return contract
