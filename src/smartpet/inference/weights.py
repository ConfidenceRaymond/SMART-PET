from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from smartpet import __version__
from smartpet.checkpoint_io import safe_torch_load, sha256_file
from smartpet.models import (
    ARCHITECTURE_CONFIG_FIELDS,
    OUTPUT_MODES,
    SmartPETGenerator,
    normalize_architecture_config,
)
from smartpet.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_save,
)

INFERENCE_WEIGHTS_FORMAT_VERSION = 2
SUPPORTED_INFERENCE_WEIGHTS_FORMATS = (1, 2)
INFERENCE_WEIGHTS_ARTIFACT_TYPE = "smartpet_inference_weights"
_BASE_INFERENCE_CONFIG_FIELDS = (
    "base_channels",
    "attention_levels",
    "output_mode",
    "asinh_scale",
    "patch_size",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _required(payload: Mapping[str, Any], field: str) -> Any:
    if field not in payload:
        raise RuntimeError(f"Inference weights are missing required field {field!r}")
    return payload[field]


def _validate_sha256(value: object, *, field_name: str) -> str:
    digest = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(digest):
        raise RuntimeError(f"{field_name} must be a 64-character hexadecimal SHA-256 digest")
    return digest


def _validated_config(
    raw: object,
    *,
    allow_v030_architecture: bool,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuntimeError("Inference weights config must be a mapping")
    missing = [field for field in _BASE_INFERENCE_CONFIG_FIELDS if raw.get(field) is None]
    if missing:
        raise RuntimeError(f"Inference weights config is missing required fields: {missing}")

    base_channels = raw["base_channels"]
    if isinstance(base_channels, bool) or not isinstance(base_channels, int):
        raise RuntimeError("Inference weights base_channels must be an integer")
    if base_channels <= 0:
        raise RuntimeError("Inference weights base_channels must be positive")

    levels_raw = raw["attention_levels"]
    if not isinstance(levels_raw, list | tuple):
        raise RuntimeError("Inference weights attention_levels must be an array")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in levels_raw):
        raise RuntimeError("Inference weights attention_levels must contain integers")
    attention_levels = tuple(int(value) for value in levels_raw)
    if len(set(attention_levels)) != len(attention_levels):
        raise RuntimeError("Inference weights attention_levels contains duplicate entries")
    if any(value < 0 or value > 6 for value in attention_levels):
        raise RuntimeError("Inference weights attention_levels must be within [0, 6]")

    output_mode = raw["output_mode"]
    if not isinstance(output_mode, str) or output_mode not in OUTPUT_MODES:
        raise RuntimeError(
            f"Inference weights output_mode={output_mode!r} is unsupported; expected {OUTPUT_MODES}"
        )

    asinh_scale = raw["asinh_scale"]
    if isinstance(asinh_scale, bool) or not isinstance(asinh_scale, int | float):
        raise RuntimeError("Inference weights asinh_scale must be numeric")
    if not math.isfinite(float(asinh_scale)) or float(asinh_scale) <= 0:
        raise RuntimeError("Inference weights asinh_scale must be finite and positive")

    patch_raw = raw["patch_size"]
    if not isinstance(patch_raw, list | tuple) or len(patch_raw) != 3:
        raise RuntimeError("Inference weights patch_size must contain exactly three integers")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in patch_raw):
        raise RuntimeError("Inference weights patch_size must contain integers")
    patch_size = tuple(int(value) for value in patch_raw)
    if any(value <= 0 for value in patch_size):
        raise RuntimeError("Inference weights patch_size values must be positive")

    try:
        architecture = normalize_architecture_config(
            raw,
            allow_v030_defaults=allow_v030_architecture,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid inference architecture config: {exc}") from exc

    return {
        "base_channels": int(base_channels),
        "attention_levels": list(attention_levels),
        "output_mode": output_mode,
        "asinh_scale": float(asinh_scale),
        "patch_size": list(patch_size),
        **architecture,
    }


def _validate_state_dict(raw: object) -> dict[str, torch.Tensor]:
    if not isinstance(raw, Mapping) or not raw:
        raise RuntimeError("Inference weights generator_state must be a non-empty mapping")
    invalid_keys = [key for key in raw if not isinstance(key, str)]
    if invalid_keys:
        raise RuntimeError("Inference weights generator_state keys must be strings")
    invalid_values = [key for key, value in raw.items() if not isinstance(value, torch.Tensor)]
    if invalid_values:
        raise RuntimeError(
            "Inference weights generator_state must contain tensors only; "
            f"invalid keys={invalid_values[:5]}"
        )
    return dict(raw)


def _generator_from_config(config: Mapping[str, Any]) -> SmartPETGenerator:
    return SmartPETGenerator(
        int(config["base_channels"]),
        attention_levels=tuple(config["attention_levels"]),
        output_mode=str(config["output_mode"]),
        similarity_mode=str(config["similarity_mode"]),
        encoder_convs_per_level=int(config["encoder_convs_per_level"]),
        channel_spatial_input_projection=bool(config["channel_spatial_input_projection"]),
        generator_spectral_norm=bool(config["generator_spectral_norm"]),
    )


def _validate_model_state(
    config: Mapping[str, Any],
    generator_state: Mapping[str, torch.Tensor],
) -> None:
    model = _generator_from_config(config)
    try:
        model.load_state_dict(generator_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Inference weights generator_state does not match the recorded architecture"
        ) from exc


def export_inference_weights(
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Export a full safe training checkpoint as inference-only generator weights."""

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if checkpoint_path.resolve() == output_path.resolve():
        raise ValueError("Input checkpoint and output weights path must be different")

    full = safe_torch_load(checkpoint_path)
    if full.get("artifact_type") != "smartpet_training_checkpoint":
        raise RuntimeError(
            "Source artifact is not a SMART-PET full training checkpoint; "
            f"artifact_type={full.get('artifact_type')!r}"
        )
    source_format = int(full.get("format_version", 0))
    if source_format < CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"Source checkpoint format_version={source_format} predates the safe export "
            f"format {CHECKPOINT_FORMAT_VERSION}; convert the trusted legacy file first."
        )

    # Format-4 v0.3.0 checkpoints predate explicit S4 architecture fields. The
    # all-absent case is the immutable v0.3.0 schema; partial metadata is refused.
    config = _validated_config(
        _required(full, "config"),
        allow_v030_architecture=True,
    )
    generator_state = _validate_state_dict(_required(full, "generator_state"))
    _validate_model_state(config, generator_state)

    global_step = int(_required(full, "global_step"))
    epoch = int(_required(full, "epoch"))
    best_metric = float(_required(full, "best_metric"))
    if global_step < 0 or epoch < -1 or not math.isfinite(best_metric):
        raise RuntimeError(
            "Source checkpoint provenance is invalid: "
            f"global_step={global_step}, epoch={epoch}, best_metric={best_metric}"
        )

    payload: dict[str, Any] = {
        "format_version": INFERENCE_WEIGHTS_FORMAT_VERSION,
        "artifact_type": INFERENCE_WEIGHTS_ARTIFACT_TYPE,
        "generator_state": generator_state,
        "config": config,
        "smartpet_version": __version__,
        "source_smartpet_version": str(full.get("smartpet_version", "unknown")),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_checkpoint_format_version": source_format,
        "source_global_step": global_step,
        "source_epoch": epoch,
        "source_best_metric": best_metric,
    }
    atomic_save(payload, output_path)
    return payload


def audit_inference_weights(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    validate_model_state: bool = True,
) -> dict[str, Any]:
    """Audit an inference-only artifact without applying training invariants."""

    artifact_path = Path(path)
    actual_sha256 = sha256_file(artifact_path)
    if expected_sha256 is not None:
        expected = _validate_sha256(expected_sha256, field_name="expected_sha256")
        if actual_sha256 != expected:
            raise RuntimeError(f"SHA-256 mismatch: expected {expected}, got {actual_sha256}")

    payload = safe_torch_load(artifact_path)
    artifact_type = _required(payload, "artifact_type")
    if artifact_type != INFERENCE_WEIGHTS_ARTIFACT_TYPE:
        raise RuntimeError(
            f"Expected artifact_type={INFERENCE_WEIGHTS_ARTIFACT_TYPE!r}, "
            f"found {artifact_type!r}"
        )
    format_version = int(_required(payload, "format_version"))
    if format_version not in SUPPORTED_INFERENCE_WEIGHTS_FORMATS:
        raise RuntimeError(
            f"Unsupported inference weights format_version={format_version}; "
            f"expected one of {SUPPORTED_INFERENCE_WEIGHTS_FORMATS}"
        )

    config = _validated_config(
        _required(payload, "config"),
        allow_v030_architecture=(format_version == 1),
    )
    generator_state = _validate_state_dict(_required(payload, "generator_state"))
    parent_sha256 = _validate_sha256(
        _required(payload, "source_checkpoint_sha256"),
        field_name="source_checkpoint_sha256",
    )
    source_format_version = int(_required(payload, "source_checkpoint_format_version"))
    if source_format_version < CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            "Inference weights source_checkpoint_format_version predates the safe "
            f"training format {CHECKPOINT_FORMAT_VERSION}"
        )
    source_smartpet_version = str(_required(payload, "source_smartpet_version")).strip()
    if not source_smartpet_version:
        raise RuntimeError("Inference weights source_smartpet_version cannot be empty")
    smartpet_version = str(_required(payload, "smartpet_version")).strip()
    if not smartpet_version:
        raise RuntimeError("Inference weights smartpet_version cannot be empty")

    source_global_step = int(_required(payload, "source_global_step"))
    source_epoch = int(_required(payload, "source_epoch"))
    source_best_metric = float(_required(payload, "source_best_metric"))
    if source_global_step < 0 or source_epoch < -1 or not math.isfinite(source_best_metric):
        raise RuntimeError(
            "Inference weights source provenance is invalid: "
            f"global_step={source_global_step}, epoch={source_epoch}, "
            f"best_metric={source_best_metric}"
        )

    if validate_model_state:
        _validate_model_state(config, generator_state)

    return {
        "path": str(artifact_path),
        "sha256": actual_sha256,
        "format_version": format_version,
        "artifact_type": artifact_type,
        "smartpet_version": smartpet_version,
        "config": config,
        "architecture_config_fields": list(ARCHITECTURE_CONFIG_FIELDS),
        "source_checkpoint_sha256": parent_sha256,
        "source_checkpoint_format_version": source_format_version,
        "source_smartpet_version": source_smartpet_version,
        "source_global_step": source_global_step,
        "source_epoch": source_epoch,
        "source_best_metric": source_best_metric,
    }
