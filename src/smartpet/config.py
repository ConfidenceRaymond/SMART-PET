from __future__ import annotations

import json
import os
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

from smartpet.training.trainer import TrainConfig

_TUPLE_FIELDS = {"patch_size", "preview_stride", "attention_levels"}
_ALIAS_FIELDS = {"batch_size_per_rank": "batch_size"}
_IGNORED_METADATA_FIELDS = {"profile", "recommended_world_size", "global_batch_size"}


def _coerce_value(name: str, value: Any) -> Any:
    if name in _TUPLE_FIELDS:
        if not isinstance(value, list | tuple):
            raise TypeError(f"Configuration field {name!r} must be a JSON array")
        return tuple(int(item) for item in value)
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def load_train_config(path: str | Path, *, overrides: dict[str, Any] | None = None) -> TrainConfig:
    """Load a strict JSON training configuration.

    Unknown keys are rejected. Command-line overrides are applied after the file.
    Environment variables and ``~`` are expanded in string values.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Training configuration must be a JSON object")

    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _IGNORED_METADATA_FIELDS:
            continue
        canonical = _ALIAS_FIELDS.get(key, key)
        if canonical in normalized:
            raise ValueError(f"Configuration defines {canonical!r} more than once")
        normalized[canonical] = value

    for key, value in (overrides or {}).items():
        if value is not None:
            normalized[key] = value

    valid_fields = {field.name: field for field in fields(TrainConfig)}
    unknown = sorted(set(normalized) - set(valid_fields))
    if unknown:
        raise ValueError(f"Unknown training configuration fields: {unknown}")

    payload: dict[str, Any] = {}
    missing: list[str] = []
    for name, field in valid_fields.items():
        if name in normalized:
            payload[name] = _coerce_value(name, normalized[name])
            continue
        if field.default is MISSING and field.default_factory is MISSING:
            missing.append(name)
    if missing:
        raise ValueError("Training configuration is missing required fields: " + ", ".join(missing))
    return TrainConfig(**payload)


def parse_set_overrides(items: list[str]) -> dict[str, Any]:
    """Parse repeated ``--set key=json_value`` overrides."""

    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value {item!r}; expected key=json_value")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--set key cannot be empty")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        overrides[key] = value
    return overrides
