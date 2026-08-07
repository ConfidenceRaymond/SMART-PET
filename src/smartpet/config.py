from __future__ import annotations

import json
import os
import types
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from smartpet.training.trainer import TrainConfig

_TUPLE_FIELDS = {"patch_size", "preview_stride", "attention_levels"}
_TUPLE_ARITY = {"patch_size": 3, "preview_stride": 3}
_ALIAS_FIELDS = {"batch_size_per_rank": "batch_size"}
_IGNORED_METADATA_FIELDS = {"profile", "recommended_world_size", "global_batch_size"}


def _canonical_name(name: str) -> str:
    return _ALIAS_FIELDS.get(name, name)


def _coerce_value(name: str, value: Any) -> Any:
    if name in _TUPLE_FIELDS:
        if not isinstance(value, list | tuple):
            raise TypeError(f"Configuration field {name!r} must be a JSON array")
        arity = _TUPLE_ARITY.get(name)
        if arity is not None and len(value) != arity:
            raise ValueError(
                f"Configuration field {name!r} must contain exactly {arity} elements; "
                f"received {len(value)}"
            )
        converted: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"Configuration field {name!r} entries must be integers; "
                    f"received {item!r} ({type(item).__name__})"
                )
            converted.append(item)
        return tuple(converted)
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def _check_type(name: str, value: Any, annotation: Any) -> None:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (types.UnionType,):
        if any(argument is type(None) and value is None for argument in args):
            return
        candidates = [argument for argument in args if argument is not type(None)]
        for candidate in candidates:
            try:
                _check_type(name, value, candidate)
            except TypeError:
                continue
            return
        expected = " | ".join(getattr(candidate, "__name__", str(candidate)) for candidate in args)
        raise TypeError(
            f"Configuration field {name!r} must be {expected}; "
            f"received {value!r} ({type(value).__name__})"
        )

    if origin is tuple:
        if not isinstance(value, tuple):
            raise TypeError(f"Configuration field {name!r} must be a tuple")
        return

    if annotation is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif annotation is float:
        valid = isinstance(value, int | float) and not isinstance(value, bool)
    elif annotation is bool:
        valid = isinstance(value, bool)
    elif annotation is str:
        valid = isinstance(value, str)
    else:
        valid = True
    if not valid:
        expected = getattr(annotation, "__name__", str(annotation))
        raise TypeError(
            f"Configuration field {name!r} must be {expected}; "
            f"received {value!r} ({type(value).__name__})"
        )


def _normalize_mapping(raw: dict[str, Any], *, source: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _IGNORED_METADATA_FIELDS:
            continue
        canonical = _canonical_name(key)
        if canonical in normalized:
            raise ValueError(
                f"{source} defines configuration field {canonical!r} more than once"
            )
        normalized[canonical] = value
    return normalized


def load_train_config(path: str | Path, *, overrides: dict[str, Any] | None = None) -> TrainConfig:
    """Load a strict JSON training configuration.

    Unknown keys and incompatible value types are rejected before any GPU,
    distributed process group, dataset, or output directory is initialized.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Training configuration must be a JSON object")

    normalized = _normalize_mapping(raw, source="Configuration file")
    override_values = _normalize_mapping(dict(overrides or {}), source="Command-line overrides")
    normalized.update({key: value for key, value in override_values.items() if value is not None})

    valid_fields = {field.name: field for field in fields(TrainConfig)}
    type_hints = get_type_hints(TrainConfig)
    unknown = sorted(set(normalized) - set(valid_fields))
    if unknown:
        raise ValueError(f"Unknown training configuration fields: {unknown}")

    payload: dict[str, Any] = {}
    missing: list[str] = []
    for name, field in valid_fields.items():
        if name in normalized:
            value = _coerce_value(name, normalized[name])
            _check_type(name, value, type_hints[name])
            payload[name] = value
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
        canonical = _canonical_name(key)
        if canonical in overrides:
            raise ValueError(f"Configuration override {canonical!r} was supplied more than once")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        overrides[canonical] = value
    return overrides
