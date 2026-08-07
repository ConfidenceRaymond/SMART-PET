from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from smartpet import __version__
from smartpet.checkpoint_io import sha256_file
from smartpet.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    _serialize_numpy_rng,
    _serialize_python_rng,
    atomic_save,
)

_CONFIRMATION = "I_UNDERSTAND_UNSAFE_PICKLE"
_ATTENTION_KEY = re.compile(r"^(?:module\.)?attention\.(\d+)\.")
_REQUIRED_FULL_FIELDS = (
    "generator_state",
    "discriminator_state",
    "g_optimizer_state",
    "d_optimizer_state",
    "global_step",
    "g_optimizer_updates",
    "d_optimizer_updates",
    "rng_states",
    "world_size",
    "config",
    "best_metric",
)


def _normalize_attention_levels(raw: Sequence[int]) -> tuple[int, ...]:
    levels: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError("attention_levels must contain integers")
        level = int(value)
        if level < 0 or level > 6:
            raise RuntimeError("attention_levels must contain values within [0, 6]")
        levels.append(level)
    normalized = tuple(levels)
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("attention_levels must not contain duplicates")
    return normalized


def _state_attention_levels(raw: object) -> tuple[int, ...]:
    if not isinstance(raw, Mapping):
        raise RuntimeError("Legacy generator_state must be a mapping")
    levels: set[int] = set()
    for key in raw:
        if not isinstance(key, str):
            raise RuntimeError("Legacy generator_state keys must be strings")
        match = _ATTENTION_KEY.match(key)
        if match is not None:
            levels.add(int(match.group(1)))
    return tuple(sorted(levels))


def _migrate_config(
    raw_config: object,
    generator_state: object,
    *,
    attention_levels: Sequence[int] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(raw_config, Mapping):
        raise RuntimeError("Legacy checkpoint config must be a mapping")

    config = dict(raw_config)
    state_levels = _state_attention_levels(generator_state)
    supplied_levels = (
        None if attention_levels is None else _normalize_attention_levels(attention_levels)
    )
    recorded_raw = config.get("attention_levels")

    overrides: dict[str, Any] = {}
    if recorded_raw is None:
        if supplied_levels is None:
            raise RuntimeError(
                "Legacy checkpoint config is missing 'attention_levels'. Supply the "
                "known training architecture explicitly with --attention-levels; "
                "SMART-PET will not guess architecture metadata during conversion."
            )
        if state_levels and supplied_levels != state_levels:
            raise RuntimeError(
                "Supplied attention_levels do not match generator_state: "
                f"supplied={list(supplied_levels)}, state={list(state_levels)}"
            )
        config["attention_levels"] = list(supplied_levels)
        overrides["attention_levels"] = list(supplied_levels)
    else:
        if not isinstance(recorded_raw, list | tuple):
            raise RuntimeError("Legacy config attention_levels must be an array")
        recorded_levels = _normalize_attention_levels(recorded_raw)
        if supplied_levels is not None and supplied_levels != recorded_levels:
            raise RuntimeError(
                "Supplied attention_levels conflict with the value recorded in the "
                f"legacy config: supplied={list(supplied_levels)}, "
                f"recorded={list(recorded_levels)}"
            )
        if state_levels and recorded_levels != state_levels:
            raise RuntimeError(
                "Legacy config attention_levels do not match generator_state: "
                f"recorded={list(recorded_levels)}, state={list(state_levels)}"
            )
        config["attention_levels"] = list(recorded_levels)

    return config, {
        "config_overrides": overrides,
        "state_attention_levels": list(state_levels),
    }


def _convert_rng_state(raw: object, *, rank: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError(f"Legacy RNG state for rank {rank} must be a dictionary")
    converted = dict(raw)
    if isinstance(converted.get("python"), tuple):
        converted["python"] = _serialize_python_rng(converted["python"])
    if isinstance(converted.get("numpy"), tuple):
        converted["numpy"] = _serialize_numpy_rng(converted["numpy"])
    for field in ("torch_cpu",):
        value = converted.get(field)
        if value is not None and not isinstance(value, torch.Tensor):
            converted[field] = torch.as_tensor(value, dtype=torch.uint8, device="cpu")
    if "torch_cuda" in converted:
        cuda_states = converted["torch_cuda"]
        if not isinstance(cuda_states, list | tuple):
            raise RuntimeError(f"Legacy torch_cuda RNG state for rank {rank} must be an array")
        converted["torch_cuda"] = [
            value.detach().cpu()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, dtype=torch.uint8, device="cpu")
            for value in cuda_states
        ]
    return converted


def convert_legacy_checkpoint(
    input_path: str | Path,
    output_path: str | Path,
    *,
    expected_sha256: str,
    confirmation: str,
    attention_levels: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Convert one explicitly trusted v3 checkpoint to the tensor-safe v4 contract.

    This function intentionally performs unsafe pickle loading. It must only be
    run in a disposable isolated environment after the input digest is verified.
    Normal SMART-PET code paths never call it.
    """

    source = Path(input_path)
    output = Path(output_path)
    if confirmation != _CONFIRMATION:
        raise RuntimeError(
            "Legacy conversion requires the exact confirmation token "
            f"{_CONFIRMATION!r}"
        )
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must be different")

    expected = expected_sha256.strip().lower()
    actual = sha256_file(source)
    if len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    if actual != expected:
        raise RuntimeError(f"Legacy checkpoint SHA-256 mismatch: expected {expected}, got {actual}")

    # SECURITY BOUNDARY: this is the sole intentional unsafe loader in SMART-PET.
    # Execute only in a disposable container with no credentials, patient data,
    # network access, or writable host mounts.
    raw = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Legacy checkpoint must contain a dictionary, got {type(raw).__name__}")
    missing = [field for field in _REQUIRED_FULL_FIELDS if field not in raw]
    if missing:
        raise RuntimeError(f"Legacy checkpoint is missing required full-state fields: {missing}")
    source_format = int(raw.get("format_version", 0))
    if source_format != 3:
        raise RuntimeError(
            f"This converter accepts only SMART-PET format_version=3, found {source_format}"
        )

    rng_states = raw["rng_states"]
    if not isinstance(rng_states, list) or len(rng_states) != int(raw["world_size"]):
        raise RuntimeError("Legacy RNG state count does not match world_size")

    migrated_config, migration_metadata = _migrate_config(
        raw["config"],
        raw["generator_state"],
        attention_levels=attention_levels,
    )

    converted = dict(raw)
    converted["format_version"] = CHECKPOINT_FORMAT_VERSION
    converted["artifact_type"] = "smartpet_training_checkpoint"
    converted["smartpet_version"] = str(raw.get("smartpet_version", "legacy-v0.3.0"))
    converted["config"] = migrated_config
    converted["rng_states"] = [
        _convert_rng_state(state, rank=rank) for rank, state in enumerate(rng_states)
    ]
    converted["conversion"] = {
        "converted_by_smartpet_version": __version__,
        "source_format_version": source_format,
        "source_sha256": actual,
        **migration_metadata,
    }
    atomic_save(converted, output)
    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "source_sha256": actual,
        "source_format_version": source_format,
        "output_format_version": CHECKPOINT_FORMAT_VERSION,
        "attention_levels": list(migrated_config["attention_levels"]),
        "config_overrides": migration_metadata["config_overrides"],
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert an explicitly trusted SMART-PET v0.3.0 format-3 checkpoint "
            "to the safe format-4 contract. Run only in an isolated disposable environment."
        )
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument(
        "--attention-levels",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit legacy generator attention levels, for example "
            "--attention-levels 2 3. Required when the format-3 config omitted "
            "this architecture field."
        ),
    )
    p.add_argument(
        "--confirmation",
        required=True,
        help=f"Must equal {_CONFIRMATION}",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = convert_legacy_checkpoint(
        args.input,
        args.output,
        expected_sha256=args.expected_sha256,
        confirmation=args.confirmation,
        attention_levels=args.attention_levels,
    )
    print(f"[OK] converted checkpoint: {Path(result['output']).resolve()}")
    print(f"[OK] source SHA-256: {result['source_sha256']}")
    print(f"[OK] output SHA-256: {result['output_sha256']}")
    print(f"[OK] output format_version={result['output_format_version']}")
    print(f"[OK] attention_levels={result['attention_levels']}")
    print(f"[OK] config_overrides={result['config_overrides']}")
    print("[OK] SMART-PET LEGACY CHECKPOINT CONVERSION PASSED")


if __name__ == "__main__":
    main()
