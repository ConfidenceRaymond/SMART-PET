from __future__ import annotations

import argparse
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

    converted = dict(raw)
    converted["format_version"] = CHECKPOINT_FORMAT_VERSION
    converted["artifact_type"] = "smartpet_training_checkpoint"
    converted["smartpet_version"] = str(raw.get("smartpet_version", "legacy-v0.3.0"))
    converted["rng_states"] = [
        _convert_rng_state(state, rank=rank) for rank, state in enumerate(rng_states)
    ]
    converted["conversion"] = {
        "converted_by_smartpet_version": __version__,
        "source_format_version": source_format,
        "source_sha256": actual,
    }
    atomic_save(converted, output)
    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "source_sha256": actual,
        "source_format_version": source_format,
        "output_format_version": CHECKPOINT_FORMAT_VERSION,
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
    )
    print(f"[OK] converted checkpoint: {Path(result['output']).resolve()}")
    print(f"[OK] source SHA-256: {result['source_sha256']}")
    print(f"[OK] output SHA-256: {result['output_sha256']}")
    print(f"[OK] output format_version={result['output_format_version']}")
    print("[OK] SMART-PET LEGACY CHECKPOINT CONVERSION PASSED")


if __name__ == "__main__":
    main()
