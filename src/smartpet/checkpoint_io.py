from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_torch_load(path: str | Path) -> dict[str, Any]:
    """Load a tensor-only SMART-PET artifact without executing pickle globals.

    ``weights_only=True`` is a security boundary. A file that cannot be loaded
    under this policy must be converted in an isolated environment rather than
    opened by relaxing the loader.
    """

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    try:
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Refusing to load {artifact}: the file is not compatible with "
            "SMART-PET safe checkpoint loading (weights_only=True). It may be a "
            "legacy checkpoint or an unsafe/malformed artifact. Convert trusted "
            "legacy files in an isolated environment; do not disable weights_only."
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError(
            "SMART-PET artifact must contain a dictionary, "
            f"got {type(payload).__name__}"
        )
    return payload
