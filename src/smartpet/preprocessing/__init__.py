from __future__ import annotations

from .metadata import ExternalPairRecord, read_external_pair_metadata
from .suv import suv_from_activity_concentration

__all__ = [
    "ExternalPairRecord",
    "INPUT_KINDS",
    "read_external_pair_metadata",
    "run_external_preprocessing",
    "suv_from_activity_concentration",
]


def __getattr__(name: str):
    if name in {"INPUT_KINDS", "run_external_preprocessing"}:
        from .pipeline import INPUT_KINDS, run_external_preprocessing

        return {
            "INPUT_KINDS": INPUT_KINDS,
            "run_external_preprocessing": run_external_preprocessing,
        }[name]
    raise AttributeError(name)
