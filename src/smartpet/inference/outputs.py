from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OutputPlan:
    legacy_output: Path | None
    legacy_domain: str | None
    normalized_output: Path | None
    suv_output: Path | None
    metadata_json: Path | None

    @property
    def is_shared_output_mode(self) -> bool:
        return self.normalized_output is not None or self.suv_output is not None


def _path_or_none(value: str | Path | None, *, name: str) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return Path(text)


def _without_nifti_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def resolve_output_plan(
    *,
    output: str | Path | None,
    output_domain: str | None,
    normalized_output: str | Path | None,
    suv_output: str | Path | None,
    metadata_json: str | Path | None,
) -> OutputPlan:
    legacy = _path_or_none(output, name="--output")
    normalized = _path_or_none(normalized_output, name="--normalized-output")
    suv = _path_or_none(suv_output, name="--suv-output")
    metadata = _path_or_none(metadata_json, name="--metadata-json")

    if legacy is not None and (normalized is not None or suv is not None):
        raise ValueError(
            "Use either legacy --output/--output-domain or shared-output flags "
            "--normalized-output and/or --suv-output, not both"
        )
    if legacy is None and normalized is None and suv is None:
        raise ValueError("Provide --output, --normalized-output, or --suv-output")
    if legacy is not None:
        domain = output_domain or "suv"
        return OutputPlan(
            legacy_output=legacy,
            legacy_domain=domain,
            normalized_output=None,
            suv_output=None,
            metadata_json=metadata,
        )
    if output_domain is not None:
        raise ValueError(
            "--output-domain is only valid with legacy --output; shared-output flags "
            "already define their domains"
        )
    if normalized is not None and suv is not None:
        if normalized.expanduser().resolve() == suv.expanduser().resolve():
            raise ValueError("--normalized-output and --suv-output must be different files")

    if metadata is None:
        anchor = normalized or suv
        assert anchor is not None
        metadata = anchor.with_name(f"{_without_nifti_suffix(anchor)}_prediction.json")

    return OutputPlan(
        legacy_output=None,
        legacy_domain=None,
        normalized_output=normalized,
        suv_output=suv,
        metadata_json=metadata,
    )


def prediction_identifier(normalized_prediction: np.ndarray) -> str:
    canonical = np.ascontiguousarray(normalized_prediction, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
