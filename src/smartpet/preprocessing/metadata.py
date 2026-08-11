from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_REQUIRED_COLUMNS = (
    "subject_id",
    "source_image_path",
    "target_image_path",
)

ACTIVITY_REQUIRED_COLUMNS = (
    "weight_kg",
    "source_net_injected_dose_mbq",
    "target_net_injected_dose_mbq",
    "source_activity_unit",
    "target_activity_unit",
    "source_decay_reference",
    "target_decay_reference",
    "source_count_scaling",
    "target_count_scaling",
    "source_count_fraction",
    "target_count_fraction",
)

COUNT_PROTOCOL_COLUMNS = (
    "source_sampling_scheme",
    "source_chunk_duration_seconds",
    "source_number_of_chunks",
    "source_total_duration_seconds",
    "target_total_duration_seconds",
    "selection_window_start_minutes",
    "selection_window_end_minutes",
)

_ACTIVITY_FACTORS = {
    "bq/ml": 1.0,
    "kbq/ml": 1_000.0,
    "mbq/ml": 1_000_000.0,
}

_DECAY_REFERENCES = {"ADMIN", "START"}
_COUNT_SCALING_MODES = {"quantitative", "count_scaled"}
_SAMPLING_SCHEMES = {
    "random_noncontiguous",
    "contiguous",
    "full_window",
    "unknown",
}
_DURATION_ABS_TOL_SECONDS = 0.5
_FRACTION_ABS_TOL = 1e-6


def normalize_activity_unit(value: str) -> str:
    unit = str(value).strip().lower().replace(" ", "")
    aliases = {
        "bq/ml": "bq/ml",
        "bqml": "bq/ml",
        "kbq/ml": "kbq/ml",
        "kbqml": "kbq/ml",
        "mbq/ml": "mbq/ml",
        "mbqml": "mbq/ml",
    }
    if unit not in aliases:
        raise ValueError(
            f"Unsupported activity unit {value!r}; use Bq/mL, kBq/mL, or MBq/mL"
        )
    return aliases[unit]


def activity_factor_to_bq_per_ml(unit: str) -> float:
    return float(_ACTIVITY_FACTORS[normalize_activity_unit(unit)])


def normalize_decay_reference(value: str) -> str:
    reference = str(value).strip().upper()
    if reference not in _DECAY_REFERENCES:
        raise ValueError(
            f"Unsupported decay reference {value!r}; use ADMIN or START"
        )
    return reference


def normalize_count_scaling(value: str) -> str:
    mode = str(value).strip().lower()
    aliases = {
        "quantitative": "quantitative",
        "calibrated": "quantitative",
        "full_scale": "quantitative",
        "count_scaled": "count_scaled",
        "proportional": "count_scaled",
        "fraction_scaled": "count_scaled",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported count scaling {value!r}; use quantitative or count_scaled"
        )
    return aliases[mode]


def normalize_sampling_scheme(value: str) -> str:
    scheme = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "random": "random_noncontiguous",
        "random_chunks": "random_noncontiguous",
        "random_non_contiguous": "random_noncontiguous",
        "random_noncontiguous": "random_noncontiguous",
        "noncontiguous": "random_noncontiguous",
        "contiguous": "contiguous",
        "continuous": "contiguous",
        "full_window": "full_window",
        "full": "full_window",
        "unknown": "unknown",
    }
    if scheme not in aliases:
        raise ValueError(
            f"Unsupported source_sampling_scheme {value!r}; use one of "
            + ", ".join(sorted(_SAMPLING_SCHEMES))
        )
    return aliases[scheme]


def _optional_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any, *, name: str, subject_id: str) -> float | None:
    text = _optional_text(value)
    if text is None:
        return None
    result = float(text)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite for {subject_id}")
    return result


def _optional_int(value: Any, *, name: str, subject_id: str) -> int | None:
    parsed = _optional_float(value, name=name, subject_id=subject_id)
    if parsed is None:
        return None
    rounded = int(round(parsed))
    if not math.isclose(parsed, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must be an integer for {subject_id}: {parsed}")
    return rounded


def _optional_datetime(value: Any, *, name: str, subject_id: str) -> datetime | None:
    text = _optional_text(value)
    if text is None:
        return None
    try:
        parsed = pd.Timestamp(text)
    except Exception as exc:
        raise ValueError(
            f"{name} must be an ISO-8601 date/time for {subject_id}: {text!r}"
        ) from exc
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed.to_pydatetime()


@dataclass(frozen=True)
class ExternalPairRecord:
    subject_id: str
    source_image_path: Path
    target_image_path: Path
    sex: str | None = None
    age_years: float | None = None
    weight_kg: float | None = None
    source_net_injected_dose_mbq: float | None = None
    target_net_injected_dose_mbq: float | None = None
    source_activity_unit: str | None = None
    target_activity_unit: str | None = None
    source_decay_reference: str | None = None
    target_decay_reference: str | None = None
    source_injection_datetime: datetime | None = None
    target_injection_datetime: datetime | None = None
    source_acquisition_datetime: datetime | None = None
    target_acquisition_datetime: datetime | None = None
    radionuclide_half_life_seconds: float | None = None
    source_count_scaling: str | None = None
    target_count_scaling: str | None = None
    source_count_fraction: float | None = None
    target_count_fraction: float | None = None
    source_sampling_scheme: str | None = None
    source_chunk_duration_seconds: float | None = None
    source_number_of_chunks: int | None = None
    source_total_duration_seconds: float | None = None
    target_total_duration_seconds: float | None = None
    selection_window_start_minutes: float | None = None
    selection_window_end_minutes: float | None = None
    derived_source_count_fraction: float | None = None
    count_protocol_status: str | None = None


def _resolve_path(value: object, *, data_root: Path, csv_parent: Path) -> Path:
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        return path
    root = data_root if str(data_root) else csv_parent
    return (root / path).resolve()


def _get(row: Any, name: str) -> Any:
    return getattr(row, name, None)


def _validate_count_protocol(
    row: Any,
    *,
    subject_id: str,
    source_fraction: float,
    target_fraction: float,
) -> dict[str, Any]:
    values_present = any(
        _optional_text(_get(row, name)) is not None
        for name in COUNT_PROTOCOL_COLUMNS
    )
    if not values_present:
        return {
            "source_sampling_scheme": None,
            "source_chunk_duration_seconds": None,
            "source_number_of_chunks": None,
            "source_total_duration_seconds": None,
            "target_total_duration_seconds": None,
            "selection_window_start_minutes": None,
            "selection_window_end_minutes": None,
            "derived_source_count_fraction": None,
            "count_protocol_status": "not_provided",
        }

    scheme_text = _optional_text(_get(row, "source_sampling_scheme"))
    scheme = normalize_sampling_scheme(scheme_text) if scheme_text is not None else None
    chunk_seconds = _optional_float(
        _get(row, "source_chunk_duration_seconds"),
        name="source_chunk_duration_seconds",
        subject_id=subject_id,
    )
    n_chunks = _optional_int(
        _get(row, "source_number_of_chunks"),
        name="source_number_of_chunks",
        subject_id=subject_id,
    )
    source_total = _optional_float(
        _get(row, "source_total_duration_seconds"),
        name="source_total_duration_seconds",
        subject_id=subject_id,
    )
    target_total = _optional_float(
        _get(row, "target_total_duration_seconds"),
        name="target_total_duration_seconds",
        subject_id=subject_id,
    )
    window_start = _optional_float(
        _get(row, "selection_window_start_minutes"),
        name="selection_window_start_minutes",
        subject_id=subject_id,
    )
    window_end = _optional_float(
        _get(row, "selection_window_end_minutes"),
        name="selection_window_end_minutes",
        subject_id=subject_id,
    )

    if source_total is None or target_total is None:
        raise ValueError(
            "Count-decimation protocol metadata requires source_total_duration_seconds "
            f"and target_total_duration_seconds for {subject_id}"
        )
    if source_total <= 0 or target_total <= 0:
        raise ValueError(f"Count-decimation durations must be positive for {subject_id}")
    if source_total > target_total + _DURATION_ABS_TOL_SECONDS:
        raise ValueError(
            "source_total_duration_seconds cannot exceed "
            f"target_total_duration_seconds for {subject_id}"
        )
    if not math.isclose(target_fraction, 1.0, rel_tol=0.0, abs_tol=_FRACTION_ABS_TOL):
        raise ValueError(
            "target_count_fraction must be 1.0 when target_total_duration_seconds "
            f"defines the full reference duration for {subject_id}"
        )

    if (chunk_seconds is None) != (n_chunks is None):
        raise ValueError(
            "source_chunk_duration_seconds and source_number_of_chunks must be provided "
            f"together for {subject_id}"
        )
    if scheme == "random_noncontiguous" and (chunk_seconds is None or n_chunks is None):
        raise ValueError(
            "source_sampling_scheme=random_noncontiguous requires source_chunk_duration_seconds "
            f"and source_number_of_chunks for {subject_id}"
        )
    if chunk_seconds is not None:
        if chunk_seconds <= 0 or n_chunks is None or n_chunks <= 0:
            raise ValueError(
                "Chunk duration and number of chunks must be positive "
                f"for {subject_id}"
            )
        chunk_total = chunk_seconds * n_chunks
        if not math.isclose(
            chunk_total,
            source_total,
            rel_tol=0.0,
            abs_tol=_DURATION_ABS_TOL_SECONDS,
        ):
            raise ValueError(
                "source_chunk_duration_seconds × source_number_of_chunks must equal "
                f"source_total_duration_seconds for {subject_id}: "
                f"{chunk_seconds} × {n_chunks} = {chunk_total}, stated {source_total}"
            )

    if (window_start is None) != (window_end is None):
        raise ValueError(
            "selection_window_start_minutes and selection_window_end_minutes must be "
            f"provided together for {subject_id}"
        )
    if window_start is not None and window_end is not None:
        if window_start < 0 or window_end <= window_start:
            raise ValueError(f"Invalid selection window for {subject_id}")
        window_seconds = (window_end - window_start) * 60.0
        if not math.isclose(
            window_seconds,
            target_total,
            rel_tol=0.0,
            abs_tol=_DURATION_ABS_TOL_SECONDS,
        ):
            raise ValueError(
                "The selection-window duration must equal target_total_duration_seconds "
                f"for {subject_id}: ({window_end} - {window_start}) × 60 = "
                f"{window_seconds}, stated {target_total}"
            )

    derived_fraction = source_total / target_total
    if not math.isclose(
        derived_fraction,
        source_fraction,
        rel_tol=0.0,
        abs_tol=_FRACTION_ABS_TOL,
    ):
        raise ValueError(
            "source_count_fraction must equal source_total_duration_seconds / "
            f"target_total_duration_seconds for {subject_id}: derived "
            f"{derived_fraction:.12g}, stated {source_fraction:.12g}"
        )

    return {
        "source_sampling_scheme": scheme,
        "source_chunk_duration_seconds": chunk_seconds,
        "source_number_of_chunks": n_chunks,
        "source_total_duration_seconds": source_total,
        "target_total_duration_seconds": target_total,
        "selection_window_start_minutes": window_start,
        "selection_window_end_minutes": window_end,
        "derived_source_count_fraction": derived_fraction,
        "count_protocol_status": "validated_duration_fraction",
    }


def _validate_activity_metadata(
    row: Any,
    *,
    subject_id: str,
) -> dict[str, Any]:
    weight = float(_get(row, "weight_kg"))
    source_dose = float(_get(row, "source_net_injected_dose_mbq"))
    target_dose = float(_get(row, "target_net_injected_dose_mbq"))
    values = np.asarray([weight, source_dose, target_dose], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite SUV metadata for {subject_id}")
    if weight <= 0:
        raise ValueError(f"weight_kg must be positive for {subject_id}")
    if source_dose <= 0 or target_dose <= 0:
        raise ValueError(f"Net injected doses must be positive for {subject_id}")

    source_fraction = float(_get(row, "source_count_fraction"))
    target_fraction = float(_get(row, "target_count_fraction"))
    for name, value in (
        ("source_count_fraction", source_fraction),
        ("target_count_fraction", target_fraction),
    ):
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0,1] for {subject_id}: {value}")

    source_decay = normalize_decay_reference(_get(row, "source_decay_reference"))
    target_decay = normalize_decay_reference(_get(row, "target_decay_reference"))
    source_scaling = normalize_count_scaling(_get(row, "source_count_scaling"))
    target_scaling = normalize_count_scaling(_get(row, "target_count_scaling"))

    half_life = _optional_float(
        _get(row, "radionuclide_half_life_seconds"),
        name="radionuclide_half_life_seconds",
        subject_id=subject_id,
    )
    if half_life is not None and half_life <= 0:
        raise ValueError(
            f"radionuclide_half_life_seconds must be positive for {subject_id}"
        )

    source_injection = _optional_datetime(
        _get(row, "source_injection_datetime"),
        name="source_injection_datetime",
        subject_id=subject_id,
    )
    target_injection = _optional_datetime(
        _get(row, "target_injection_datetime"),
        name="target_injection_datetime",
        subject_id=subject_id,
    )
    source_acquisition = _optional_datetime(
        _get(row, "source_acquisition_datetime"),
        name="source_acquisition_datetime",
        subject_id=subject_id,
    )
    target_acquisition = _optional_datetime(
        _get(row, "target_acquisition_datetime"),
        name="target_acquisition_datetime",
        subject_id=subject_id,
    )

    if source_decay == "START":
        if source_injection is None or source_acquisition is None or half_life is None:
            raise ValueError(
                "source_decay_reference=START requires source_injection_datetime, "
                f"source_acquisition_datetime, and radionuclide_half_life_seconds for {subject_id}"
            )
    if target_decay == "START":
        if target_injection is None or target_acquisition is None or half_life is None:
            raise ValueError(
                "target_decay_reference=START requires target_injection_datetime, "
                f"target_acquisition_datetime, and radionuclide_half_life_seconds for {subject_id}"
            )

    protocol_values = _validate_count_protocol(
        row,
        subject_id=subject_id,
        source_fraction=source_fraction,
        target_fraction=target_fraction,
    )

    return {
        "weight_kg": weight,
        "source_net_injected_dose_mbq": source_dose,
        "target_net_injected_dose_mbq": target_dose,
        "source_activity_unit": normalize_activity_unit(_get(row, "source_activity_unit")),
        "target_activity_unit": normalize_activity_unit(_get(row, "target_activity_unit")),
        "source_decay_reference": source_decay,
        "target_decay_reference": target_decay,
        "source_injection_datetime": source_injection,
        "target_injection_datetime": target_injection,
        "source_acquisition_datetime": source_acquisition,
        "target_acquisition_datetime": target_acquisition,
        "radionuclide_half_life_seconds": half_life,
        "source_count_scaling": source_scaling,
        "target_count_scaling": target_scaling,
        "source_count_fraction": source_fraction,
        "target_count_fraction": target_fraction,
        **protocol_values,
    }


def read_external_pair_metadata(
    csv_path: str | Path,
    *,
    data_root: str | Path | None = None,
    input_kind: str = "raw_activity",
    require_files: bool = True,
) -> list[ExternalPairRecord]:
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    required = list(BASE_REQUIRED_COLUMNS)
    if input_kind in {"raw_activity", "mni_activity"}:
        required.extend(ACTIVITY_REQUIRED_COLUMNS)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "External preprocessing CSV is missing required columns for "
            f"input_kind={input_kind}: " + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("External preprocessing CSV is empty")

    root = Path(data_root).expanduser().resolve() if data_root else csv_path.parent.resolve()
    records: list[ExternalPairRecord] = []
    seen: set[str] = set()
    for row in frame.itertuples(index=False):
        subject_id = str(row.subject_id).strip()
        if not subject_id:
            raise ValueError("subject_id cannot be blank")
        if subject_id in seen:
            raise ValueError(f"Duplicate subject_id in external metadata: {subject_id}")
        seen.add(subject_id)

        source_path = _resolve_path(
            row.source_image_path, data_root=root, csv_parent=csv_path.parent
        )
        target_path = _resolve_path(
            row.target_image_path, data_root=root, csv_parent=csv_path.parent
        )
        if source_path == target_path:
            raise ValueError(f"Source and target paths are identical for {subject_id}")
        if require_files:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if not target_path.is_file():
                raise FileNotFoundError(target_path)

        age = _optional_float(
            _get(row, "age_years"), name="age_years", subject_id=subject_id
        )
        if age is not None and not 0 <= age <= 120:
            raise ValueError(f"age_years outside [0,120] for {subject_id}: {age}")

        activity_values: dict[str, Any] = {}
        if input_kind in {"raw_activity", "mni_activity"}:
            activity_values = _validate_activity_metadata(row, subject_id=subject_id)

        records.append(
            ExternalPairRecord(
                subject_id=subject_id,
                source_image_path=source_path,
                target_image_path=target_path,
                sex=_optional_text(_get(row, "sex")),
                age_years=age,
                **activity_values,
            )
        )
    return records
