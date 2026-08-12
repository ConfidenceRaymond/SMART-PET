from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smartpet.preprocessing.metadata import (
    ACTIVITY_REQUIRED_COLUMNS,
    BASE_REQUIRED_COLUMNS,
    IMAGE_TIMING_COLUMNS,
    read_external_pair_metadata,
)


def _activity_row() -> dict[str, object]:
    return {
        "subject_id": "s1",
        "source_image_path": "source.nii.gz",
        "target_image_path": "target.nii.gz",
        "sex": "F",
        "age_years": 40,
        "weight_kg": 65,
        "source_net_injected_dose_mbq": 180,
        "target_net_injected_dose_mbq": 180,
        "source_activity_unit": "kBq/mL",
        "target_activity_unit": "kBq/mL",
        "source_decay_reference": "ADMIN",
        "target_decay_reference": "ADMIN",
        "source_count_scaling": "count_scaled",
        "target_count_scaling": "quantitative",
        "source_count_fraction": 0.10,
        "target_count_fraction": 1.0,
        "source_sampling_scheme": "random_noncontiguous",
        "source_chunk_duration_seconds": 30,
        "source_number_of_chunks": 4,
        "source_total_duration_seconds": 120,
        "target_total_duration_seconds": 1200,
        "selection_window_start_minutes": 40,
        "selection_window_end_minutes": 60,
        "source_injection_datetime": "2026-01-01T10:00:00",
        "target_injection_datetime": "2026-01-01T10:00:00",
        "source_acquisition_datetime": "2026-01-01T11:00:00",
        "target_acquisition_datetime": "2026-01-01T11:00:00",
        "radionuclide_half_life_seconds": 6586.2,
    }


def test_external_activity_metadata_resolves_paths_and_validates_fields(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    csv_path = tmp_path / "pairs.csv"
    pd.DataFrame([_activity_row()]).to_csv(csv_path, index=False)
    records = read_external_pair_metadata(
        csv_path,
        data_root=data_root,
        input_kind="raw_activity",
        require_files=False,
    )
    assert len(records) == 1
    record = records[0]
    assert record.source_image_path == (data_root / "source.nii.gz").resolve()
    assert record.source_activity_unit == "kbq/ml"
    assert record.source_decay_reference == "ADMIN"
    assert record.source_count_scaling == "count_scaled"
    assert record.source_count_fraction == pytest.approx(0.10)
    assert record.source_sampling_scheme == "random_noncontiguous"
    assert record.source_total_duration_seconds == pytest.approx(120)
    assert record.target_total_duration_seconds == pytest.approx(1200)
    assert record.derived_source_count_fraction == pytest.approx(0.10)
    assert record.count_protocol_status == "validated_duration_fraction"


def test_mni_suv_metadata_needs_only_paths(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "s1",
                "source_image_path": "source.nii.gz",
                "target_image_path": "target.nii.gz",
            }
        ]
    ).to_csv(path, index=False)
    records = read_external_pair_metadata(
        path, input_kind="mni_suv", require_files=False
    )
    assert len(records) == 1
    assert records[0].weight_kg is None


def test_activity_metadata_rejects_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"subject_id": "s1"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        read_external_pair_metadata(
            path, input_kind="raw_activity", require_files=False
        )


def test_start_decay_reference_requires_timing_and_half_life(tmp_path: Path) -> None:
    row = _activity_row()
    row["source_decay_reference"] = "START"
    row["source_injection_datetime"] = ""
    path = tmp_path / "bad_start.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="source_decay_reference=START requires"):
        read_external_pair_metadata(
            path, input_kind="mni_activity", require_files=False
        )


def test_none_decay_reference_requires_image_duration(tmp_path: Path) -> None:
    row = _activity_row()
    row["source_decay_reference"] = "NONE"
    row["source_image_duration_seconds"] = ""
    path = tmp_path / "bad_none.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="source_decay_reference=NONE requires"):
        read_external_pair_metadata(
            path, input_kind="mni_activity", require_files=False
        )


def test_none_decay_reference_accepts_complete_timing(tmp_path: Path) -> None:
    row = _activity_row()
    row["source_decay_reference"] = "NONE"
    row["source_image_duration_seconds"] = 120
    path = tmp_path / "none.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    record = read_external_pair_metadata(
        path, input_kind="mni_activity", require_files=False
    )[0]
    assert record.source_decay_reference == "NONE"
    assert record.source_image_duration_seconds == pytest.approx(120)


def test_external_metadata_can_be_read_from_xlsx(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "pairs.xlsx"
    pd.DataFrame([_activity_row()]).to_excel(path, index=False)
    record = read_external_pair_metadata(
        path, input_kind="mni_activity", require_files=False
    )[0]
    assert record.subject_id == "s1"
    assert record.source_activity_unit == "kbq/ml"


def test_external_metadata_prefers_bundled_template_sheet_name(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "template.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"Instructions": ["Do not parse this sheet"]}).to_excel(
            writer, sheet_name="Instructions", index=False
        )
        pd.DataFrame([_activity_row()]).to_excel(
            writer, sheet_name="Raw_Activity_Template", index=False
        )
        pd.DataFrame([{**_activity_row(), "subject_id": "example"}]).to_excel(
            writer, sheet_name="Example", index=False
        )
    record = read_external_pair_metadata(
        path, input_kind="mni_activity", require_files=False
    )[0]
    assert record.subject_id == "s1"


def test_external_metadata_sheet_can_be_selected_explicitly(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([{**_activity_row(), "subject_id": "first"}]).to_excel(
            writer, sheet_name="First", index=False
        )
        pd.DataFrame([{**_activity_row(), "subject_id": "second"}]).to_excel(
            writer, sheet_name="Second", index=False
        )
    record = read_external_pair_metadata(
        path,
        input_kind="mni_activity",
        require_files=False,
        metadata_sheet="Second",
    )[0]
    assert record.subject_id == "second"


def test_count_fraction_must_be_valid(tmp_path: Path) -> None:
    row = _activity_row()
    row["source_count_fraction"] = 0.0
    path = tmp_path / "bad_fraction.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="source_count_fraction must be in"):
        read_external_pair_metadata(
            path, input_kind="mni_activity", require_files=False
        )


def test_random_chunk_protocol_rejects_duration_mismatch(tmp_path: Path) -> None:
    row = _activity_row()
    row["source_number_of_chunks"] = 3
    path = tmp_path / "bad_chunk_total.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="must equal source_total_duration_seconds"):
        read_external_pair_metadata(
            path, input_kind="mni_activity", require_files=False
        )


def test_random_chunk_protocol_rejects_fraction_mismatch(tmp_path: Path) -> None:
    row = _activity_row()
    row["source_count_fraction"] = 0.075
    path = tmp_path / "bad_fraction_duration.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="must equal source_total_duration_seconds"):
        read_external_pair_metadata(
            path, input_kind="mni_activity", require_files=False
        )


def test_random_chunk_protocol_rejects_window_mismatch(tmp_path: Path) -> None:
    row = _activity_row()
    row["selection_window_end_minutes"] = 59
    path = tmp_path / "bad_window.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="selection-window duration"):
        read_external_pair_metadata(
            path, input_kind="mni_activity", require_files=False
        )


def test_activity_metadata_allows_missing_optional_duration_protocol(tmp_path: Path) -> None:
    row = _activity_row()
    for key in (
        "source_sampling_scheme",
        "source_chunk_duration_seconds",
        "source_number_of_chunks",
        "source_total_duration_seconds",
        "target_total_duration_seconds",
        "selection_window_start_minutes",
        "selection_window_end_minutes",
    ):
        row.pop(key)
    path = tmp_path / "no_duration_protocol.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    record = read_external_pair_metadata(
        path, input_kind="mni_activity", require_files=False
    )[0]
    assert record.derived_source_count_fraction is None
    assert record.count_protocol_status == "not_provided"


def test_bundled_udunna_random_chunk_example_is_valid() -> None:
    example = (
        Path(__file__).resolve().parents[1]
        / "manifests"
        / "examples"
        / "udunna_random_30s_chunks.csv"
    )
    record = read_external_pair_metadata(
        example,
        input_kind="mni_activity",
        require_files=False,
    )[0]
    assert record.source_net_injected_dose_mbq == pytest.approx(
        record.target_net_injected_dose_mbq
    )
    assert record.source_total_duration_seconds == pytest.approx(120)
    assert record.target_total_duration_seconds == pytest.approx(1200)
    assert record.derived_source_count_fraction == pytest.approx(0.10)


def test_bundled_excel_template_has_safe_data_sheets() -> None:
    pytest.importorskip("openpyxl")
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "external_activity_metadata_template.xlsx"
    )
    workbook = pd.ExcelFile(path, engine="openpyxl")
    assert workbook.sheet_names == [
        "Instructions",
        "Raw_Activity_Template",
        "Example",
    ]

    template = pd.read_excel(
        workbook, sheet_name="Raw_Activity_Template", engine="openpyxl"
    )
    assert template.empty
    assert set(BASE_REQUIRED_COLUMNS).issubset(template.columns)
    assert set(ACTIVITY_REQUIRED_COLUMNS).issubset(template.columns)
    assert set(IMAGE_TIMING_COLUMNS).issubset(template.columns)

    example = pd.read_excel(workbook, sheet_name="Example", engine="openpyxl")
    assert len(example) == 1
    assert example.iloc[0]["subject_id"] == "example-001"
