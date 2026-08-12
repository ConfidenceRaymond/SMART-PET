from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from smartpet.data.nifti import MNIContract
from smartpet.preprocessing import pipeline
from smartpet.preprocessing.metadata import ExternalPairRecord
from smartpet.preprocessing.suv import uncorrected_frame_to_admin_factor


def _write_nifti(
    path: Path,
    *,
    affine: np.ndarray | None = None,
) -> None:
    if affine is None:
        affine = np.eye(4, dtype=np.float64)

    image = nib.Nifti1Image(
        np.ones((8, 9, 10), dtype=np.float32),
        affine,
    )
    nib.save(image, path)


def _write_4d_nifti(path: Path) -> None:
    image = nib.Nifti1Image(
        np.ones((8, 9, 10, 2), dtype=np.float32),
        np.eye(4, dtype=np.float64),
    )
    nib.save(image, path)


def test_raw_pair_registration_estimates_one_target_transform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.nii.gz"
    target = tmp_path / "target.nii.gz"
    reference = tmp_path / "reference.nii.gz"

    _write_nifti(source)
    _write_nifti(target)
    _write_nifti(reference)

    source_output = tmp_path / "source_mni.nii.gz"
    target_output = tmp_path / "target_mni.nii.gz"
    provenance = tmp_path / "registration.json"
    work = tmp_path / "work"

    calls: list[list[str]] = []

    def fake_which(name: str) -> str:
        return name

    def fake_run(
        command: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> None:
        assert check is True
        calls.append(command.copy())

        if command[0] == "antsRegistrationSyNQuick.sh":
            prefix = Path(command[command.index("-o") + 1])

            Path(f"{prefix}0GenericAffine.mat").write_text(
                "mock affine\n"
            )
            Path(f"{prefix}1Warp.nii.gz").write_bytes(b"mock warp")
            return

        if command[0] == "antsApplyTransforms":
            input_path = Path(command[command.index("-i") + 1])
            output_path = Path(command[command.index("-o") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return

        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(pipeline.shutil, "which", fake_which)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    actual_source, actual_target = pipeline._register_pair(
        source,
        target,
        reference=reference,
        source_destination=source_output,
        target_destination=target_output,
        provenance_path=provenance,
        work_dir=work,
        threads=2,
        transform_type="s",
        force=False,
    )

    assert actual_source == source_output
    assert actual_target == target_output
    assert source_output.is_file()
    assert target_output.is_file()

    registration_calls = [
        call
        for call in calls
        if call[0] == "antsRegistrationSyNQuick.sh"
    ]
    apply_calls = [
        call
        for call in calls
        if call[0] == "antsApplyTransforms"
    ]

    assert len(registration_calls) == 1
    assert len(apply_calls) == 2

    registration = registration_calls[0]

    assert Path(
        registration[registration.index("-m") + 1]
    ) == target

    apply_inputs = {
        Path(call[call.index("-i") + 1])
        for call in apply_calls
    }

    assert apply_inputs == {source, target}

    transform_stacks = []

    for call in apply_calls:
        transforms = [
            call[index + 1]
            for index, value in enumerate(call)
            if value == "-t"
        ]
        transform_stacks.append(transforms)

    assert transform_stacks[0] == transform_stacks[1]
    assert len(transform_stacks[0]) == 2
    assert transform_stacks[0][0].endswith("1Warp.nii.gz")
    assert transform_stacks[0][1].endswith("0GenericAffine.mat")

    contract = json.loads(provenance.read_text())

    assert contract["strategy"] == "target_estimated_shared_transform"
    assert contract["registration_driver"] == "target"
    assert contract["interpolation"] == "Linear"
    assert contract["native_pair_geometry_checked"] is True
    assert contract["transform_type_label"] == "SyN"


def test_raw_pair_registration_rejects_native_geometry_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nii.gz"
    target = tmp_path / "target.nii.gz"

    source_affine = np.eye(4, dtype=np.float64)
    target_affine = np.eye(4, dtype=np.float64)
    target_affine[0, 3] = 2.0

    _write_nifti(source, affine=source_affine)
    _write_nifti(target, affine=target_affine)

    with pytest.raises(
        ValueError,
        match="same native physical geometry",
    ):
        pipeline._assert_same_native_geometry(source, target)


def test_raw_pair_registration_refuses_unprovenanced_existing_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.nii.gz"
    target = tmp_path / "target.nii.gz"
    reference = tmp_path / "reference.nii.gz"

    _write_nifti(source)
    _write_nifti(target)
    _write_nifti(reference)

    source_output = tmp_path / "source_mni.nii.gz"
    target_output = tmp_path / "target_mni.nii.gz"

    _write_nifti(source_output)
    _write_nifti(target_output)

    with pytest.raises(
        RuntimeError,
        match="shared-transform provenance",
    ):
        pipeline._register_pair(
            source,
            target,
            reference=reference,
            source_destination=source_output,
            target_destination=target_output,
            provenance_path=tmp_path / "missing.json",
            work_dir=tmp_path / "work",
            threads=2,
            transform_type="s",
            force=False,
        )


def test_raw_pair_registration_rejects_dynamic_4d_pet(tmp_path: Path) -> None:
    source = tmp_path / "source_4d.nii.gz"
    target = tmp_path / "target_4d.nii.gz"
    _write_4d_nifti(source)
    _write_4d_nifti(target)

    with pytest.raises(ValueError, match="Dynamic 4D PET"):
        pipeline._assert_same_native_geometry(source, target)


def test_mni_activity_none_is_corrected_to_admin_before_suv(tmp_path: Path) -> None:
    shape = (8, 9, 10)
    affine = np.eye(4, dtype=np.float64)
    reference = tmp_path / "reference.nii.gz"
    source = tmp_path / "source.nii.gz"
    target = tmp_path / "target.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), reference)
    nib.save(nib.Nifti1Image(np.full(shape, 100.0, dtype=np.float32), affine), source)
    nib.save(nib.Nifti1Image(np.full(shape, 120.0, dtype=np.float32), affine), target)

    injection = datetime(2026, 1, 1, 10, 0, 0)
    source_acquisition = injection + timedelta(minutes=40)
    target_acquisition = injection + timedelta(minutes=30)
    half_life = 6586.2

    record = ExternalPairRecord(
        subject_id="s1",
        source_image_path=source,
        target_image_path=target,
        weight_kg=70.0,
        source_net_injected_dose_mbq=300.0,
        target_net_injected_dose_mbq=300.0,
        source_activity_unit="bq/ml",
        target_activity_unit="bq/ml",
        source_decay_reference="NONE",
        target_decay_reference="NONE",
        source_injection_datetime=injection,
        target_injection_datetime=injection,
        source_acquisition_datetime=source_acquisition,
        target_acquisition_datetime=target_acquisition,
        radionuclide_half_life_seconds=half_life,
        source_image_duration_seconds=120.0,
        target_image_duration_seconds=1800.0,
        source_count_scaling="quantitative",
        target_count_scaling="quantitative",
        source_count_fraction=0.1,
        target_count_fraction=1.0,
    )

    output_root = tmp_path / "prepared"
    report = pipeline._prepare_one(
        record,
        input_kind="mni_activity",
        output_root=output_root,
        reference=reference,
        contract=MNIContract.from_reference(reference),
        run_work=tmp_path / "work",
        asinh_scale=1.0,
        threads=1,
        transform_type="s",
        force=False,
    )

    source_factor = uncorrected_frame_to_admin_factor(
        injection_datetime=injection,
        acquisition_datetime=source_acquisition,
        image_duration_seconds=120.0,
        radionuclide_half_life_seconds=half_life,
    )
    target_factor = uncorrected_frame_to_admin_factor(
        injection_datetime=injection,
        acquisition_datetime=target_acquisition,
        image_duration_seconds=1800.0,
        radionuclide_half_life_seconds=half_life,
    )
    suv_scale = (70.0 * 1000.0) / (300.0 * 1_000_000.0)

    source_suv = np.asarray(
        nib.load(report["source_suv_path"]).dataobj, dtype=np.float32
    )
    target_suv = np.asarray(
        nib.load(report["target_suv_path"]).dataobj, dtype=np.float32
    )

    assert np.allclose(source_suv, 100.0 * source_factor * suv_scale)
    assert np.allclose(target_suv, 120.0 * target_factor * suv_scale)
    assert report["source_suv_denominator_mbq"] == pytest.approx(300.0)
    assert report["target_suv_denominator_mbq"] == pytest.approx(300.0)
    assert report["source_activity_decay_correction_to_admin"] == pytest.approx(
        source_factor
    )
    assert report["target_activity_decay_correction_to_admin"] == pytest.approx(
        target_factor
    )
