from __future__ import annotations

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from smartpet.preprocessing import pipeline


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
