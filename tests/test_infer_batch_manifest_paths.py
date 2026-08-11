from pathlib import Path

import numpy as np

from smartpet.cli.infer_batch import _optional_path, _resolve_manifest_path


def test_relative_input_path_resolves_from_manifest_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()

    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    resolved = _resolve_manifest_path(
        "../data/input.nii.gz",
        manifest_dir=manifest_dir,
    )

    assert resolved == (manifest_dir / "../data/input.nii.gz").resolve()


def test_absolute_input_path_is_preserved(tmp_path: Path) -> None:
    absolute = (tmp_path / "input.nii.gz").resolve()

    resolved = _resolve_manifest_path(
        str(absolute),
        manifest_dir=tmp_path / "manifests",
    )

    assert resolved == absolute


def test_optional_output_path_resolves_from_manifest_directory(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()

    assert _optional_path("", manifest_dir=manifest_dir) is None
    assert _optional_path(np.nan, manifest_dir=manifest_dir) is None

    resolved = _optional_path(
        "../outputs/prediction_suv.nii.gz",
        manifest_dir=manifest_dir,
    )

    assert resolved == (manifest_dir / "../outputs/prediction_suv.nii.gz").resolve()
