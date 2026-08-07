from pathlib import Path

import pandas as pd
import pytest

from smartpet.data.dataset import read_manifest


def test_manifest_requires_neutral_pair_columns(tmp_path: Path):
    path = tmp_path / "pairs.csv"
    pd.DataFrame(
        {"subject_id": ["a"], "source_path": ["x"], "target_path": ["y"]}
    ).to_csv(path, index=False)
    assert len(read_manifest(path)) == 1
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"dose": [10]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        read_manifest(bad)


def test_manifest_preserves_zero_padded_ids_and_resolves_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    path = manifest_dir / "pairs.csv"
    path.write_text(
        "subject_id,source_path,target_path\n"
        "007,data/source_007.nii.gz,data/target_007.nii.gz\n"
        "0007,data/source_0007.nii.gz,data/target_0007.nii.gz\n",
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    records = read_manifest(path)

    assert [record.subject_id for record in records] == ["007", "0007"]
    assert records[0].source_path == (manifest_dir / "data/source_007.nii.gz").resolve()
    assert records[1].target_path == (manifest_dir / "data/target_0007.nii.gz").resolve()


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("id,,target.nii.gz", "empty source_path"),
        ("id,   ,target.nii.gz", "empty source_path"),
        ("id,source.nii.gz,source.nii.gz", "source and target are the same file"),
    ],
)
def test_manifest_rejects_unsafe_pair_rows(tmp_path: Path, row: str, message: str) -> None:
    path = tmp_path / "pairs.csv"
    path.write_text(
        "subject_id,source_path,target_path\n" + row + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        read_manifest(path)
