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
