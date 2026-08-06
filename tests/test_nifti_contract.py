from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
from smartpet.data.nifti import MNIContract, load_mni_volume  # noqa: E402


def test_contract_accepts_match_and_rejects_shape(tmp_path: Path):
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    ref = tmp_path / "ref.nii.gz"
    good = tmp_path / "good.nii.gz"
    bad = tmp_path / "bad.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((9, 10, 11), np.float32), affine), ref)
    nib.save(nib.Nifti1Image(np.ones((9, 10, 11), np.float32), affine), good)
    nib.save(nib.Nifti1Image(np.ones((9, 10, 12), np.float32), affine), bad)
    contract = MNIContract.from_reference(ref)
    _, data = load_mni_volume(good, contract)
    assert data.shape == (9, 10, 11)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_mni_volume(bad, contract)
