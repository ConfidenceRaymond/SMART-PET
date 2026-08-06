from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from smartpet.cli.audit_inference import audit_inference  # noqa: E402


def _save(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))


def test_audit_inference_accepts_finite_nonnegative_output(tmp_path: Path) -> None:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    reference = tmp_path / "reference.nii.gz"
    output = tmp_path / "output.nii.gz"
    _save(reference, np.zeros((8, 9, 10), dtype=np.float32), affine)
    _save(output, np.ones((8, 9, 10), dtype=np.float32), affine)
    result = audit_inference(output, mni_reference=reference)
    assert result["output_stats"]["negative_voxel_count"] == 0


def test_audit_inference_rejects_negative_output(tmp_path: Path) -> None:
    affine = np.eye(4)
    reference = tmp_path / "reference.nii.gz"
    output = tmp_path / "output.nii.gz"
    _save(reference, np.zeros((4, 4, 4), dtype=np.float32), affine)
    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[0, 0, 0] = -0.5
    _save(output, data, affine)
    with pytest.raises(RuntimeError, match="negative values"):
        audit_inference(output, mni_reference=reference)
