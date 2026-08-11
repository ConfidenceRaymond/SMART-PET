from __future__ import annotations

import nibabel as nib
import numpy as np

from smartpet.data.nifti import MNIContract, load_mni_volume


def test_las_reference_accepts_equivalent_canonical_ras_volume(
    tmp_path,
) -> None:
    shape = (8, 9, 10)

    las_affine = np.array(
        [
            [-1.0, 0.0, 0.0, 7.0],
            [0.0, 1.0, 0.0, -4.0],
            [0.0, 0.0, 1.0, -5.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    data = np.arange(
        np.prod(shape),
        dtype=np.float32,
    ).reshape(shape)

    ref_path = tmp_path / "reference_las.nii.gz"
    input_path = tmp_path / "input_las.nii.gz"
    ras_path = tmp_path / "prediction_ras.nii.gz"

    nib.save(
        nib.Nifti1Image(
            np.zeros(shape, dtype=np.float32),
            las_affine,
        ),
        ref_path,
    )

    nib.save(
        nib.Nifti1Image(
            data,
            las_affine,
        ),
        input_path,
    )

    canonical = nib.as_closest_canonical(
        nib.load(str(input_path))
    )

    nib.save(canonical, ras_path)

    assert nib.aff2axcodes(
        nib.load(str(ref_path)).affine
    ) == ("L", "A", "S")

    assert nib.aff2axcodes(
        nib.load(str(ras_path)).affine
    ) == ("R", "A", "S")

    contract = MNIContract.from_reference(ref_path)

    image, loaded = load_mni_volume(
        ras_path,
        contract,
    )

    assert image.shape == shape
    assert nib.aff2axcodes(image.affine) == ("R", "A", "S")
    assert np.isfinite(loaded).all()
