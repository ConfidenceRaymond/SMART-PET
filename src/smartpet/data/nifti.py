from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class MNIContract:
    shape: tuple[int, int, int]
    affine: np.ndarray
    zooms: tuple[float, float, float]
    affine_atol: float = 1e-4
    zoom_atol: float = 1e-4

    @classmethod
    def from_reference(cls, path: str | Path) -> MNIContract:
        image = nib.as_closest_canonical(nib.load(str(path)))
        if len(image.shape) != 3:
            raise ValueError(f"MNI reference must be 3D, got {image.shape}")
        return cls(
            shape=tuple(map(int, image.shape)),
            affine=np.asarray(image.affine, dtype=np.float64),
            zooms=tuple(map(float, image.header.get_zooms()[:3])),
        )


def load_mni_volume(
    path: str | Path,
    contract: MNIContract,
) -> tuple[nib.Nifti1Image, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    image = nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape) != 3:
        raise ValueError(f"Expected scalar 3D NIfTI, got shape {image.shape}: {path}")
    if tuple(image.shape) != contract.shape:
        raise ValueError(f"MNI shape mismatch for {path}: {image.shape} != {contract.shape}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.allclose(affine, contract.affine, atol=contract.affine_atol, rtol=0.0):
        raise ValueError(f"MNI affine mismatch for {path}")
    zooms = tuple(map(float, image.header.get_zooms()[:3]))
    if not np.allclose(zooms, contract.zooms, atol=contract.zoom_atol, rtol=0.0):
        raise ValueError(f"MNI voxel-spacing mismatch for {path}: {zooms} != {contract.zooms}")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values in {path}")
    return image, np.asarray(data, dtype=np.float32)


def save_like(data: np.ndarray, reference: nib.Nifti1Image, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), reference.affine, header)
    nib.save(image, str(path))
    return path
