from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np


def _triplet(values: Sequence[int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"Expected three values, got {values}")
    out = tuple(int(v) for v in values)
    if any(v <= 0 for v in out):
        raise ValueError(f"Values must be positive, got {out}")
    return out


def pad_to_patch(
    volume: np.ndarray,
    patch_size: Sequence[int],
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    patch = _triplet(patch_size)
    pads = []
    for size, target in zip(volume.shape, patch, strict=True):
        missing = max(0, target - int(size))
        before = missing // 2
        pads.append((before, missing - before))
    return np.pad(volume, pads, mode="constant"), tuple(pads)


def unpad(volume: np.ndarray, pads: tuple[tuple[int, int], ...]) -> np.ndarray:
    slices = tuple(
        slice(lo, volume.shape[i] - hi if hi else None)
        for i, (lo, hi) in enumerate(pads)
    )
    return volume[slices]


def grid_starts(
    shape: Sequence[int],
    patch_size: Sequence[int],
    stride: Sequence[int],
) -> list[tuple[int, int, int]]:
    shape3, patch, stride3 = _triplet(shape), _triplet(patch_size), _triplet(stride)
    axes: list[list[int]] = []
    for size, width, step in zip(shape3, patch, stride3, strict=True):
        if width > size:
            raise ValueError(f"Patch {patch} exceeds padded shape {shape3}")
        starts = list(range(0, size - width + 1, step))
        if starts[-1] != size - width:
            starts.append(size - width)
        axes.append(starts)
    return list(product(*axes))


def extract_patch(
    volume: np.ndarray,
    origin: Sequence[int],
    patch_size: Sequence[int],
) -> np.ndarray:
    origin3, patch = _triplet([int(v) + 1 for v in origin]), _triplet(patch_size)
    origin0 = tuple(v - 1 for v in origin3)
    slices = tuple(
        slice(o, o + p) for o, p in zip(origin0, patch, strict=True)
    )
    result = volume[slices]
    if result.shape != patch:
        raise ValueError(f"Invalid patch shape {result.shape}; expected {patch}")
    return result


def random_origin(
    shape: Sequence[int],
    patch_size: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    shape3, patch = _triplet(shape), _triplet(patch_size)
    return tuple(
        int(rng.integers(0, size - width + 1))
        for size, width in zip(shape3, patch, strict=True)
    )


def blend_window(patch_size: Sequence[int], floor: float = 1e-3) -> np.ndarray:
    patch = _triplet(patch_size)
    axes = []
    for size in patch:
        if size == 1:
            axes.append(np.ones(1, dtype=np.float32))
        else:
            axes.append(np.maximum(np.hanning(size).astype(np.float32), np.float32(floor)))
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]


class PatchAccumulator:
    def __init__(self, shape: Sequence[int], patch_size: Sequence[int]) -> None:
        self.shape = _triplet(shape)
        self.patch_size = _triplet(patch_size)
        self.sum = np.zeros(self.shape, dtype=np.float32)
        self.weight = np.zeros(self.shape, dtype=np.float32)
        self.window = blend_window(self.patch_size)

    def add(self, patch: np.ndarray, origin: Sequence[int]) -> None:
        origin3 = tuple(int(v) for v in origin)
        slices = tuple(
            slice(o, o + p)
            for o, p in zip(origin3, self.patch_size, strict=True)
        )
        if patch.shape != self.patch_size:
            raise ValueError(f"Patch shape {patch.shape} != {self.patch_size}")
        self.sum[slices] += patch.astype(np.float32) * self.window
        self.weight[slices] += self.window

    def finalize(self) -> np.ndarray:
        if np.any(self.weight <= 0):
            raise RuntimeError("Patch grid did not cover the full volume")
        return (self.sum / self.weight).astype(np.float32)
