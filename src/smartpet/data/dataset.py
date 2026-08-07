from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .patches import extract_patch, pad_to_patch, random_origin

REQUIRED_COLUMNS = ("subject_id", "source_path", "target_path")
PATCH_MODES = ("random", "random_foreground", "fixed_random", "center")


@dataclass(frozen=True)
class PairRecord:
    subject_id: str
    source_path: Path
    target_path: Path


def read_manifest(path: str | Path) -> list[PairRecord]:
    path = Path(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Manifest {path} missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"Manifest {path} is empty")

    manifest_dir = path.resolve().parent
    records: list[PairRecord] = []
    for line, row in enumerate(frame.itertuples(index=False), start=2):
        subject_id = str(row.subject_id).strip()
        if not subject_id:
            raise ValueError(f"{path}:{line} has an empty subject_id")

        resolved_paths: dict[str, Path] = {}
        for field in ("source_path", "target_path"):
            text = str(getattr(row, field)).strip()
            if not text:
                raise ValueError(f"{path}:{line} ({subject_id}) has an empty {field}")
            candidate = Path(text)
            if not candidate.is_absolute():
                candidate = manifest_dir / candidate
            resolved_paths[field] = candidate.resolve()

        source_path = resolved_paths["source_path"]
        target_path = resolved_paths["target_path"]
        if source_path == target_path:
            raise ValueError(
                f"{path}:{line} ({subject_id}): source and target are the same file; "
                "this would train the identity map"
            )
        records.append(PairRecord(subject_id, source_path, target_path))

    duplicates = sorted(
        subject_id
        for subject_id, count in Counter(record.subject_id for record in records).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"{path}: duplicate subject_id values: {duplicates[:10]}")
    return records


def _center_origin(
    shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(
        max(0, (int(dim) - int(patch)) // 2)
        for dim, patch in zip(shape, patch_size, strict=True)
    )


def _foreground_fraction(patch: np.ndarray, threshold: float) -> float:
    return float(np.count_nonzero(patch > float(threshold)) / patch.size)


class PairedMNIPatchDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        manifest: str | Path,
        mni_reference: str | Path,
        patch_size: tuple[int, int, int] = (128, 128, 128),
        seed: int = 2023,
        *,
        patch_mode: str = "random",
        min_foreground_fraction: float = 0.0,
        foreground_threshold: float = 0.0,
        foreground_attempts: int = 16,
    ) -> None:
        if patch_mode not in PATCH_MODES:
            raise ValueError(f"Unsupported patch_mode={patch_mode!r}; expected {PATCH_MODES}")
        if not 0.0 <= float(min_foreground_fraction) <= 1.0:
            raise ValueError("min_foreground_fraction must be in [0, 1]")
        self.records = read_manifest(manifest)
        from .nifti import MNIContract

        self.contract = MNIContract.from_reference(mni_reference)
        self.patch_size = tuple(map(int, patch_size))
        self.seed = int(seed)
        self.epoch = 0
        self.patch_mode = str(patch_mode)
        self.min_foreground_fraction = float(min_foreground_fraction)
        self.foreground_threshold = float(foreground_threshold)
        self.foreground_attempts = max(1, int(foreground_attempts))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _choose_origin(
        self,
        target: np.ndarray,
        *,
        index: int,
    ) -> tuple[tuple[int, int, int], float]:
        if self.patch_mode == "center":
            origin = _center_origin(target.shape, self.patch_size)
            patch = extract_patch(target, origin, self.patch_size)
            return origin, _foreground_fraction(patch, self.foreground_threshold)

        epoch_component = 0 if self.patch_mode == "fixed_random" else self.epoch
        rng = np.random.default_rng(
            self.seed + epoch_component * len(self.records) + int(index)
        )
        best_origin = random_origin(target.shape, self.patch_size, rng)
        best_patch = extract_patch(target, best_origin, self.patch_size)
        best_fraction = _foreground_fraction(best_patch, self.foreground_threshold)

        if self.patch_mode != "random_foreground" or self.min_foreground_fraction <= 0:
            return best_origin, best_fraction

        for _ in range(self.foreground_attempts - 1):
            if best_fraction >= self.min_foreground_fraction:
                break
            candidate_origin = random_origin(target.shape, self.patch_size, rng)
            candidate_patch = extract_patch(target, candidate_origin, self.patch_size)
            candidate_fraction = _foreground_fraction(
                candidate_patch,
                self.foreground_threshold,
            )
            if candidate_fraction > best_fraction:
                best_origin = candidate_origin
                best_fraction = candidate_fraction
        return best_origin, best_fraction

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        from .nifti import load_mni_volume

        _, source = load_mni_volume(record.source_path, self.contract)
        _, target = load_mni_volume(record.target_path, self.contract)
        tolerance = 1e-6
        source_min = float(source.min())
        target_min = float(target.min())
        if source_min < -tolerance or target_min < -tolerance:
            raise ValueError(
                "Modernized SMART-PET requires non-negative normalized PET values: "
                f"subject={record.subject_id}, source_min={source_min:.8g}, "
                f"target_min={target_min:.8g}"
            )
        # Remove only negligible interpolation/numerical undershoot inside tolerance.
        source = np.clip(source, 0.0, None).astype(np.float32, copy=False)
        target = np.clip(target, 0.0, None).astype(np.float32, copy=False)
        source, pads = pad_to_patch(source, self.patch_size)
        target, target_pads = pad_to_patch(target, self.patch_size)
        if pads != target_pads or source.shape != target.shape:
            raise ValueError(f"Source/target geometry mismatch for {record.subject_id}")
        origin, foreground_fraction = self._choose_origin(target, index=index)
        source_patch = extract_patch(source, origin, self.patch_size)
        target_patch = extract_patch(target, origin, self.patch_size)
        return {
            "subject_id": record.subject_id,
            "source": torch.from_numpy(source_patch[None]).float(),
            "target": torch.from_numpy(target_patch[None]).float(),
            "origin": torch.tensor(origin, dtype=torch.int64),
            "foreground_fraction": torch.tensor(foreground_fraction, dtype=torch.float32),
        }
