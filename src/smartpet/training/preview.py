from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from smartpet.data.dataset import PairRecord, read_manifest
from smartpet.data.normalization import asinh_denormalize
from smartpet.inference.sliding_window import predict_volume
from smartpet.metrics import image_quality_metrics, legacy_vgg19_feature_l1
from smartpet.training.precision import PrecisionPolicy

PREVIEW_SELECTIONS = ("first", "fixed_random", "random_each_epoch")


def select_preview_record(
    records: list[PairRecord],
    *,
    subject_id: str | None,
    selection: str,
    seed: int,
    epoch: int,
) -> PairRecord:
    if not records:
        raise ValueError("Preview manifest is empty")
    if subject_id:
        matches = [record for record in records if record.subject_id == subject_id]
        if not matches:
            raise ValueError(f"Preview subject_id not found in validation manifest: {subject_id}")
        return matches[0]
    if selection not in PREVIEW_SELECTIONS:
        raise ValueError(
            f"Unsupported preview selection {selection!r}; "
            f"expected {PREVIEW_SELECTIONS}"
        )
    if selection == "first":
        return records[0]
    epoch_offset = int(epoch) if selection == "random_each_epoch" else 0
    rng = random.Random(int(seed) + 1_000_003 * epoch_offset)
    return records[rng.randrange(len(records))]


def _prediction_id(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _metric_pair(
    volume: np.ndarray,
    target: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, float]:
    pred = torch.from_numpy(np.asarray(volume, dtype=np.float32))[None, None].to(device)
    true = torch.from_numpy(np.asarray(target, dtype=np.float32))[None, None].to(device)
    with torch.no_grad():
        return image_quality_metrics(pred, true)


def _slice_indices(target: np.ndarray) -> tuple[int, int, int]:
    positive = np.clip(target, 0.0, None)
    sagittal = int(np.argmax(positive.sum(axis=(1, 2))))
    coronal = int(np.argmax(positive.sum(axis=(0, 2))))
    axial = int(np.argmax(positive.sum(axis=(0, 1))))
    return sagittal, coronal, axial


def _plane(volume: np.ndarray, orientation: str, index: int) -> np.ndarray:
    if orientation == "sagittal":
        image = volume[index, :, :]
    elif orientation == "coronal":
        image = volume[:, index, :]
    elif orientation == "axial":
        image = volume[:, :, index]
    else:  # pragma: no cover
        raise ValueError(orientation)
    return np.rot90(image)


def save_preview_panel(
    *,
    lowdose_suv: np.ndarray,
    prediction_suv: np.ndarray,
    target_suv: np.ndarray,
    output_path: str | Path,
    subject_id: str,
    epoch_label: str,
    metrics: dict[str, Any],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices = _slice_indices(target_suv)
    positive = target_suv[target_suv > 0]
    vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    vmax = max(vmax, 1e-6)
    error = np.abs(prediction_suv - target_suv)
    error_positive = error[error > 0]
    error_vmax = float(np.percentile(error_positive, 99.0)) if error_positive.size else 1.0
    error_vmax = max(error_vmax, 1e-6)

    orientations = (
        ("Sagittal", "sagittal", indices[0]),
        ("Coronal", "coronal", indices[1]),
        ("Axial", "axial", indices[2]),
    )
    columns = (
        ("Low-dose", lowdose_suv, vmax),
        ("SMART-PET", prediction_suv, vmax),
        ("Standard-dose", target_suv, vmax),
        ("Absolute error", error, error_vmax),
    )
    figure, axes = plt.subplots(3, 4, figsize=(15, 10.5), constrained_layout=True)
    pet_mappable = None
    error_mappable = None
    for row, (orientation_title, orientation, index) in enumerate(orientations):
        for column, (title, volume, upper) in enumerate(columns):
            axis = axes[row, column]
            cmap = "magma" if column < 3 else "viridis"
            mappable = axis.imshow(
                _plane(volume, orientation, index),
                cmap=cmap,
                vmin=0.0,
                vmax=upper,
                interpolation="nearest",
            )
            if column < 3:
                pet_mappable = mappable
            else:
                error_mappable = mappable
            axis.set_axis_off()
            if row == 0:
                axis.set_title(title, fontsize=12, fontweight="bold")
            if column == 0:
                axis.text(
                    -0.04,
                    0.5,
                    f"{orientation_title}\nindex {index}",
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                )
    if pet_mappable is not None:
        figure.colorbar(
            pet_mappable,
            ax=axes[:, :3],
            location="right",
            shrink=0.82,
            pad=0.015,
            label="SUV",
        )
    if error_mappable is not None:
        figure.colorbar(
            error_mappable,
            ax=axes[:, 3],
            location="right",
            shrink=0.82,
            pad=0.015,
            label="Absolute SUV error",
        )
    input_metrics = metrics["source_vs_target"]
    prediction_metrics = metrics["prediction_vs_target"]
    figure.suptitle(
        f"SMART-PET whole-volume validation — {subject_id} — {epoch_label}\n"
        f"PSNR {input_metrics['psnr_db']:.3f}→{prediction_metrics['psnr_db']:.3f} dB | "
        f"SSIM {input_metrics['ssim']:.4f}→{prediction_metrics['ssim']:.4f} | "
        f"NRMSE {input_metrics['nrmse']:.4f}→{prediction_metrics['nrmse']:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_epoch_preview(
    *,
    model: nn.Module,
    val_manifest: str | Path,
    mni_reference: str | Path,
    out_dir: str | Path,
    device: torch.device,
    precision: PrecisionPolicy,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    asinh_scale: float,
    epoch: int,
    global_step: int,
    subject_id: str | None,
    selection: str,
    seed: int,
    save_nifti: bool,
    vgg19_weights: str | None = None,
) -> dict[str, Any]:
    from smartpet.data.nifti import MNIContract, load_mni_volume, save_like

    records = read_manifest(val_manifest)
    record = select_preview_record(
        records,
        subject_id=subject_id,
        selection=selection,
        seed=seed,
        epoch=epoch,
    )
    contract = MNIContract.from_reference(mni_reference)
    source_image, source_normalized = load_mni_volume(record.source_path, contract)
    _, target_normalized = load_mni_volume(record.target_path, contract)
    prediction_normalized = predict_volume(
        model,
        source_normalized,
        device=device,
        patch_size=patch_size,
        stride=stride,
        amp=precision.autocast_enabled,
        amp_dtype=precision.autocast_dtype,
    )
    if not np.isfinite(prediction_normalized).all():
        raise RuntimeError("Preview prediction contains non-finite values")
    negative_count = int(np.count_nonzero(prediction_normalized < 0))
    if negative_count:
        raise RuntimeError(
            "Modernized positive-output generator produced negative normalized voxels: "
            f"{negative_count}"
        )

    source_suv = asinh_denormalize(source_normalized, asinh_scale)
    target_suv = asinh_denormalize(target_normalized, asinh_scale)
    prediction_suv = asinh_denormalize(prediction_normalized, asinh_scale)
    metrics: dict[str, Any] = {
        "source_vs_target": _metric_pair(source_suv, target_suv, device=device),
        "prediction_vs_target": _metric_pair(prediction_suv, target_suv, device=device),
    }
    if vgg19_weights:
        metrics["historical_fid_proxy_vgg19_l1"] = legacy_vgg19_feature_l1(
            prediction_suv,
            target_suv,
            weights_path=vgg19_weights,
            device=device,
        )
        metrics["historical_fid_proxy_definition"] = (
            "VGG19 axial-slice feature L1 from the legacy repository; not true Frechet distance"
        )
    else:
        metrics["historical_fid_proxy_vgg19_l1"] = None
        metrics["historical_fid_proxy_status"] = "not_computed_no_explicit_vgg19_weights"

    preview_root = Path(out_dir) / "previews" / record.subject_id
    stage_name = (
        f"initial_step_{int(global_step):08d}"
        if int(epoch) < 0
        else f"epoch_{int(epoch):04d}_step_{int(global_step):08d}"
    )
    epoch_label = "initialization" if int(epoch) < 0 else f"epoch {epoch}, step {global_step}"
    epoch_dir = preview_root / stage_name
    epoch_dir.mkdir(parents=True, exist_ok=True)
    if save_nifti:
        shared_dir = preview_root / "reference"
        source_path = shared_dir / "lowdose_normalized.nii.gz"
        target_path = shared_dir / "standarddose_normalized.nii.gz"
        source_suv_path = shared_dir / "lowdose_suv.nii.gz"
        target_suv_path = shared_dir / "standarddose_suv.nii.gz"
        if not source_path.exists():
            save_like(source_normalized, source_image, source_path)
        if not target_path.exists():
            save_like(target_normalized, source_image, target_path)
        if not source_suv_path.exists():
            save_like(source_suv, source_image, source_suv_path)
        if not target_suv_path.exists():
            save_like(target_suv, source_image, target_suv_path)
        save_like(prediction_normalized, source_image, epoch_dir / "prediction_normalized.nii.gz")
        save_like(prediction_suv, source_image, epoch_dir / "prediction_suv.nii.gz")

    payload: dict[str, Any] = {
        "format_version": 1,
        "subject_id": record.subject_id,
        "source_path": str(record.source_path),
        "target_path": str(record.target_path),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "selection": selection,
        "prediction_id": _prediction_id(prediction_normalized),
        "negative_normalized_voxels": negative_count,
        "patch_size": list(map(int, patch_size)),
        "stride": list(map(int, stride)),
        "precision": precision.resolved,
        "metrics_suv": metrics,
    }
    metrics_path = epoch_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    panel_path = save_preview_panel(
        lowdose_suv=source_suv,
        prediction_suv=prediction_suv,
        target_suv=target_suv,
        output_path=epoch_dir / "lowdose_prediction_standarddose.png",
        subject_id=record.subject_id,
        epoch_label=epoch_label,
        metrics=metrics,
    )
    payload["panel_path"] = str(panel_path)
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    summary_path = Path(out_dir) / "preview_metrics.csv"
    row = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "subject_id": record.subject_id,
        **{f"input_{key}": value for key, value in metrics["source_vs_target"].items()},
        **{f"prediction_{key}": value for key, value in metrics["prediction_vs_target"].items()},
        "historical_fid_proxy_vgg19_l1": metrics["historical_fid_proxy_vgg19_l1"],
        "panel_path": str(panel_path),
    }
    exists = summary_path.exists()
    fieldnames = list(row)
    if exists:
        with summary_path.open(newline="") as handle:
            existing = csv.DictReader(handle).fieldnames
        if existing != fieldnames:
            raise RuntimeError(
                f"Preview CSV schema mismatch: existing={existing}, new={fieldnames}"
            )
    with summary_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return payload
