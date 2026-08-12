from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from smartpet.data.nifti import MNIContract, load_mni_volume, save_like
from smartpet.data.normalization import asinh_normalize

from .metadata import ExternalPairRecord, read_external_pair_metadata
from .suv import (
    effective_suv_denominator_mbq,
    suv_from_activity_concentration,
    uncorrected_frame_to_admin_factor,
)

INPUT_KINDS = ("raw_activity", "mni_activity", "mni_suv", "mni_suv_normalized")



def _copy_nifti(source: Path, destination: Path, *, force: bool) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination
    shutil.copy2(source, destination)
    return destination


def _assert_same_native_geometry(source: Path, target: Path) -> None:
    """Require a paired PET source/target to occupy the same native grid."""

    source_image = nib.load(str(source))
    target_image = nib.load(str(target))

    for label, image, path in (
        ("source", source_image, source),
        ("target", target_image, target),
    ):
        if len(image.shape) != 3:
            raise ValueError(
                "SMART-PET raw_activity preprocessing requires scalar 3D PET. "
                f"The {label} image is {image.shape}: {path}. Dynamic 4D PET must be "
                "combined into a documented static 3D image before preprocessing."
            )

    if source_image.shape != target_image.shape:
        raise ValueError(
            "Raw paired PET images must have identical native shapes before "
            "shared MNI registration: "
            f"source={source_image.shape}, target={target_image.shape}"
        )

    if not np.allclose(
        source_image.affine,
        target_image.affine,
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Raw paired PET images must share the same native physical "
            "geometry before shared MNI registration. Their NIfTI affines "
            "differ."
        )


def _forward_transform_paths(prefix: Path) -> list[Path]:
    """Return ANTs forward transforms in antsApplyTransforms order."""

    warp = Path(f"{prefix}1Warp.nii.gz")
    affine = Path(f"{prefix}0GenericAffine.mat")

    transforms: list[Path] = []

    if warp.is_file():
        transforms.append(warp)

    if affine.is_file():
        transforms.append(affine)

    if not transforms:
        raise RuntimeError(
            "ANTs registration completed but no forward transform was found "
            f"for prefix {prefix}"
        )

    return transforms


def _apply_transforms(
    source: Path,
    *,
    reference: Path,
    destination: Path,
    transforms: list[Path],
    threads: int,
) -> Path:
    executable = shutil.which("antsApplyTransforms")

    if executable is None:
        raise RuntimeError(
            "antsApplyTransforms was not found. Load ANTs or use an MNI "
            "input kind."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        executable,
        "-d",
        "3",
        "-i",
        str(source),
        "-r",
        str(reference),
        "-o",
        str(destination),
        "-n",
        "Linear",
    ]

    for transform in transforms:
        command.extend(["-t", str(transform)])

    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(int(threads))
    env["OMP_NUM_THREADS"] = str(int(threads))

    subprocess.run(command, check=True, env=env)

    if not destination.is_file():
        raise RuntimeError(
            "antsApplyTransforms completed but output is missing: "
            f"{destination}"
        )

    return destination


def _register_pair(
    source: Path,
    target: Path,
    *,
    reference: Path,
    source_destination: Path,
    target_destination: Path,
    provenance_path: Path,
    work_dir: Path,
    threads: int,
    transform_type: str,
    force: bool,
) -> tuple[Path, Path]:
    """Estimate target-to-MNI once and apply it identically to the PET pair."""

    source_exists = source_destination.exists()
    target_exists = target_destination.exists()

    if not force and source_exists and target_exists:
        if provenance_path.is_file():
            try:
                provenance = json.loads(provenance_path.read_text())
            except (OSError, json.JSONDecodeError):
                provenance = {}

            if (
                provenance.get("strategy")
                == "target_estimated_shared_transform"
            ):
                return source_destination, target_destination

        raise RuntimeError(
            "Existing raw-activity MNI outputs do not contain the current "
            "shared-transform provenance contract. Rerun with --force rather "
            "than silently reusing outputs that may have been independently "
            "registered."
        )

    if not force and (source_exists != target_exists):
        raise RuntimeError(
            "Only one member of the paired MNI output exists. Rerun with "
            "--force so source and target are regenerated together with one "
            "shared transform."
        )

    _assert_same_native_geometry(source, target)

    registration = shutil.which("antsRegistrationSyNQuick.sh")

    if registration is None:
        raise RuntimeError(
            "antsRegistrationSyNQuick.sh was not found. Load ANTs or use an "
            "MNI input kind."
        )

    if shutil.which("antsApplyTransforms") is None:
        raise RuntimeError(
            "antsApplyTransforms was not found. Load ANTs or use an MNI "
            "input kind."
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    prefix = work_dir / "ants_"

    # The full-dose target is the registration driver because it normally has
    # the highest SNR. The low-dose source never estimates a separate warp.
    command = [
        registration,
        "-d",
        "3",
        "-f",
        str(reference),
        "-m",
        str(target),
        "-o",
        str(prefix),
        "-t",
        str(transform_type),
        "-n",
        str(int(threads)),
    ]

    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(int(threads))
    env["OMP_NUM_THREADS"] = str(int(threads))

    subprocess.run(command, check=True, env=env)

    transforms = _forward_transform_paths(prefix)

    _apply_transforms(
        target,
        reference=reference,
        destination=target_destination,
        transforms=transforms,
        threads=threads,
    )

    _apply_transforms(
        source,
        reference=reference,
        destination=source_destination,
        transforms=transforms,
        threads=threads,
    )

    provenance_path.parent.mkdir(parents=True, exist_ok=True)

    provenance = {
        "schema_version": 1,
        "strategy": "target_estimated_shared_transform",
        "registration_driver": "target",
        "reference": str(reference),
        "source_input": str(source),
        "target_input": str(target),
        "source_output": str(source_destination),
        "target_output": str(target_destination),
        "transform_type": str(transform_type),
        "transform_type_label": {
            "s": "SyN",
            "r": "Rigid",
            "a": "Affine",
        }.get(str(transform_type), str(transform_type)),
        "interpolation": "Linear",
        "native_pair_geometry_checked": True,
        "forward_transform_files": [
            transform.name for transform in transforms
        ],
    }

    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    return source_destination, target_destination


def _paths(output_root: Path, subject_id: str) -> dict[str, Path]:
    return {
        "source_mni": output_root / "mni" / "source" / f"{subject_id}_source_mni.nii.gz",
        "target_mni": output_root / "mni" / "target" / f"{subject_id}_target_mni.nii.gz",
        "source_suv": output_root / "suv" / "source" / f"{subject_id}_source_mni_suv.nii.gz",
        "target_suv": output_root / "suv" / "target" / f"{subject_id}_target_mni_suv.nii.gz",
        "source_norm": output_root / "normalized" / "source" / f"{subject_id}_source_norm.nii.gz",
        "target_norm": output_root / "normalized" / "target" / f"{subject_id}_target_norm.nii.gz",
        "registration_json": (
            output_root
            / "mni"
            / "registration"
            / f"{subject_id}_shared_transform.json"
        ),
    }


def _prepare_one(
    record: ExternalPairRecord,
    *,
    input_kind: str,
    output_root: Path,
    reference: Path,
    contract: MNIContract,
    run_work: Path,
    asinh_scale: float,
    threads: int,
    transform_type: str,
    force: bool,
) -> dict[str, object]:
    paths = _paths(output_root, record.subject_id)

    registration_qc: dict[str, object] = {}

    if input_kind == "raw_activity":
        source_mni, target_mni = _register_pair(
            record.source_image_path,
            record.target_image_path,
            reference=reference,
            source_destination=paths["source_mni"],
            target_destination=paths["target_mni"],
            provenance_path=paths["registration_json"],
            work_dir=run_work / record.subject_id / "pair_registration",
            threads=threads,
            transform_type=transform_type,
            force=force,
        )
        registration_qc = {
            "registration_strategy": "target_estimated_shared_transform",
            "registration_driver": "target",
            "registration_transform_type": str(transform_type),
            "registration_transform_label": {
                "s": "SyN",
                "r": "Rigid",
                "a": "Affine",
            }.get(str(transform_type), str(transform_type)),
            "registration_provenance_path": str(
                paths["registration_json"]
            ),
        }
    else:
        source_mni = _copy_nifti(
            record.source_image_path,
            paths["source_mni"],
            force=force,
        )
        target_mni = _copy_nifti(
            record.target_image_path,
            paths["target_mni"],
            force=force,
        )

    source_image, source_data = load_mni_volume(source_mni, contract)
    target_image, target_data = load_mni_volume(target_mni, contract)

    dose_qc: dict[str, float] = {}
    if input_kind in {"raw_activity", "mni_activity"}:
        if record.weight_kg is None:
            raise ValueError(f"Missing weight_kg for activity input: {record.subject_id}")

        source_decay_reference = str(record.source_decay_reference)
        target_decay_reference = str(record.target_decay_reference)

        source_decay_factor = 1.0
        target_decay_factor = 1.0
        if source_decay_reference == "NONE":
            assert record.source_injection_datetime is not None
            assert record.source_acquisition_datetime is not None
            assert record.source_image_duration_seconds is not None
            assert record.radionuclide_half_life_seconds is not None
            source_decay_factor = uncorrected_frame_to_admin_factor(
                injection_datetime=record.source_injection_datetime,
                acquisition_datetime=record.source_acquisition_datetime,
                image_duration_seconds=record.source_image_duration_seconds,
                radionuclide_half_life_seconds=record.radionuclide_half_life_seconds,
            )
            source_data = source_data * np.float32(source_decay_factor)
            source_decay_reference = "ADMIN"

        if target_decay_reference == "NONE":
            assert record.target_injection_datetime is not None
            assert record.target_acquisition_datetime is not None
            assert record.target_image_duration_seconds is not None
            assert record.radionuclide_half_life_seconds is not None
            target_decay_factor = uncorrected_frame_to_admin_factor(
                injection_datetime=record.target_injection_datetime,
                acquisition_datetime=record.target_acquisition_datetime,
                image_duration_seconds=record.target_image_duration_seconds,
                radionuclide_half_life_seconds=record.radionuclide_half_life_seconds,
            )
            target_data = target_data * np.float32(target_decay_factor)
            target_decay_reference = "ADMIN"

        source_reference_dose, source_denominator = effective_suv_denominator_mbq(
            net_injected_dose_mbq=float(record.source_net_injected_dose_mbq),
            decay_reference=source_decay_reference,
            count_scaling=str(record.source_count_scaling),
            count_fraction=float(record.source_count_fraction),
            injection_datetime=record.source_injection_datetime,
            acquisition_datetime=record.source_acquisition_datetime,
            radionuclide_half_life_seconds=record.radionuclide_half_life_seconds,
        )
        target_reference_dose, target_denominator = effective_suv_denominator_mbq(
            net_injected_dose_mbq=float(record.target_net_injected_dose_mbq),
            decay_reference=target_decay_reference,
            count_scaling=str(record.target_count_scaling),
            count_fraction=float(record.target_count_fraction),
            injection_datetime=record.target_injection_datetime,
            acquisition_datetime=record.target_acquisition_datetime,
            radionuclide_half_life_seconds=record.radionuclide_half_life_seconds,
        )
        source_suv = suv_from_activity_concentration(
            source_data,
            weight_kg=record.weight_kg,
            injected_dose_mbq=source_denominator,
            activity_unit=str(record.source_activity_unit),
        )
        target_suv = suv_from_activity_concentration(
            target_data,
            weight_kg=record.weight_kg,
            injected_dose_mbq=target_denominator,
            activity_unit=str(record.target_activity_unit),
        )
        dose_qc = {
            "source_dose_at_image_reference_mbq": source_reference_dose,
            "target_dose_at_image_reference_mbq": target_reference_dose,
            "source_suv_denominator_mbq": source_denominator,
            "target_suv_denominator_mbq": target_denominator,
            "source_activity_decay_correction_to_admin": source_decay_factor,
            "target_activity_decay_correction_to_admin": target_decay_factor,
        }
        save_like(source_suv, source_image, paths["source_suv"])
        save_like(target_suv, target_image, paths["target_suv"])
    elif input_kind == "mni_suv":
        source_suv = source_data
        target_suv = target_data
        save_like(source_suv, source_image, paths["source_suv"])
        save_like(target_suv, target_image, paths["target_suv"])
    elif input_kind == "mni_suv_normalized":
        source_norm = source_data
        target_norm = target_data
        save_like(source_norm, source_image, paths["source_norm"])
        save_like(target_norm, target_image, paths["target_norm"])
        return {
            **asdict(record),
            "input_kind": input_kind,
            "source_mni_path": str(source_mni),
            "target_mni_path": str(target_mni),
            "source_suv_path": "",
            "target_suv_path": "",
            "source_normalized_path": str(paths["source_norm"]),
            "target_normalized_path": str(paths["target_norm"]),
            "source_norm_mean": float(np.mean(source_norm)),
            "target_norm_mean": float(np.mean(target_norm)),
            "status": "ok",
        }
    else:
        raise ValueError(f"Unsupported input_kind={input_kind}")

    source_norm = asinh_normalize(source_suv, asinh_scale)
    target_norm = asinh_normalize(target_suv, asinh_scale)
    save_like(source_norm, source_image, paths["source_norm"])
    save_like(target_norm, target_image, paths["target_norm"])

    return {
        **asdict(record),
        "input_kind": input_kind,
        "source_mni_path": str(source_mni),
        "target_mni_path": str(target_mni),
        "source_suv_path": str(paths["source_suv"]),
        "target_suv_path": str(paths["target_suv"]),
        "source_normalized_path": str(paths["source_norm"]),
        "target_normalized_path": str(paths["target_norm"]),
        **registration_qc,
        **dose_qc,
        "source_suv_mean": float(np.mean(source_suv)),
        "target_suv_mean": float(np.mean(target_suv)),
        "source_norm_mean": float(np.mean(source_norm)),
        "target_norm_mean": float(np.mean(target_norm)),
        "status": "ok",
    }


def run_external_preprocessing(
    *,
    metadata_csv: str | Path,
    data_root: str | Path | None,
    metadata_sheet: str | int | None = None,
    output_root: str | Path,
    mni_reference: str | Path,
    input_kind: str,
    asinh_scale: float = 1.0,
    threads: int = 4,
    transform_type: str = "s",
    work_dir: str | Path | None = None,
    keep_work: bool = False,
    force: bool = False,
) -> Path:
    if input_kind not in INPUT_KINDS:
        raise ValueError(f"input_kind must be one of {INPUT_KINDS}, got {input_kind}")
    if asinh_scale <= 0:
        raise ValueError("asinh_scale must be positive")
    output_root = Path(output_root).expanduser().resolve()
    reference = Path(mni_reference).expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError(reference)
    output_root.mkdir(parents=True, exist_ok=True)
    persistent_work = (
        Path(work_dir).expanduser().resolve() if work_dir else output_root / "work"
    )
    persistent_work.mkdir(parents=True, exist_ok=True)
    run_work = persistent_work / (
        "preprocess_" + datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    )
    run_work.mkdir(parents=True, exist_ok=False)

    contract = MNIContract.from_reference(reference)
    records = read_external_pair_metadata(
        metadata_csv,
        data_root=data_root,
        input_kind=input_kind,
        require_files=True,
        metadata_sheet=metadata_sheet,
    )
    report_rows: list[dict[str, object]] = []
    failed = False
    try:
        for record in records:
            try:
                report_rows.append(
                    _prepare_one(
                        record,
                        input_kind=input_kind,
                        output_root=output_root,
                        reference=reference,
                        contract=contract,
                        run_work=run_work,
                        asinh_scale=asinh_scale,
                        threads=threads,
                        transform_type=transform_type,
                        force=force,
                    )
                )
                print(f"[OK] {record.subject_id}")
            except Exception as exc:
                failed = True
                report_rows.append(
                    {
                        **asdict(record),
                        "input_kind": input_kind,
                        "status": f"error:{type(exc).__name__}:{exc}",
                    }
                )
                print(f"[ERROR] {record.subject_id}: {exc}")

        qc_dir = output_root / "qc"
        manifest_dir = output_root / "manifests"
        qc_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        report = pd.DataFrame(report_rows)
        report_path = qc_dir / "preprocessing_report.csv"
        report.to_csv(report_path, index=False)

        success = report[report["status"].eq("ok")].copy()
        training_manifest = success[
            ["subject_id", "source_normalized_path", "target_normalized_path"]
        ].rename(
            columns={
                "source_normalized_path": "source_path",
                "target_normalized_path": "target_path",
            }
        )
        manifest_path = manifest_dir / "pairs_normalized.csv"
        training_manifest.to_csv(manifest_path, index=False)

        protocol_status_counts: dict[str, int] = {}
        if "count_protocol_status" in success.columns:
            protocol_status_counts = {
                str(key): int(value)
                for key, value in success["count_protocol_status"]
                .fillna("not_applicable")
                .value_counts()
                .to_dict()
                .items()
            }

        summary = {
            "metadata_csv": str(Path(metadata_csv).resolve()),
            "data_root": str(Path(data_root).resolve()) if data_root else None,
            "output_root": str(output_root),
            "mni_reference": str(reference),
            "input_kind": input_kind,
            "asinh_scale": asinh_scale,
            "n_total": len(records),
            "n_success": int(len(success)),
            "n_failed": int(len(records) - len(success)),
            "count_protocol_status_counts": protocol_status_counts,
            "training_manifest": str(manifest_path),
            "work_dir": str(run_work),
            "work_retained": bool(keep_work or failed),
        }
        (qc_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2))
        if failed:
            raise RuntimeError(
                f"Preprocessing failed for {len(records) - len(success)} subject(s). "
                f"Inspect {report_path}; work retained at {run_work}."
            )
        return manifest_path
    finally:
        if run_work.exists() and not keep_work and not failed:
            shutil.rmtree(run_work)
