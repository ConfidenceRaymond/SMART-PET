from __future__ import annotations

import argparse

INPUT_KINDS = ("raw_activity", "mni_activity", "mni_suv", "mni_suv_normalized")
TRANSFORM_TYPES = ("s", "r", "a")

_HELP = """
Metadata files
--------------
CSV is supported by the base installation. XLSX/XLSM is supported when SMART-PET
is installed with the optional Excel dependency: python -m pip install '.[excel]'
The bundled workbook uses the Raw_Activity_Template sheet by default. For other
workbooks with multiple sheets, use --metadata-sheet SHEET_NAME.

For raw_activity and mni_activity, required columns are:
  subject_id, source_image_path, target_image_path, weight_kg,
  source_net_injected_dose_mbq, target_net_injected_dose_mbq,
  source_activity_unit, target_activity_unit,
  source_decay_reference, target_decay_reference,
  source_count_scaling, target_count_scaling,
  source_count_fraction, target_count_fraction

Accepted activity units: Bq/mL, kBq/mL, MBq/mL.
Accepted decay references:
  ADMIN  image values are decay-corrected to administration time.
  START  image values are decay-corrected to acquisition start; injection and
         acquisition datetimes plus radionuclide half-life are required.
  NONE   image values are calibrated but not decay-corrected and represent a
         frame-average activity concentration. Injection/acquisition datetimes,
         radionuclide half-life, and source/target_image_duration_seconds are
         required so SMART-PET can correct the image to ADMIN before SUVbw.

Accepted count scaling:
  quantitative  calibrated activity scale is preserved; count fraction is provenance.
  count_scaled  voxel values scale with retained counts; the SUV denominator is
                multiplied by count fraction.

All PET inputs must be scalar 3D NIfTI volumes. Dynamic 4D PET must first be
combined into a scientifically documented static 3D image.

raw_activity additionally requires ANTs executables antsRegistrationSyNQuick.sh
and antsApplyTransforms. MNI-domain input kinds do not run registration.
"""


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Prepare paired PET images for SMART-PET: optional MNI registration, "
            "SUV conversion, reversible asinh normalization, QC, and a neutral training manifest."
        ),
        epilog=_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--metadata-csv",
        "--metadata-file",
        dest="metadata_csv",
        required=True,
        help="External metadata table (.csv, .xlsx, or .xlsm).",
    )
    p.add_argument("--data-root", help="Root used to resolve relative image paths in metadata")
    p.add_argument(
        "--metadata-sheet",
        help=(
            "Excel sheet containing metadata. The bundled Raw_Activity_Template "
            "sheet is selected automatically when present."
        ),
    )
    p.add_argument("--output-root", required=True)
    p.add_argument("--mni-reference", required=True)
    p.add_argument("--input-kind", required=True, choices=INPUT_KINDS)
    p.add_argument("--asinh-scale", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--transform-type",
        choices=TRANSFORM_TYPES,
        default="s",
        help="ANTs SyNQuick transform: s=SyN (default), r=rigid, a=affine.",
    )
    p.add_argument(
        "--work-dir",
        help="Temporary workspace. Defaults to OUTPUT_ROOT/work and is deleted on success.",
    )
    p.add_argument("--keep-work", action="store_true")
    p.add_argument("--force", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    from smartpet.preprocessing.pipeline import run_external_preprocessing

    manifest = run_external_preprocessing(
        metadata_csv=args.metadata_csv,
        data_root=args.data_root,
        metadata_sheet=args.metadata_sheet,
        output_root=args.output_root,
        mni_reference=args.mni_reference,
        input_kind=args.input_kind,
        asinh_scale=args.asinh_scale,
        threads=args.threads,
        transform_type=args.transform_type,
        work_dir=args.work_dir,
        keep_work=args.keep_work,
        force=args.force,
    )
    print(f"[OK] training manifest: {manifest.resolve()}")


if __name__ == "__main__":
    main()
