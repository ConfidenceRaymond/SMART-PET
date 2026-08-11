from __future__ import annotations

import argparse

INPUT_KINDS = ("raw_activity", "mni_activity", "mni_suv", "mni_suv_normalized")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Prepare paired PET images for SMART-PET: optional MNI registration, "
            "SUV conversion, reversible asinh normalization, QC, and a neutral training manifest."
        )
    )
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--data-root", help="Root used to resolve relative image paths in the CSV")
    p.add_argument("--output-root", required=True)
    p.add_argument("--mni-reference", required=True)
    p.add_argument("--input-kind", required=True, choices=INPUT_KINDS)
    p.add_argument("--asinh-scale", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--transform-type", default="s")
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
