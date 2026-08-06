from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from smartpet.inference.engine import InferenceEngine

REQUIRED_COLUMNS = ("subject_id", "input_path", "normalized_output", "suv_output")


def _optional_path(value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return Path(text) if text else None


def main() -> None:
    p = argparse.ArgumentParser(description="Batch whole-volume SMART-PET inference.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mni-reference", required=True)
    p.add_argument("--input-domain", choices=["suv", "normalized"], default="suv")
    p.add_argument("--patch-size", nargs=3, type=int)
    p.add_argument("--stride", nargs=3, type=int)
    p.add_argument("--asinh-scale", type=float)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--report-csv")
    args = p.parse_args()

    frame = pd.read_csv(args.manifest)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        p.error(f"Inference manifest missing columns: {missing}")
    if frame.empty:
        p.error("Inference manifest is empty")
    if frame["subject_id"].astype(str).duplicated().any():
        p.error("subject_id must be unique")

    report_path = (
        Path(args.report_csv)
        if args.report_csv
        else Path(args.manifest).with_name("inference_report.csv")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    engine = InferenceEngine(
        checkpoint=args.checkpoint,
        mni_reference=args.mni_reference,
        patch_size=tuple(args.patch_size) if args.patch_size else None,
        stride=tuple(args.stride) if args.stride else None,
        asinh_scale=args.asinh_scale,
        amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
        device=args.device,
    )

    rows: list[dict[str, object]] = []
    failures = 0
    for index, row in frame.iterrows():
        subject_id = str(row["subject_id"])
        input_path = Path(str(row["input_path"]))
        normalized_output = _optional_path(row["normalized_output"])
        suv_output = _optional_path(row["suv_output"])
        if normalized_output is None and suv_output is None:
            raise ValueError(f"{subject_id}: at least one output path is required")
        outputs = [path for path in (normalized_output, suv_output) if path is not None]
        if not args.overwrite and outputs and all(path.is_file() for path in outputs):
            rows.append({"subject_id": subject_id, "status": "SKIPPED_EXISTS", "error": ""})
            print(f"[{index + 1:04d}/{len(frame):04d}] {subject_id} SKIPPED_EXISTS", flush=True)
            continue
        try:
            result = engine.predict(input_path, input_domain=args.input_domain)
            saved = engine.save(
                result,
                normalized_output=normalized_output,
                suv_output=suv_output,
            )
            metadata_anchor = normalized_output or suv_output
            assert metadata_anchor is not None
            metadata_stem = metadata_anchor.name.replace(".nii.gz", "").replace(
                ".nii", ""
            )
            metadata = metadata_anchor.with_name(f"{metadata_stem}_prediction.json")
            metadata.write_text(
                json.dumps(
                    {
                        "subject_id": subject_id,
                        "prediction_id": result.prediction_id,
                        "checkpoint_sha256": engine.checkpoint_sha256,
                        "input": str(input_path.resolve()),
                        "normalized_output": str(saved.get("normalized", "")),
                        "suv_output": str(saved.get("suv", "")),
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n"
            )
            rows.append({"subject_id": subject_id, "status": "PASS", "error": ""})
            print(f"[{index + 1:04d}/{len(frame):04d}] {subject_id} PASS", flush=True)
        except Exception as exc:
            failures += 1
            rows.append({"subject_id": subject_id, "status": "FAILED", "error": repr(exc)})
            print(f"[{index + 1:04d}/{len(frame):04d}] {subject_id} FAILED: {exc}", flush=True)
            if not args.continue_on_error:
                break

    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "status", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"report_csv={report_path.resolve()}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
