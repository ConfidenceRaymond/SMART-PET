from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from smartpet.inference.engine import InferenceEngine
from smartpet.inference.outputs import resolve_output_plan


def _write_metadata(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Whole-volume SMART-PET NIfTI inference.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--mni-reference", required=True)
    p.add_argument("--output", help="Single legacy output path; defaults to SUV domain")
    p.add_argument("--normalized-output")
    p.add_argument("--suv-output")
    p.add_argument("--metadata-json")
    p.add_argument("--input-domain", choices=["suv", "normalized"], default="suv")
    p.add_argument("--output-domain", choices=["suv", "normalized"], default=None)
    p.add_argument("--patch-size", nargs=3, type=int)
    p.add_argument("--stride", nargs=3, type=int)
    p.add_argument("--asinh-scale", type=float)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--no-amp", action="store_true")
    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    try:
        outputs = resolve_output_plan(
            output=args.output,
            output_domain=args.output_domain,
            normalized_output=args.normalized_output,
            suv_output=args.suv_output,
            metadata_json=args.metadata_json,
        )
    except ValueError as exc:
        p.error(str(exc))

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
    result = engine.predict(args.input, input_domain=args.input_domain)

    normalized_output = outputs.normalized_output
    suv_output = outputs.suv_output
    if outputs.legacy_output is not None:
        if outputs.legacy_domain == "normalized":
            normalized_output = outputs.legacy_output
        else:
            suv_output = outputs.legacy_output
    saved = engine.save(
        result,
        normalized_output=normalized_output,
        suv_output=suv_output,
    )

    metadata_path: Path | None = None
    if outputs.metadata_json is not None:
        metadata_path = _write_metadata(
            outputs.metadata_json,
            {
                "format_version": 2,
                "prediction_id": result.prediction_id,
                "checkpoint": str(engine.checkpoint_path.resolve()),
                "checkpoint_sha256": engine.checkpoint_sha256,
                "input": str(Path(args.input).resolve()),
                "mni_reference": str(engine.mni_reference.resolve()),
                "input_domain": args.input_domain,
                "asinh_scale": engine.asinh_scale,
                "precision_requested": args.amp_dtype,
                "precision_resolved": engine.precision.resolved,
                "patch_size": list(engine.patch_size),
                "stride": list(engine.stride),
                "model_output_mode": engine.output_mode,
                "negative_normalized_voxels": result.negative_count,
                "normalized_output": (
                    str(saved["normalized"].resolve())
                    if "normalized" in saved
                    else None
                ),
                "suv_output": str(saved["suv"].resolve()) if "suv" in saved else None,
                "shared_forward_pass": True,
            },
        )

    print(f"input_domain={args.input_domain}")
    print(f"precision_resolved={engine.precision.resolved}")
    print(f"patch_size={','.join(map(str, engine.patch_size))}")
    print(f"stride={','.join(map(str, engine.stride))}")
    print(f"shared_prediction_id={result.prediction_id}")
    for domain, path in saved.items():
        print(f"{domain}_output={path.resolve()}")
    if metadata_path is not None:
        print(f"metadata_json={metadata_path.resolve()}")


if __name__ == "__main__":
    main()
