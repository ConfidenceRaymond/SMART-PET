from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from smartpet.data.nifti import MNIContract, load_mni_volume
from smartpet.data.normalization import asinh_denormalize
from smartpet.metrics.masked import masked_image_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_suv(data: np.ndarray, domain: str, scale: float) -> np.ndarray:
    if domain == "suv":
        return np.asarray(data, dtype=np.float32)
    return asinh_denormalize(np.clip(data, 0.0, None), scale)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate a SMART-PET prediction inside a fixed brain mask."
    )
    p.add_argument("--prediction", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--brain-mask", required=True)
    p.add_argument("--mni-reference", required=True)
    p.add_argument("--prediction-domain", choices=["suv", "normalized"], default="suv")
    p.add_argument("--target-domain", choices=["suv", "normalized"], default="suv")
    p.add_argument("--asinh-scale", type=float, default=1.0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output-json", required=True)
    args = p.parse_args()
    if args.asinh_scale <= 0:
        p.error("--asinh-scale must be positive")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)

    contract = MNIContract.from_reference(args.mni_reference)
    _, prediction = load_mni_volume(args.prediction, contract)
    _, target = load_mni_volume(args.target, contract)
    _, mask_raw = load_mni_volume(args.brain_mask, contract)
    mask = mask_raw > 0.5
    fraction = float(mask.mean())
    if not 0.05 < fraction < 0.75:
        raise RuntimeError(f"Implausible brain-mask volume fraction: {fraction:.6f}")
    prediction_suv = _to_suv(prediction, args.prediction_domain, args.asinh_scale)
    target_suv = _to_suv(target, args.target_domain, args.asinh_scale)
    metrics = masked_image_metrics(prediction_suv, target_suv, mask, device=device)
    payload = {
        "format_version": 1,
        "metric_domain": "suv",
        "mask_threshold": ">0.5",
        "mask_voxels": int(mask.sum()),
        "mask_volume_fraction": fraction,
        "prediction": str(Path(args.prediction).resolve()),
        "target": str(Path(args.target).resolve()),
        "brain_mask": str(Path(args.brain_mask).resolve()),
        "brain_mask_sha256": _sha256(Path(args.brain_mask)),
        "mni_reference": str(Path(args.mni_reference).resolve()),
        "mni_reference_sha256": _sha256(Path(args.mni_reference)),
        "metrics": metrics,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
