from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from smartpet.data.nifti import MNIContract, load_mni_volume


def _stats(volume: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(volume, dtype=np.float32)
    return {
        "voxel_count": int(array.size),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "p01": float(np.percentile(array, 1)),
        "p50": float(np.percentile(array, 50)),
        "p99": float(np.percentile(array, 99)),
        "negative_voxel_count": int(np.count_nonzero(array < 0.0)),
        "nonzero_voxel_count": int(np.count_nonzero(array)),
    }


def audit_inference(
    output_path: str | Path,
    *,
    mni_reference: str | Path,
    input_path: str | Path | None = None,
    target_path: str | Path | None = None,
    require_nonnegative: bool = True,
) -> dict[str, Any]:
    contract = MNIContract.from_reference(mni_reference)
    _, output = load_mni_volume(output_path, contract)
    result: dict[str, Any] = {
        "output": str(Path(output_path)),
        "shape": list(contract.shape),
        "zooms": list(contract.zooms),
        "output_stats": _stats(output),
    }
    if require_nonnegative and float(output.min()) < -1e-6:
        raise RuntimeError(f"Inference output contains negative values: min={float(output.min())}")
    if input_path is not None:
        _, source = load_mni_volume(input_path, contract)
        result["input"] = str(Path(input_path))
        result["input_stats"] = _stats(source)
    if target_path is not None:
        _, target = load_mni_volume(target_path, contract)
        result["target"] = str(Path(target_path))
        result["target_stats"] = _stats(target)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit SMART-PET whole-volume inference output.")
    p.add_argument("--output", required=True)
    p.add_argument("--mni-reference", required=True)
    p.add_argument("--input")
    p.add_argument("--target")
    p.add_argument("--json-output")
    p.add_argument("--allow-negative", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    result = audit_inference(
        args.output,
        mni_reference=args.mni_reference,
        input_path=args.input,
        target_path=args.target,
        require_nonnegative=not args.allow_negative,
    )
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"[OK] audit JSON saved: {path.resolve()}")
    print(f"[OK] MNI shape={tuple(result['shape'])}")
    print(f"[OK] MNI zooms={tuple(result['zooms'])}")
    print("[OK] output is finite")
    print(f"[OK] output minimum={result['output_stats']['minimum']:.8g}")
    print(f"[OK] output maximum={result['output_stats']['maximum']:.8g}")
    print(f"[OK] negative voxels={result['output_stats']['negative_voxel_count']}")
    print("[OK] SMART-PET INFERENCE AUDIT PASSED")


if __name__ == "__main__":
    main()
