from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartpet.checkpoint_io import sha256_file
from smartpet.inference.weights import export_inference_weights


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export a full SMART-PET training checkpoint as inference-only weights."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--json-output")
    return p


def main() -> None:
    args = parser().parse_args()
    output = Path(args.output)
    payload = export_inference_weights(args.checkpoint, output)
    result = {
        "output": str(output.resolve()),
        "sha256": sha256_file(output),
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "source_global_step": payload["source_global_step"],
        "source_epoch": payload["source_epoch"],
        "config": payload["config"],
    }
    if args.json_output:
        json_output = Path(args.json_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"[OK] export JSON saved: {json_output.resolve()}")
    print(f"[OK] inference weights saved: {output.resolve()}")
    print(f"[OK] inference weights SHA-256: {result['sha256']}")
    print(f"[OK] source checkpoint SHA-256: {result['source_checkpoint_sha256']}")
    print("[OK] SMART-PET INFERENCE WEIGHTS EXPORT PASSED")


if __name__ == "__main__":
    main()
