from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartpet.inference.weights import audit_inference_weights


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit SMART-PET inference-only weights.")
    p.add_argument("--weights", required=True)
    p.add_argument("--expected-sha256")
    p.add_argument("--json-output")
    p.add_argument(
        "--skip-model-state-check",
        action="store_true",
        help="Skip strict generator construction/state loading; integrity checks still run.",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    result = audit_inference_weights(
        args.weights,
        expected_sha256=args.expected_sha256,
        validate_model_state=not args.skip_model_state_check,
    )
    if args.json_output:
        json_output = Path(args.json_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"[OK] audit JSON saved: {json_output.resolve()}")
    print(f"[OK] artifact_type={result['artifact_type']}")
    print(f"[OK] format_version={result['format_version']}")
    print(f"[OK] SHA-256={result['sha256']}")
    print(f"[OK] source checkpoint SHA-256={result['source_checkpoint_sha256']}")
    print(f"[OK] source global_step={result['source_global_step']}")
    print(f"[OK] source epoch={result['source_epoch']}")
    print("[OK] SMART-PET INFERENCE WEIGHTS AUDIT PASSED")


if __name__ == "__main__":
    main()
