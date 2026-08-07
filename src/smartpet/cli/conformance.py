from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from smartpet.conformance.architecture import (
    architecture_report,
    load_legacy_architecture_contract,
)
from smartpet.conformance.gradients import generator_gradient_attribution
from smartpet.conformance.legacy_reference import attention_comparison_report
from smartpet.models import SIMILARITY_MODES
from smartpet.models.discriminator import PatchDiscriminator3D
from smartpet.models.generator import SmartPETGenerator


def _write_json(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    print(f"[OK] saved: {output}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_legacy(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[dict[str, str]] = []
    verified = []
    repository_root = root.parent.parent
    for record in manifest["files"]:
        path = repository_root / record["path"]
        actual = _sha256(path) if path.is_file() else None
        if actual != record["sha256"]:
            failures.append(
                {
                    "path": record["path"],
                    "expected": record["sha256"],
                    "actual": actual or "missing",
                }
            )
        else:
            verified.append(record["path"])
    if failures:
        raise RuntimeError(f"Legacy reference verification failed: {failures}")
    return {
        "snapshot_id": manifest["snapshot_id"],
        "verified_file_count": len(verified),
        "verified_files": verified,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit SMART-PET historical-versus-modern conformance."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-legacy", help="Verify the frozen legacy snapshot")
    verify.add_argument("--root", type=Path, default=Path("reference/legacy"))
    verify.add_argument("--output", type=Path)

    architecture = commands.add_parser("architecture", help="Report current model topology")
    architecture.add_argument("--base-channels", type=int, default=32)
    architecture.add_argument("--attention-levels", type=int, nargs="*", default=[2, 3])
    architecture.add_argument(
        "--output-mode",
        choices=("linear", "positive_softplus_residual"),
        default="positive_softplus_residual",
    )
    architecture.add_argument(
        "--similarity-mode",
        choices=SIMILARITY_MODES,
        default="v030_luminance",
    )
    architecture.add_argument(
        "--encoder-convs-per-level",
        type=int,
        choices=(1, 2),
        default=1,
    )
    architecture.add_argument(
        "--channel-spatial-input-projection",
        action="store_true",
    )
    architecture.add_argument("--generator-spectral-norm", action="store_true")
    architecture.add_argument("--discriminator-spectral-norm", action="store_true")
    architecture.add_argument(
        "--legacy-contract",
        type=Path,
        default=Path("reference/legacy/architecture_contract.json"),
    )
    architecture.add_argument("--output", type=Path)

    attention = commands.add_parser("attention", help="Compare similarity statistics")
    attention.add_argument("--seed", type=int, default=2023)
    attention.add_argument("--shape", type=int, nargs=5, default=[1, 2, 16, 16, 16])
    attention.add_argument("--scales", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    attention.add_argument("--output", type=Path)

    gradients = commands.add_parser("gradients", help="Audit generator gradient connectivity")
    gradients.add_argument("--seed", type=int, default=2023)
    gradients.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "verify-legacy":
        payload = _verify_legacy(args.root)
    elif args.command == "architecture":
        generator = SmartPETGenerator(
            base_channels=args.base_channels,
            attention_levels=tuple(args.attention_levels),
            output_mode=args.output_mode,
            similarity_mode=args.similarity_mode,
            encoder_convs_per_level=args.encoder_convs_per_level,
            channel_spatial_input_projection=args.channel_spatial_input_projection,
            generator_spectral_norm=args.generator_spectral_norm,
        )
        discriminator = PatchDiscriminator3D(
            base_channels=args.base_channels,
            spectral_norm=args.discriminator_spectral_norm,
        )
        payload = architecture_report(generator, discriminator)
        payload["legacy_reference"] = load_legacy_architecture_contract(args.legacy_contract)
    elif args.command == "attention":
        if any(value <= 0 for value in args.shape):
            raise ValueError("All --shape values must be positive")
        torch.manual_seed(args.seed)
        sample = torch.randn(*args.shape, dtype=torch.float32)
        payload = attention_comparison_report(sample, scales=args.scales)
        payload["seed"] = args.seed
    else:
        payload = generator_gradient_attribution(seed=args.seed)
    _write_json(payload, args.output)


if __name__ == "__main__":
    main()
