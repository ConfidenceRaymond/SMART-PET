from __future__ import annotations

import argparse

from smartpet.config import load_train_config, parse_set_overrides
from smartpet.training.trainer import run


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train, exactly resume, or fine-tune SMART-PET from a strict "
            "JSON configuration."
        )
    )
    p.add_argument("--config", required=True, help="Path to a SMART-PET training JSON file")
    p.add_argument("--train-csv", help="Override train_csv from the configuration")
    p.add_argument("--val-csv", help="Override val_csv from the configuration")
    p.add_argument("--mni-reference", help="Override mni_reference from the configuration")
    p.add_argument("--out-dir", help="Override out_dir from the configuration")
    p.add_argument(
        "--backend",
        choices=["auto", "single", "ddp"],
        help="auto selects DDP under torchrun and single-process execution otherwise.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        help=(
            "Exact continuation from a full checkpoint. Restores models, optimizers, "
            "schedulers, progress, and rank-specific RNG state."
        ),
    )
    mode.add_argument(
        "--init-checkpoint",
        help=(
            "Start a new fine-tuning run from pretrained model weights. Optimizers, "
            "schedulers, progress, and RNG state are reset."
        ),
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="Override any TrainConfig field; may be supplied multiple times.",
    )
    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    try:
        overrides = parse_set_overrides(args.set)
        explicit = {
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "mni_reference": args.mni_reference,
            "out_dir": args.out_dir,
            "backend": args.backend,
            "resume": args.resume,
            "init_checkpoint": args.init_checkpoint,
        }
        overrides.update({key: value for key, value in explicit.items() if value is not None})
        config = load_train_config(args.config, overrides=overrides)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        p.error(str(exc))
    run(config)


if __name__ == "__main__":
    main()
