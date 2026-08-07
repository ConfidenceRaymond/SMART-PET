from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from smartpet.checkpoint_io import safe_torch_load
from smartpet.training.checkpoint import CHECKPOINT_FORMAT_VERSION
from smartpet.training.precision import optimizer_max_step


def audit_checkpoint(
    checkpoint_path: str | Path,
    *,
    metrics_path: str | Path | None = None,
    expected_step: int | None = None,
    expected_world_size: int | None = None,
    expected_precision: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = safe_torch_load(checkpoint_path)
    artifact_type = checkpoint.get("artifact_type")
    if artifact_type != "smartpet_training_checkpoint":
        raise RuntimeError(
            "smartpet-audit-checkpoint accepts only full training checkpoints; "
            f"artifact_type={artifact_type!r}. Use smartpet-audit-weights for "
            "inference-only artifacts."
        )
    smartpet_version = str(checkpoint.get("smartpet_version", "")).strip()
    if not smartpet_version:
        raise RuntimeError("Training checkpoint is missing smartpet_version provenance")
    format_version = int(checkpoint.get("format_version", 0))
    if format_version < CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"Checkpoint format_version={format_version} predates the safe production "
            f"format {CHECKPOINT_FORMAT_VERSION} and is not accepted for audit/resume."
        )

    global_step = int(checkpoint["global_step"])
    world_size = int(checkpoint["world_size"])
    if expected_step is not None and global_step != int(expected_step):
        raise RuntimeError(f"Expected global_step={expected_step}, found {global_step}")
    if expected_world_size is not None and world_size != int(expected_world_size):
        raise RuntimeError(f"Expected world_size={expected_world_size}, found {world_size}")

    g_state = checkpoint["g_optimizer_state"]
    d_state = checkpoint["d_optimizer_state"]
    g_actual = optimizer_max_step(g_state)
    d_actual = optimizer_max_step(d_state)
    g_recorded = int(checkpoint.get("g_optimizer_updates", -1))
    d_recorded = int(checkpoint.get("d_optimizer_updates", -1))
    if not (g_actual == g_recorded == global_step):
        raise RuntimeError(
            "Generator optimizer progress mismatch: "
            f"actual={g_actual}, recorded={g_recorded}, global_step={global_step}"
        )
    if not (d_actual == d_recorded == global_step):
        raise RuntimeError(
            "Discriminator optimizer progress mismatch: "
            f"actual={d_actual}, recorded={d_recorded}, global_step={global_step}"
        )

    precision = str(checkpoint.get("precision", {}).get("resolved", "unknown"))
    if expected_precision is not None and precision != expected_precision:
        raise RuntimeError(f"Expected precision={expected_precision}, found {precision}")

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Training checkpoint config must be a mapping")
    if config.get("output_mode") is None:
        raise RuntimeError("Training checkpoint config is missing output_mode")
    if not checkpoint.get("generator_state"):
        raise RuntimeError("Generator state is empty")
    if not checkpoint.get("discriminator_state"):
        raise RuntimeError("Discriminator state is empty")
    if len(checkpoint.get("rng_states", [])) != world_size:
        raise RuntimeError(
            "RNG state count does not match world size: "
            f"rng_states={len(checkpoint.get('rng_states', []))}, world_size={world_size}"
        )
    if not math.isfinite(float(checkpoint["best_metric"])):
        raise RuntimeError("best_metric is not finite")

    metrics_last: dict[str, str] | None = None
    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        with metrics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError("metrics.csv is empty")
        metrics_last = rows[-1]
        if int(metrics_last["global_step"]) != global_step:
            raise RuntimeError(
                "metrics.csv global_step does not match checkpoint: "
                f"metrics={metrics_last['global_step']}, checkpoint={global_step}"
            )
        metric_groups = (
            ("train_g_total", "train_g"),
            ("train_d_total", "train_d"),
            ("val_l1_normalized", "val_l1"),
        )
        for alternatives in metric_groups:
            field = next((name for name in alternatives if name in metrics_last), None)
            if field is None:
                raise RuntimeError(f"metrics.csv missing required alternatives: {alternatives}")
            if not math.isfinite(float(metrics_last[field])):
                raise RuntimeError(f"Non-finite metric {field}={metrics_last[field]}")

    return {
        "checkpoint": str(checkpoint_path),
        "artifact_type": artifact_type,
        "smartpet_version": smartpet_version,
        "format_version": format_version,
        "global_step": global_step,
        "world_size": world_size,
        "generator_updates": g_actual,
        "discriminator_updates": d_actual,
        "precision": precision,
        "epoch": int(checkpoint.get("epoch", -1)),
        "epoch_complete": bool(checkpoint.get("epoch_complete", True)),
        "batch_in_epoch": int(checkpoint.get("batch_in_epoch", 0)),
        "samples_seen": int(checkpoint.get("samples_seen", 0)),
        "output_mode": str(config["output_mode"]),
        "metrics_last": metrics_last,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit a SMART-PET training checkpoint.")
    location = p.add_mutually_exclusive_group(required=True)
    location.add_argument("--run-dir")
    location.add_argument("--checkpoint")
    p.add_argument("--metrics")
    p.add_argument("--expected-step", type=int)
    p.add_argument("--expected-world-size", type=int)
    p.add_argument("--expected-precision", choices=["fp32", "bf16", "fp16"])
    return p


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.run_dir is not None and not args.run_dir.strip():
        argument_parser.error("--run-dir cannot be empty")
    if args.checkpoint is not None and not args.checkpoint.strip():
        argument_parser.error("--checkpoint cannot be empty")
    if args.run_dir:
        run_dir = Path(args.run_dir)
        checkpoint = run_dir / "checkpoints" / "last.pt"
        metrics = Path(args.metrics) if args.metrics else run_dir / "metrics.csv"
    else:
        checkpoint = Path(args.checkpoint)
        metrics = Path(args.metrics) if args.metrics else None

    result = audit_checkpoint(
        checkpoint,
        metrics_path=metrics,
        expected_step=args.expected_step,
        expected_world_size=args.expected_world_size,
        expected_precision=args.expected_precision,
    )
    print(f"[OK] artifact_type={result['artifact_type']}")
    print(f"[OK] smartpet_version={result['smartpet_version']}")
    print(f"[OK] checkpoint format_version={result['format_version']}")
    print(f"[OK] global_step={result['global_step']}")
    print(f"[OK] world_size={result['world_size']}")
    print(f"[OK] generator optimizer updates={result['generator_updates']}")
    print(f"[OK] discriminator optimizer updates={result['discriminator_updates']}")
    print(f"[OK] precision={result['precision']}")
    print(f"[OK] epoch={result['epoch']} epoch_complete={result['epoch_complete']}")
    print(f"[OK] batch_in_epoch={result['batch_in_epoch']}")
    print(f"[OK] samples_seen={result['samples_seen']}")
    print(f"[OK] output_mode={result['output_mode']}")
    print("[OK] SMART-PET CHECKPOINT AUDIT PASSED")


if __name__ == "__main__":
    main()
