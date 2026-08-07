# Checkpoint and weight contracts

SMART-PET deliberately separates two artifact types.

## Full training checkpoints

Full checkpoints use `artifact_type=smartpet_training_checkpoint` and format
version 4. They contain:

- generator and discriminator states;
- both Adam optimizer states;
- learning-rate schedulers;
- AMP scaler state when applicable;
- epoch, batch position, global step, and samples seen;
- optimizer-update integrity counters;
- rank-specific Python, NumPy, CPU Torch, and CUDA RNG states;
- training configuration, SMART-PET version, and precision policy;
- current best validation metric.

Use `smartpet-audit-checkpoint` before state-complete continuation or adversarial
fine-tuning. Bitwise restart equivalence additionally requires `deterministic=true`
in both the checkpoint and the resumed run. Do not store multi-gigabyte
checkpoints directly in Git.

## Inference-only weights

Inference exports use `artifact_type=smartpet_inference_weights` and their own
format version. Format 2 records the complete configurable S4 architecture.
Format 1 remains readable as the immutable v0.3.0 architecture profile; partial
architecture metadata is rejected. Exports contain only the generator state,
inference-critical configuration, source-checkpoint digest, source step/epoch,
and export version. They contain no discriminator, optimizer, scheduler, scaler,
or RNG state.

Create them reproducibly:

```bash
smartpet-export-weights \
  --checkpoint /path/to/full_training_checkpoint.pt \
  --output /path/to/smartpet_inference.pt \
  --json-output /path/to/smartpet_inference.export.json
```

Audit them with the correct tool:

```bash
smartpet-audit-weights \
  --weights /path/to/smartpet_inference.pt \
  --expected-sha256 <SHA256>
```

Do not pass inference-only weights to `--resume` or `--init-checkpoint`.
Adversarial fine-tuning requires a full checkpoint with a trained discriminator.

## Legacy format-3 checkpoints

Legacy format-3 checkpoints must be converted in a disposable isolated
environment before use. See `docs/LEGACY_CHECKPOINT_CONVERSION.md`.

Publish release weights through an immutable archive or institutional repository
and commit the release SHA-256 manifest to Git. A mutable mirror may be offered
for convenience but is not the authoritative integrity record.
