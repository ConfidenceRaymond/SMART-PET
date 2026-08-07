# Training and exact continuation

## Validate inputs

```bash
smartpet-validate-manifest --manifest train.csv --mni-reference reference.nii.gz
smartpet-validate-manifest --manifest val.csv --mni-reference reference.nii.gz
```

## Single GPU

```bash
smartpet-train --config configs/train_from_scratch.json --backend single
```

## DDP on one node

```bash
torchrun --standalone --nproc-per-node=2 -m smartpet.cli.train \
  --config configs/train_from_scratch.json --backend ddp
```

`batch_size` is per rank. The run records per-rank and global batch sizes, successful optimizer updates, subject exposures, manifests, reference hashes, precision, and model configuration.

DDP initialization uses a 30-minute collective timeout by default. Set
`SMARTPET_DIST_TIMEOUT_MIN` to a positive integer to use a site-specific limit.
Validation must contain at least one subject per rank; training stops before the
first validation collective when the split is smaller than the DDP world size.

With fp16, a GradScaler overflow skips both GAN optimizers together and does not
advance `global_step`. Each epoch records `fp16_scaler_skipped_batches` and the
final scaler value. The run stops after `max_consecutive_scaler_skips`
consecutive skipped batches because sustained overflow indicates divergence.

## Exact continuation

```bash
smartpet-train --config /path/to/original/config.json \
  --resume /path/to/run/checkpoints/last.pt \
  --set epochs=50
```

Exact resume restores both models, both optimizers, schedulers, scaler, global step, batch position, metric accumulators, and rank-specific RNG states. Architecture, data, precision, world size, and sampling-policy mismatches are rejected. `epochs` is the new total target, not an additional epoch count.

`num_workers=0` is the conservative production default because it was the stable exact-resume configuration on Narval. Increase it only after a site-specific continuation smoke test.
