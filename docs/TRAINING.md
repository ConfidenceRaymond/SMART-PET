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

## Exact continuation

```bash
smartpet-train --config /path/to/original/config.json \
  --resume /path/to/run/checkpoints/last.pt \
  --set epochs=50
```

Exact resume restores both models, both optimizers, schedulers, scaler, global step, batch position, metric accumulators, and rank-specific RNG states. Architecture, data, precision, world size, and sampling-policy mismatches are rejected. `epochs` is the new total target, not an additional epoch count.

`num_workers=0` is the conservative production default because it was the stable exact-resume configuration on Narval. Increase it only after a site-specific continuation smoke test.
