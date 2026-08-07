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

## Architecture profiles

The frozen v0.3.0 profile is `configs/train_from_scratch.json`. The corrected
Phase 2B candidate is `configs/train_corrected_s4.json`. The five S4 architecture
fields are mandatory in maintained configuration files:

| Field | Frozen v0.3.0 | Corrected candidate |
|---|---|---|
| `similarity_mode` | `v030_luminance` | `scale_consistent` |
| `encoder_convs_per_level` | `1` | `2` |
| `channel_spatial_input_projection` | `false` | `true` |
| `generator_spectral_norm` | `false` | `false` |
| `discriminator_spectral_norm` | `false` | `true` |

Run the corrected candidate with:

```bash
smartpet-train --config configs/train_corrected_s4.json --backend single
```

Do not resume or fine-tune across architecture profiles. Full checkpoints record
these fields, and SMART-PET rejects mismatches before training continues. See
`docs/CORRECTED_S4.md` for the operator-level contract.

## Exact continuation

```bash
smartpet-train --config /path/to/original/config.json \
  --resume /path/to/run/checkpoints/last.pt \
  --set epochs=50
```

Checkpoint continuation restores both models, both optimizers, schedulers, scaler, global step, batch position, metric accumulators, and rank-specific RNG states. Architecture, data, precision, world size, deterministic mode, and sampling-policy mismatches are rejected. `epochs` is the new total target, not an additional epoch count.

Set `deterministic=true` when bitwise restart equivalence is required. In this mode SMART-PET enables deterministic PyTorch algorithms, deterministic cuDNN, deterministic cuBLAS workspace configuration, disables TF32, and forces the math scaled-dot-product-attention backend. For the frozen `v030_luminance` profile, deterministic mode replaces only the parameter-free `AvgPool3d` smoothing operation with an equivalent grouped box convolution because CUDA does not implement deterministic `avg_pool3d` backward. The state-dict and trainable architecture remain unchanged. The default remains `false` because deterministic kernels can reduce throughput.

DataLoader iterator bookkeeping uses dedicated per-rank/per-epoch generators, so creating an iterator during a mid-epoch restart does not advance the checkpointed model CPU RNG stream. When discriminator spectral normalization is enabled, DDP broadcasts discriminator buffers so the stateful spectral-normalization power-iteration buffers remain synchronized across ranks.

`num_workers=0` remains the conservative production default. Increase it only after a site-specific continuation smoke test.
