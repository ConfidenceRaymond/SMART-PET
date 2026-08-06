# Fine-tuning

Fine-tuning and exact resume are intentionally different operations.

```bash
smartpet-train \
  --config configs/finetune.json \
  --init-checkpoint /path/to/pretrained.pt \
  --train-csv /path/to/new_train.csv \
  --val-csv /path/to/new_val.csv \
  --mni-reference /path/to/reference.nii.gz \
  --out-dir /path/to/new_finetune_run
```

Fine-tuning loads generator weights and, when present, discriminator weights. It then creates new optimizers, schedulers, progress counters, and RNG streams. The parent checkpoint path and SHA-256 are recorded in `run_manifest.json`.

The checkpoint architecture must match `base_channels`, `attention_levels`, and `output_mode`. A configuration change that alters tensor shapes is rejected rather than partially loaded.

Recommended first fine-tuning study:

1. freeze the current epoch-4 baseline and its validation evidence;
2. use a lower learning rate such as `1e-5`;
3. run a short pilot with the same fixed validation protocol;
4. compare checkpoints using the predefined validation criterion;
5. reserve a patient-disjoint test split for the final claim.
