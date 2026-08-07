# Fine-tuning

Fine-tuning and exact resume are intentionally different operations.

```bash
smartpet-train \
  --config configs/finetune.json \
  --init-checkpoint /path/to/full_training_checkpoint.pt \
  --train-csv /path/to/new_train.csv \
  --val-csv /path/to/new_val.csv \
  --mni-reference /path/to/reference.nii.gz \
  --out-dir /path/to/new_finetune_run
```

Fine-tuning requires a **full training checkpoint** containing both generator
and discriminator states. Inference-only weights are rejected. Pairing a
trained generator with a newly randomized discriminator would change the
adversarial objective at initialization and is not an approved SMART-PET
fine-tuning path.

Fine-tuning loads both model states and creates new optimizers, schedulers,
progress counters, and RNG streams. The parent checkpoint path and SHA-256 are
recorded in `run_manifest.json`.

The checkpoint architecture must match `base_channels`, `attention_levels`,
`output_mode`, `asinh_scale`, and all configurable S4 fields described in
`docs/CORRECTED_S4.md`. A configuration change that alters the model or physical
output contract is rejected rather than partially loaded.

Recommended first fine-tuning study:

1. freeze the parent checkpoint and its validation evidence;
2. use a lower learning rate such as `1e-5`;
3. run a short pilot with the same fixed validation protocol;
4. compare checkpoints using the predefined validation criterion;
5. reserve a patient-disjoint test split for the final claim.
