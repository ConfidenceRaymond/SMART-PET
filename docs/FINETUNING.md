# Fine-tuning

Fine-tuning and exact resume are intentionally different operations.

## Public parent checkpoint

Fine-tuning starts from the full parent training checkpoint:

```text
smartpet_g001_parent_v0.3.1_full_checkpoint.pt
```

Download it with the verified `finetune` asset profile or from the public asset folder:

```bash
smartpet-download-assets --profile finetune --output-dir resources
```

SHA-256:

```text
2c974d4196e4514e5a0b877923d6b9b0a0c35ad4b447d06cd73d1bbc7abb8dee
```

Run:

```bash
smartpet-train \
  --config configs/finetune.json \
  --init-checkpoint resources/checkpoints/smartpet_g001_parent_v0.3.1_full_checkpoint.pt \
  --train-csv /path/to/new_train.csv \
  --val-csv /path/to/new_val.csv \
  --mni-reference resources/templates/csymT.nii.gz \
  --out-dir /path/to/new_finetune_run
```

Fine-tuning requires a **full training checkpoint** containing both generator
and discriminator states. Inference-only weights are rejected. Pairing a
trained generator with a newly initialized discriminator would change the
adversarial objective at initialization and is not the released SMART-PET
fine-tuning contract.

Fine-tuning loads both trained model states and creates new optimizers,
schedulers, progress counters, and RNG streams. The parent checkpoint path and
SHA-256 are recorded in `run_manifest.json`.

The checkpoint architecture must match `base_channels`, `attention_levels`,
`output_mode`, `asinh_scale`, and all configurable architecture fields. An
incompatible model or physical-output contract is rejected rather than
partially loaded.

## Launching fine-tuning

All fine-tuning commands start a new run from the full parent checkpoint.
Supply your own training and validation manifests and a new output directory.

### Single GPU

```bash
export SMARTPET_INIT_CHECKPOINT=resources/checkpoints/smartpet_g001_parent_v0.3.1_full_checkpoint.pt

bash scripts/finetune_single_gpu.sh configs/finetune.json \
  --train-csv /path/to/new_train.csv \
  --val-csv /path/to/new_val.csv \
  --mni-reference resources/templates/csymT.nii.gz \
  --out-dir /path/to/new_finetune_run
```

### Multi-GPU on one machine

```bash
export SMARTPET_INIT_CHECKPOINT=resources/checkpoints/smartpet_g001_parent_v0.3.1_full_checkpoint.pt
export SMARTPET_NPROC_PER_NODE=2

bash scripts/finetune_ddp.sh configs/finetune.json \
  --train-csv /path/to/new_train.csv \
  --val-csv /path/to/new_val.csv \
  --mni-reference resources/templates/csymT.nii.gz \
  --out-dir /path/to/new_finetune_run
```

### Alliance Canada / Narval

```bash
export SMARTPET_INIT_CHECKPOINT=resources/checkpoints/smartpet_g001_parent_v0.3.1_full_checkpoint.pt

sbatch scripts/slurm/narval_finetune_ddp.sh \
  --train-csv /path/to/new_train.csv \
  --val-csv /path/to/new_val.csv \
  --mni-reference resources/templates/csymT.nii.gz \
  --out-dir /path/to/new_finetune_run
```

The SLURM wrapper forwards these arguments to the training CLI.

These are fine-tuning operations. Use `--resume` only to continue an
interrupted run from a checkpoint produced by that same run.

## Choosing a fine-tuning schedule

`configs/finetune.json` is a reproducible example, not a universal
hyperparameter prescription.

The released external-adapted model was initialized from the parent checkpoint
and trained for 15 epochs with an initial learning rate of `1e-5`. That setting
worked for that adaptation study, but a new dataset should still use a
patient-disjoint validation set and a predefined model-selection criterion.

A practical fine-tuning study should:

1. freeze the parent checkpoint and its validation evidence;
2. define the train, validation, and final test cohorts before tuning;
3. start with conservative optimization settings;
4. select checkpoints only on the validation cohort;
5. reserve the final test cohort for the final claim.

Exact continuation of an interrupted run is different from fine-tuning. Use
`--resume` with the checkpoint produced by that same run when exact
continuation is required.
