# Reproducibility assets

SMART-PET model and reference assets are distributed through the public model
folder linked from the repository README.

The repository does not automatically download these files.

## Public model artifacts

### Inference weights

```text
weights/
├── smartpet_g001_parent_v0.3.1.pt
├── smartpet_g001_external_adapted_v0.3.1.pt
└── smartpet_v0.3.0_epoch4_inference.pt
```

`smartpet_g001_parent_v0.3.1.pt` is the recommended general pretrained
inference model.

`smartpet_g001_external_adapted_v0.3.1.pt` is a domain-specific adapted model.
It should not be treated as a universal replacement for the parent model.

`smartpet_v0.3.0_epoch4_inference.pt` is retained as a historical v0.3.0
inference artifact.

The v0.3.1 inference-weight SHA-256 values are:

```text
smartpet_g001_parent_v0.3.1.pt
f26b89db433368167bb67242d0ed2e5351651a2155a92f41f6fce991649f91b0

smartpet_g001_external_adapted_v0.3.1.pt
aecd3b0c15f0b0b90fc6e2142412562ceacc7a5aacd440d37c3476e7dc89b797
```

Inference weights contain the trained generator and are intended for
`smartpet-infer` and `smartpet-infer-batch`.

### Fine-tuning checkpoint

```text
checkpoints/
└── smartpet_g001_parent_v0.3.1_full_checkpoint.pt
```

The full parent checkpoint is the supported initialization artifact for
fine-tuning.

SHA-256:

```text
2c974d4196e4514e5a0b877923d6b9b0a0c35ad4b447d06cd73d1bbc7abb8dee
```

It contains both trained networks and the training-checkpoint metadata required
by the SMART-PET fine-tuning contract.

Inference-only weights cannot be substituted for this full checkpoint.

## Reference resources

The reproducibility folder may also contain the exact MNI reference, fixed
whole-brain evaluation mask, SHA-256 manifest, and third-party notices used by
the release.

Use the same MNI reference for preprocessing, training, inference, and
evaluation.

Third-party template material remains governed by its original licensing and
is not relicensed as SMART-PET source code.

## Auditing downloaded models

Audit inference-only weights with:

```bash
smartpet-audit-weights \
  --weights /path/to/smartpet_g001_parent_v0.3.1.pt \
  --expected-sha256 f26b89db433368167bb67242d0ed2e5351651a2155a92f41f6fce991649f91b0
```

Audit the full fine-tuning checkpoint with:

```bash
smartpet-audit-checkpoint \
  --checkpoint /path/to/smartpet_g001_parent_v0.3.1_full_checkpoint.pt
```

For a downloaded file, independently checking SHA-256 before use is strongly
recommended.

## Artifact lifecycle

Inference-only weights are exported from a full SMART-PET training checkpoint
with:

```bash
smartpet-export-weights \
  --checkpoint /trusted/checkpoints/best.pt \
  --output /assets/weights/model_inference.pt \
  --json-output /assets/weights/model_inference.export.json
```

The exported model must then pass `smartpet-audit-weights`.

Full training checkpoints and inference-only weights serve different purposes:

```text
inference-only weights
    → smartpet-infer / smartpet-infer-batch

full parent checkpoint
    → smartpet-train --init-checkpoint

checkpoint produced by the same interrupted run
    → smartpet-train --resume
```

## Scientific scope

The original SMART-PET paper and the current software release should not be
treated as identical implementations. The repository documents the software
changes explicitly in `docs/CHANGES_FROM_PAPER.md`.

The external-adapted model is domain specific. The parent model remains the
recommended general pretrained model.

## Integrity, privacy, and licensing

No patient data are distributed with the public assets.

SMART-PET model artifacts are released under the repository's
CC BY-NC-SA 4.0 terms. Commercial use is prohibited.

The SHA-256 values distributed with the release are the integrity reference for
downloaded model files.
