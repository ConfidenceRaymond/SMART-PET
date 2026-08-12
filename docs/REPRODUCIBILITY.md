# Reproducibility assets

SMART-PET model and reference assets are distributed separately from source code. The public folder is linked from the repository README.

The authoritative release checksum list is maintained in the source repository:

```text
docs/PUBLIC_ASSET_SHA256_v0.3.1.txt
```

See [Public assets and integrity](PUBLIC_ASSETS.md) for automated download and verification.

## Public model artifacts

### Inference weights

```text
weights/
├── smartpet_g001_parent_v0.3.1.pt
├── smartpet_g001_external_adapted_v0.3.1.pt
└── smartpet_v0.3.0_epoch4_inference.pt
```

`smartpet_g001_parent_v0.3.1.pt` is the recommended general pretrained inference model.

`smartpet_g001_external_adapted_v0.3.1.pt` is domain specific and is not a universal replacement for the parent model.

`smartpet_v0.3.0_epoch4_inference.pt` is retained for historical reproducibility.

### Fine-tuning checkpoint

```text
checkpoints/
└── smartpet_g001_parent_v0.3.1_full_checkpoint.pt
```

Fine-tuning requires this full training checkpoint because it contains both trained networks and the training-checkpoint metadata required by the SMART-PET initialization contract. Inference-only weights cannot be substituted.

## Reference resources

```text
templates/csymT.nii.gz
templates/MNI152_T1_1mm_brain_mask.nii.gz
```

Use the same `csymT.nii.gz` reference for external preprocessing, training, inference, validation, and auditing. The evaluation mask is fixed independently of model performance.

The template files are third-party resources and remain governed by their original notices.

## Automated asset profiles

```bash
python -m pip install '.[assets]'

smartpet-download-assets --profile inference --output-dir resources
smartpet-download-assets --profile finetune --output-dir resources
smartpet-download-assets --profile all --output-dir resources
```

The downloader verifies each pinned asset after transfer. No model or template file is downloaded implicitly during package installation.

## Model auditing

Inference weights:

```bash
smartpet-audit-weights \
  --weights resources/weights/smartpet_g001_parent_v0.3.1.pt \
  --expected-sha256 f26b89db433368167bb67242d0ed2e5351651a2155a92f41f6fce991649f91b0
```

Full checkpoint:

```bash
smartpet-audit-checkpoint \
  --checkpoint resources/checkpoints/smartpet_g001_parent_v0.3.1_full_checkpoint.pt
```

## Artifact lifecycle

```text
inference-only weights
    → smartpet-infer / smartpet-infer-batch

full parent checkpoint
    → smartpet-train --init-checkpoint

checkpoint produced by the same interrupted run
    → smartpet-train --resume
```

The original SMART-PET paper and the current software release are related but not identical implementations. See `docs/CHANGES_FROM_PAPER.md`.
