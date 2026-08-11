# SMART-PET

**SMART-PET reconstructs standard-dose brain PET from 10% low-dose PET using a self-similarity-aware 3D generative model.**

![SMART-PET qualitative reconstruction examples](assets/smartpet_qualitative_examples.png)

[**Published paper**](https://www.frontiersin.org/journals/nuclear-medicine/articles/10.3389/fnume.2024.1469490/full) ·
[**Software**](https://github.com/ConfidenceRaymond/SMART-PET) ·
[**Model weights**](https://drive.google.com/drive/folders/1XqEI6W30OsrWusMycX0QB8E8DoFURhWh?usp=drive_link)

> **Research software.** SMART-PET is not a clinical diagnostic device.
>
> **License:** CC BY-NC-SA 4.0. Commercial use is prohibited.

## Overview

SMART-PET is a 3D PET restoration framework for recovering standard-dose brain PET from low-dose PET.

The current repository extends the method described in the 2024 SMART-PET paper with a reproducible PET preprocessing pathway, MNI-space patch training, whole-volume reconstruction, multi-GPU training, explicit fine-tuning, model auditing, and quantitative evaluation.

The network receives PET images only. Clinical and acquisition metadata such as body weight, injected activity, age, sex, scanner, and tracer are not model inputs. Some of these fields are used during preprocessing when SUV conversion is required.

The usual workflow is:

```text
PET images
    ↓
preprocess to MNI SUV / normalized MNI SUV
    ↓
train + validation CSV files
    ↓
validate data
    ↓
pretrained inference
       or
training / fine-tuning
    ↓
whole-volume prediction
    ↓
evaluation
```

## Quick start

### 1. Install

Install a CUDA-enabled PyTorch build appropriate for your system, then:

```bash
git clone https://github.com/ConfidenceRaymond/SMART-PET.git
cd SMART-PET

python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e '.[dev]'
pytest
```

SMART-PET does not download model weights automatically.

Download the pretrained models from the
[SMART-PET model folder](https://drive.google.com/drive/folders/1XqEI6W30OsrWusMycX0QB8E8DoFURhWh?usp=drive_link).

### 2. Prepare data

A clean project layout can look like this:

```text
data/
├── raw/
│   ├── sub-001/
│   │   ├── lowdose.nii.gz
│   │   └── standarddose.nii.gz
│   └── sub-002/
│       ├── lowdose.nii.gz
│       └── standarddose.nii.gz
│
├── metadata.csv
└── prepared/
```

This layout is recommended, not required.

A flat folder is also supported:

```text
data/
├── sub-001_lowdose.nii.gz
├── sub-001_standarddose.nii.gz
├── sub-002_lowdose.nii.gz
├── sub-002_standarddose.nii.gz
└── metadata.csv
```

SMART-PET uses the image paths in the CSV, so the images do not have to be reorganized into a particular folder hierarchy.

Four preprocessing entry points are supported:

| Input kind | Starting data | Processing performed |
|---|---|---|
| `raw_activity` | Calibrated activity PET outside MNI space | MNI registration → SUVbw → normalization |
| `mni_activity` | Calibrated activity PET already in MNI space | SUVbw → normalization |
| `mni_suv` | SUV PET already in MNI space | normalization |
| `mni_suv_normalized` | SMART-PET normalized MNI SUV | validation and staging |

If the images are already MNI SUV, the preprocessing CSV can be as simple as:

```csv
subject_id,source_image_path,target_image_path
sub-001,raw/sub-001/lowdose.nii.gz,raw/sub-001/standarddose.nii.gz
sub-002,raw/sub-002/lowdose.nii.gz,raw/sub-002/standarddose.nii.gz
```

Run:

```bash
smartpet-prepare-external \
  --metadata-csv data/metadata.csv \
  --data-root data \
  --output-root data/prepared \
  --mni-reference /path/to/mni_reference.nii.gz \
  --input-kind mni_suv
```

SMART-PET writes prepared MNI SUV images, normalized images, preprocessing QC, and a model-ready paired manifest:

```text
data/prepared/manifests/pairs_normalized.csv
```

Starting from calibrated activity PET requires additional SUV metadata such as body weight and administered activity. See [Data preparation](docs/DATA_PREPARATION.md).

### 3. Make CSV files

Training and validation use three columns:

```csv
subject_id,source_path,target_path
sub-001,/data/prepared/normalized/source/sub-001_source_norm.nii.gz,/data/prepared/normalized/target/sub-001_target_norm.nii.gz
sub-002,/data/prepared/normalized/source/sub-002_source_norm.nii.gz,/data/prepared/normalized/target/sub-002_target_norm.nii.gz
```

If you use `smartpet-prepare-external`, this format is generated automatically as `pairs_normalized.csv`.

Split the subjects into:

```text
data/train.csv
data/validation.csv
```

Keep all scans belonging to the same participant in the same split.

Users who already have prepared normalized MNI PET can create the model-facing CSV files directly.

Templates are available under [`examples/`](examples/).

### 4. Validate

Validate the image geometry, intensity domain, and train/validation separation before training:

```bash
smartpet-validate-manifest \
  --manifest data/train.csv \
  --other-manifest data/validation.csv \
  --mni-reference /path/to/mni_reference.nii.gz
```

The validator rejects incompatible MNI geometry and subject overlap between the two splits.

### 5. Run the pretrained model

For general pretrained inference, use:

```text
smartpet_g001_parent_v0.3.1.pt
```

Download it from the
[SMART-PET model folder](https://drive.google.com/drive/folders/1XqEI6W30OsrWusMycX0QB8E8DoFURhWh?usp=drive_link).

Run whole-volume inference from MNI SUV:

```bash
mkdir -p outputs

smartpet-infer \
  --checkpoint /path/to/smartpet_g001_parent_v0.3.1.pt \
  --input /path/to/lowdose_mni_suv.nii.gz \
  --input-domain suv \
  --mni-reference /path/to/mni_reference.nii.gz \
  --suv-output outputs/prediction_suv.nii.gz \
  --normalized-output outputs/prediction_normalized.nii.gz \
  --metadata-json outputs/prediction.json
```

Use `--input-domain normalized` when the input already contains SMART-PET normalized MNI SUV.

Whole-volume inference uses overlapping 128 × 128 × 128 patches and blends them into one output volume.

## Train from scratch

The reference training configuration is:

```text
configs/train_from_scratch.json
```

Single GPU:

```bash
smartpet-train \
  --config configs/train_from_scratch.json \
  --train-csv data/train.csv \
  --val-csv data/validation.csv \
  --mni-reference /path/to/mni_reference.nii.gz \
  --out-dir runs/from_scratch \
  --backend single
```

Multiple GPUs on one node:

```bash
torchrun --standalone --nproc-per-node=2 -m smartpet.cli.train \
  --config configs/train_from_scratch.json \
  --train-csv data/train.csv \
  --val-csv data/validation.csv \
  --mni-reference /path/to/mni_reference.nii.gz \
  --out-dir runs/from_scratch \
  --backend ddp
```

`batch_size` is specified per process.

See [Training](docs/TRAINING.md) for configuration details and exact checkpoint continuation.

## Fine-tune on your data

Fine-tuning starts a new run from the **full G0.01-parent training checkpoint**:

```text
smartpet_g001_parent_v0.3.1_full_checkpoint.pt
```

Download it from the [SMART-PET model folder](https://drive.google.com/drive/folders/1XqEI6W30OsrWusMycX0QB8E8DoFURhWh?usp=drive_link), then run:

```bash
smartpet-train \
  --config configs/finetune.json \
  --init-checkpoint /path/to/smartpet_g001_parent_v0.3.1_full_checkpoint.pt \
  --train-csv data/train.csv \
  --val-csv data/validation.csv \
  --mni-reference /path/to/mni_reference.nii.gz \
  --out-dir runs/finetuned
```

Fine-tuning resets optimizer, scheduler, progress, and RNG state while initializing both trained networks from the parent checkpoint.

The inference-only `.pt` files in the `weights/` folder are intended for inference and cannot be used with `--init-checkpoint`.

The full parent checkpoint SHA-256 is:

```text
2c974d4196e4514e5a0b877923d6b9b0a0c35ad4b447d06cd73d1bbc7abb8dee
```

`configs/finetune.json` is an example configuration, not a claim that one learning rate is optimal for every dataset.

See [Fine-tuning](docs/FINETUNING.md).

## Evaluate

Evaluate a paired prediction and target within a fixed brain mask:

```bash
smartpet-evaluate \
  --prediction outputs/prediction_suv.nii.gz \
  --target /path/to/standarddose_mni_suv.nii.gz \
  --brain-mask /path/to/brain_mask.nii.gz \
  --mni-reference /path/to/mni_reference.nii.gz \
  --prediction-domain suv \
  --target-domain suv \
  --output-json outputs/metrics.json
```

See [Evaluation](docs/EVALUATION.md) for metric definitions and the evaluation contract.

## Pretrained models

Public model files are available in the
[SMART-PET model folder](https://drive.google.com/drive/folders/1XqEI6W30OsrWusMycX0QB8E8DoFURhWh?usp=drive_link).

| Artifact | Intended use |
|---|---|
| `smartpet_g001_parent_v0.3.1.pt` | **Recommended general pretrained inference model** |
| `smartpet_g001_external_adapted_v0.3.1.pt` | Domain-specific adapted inference model |
| `smartpet_g001_parent_v0.3.1_full_checkpoint.pt` | Full parent checkpoint for fine-tuning |
| `smartpet_v0.3.0_epoch4_inference.pt` | Historical v0.3.0 inference checkpoint |

The recommended parent-model SHA-256 is:

```text
f26b89db433368167bb67242d0ed2e5351651a2155a92f41f6fce991649f91b0
```

The external-adapted model SHA-256 is:

```text
aecd3b0c15f0b0b90fc6e2142412562ceacc7a5aacd440d37c3476e7dc89b797
```

Use the parent model for general inference. An adapted model should be used only when its target domain supports that choice.

See [Model provenance](docs/MODEL_PROVENANCE.md).

## Expected data layout

A complete user workspace might look like:

```text
project/
├── data/
│   ├── raw/
│   ├── prepared/
│   ├── metadata.csv
│   ├── train.csv
│   └── validation.csv
│
├── models/
│   └── smartpet_g001_parent_v0.3.1.pt
│
├── outputs/
└── runs/
```

This is a recommended organization only.

SMART-PET follows paths stored in CSV files and does not require a fixed filesystem layout.

## What changed since the 2024 paper

The 2024 article describes the original SMART-PET study. The current repository preserves that method lineage but adds a substantially more reproducible software and data workflow.

Major additions include reproducible MNI/SUV preprocessing, reversible asinh normalization, overlapping 3D patch training, whole-volume reconstruction, single- and multi-GPU training, state-complete checkpoint continuation, explicit fine-tuning, batch inference, inference-weight export and auditing, and fixed-mask quantitative evaluation.

The repository also contains explicit historical and corrected architecture modes used to document differences from the published implementation.

See [Changes from the 2024 paper](docs/CHANGES_FROM_PAPER.md).

## Documentation

Start here:

- [Data preparation](docs/DATA_PREPARATION.md)
- [Training](docs/TRAINING.md)
- [Fine-tuning](docs/FINETUNING.md)
- [Inference](docs/INFERENCE.md)
- [Evaluation](docs/EVALUATION.md)
- [Model card](docs/MODEL_CARD.md)
- [Model provenance](docs/MODEL_PROVENANCE.md)
- [Changes from the 2024 paper](docs/CHANGES_FROM_PAPER.md)

Advanced checkpoint, historical-conformance, architecture, and release documentation remains under [`docs/`](docs/).

## Citation

If you use SMART-PET, please cite the published method:

**SMART-PET**, *Frontiers in Nuclear Medicine* (2024)

[Read the published article](https://www.frontiersin.org/journals/nuclear-medicine/articles/10.3389/fnume.2024.1469490/full)

Repository citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License

SMART-PET is released under the **CC BY-NC-SA 4.0** license.

Commercial use is prohibited. See [`LICENSE`](LICENSE) for the complete terms.
