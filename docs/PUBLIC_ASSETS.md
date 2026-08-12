# Public assets and integrity

SMART-PET source code does not bundle model checkpoints or third-party MNI template files. The public reproducibility assets are distributed through the Google Drive folder linked from the repository README.

The folder display name is not an integrity identifier. Use filenames plus the SHA-256 values pinned in this repository.

## Canonical v0.3.1 assets

| Relative path | Purpose | SHA-256 |
|---|---|---|
| `templates/csymT.nii.gz` | MNI reference used by preprocessing, training, inference, and auditing | `d28d312d3c895c226dbd61947b77691c6d850396c035015399bd4cfdeed4c291` |
| `templates/MNI152_T1_1mm_brain_mask.nii.gz` | Fixed whole-brain evaluation mask | `274b41c4cf787ada4ce683524301ee052d1ef64b208569c05ce7e9c00717404e` |
| `weights/smartpet_g001_parent_v0.3.1.pt` | Recommended general inference model | `f26b89db433368167bb67242d0ed2e5351651a2155a92f41f6fce991649f91b0` |
| `weights/smartpet_g001_external_adapted_v0.3.1.pt` | Domain-specific adapted inference model | `aecd3b0c15f0b0b90fc6e2142412562ceacc7a5aacd440d37c3476e7dc89b797` |
| `weights/smartpet_v0.3.0_epoch4_inference.pt` | Historical inference artifact | `ddc79a1940032754f5b719688f6affd2612d3566b9c4f03a0d2e41ce1f5b1d25` |
| `checkpoints/smartpet_g001_parent_v0.3.1_full_checkpoint.pt` | Full parent checkpoint for fine-tuning | `2c974d4196e4514e5a0b877923d6b9b0a0c35ad4b447d06cd73d1bbc7abb8dee` |

The machine-readable checksum list is [`PUBLIC_ASSET_SHA256_v0.3.1.txt`](PUBLIC_ASSET_SHA256_v0.3.1.txt).

## Automated download

Install the optional asset helper:

```bash
python -m pip install '.[assets]'
```

Recommended inference profile:

```bash
smartpet-download-assets --profile inference --output-dir resources
```

Fine-tuning profile:

```bash
smartpet-download-assets --profile finetune --output-dir resources
```

All pinned public model artifacts:

```bash
smartpet-download-assets --profile all --output-dir resources
```

The downloader uses the pinned `gdown==6.1.0` helper because its folder `--json` interface is the contract used to resolve public Drive paths. It first enumerates the public Drive folder, selects the pinned relative paths, downloads only the requested profile, and verifies every downloaded file against the SHA-256 value compiled into this software release. It does not silently download assets during `pip install`.

For an offline/HPC installation, download on an internet-connected machine, copy the `resources/` tree to the cluster, and then verify locally:

```bash
smartpet-download-assets \
  --profile inference \
  --output-dir resources \
  --verify-only
```

## Audit model files

The inference-weight audit verifies both file integrity and the SMART-PET inference artifact contract:

```bash
smartpet-audit-weights \
  --weights resources/weights/smartpet_g001_parent_v0.3.1.pt \
  --expected-sha256 f26b89db433368167bb67242d0ed2e5351651a2155a92f41f6fce991649f91b0
```

The full parent checkpoint is a different artifact type and must be audited with:

```bash
smartpet-audit-checkpoint \
  --checkpoint resources/checkpoints/smartpet_g001_parent_v0.3.1_full_checkpoint.pt
```

## Mutable mirrors

A public Drive folder is convenient but mutable. If a mirror contains an older `SHA256SUMS.txt` or a legacy folder title, do not infer artifact identity from that text alone. The pinned repository manifest above is the release integrity reference.

Before the next public asset refresh, the mirror operator should:

1. rename the folder to match the current asset release;
2. replace the mirror checksum file with the complete pinned checksum list;
3. include the external-activity XLSX metadata template from `examples/`;
4. preserve third-party notices for template resources.
