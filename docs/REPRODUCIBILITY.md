# Reproducibility assets

The controlled SMART-PET v0.3.0 asset folder is currently available to
authorized reviewers through the private review mirror linked from the README.
The mirror is temporary and is not the final citable release location.

## Intended asset contents

- `weights/smartpet_v0.3.0_epoch4_inference.pt`
- `templates/csymT.nii.gz`
- `templates/MNI152_T1_1mm_brain_mask.nii.gz`
- `SHA256SUMS.txt`
- `THIRD_PARTY_NOTICES.txt`
- asset `README.md`

## Artifact lifecycle

The model file is an inference-only generator artifact. It must be regenerated
from its full epoch-4 training checkpoint using the version-controlled command:

```bash
smartpet-export-weights \
  --checkpoint /trusted/checkpoints/epoch_0004_complete.pt \
  --output /assets/weights/smartpet_v0.3.0_epoch4_inference.pt \
  --json-output /assets/weights/smartpet_v0.3.0_epoch4_inference.export.json
```

Then audit it and record the printed digest in both the archive's
`SHA256SUMS.txt` and the Git-tracked release manifest:

```bash
smartpet-audit-weights \
  --weights /assets/weights/smartpet_v0.3.0_epoch4_inference.pt \
  --expected-sha256 <SHA256>
```

The artifact can be used with `smartpet-infer` and `smartpet-infer-batch`. It is
not an exact-resume or fine-tuning checkpoint.

The exact `csymT.nii.gz` reference must be supplied to `--mni-reference`. The
fixed whole-brain mask is used with `smartpet-evaluate`.

```bash
smartpet-infer \
  --checkpoint /assets/weights/smartpet_v0.3.0_epoch4_inference.pt \
  --input /data/lowdose_suv.nii.gz \
  --input-domain suv \
  --mni-reference /assets/templates/csymT.nii.gz \
  --suv-output /outputs/restored_suv.nii.gz
```

## Scientific scope

The reported quantitative results come from a 103-subject validation cohort
used for model selection. They are not independent held-out test results.

## Integrity and licensing

The authoritative public release must use an immutable citable archive and a
SHA-256 manifest committed to this repository. The current restricted mirror is
not sufficient as the final integrity record.

SMART-PET model weights are licensed under CC BY-NC-SA 4.0. Commercial use is
prohibited. MNI/FSL template material remains governed by its original
third-party notices and is not relicensed as SMART-PET source code.

No patient data are distributed with these assets.
