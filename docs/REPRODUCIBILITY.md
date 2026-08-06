# Reproducibility assets

The controlled SMART-PET v0.3.0 asset folder is available here:

[Open the SMART-PET reproducibility assets folder](https://drive.google.com/drive/folders/1XqEI6W30OsrWusMycX0QB8E8DoFURhWh?usp=drive_link)

## Included files

- `weights/smartpet_v0.3.0_epoch4_inference.pt`
- `templates/csymT.nii.gz`
- `templates/MNI152_T1_1mm_brain_mask.nii.gz`
- `SHA256SUMS.txt`
- `THIRD_PARTY_NOTICES.txt`
- asset `README.md`

## Use

The model artifact is an inference-only generator checkpoint. It can be used
with `smartpet-infer` and `smartpet-infer-batch`; it is not an exact-resume
training checkpoint.

The exact `csymT.nii.gz` reference must be supplied to `--mni-reference`.
The fixed whole-brain mask is used with `smartpet-evaluate`.

Example:

```bash
smartpet-infer \
  --checkpoint /assets/weights/smartpet_v0.3.0_epoch4_inference.pt \
  --input /data/lowdose_suv.nii.gz \
  --input-domain suv \
  --mni-reference /assets/templates/csymT.nii.gz \
  --suv-output /outputs/restored_suv.nii.gz
```

## Integrity and licensing

Run:

```bash
sha256sum -c SHA256SUMS.txt
```

The SMART-PET model weights are licensed under CC BY-NC-SA 4.0. Commercial
use is prohibited. MNI/FSL template material remains governed by its original
third-party notices and is not relicensed as SMART-PET source code.

No patient data are distributed with these assets.
