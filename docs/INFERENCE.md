# Whole-volume inference

## Input contract

SMART-PET inference accepts one scalar 3D PET NIfTI in either:

- body-weight SUV (`--input-domain suv`), or
- SMART-PET normalized MNI SUV (`--input-domain normalized`).

Dynamic 4D PET is not accepted directly.

The input must already occupy the same physical MNI space as the supplied reference. SMART-PET canonicalizes NIfTI storage orientation to RAS using the affine and then validates canonical shape, affine, and voxel spacing.

This means a physically correct LAS-stored image can be accepted and reoriented to canonical RAS automatically. **This is not anatomical registration.** Native-space images must first use `smartpet-prepare-external --input-kind raw_activity` (or an independently validated equivalent registration workflow).

The public reference is:

```text
resources/templates/csymT.nii.gz
SHA-256: d28d312d3c895c226dbd61947b77691c6d850396c035015399bd4cfdeed4c291
```

## One volume

Recommended parent model:

```bash
smartpet-infer \
  --checkpoint resources/weights/smartpet_g001_parent_v0.3.1.pt \
  --input data/lowdose_mni_suv.nii.gz \
  --input-domain suv \
  --mni-reference resources/templates/csymT.nii.gz \
  --normalized-output outputs/prediction_normalized.nii.gz \
  --suv-output outputs/prediction_suv.nii.gz \
  --metadata-json outputs/prediction.json
```

The checkpoint is loaded once, the input geometry is canonicalized and validated, SUV is normalized, the complete MNI volume is covered by overlapping patches, predictions are blended with a Hann window, and SUV is restored from the shared normalized prediction.

The generated output geometry follows the canonicalized input/reference contract.

## Audit before interpretation

```bash
smartpet-audit-inference \
  --output outputs/prediction_suv.nii.gz \
  --mni-reference resources/templates/csymT.nii.gz \
  --input data/lowdose_mni_suv.nii.gz \
  --target data/standarddose_mni_suv.nii.gz \
  --json-output outputs/prediction_audit.json
```

The normal audit contract requires finite, non-negative output with the expected MNI shape and voxel spacing.

## Batch inference

```bash
smartpet-infer-batch \
  --manifest examples/inference_manifest.csv \
  --checkpoint resources/weights/smartpet_g001_parent_v0.3.1.pt \
  --mni-reference resources/templates/csymT.nii.gz \
  --input-domain suv
```

The batch CLI loads the model only once. Each case receives a content-based prediction identifier and JSON provenance. Existing complete outputs are skipped unless `--overwrite` is supplied.

## Architecture metadata

Inference reconstructs the generator from the inference artifact's recorded architecture. Format-2 inference weights require all configurable S4 fields. Existing format-1 weights resolve to the frozen historical v0.3.0 profile. SMART-PET does not guess a partial architecture configuration.
