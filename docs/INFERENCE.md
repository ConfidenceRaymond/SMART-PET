# Whole-volume inference

## One volume

```bash
smartpet-infer \
  --checkpoint /models/best.pt \
  --input /data/lowdose_suv.nii.gz \
  --input-domain suv \
  --mni-reference /data/reference.nii.gz \
  --normalized-output /outputs/prediction_normalized.nii.gz \
  --suv-output /outputs/prediction_suv.nii.gz
```

The checkpoint is loaded once, the input geometry is validated, SUV is normalized, the complete volume is covered with overlapping patches, predictions are combined with a Hann window, and SUV is restored from the shared normalized prediction. The output header and affine follow the canonicalized input.

## Batch inference

```bash
smartpet-infer-batch \
  --manifest examples/inference_manifest.csv \
  --checkpoint /models/best.pt \
  --mni-reference /data/reference.nii.gz \
  --input-domain suv
```

The batch CLI loads the model only once. Every case receives a content-based prediction identifier and JSON provenance. Existing complete outputs are skipped unless `--overwrite` is supplied.

## Architecture metadata

Inference reconstructs the generator from the artifact's recorded architecture.
Format-2 inference weights require all configurable S4 fields. Existing format-1
weights resolve to the frozen v0.3.0 profile. SMART-PET never guesses a partial
architecture configuration.
