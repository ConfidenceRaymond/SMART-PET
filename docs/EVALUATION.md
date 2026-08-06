# Evaluation contract

Primary reported validation metrics for the v0.2.6 baseline were calculated in SUV space after inverse asinh normalization and inside a fixed external whole-brain mask. The mask must have geometry identical to the MNI reference and must not be selected using model performance.

```bash
smartpet-evaluate \
  --prediction prediction_suv.nii.gz \
  --target target_suv.nii.gz \
  --prediction-domain suv \
  --target-domain suv \
  --brain-mask brain_mask.nii.gz \
  --mni-reference reference.nii.gz \
  --output-json metrics.json
```

PSNR uses both MSE and target dynamic range inside the same mask. SSIM uses mask-normalized Gaussian-window local statistics and is averaged only inside the mask. Complete-grid metrics may be retained as secondary engineering metrics but should not be compared directly with brain-masked metrics.
