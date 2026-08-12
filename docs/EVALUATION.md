# Evaluation contract

SMART-PET's public evaluator reports paired image metrics in SUV space inside a **fixed external brain mask**. The mask must have geometry identical to the canonicalized MNI reference and must not be selected using model performance.

The public resources are:

```text
MNI reference:
  resources/templates/csymT.nii.gz
  SHA-256: d28d312d3c895c226dbd61947b77691c6d850396c035015399bd4cfdeed4c291

Brain mask:
  resources/templates/MNI152_T1_1mm_brain_mask.nii.gz
  SHA-256: 274b41c4cf787ada4ce683524301ee052d1ef64b208569c05ce7e9c00717404e
```

Run:

```bash
smartpet-evaluate \
  --prediction prediction_suv.nii.gz \
  --target target_suv.nii.gz \
  --prediction-domain suv \
  --target-domain suv \
  --brain-mask resources/templates/MNI152_T1_1mm_brain_mask.nii.gz \
  --mni-reference resources/templates/csymT.nii.gz \
  --output-json metrics.json
```

Reported metrics include MAE, MSE, RMSE, NMAE, NRMSE, PSNR, SSIM, prediction mean SUV, target mean SUV, and signed mean-SUV bias.

PSNR uses target dynamic range inside the same mask. SSIM uses mask-normalized Gaussian-window local statistics and is averaged only inside the mask.

Complete-grid metrics can be useful as secondary engineering checks, but the public CLI is intentionally a fixed-brain-mask evaluator and complete-grid values should not be compared directly with brain-masked values.

A model should be compared against the unprocessed low-dose baseline under the **same mask, target, domain, and metric implementation**. Improved SSIM alone does not establish improved quantitative SUV fidelity.
