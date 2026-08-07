# Model card: SMART-PET v0.3.0

## Intended use

Research on paired restoration of low-count brain PET volumes that have already been aligned to a common reference geometry.

## Inputs and outputs

- Input: one scalar 3D PET NIfTI, either SUV or the repository's normalized domain.
- Output: normalized restored PET and/or inverse-transformed SUV PET.

## Baseline architecture

Seven-level 3D encoder-decoder, base channels 32, combined self-attention/similarity/channel-spatial SSAB3D blocks at encoder levels 2 and 3, conditional 3D PatchGAN, LSGAN adversarial MSE, and paired L1.

## v0.3.1 development candidate

The corrected S4 candidate is not yet a validated replacement for the v0.3.0
baseline. It adds explicit corrected similarity attention, two encoder
convolutions per level, the channel-spatial input projection, and discriminator
spectral normalization. Attention placement remains `[2,3]` for the primary
candidate; the reported placement ablation differences were small relative to
their uncertainty and must be retested under the modern training contract.

Patch-based training limits the nominally global attention operator to the
current 128³ patch. Registration into the common reference introduces
interpolation that can alter PET noise texture.

## Validation status

The current baseline was selected on a 103-subject validation split. The primary fixed-mask SUV result for epoch 4 was MAE 0.20008, NRMSE 0.06253, PSNR 31.0768 dB, and SSIM 0.92657. These are validation model-selection results, not independent test performance. Complete-grid PSNR was higher because background voxels were included and is retained only as a secondary engineering metric.

## Limitations

- No independent test split was available for the current baseline.
- Training validation used a deterministic center patch, while final evaluation used full-volume inference.
- Cross-scanner, cross-tracer, lesion-detection, and prospective clinical performance are not established.
- The software is not a medical device and must not be used for clinical decisions without appropriate validation and regulatory review.
