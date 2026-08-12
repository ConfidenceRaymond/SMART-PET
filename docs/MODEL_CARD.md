# Model card: SMART-PET v0.3.1 G0.01-parent

## Model role

**G0.01-parent** is the recommended general pretrained SMART-PET inference model distributed with the v0.3.1 reproducibility assets.

Inference artifact:

```text
smartpet_g001_parent_v0.3.1.pt
SHA-256: f26b89db433368167bb67242d0ed2e5351651a2155a92f41f6fce991649f91b0
```

Full fine-tuning checkpoint:

```text
smartpet_g001_parent_v0.3.1_full_checkpoint.pt
SHA-256: 2c974d4196e4514e5a0b877923d6b9b0a0c35ad4b447d06cd73d1bbc7abb8dee
```

## Intended use

Research on paired restoration of low-count brain PET volumes that have been converted to the SMART-PET MNI/SUV contract.

The model is not a medical device and is not validated for unsupervised clinical decision-making.

## Inputs and outputs

- Input: one scalar 3D brain PET NIfTI in MNI space, either SUVbw or SMART-PET normalized SUV.
- Output: restored normalized PET and/or inverse-transformed SUV PET.
- Dynamic 4D PET is not a direct model input.
- Clinical/acquisition metadata are not network inputs.

## Released architecture and optimization contract

The released G0.01-parent configuration uses:

| Field | Value |
|---|---|
| base channels | 32 |
| attention levels | `[2,3]` |
| similarity mode | `scale_consistent` |
| encoder convolutions per level | 2 |
| channel-spatial input projection | enabled |
| generator spectral normalization | disabled |
| discriminator spectral normalization | enabled |
| output mode | `positive_softplus_residual` |
| training patch | 128 × 128 × 128 |
| L1 weight | 100 |
| GAN weight | 0.01 |
| GAN objective | LSGAN |
| initial learning rate | 1e-4 |
| decay start epoch | 35 |
| training epochs | 100 |

The architecture is reconstructed from versioned checkpoint/weight metadata rather than inferred heuristically.

## Preprocessing contract

Model-facing geometry is canonical MNI RAS. NIfTI storage orientation is canonicalized from the affine; this is distinct from anatomical registration.

External native-space activity images should use the shared target-estimated registration pathway documented in `docs/DATA_PREPARATION.md`.

SUV normalization is:

```text
normalized = asinh(max(SUV, 0) / asinh_scale)
```

The released parent uses `asinh_scale=1.0`.

## Validation status

The parent model is the general pretrained reference. The domain-specific external-adapted artifact is intentionally separate and should only be used when the target domain supports that choice.

The clean-room external-user exercise validated the public preprocessing, inference, auditing, and evaluation software pathway but does not constitute a new cohort-level performance claim. See `docs/EXTERNAL_USER_VALIDATION.md`.

## Limitations

- Cross-scanner, cross-tracer, lesion-detection, pediatric, and prospective clinical performance are not established by the software release itself.
- Registration to a common reference introduces interpolation and can alter PET noise texture.
- Patch-based training/inference limits attention to the current 128³ patch.
- Performance must be re-evaluated on each new domain with a predefined, patient-independent evaluation protocol.
- The software must not be used for clinical decisions without appropriate validation and regulatory review.
