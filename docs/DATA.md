# Data contract

SMART-PET v0.3.2 model-facing data are **scalar 3D NIfTI volumes** in a common MNI geometry.

Training and validation manifests contain exactly:

```csv
subject_id,source_path,target_path
```

Subject identifiers are opaque strings, so zero padding such as `007` is preserved. Whitespace-only cells are rejected. Relative paths are resolved against the manifest directory. Source and target must resolve to different files.

## Model-facing image contract

Source and target must be spatially equivalent to the supplied MNI reference after canonical orientation handling:

- scalar 3D NIfTI;
- identical canonical shape;
- matching canonical affine within tolerance;
- matching voxel spacing;
- finite values;
- model-facing normalized domain:

```text
normalized = asinh(max(SUV, 0) / asinh_scale)
```

The public MNI reference is:

```text
templates/csymT.nii.gz
SHA-256: d28d312d3c895c226dbd61947b77691c6d850396c035015399bd4cfdeed4c291
```

SMART-PET canonicalizes valid NIfTI storage orientation to RAS using the image affine. This can convert a physically correct LAS-stored volume to canonical RAS without changing its anatomical location.

**Canonicalization is not registration.** Native-space PET must be registered through the external preprocessing pathway before model use.

## Dynamic PET

Dynamic 4D PET is not accepted directly by training, inference, or external preprocessing. Construct a scientifically justified scalar 3D image first and document the frame-selection/integration procedure, temporal window, and decay assumptions.

Do not represent arbitrary reconstructed-frame selection or image-space noise injection as true event-level count thinning.

## Validation

Run `smartpet-validate-manifest` before allocating a GPU:

```bash
smartpet-validate-manifest \
  --manifest data/train.csv \
  --other-manifest data/validation.csv \
  --mni-reference resources/templates/csymT.nii.gz
```

Training and validation must be patient-disjoint.

## External preprocessing

The public repository **does include** an external preprocessing pipeline:

- `raw_activity`: ANTs registration + SUVbw + asinh;
- `mni_activity`: SUVbw + asinh;
- `mni_suv`: asinh;
- `mni_suv_normalized`: validation/staging.

It does not include raw list-mode reconstruction or scanner-specific count calibration. Uncalibrated counts, sinograms, list-mode files, or arbitrary scanner intensities cannot be converted to SUV from weight and injected dose alone.

See [Data preparation](DATA_PREPARATION.md).

## Reference resources

The public fixed brain mask is:

```text
templates/MNI152_T1_1mm_brain_mask.nii.gz
SHA-256: 274b41c4cf787ada4ce683524301ee052d1ef64b208569c05ce7e9c00717404e
```

Both resources are distributed through the public reproducibility asset folder and are verified by the repository-pinned checksum manifest. See [Public assets and integrity](PUBLIC_ASSETS.md).
