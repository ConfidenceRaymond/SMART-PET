# Data contract

SMART-PET v0.3.0 expects **preprocessed scalar 3D NIfTI volumes**. Training and validation CSV files contain exactly:

```csv
subject_id,source_path,target_path
```

Source and target volumes must be different files with the same shape, affine, voxel spacing, and canonical orientation as the supplied MNI reference. Model-facing volumes are expected in the non-negative normalized domain:

```text
normalized = asinh(max(SUV, 0) / asinh_scale)
```

Use `smartpet-validate-manifest` on both splits before allocating a GPU. Training and validation subject IDs must be patient-disjoint; the CLI validates each manifest internally, while split-level overlap must also be checked during dataset preparation.

The repository deliberately excludes patient data, institution-specific manifests, raw-list-mode processing, and registration pipelines.
