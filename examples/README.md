# Example CSV files

These files are templates. Replace each `/path/to/...` value before you run a command.

## Preprocessing

Use `preprocessing_mni_suv.csv` when the source and target are MNI-space SUV images.

```bash
smartpet-prepare-external \
  --metadata-csv examples/preprocessing_mni_suv.csv \
  --output-root /path/to/data/prepared \
  --mni-reference /path/to/mni_reference.nii.gz \
  --input-kind mni_suv
```

Use `preprocessing_raw_activity.csv` for calibrated activity-concentration images outside MNI space.

Raw-activity preprocessing needs body weight, injected activity, activity units, decay references, and count-scaling metadata.

## Training and validation

`train_manifest.csv` and `validation_manifest.csv` use this model-facing format:

```text
subject_id,source_path,target_path
```

The source and target must contain SMART-PET normalized MNI PET.

Use different subject IDs in the training and validation files.

## Batch inference

`inference_manifest.csv` uses this format:

```text
subject_id,input_path,normalized_output,suv_output
```

Each row needs at least one output path.

Use `smartpet-infer-batch --help` for all batch-inference options.
