# Example metadata and manifest files

These files are templates. Replace example paths and metadata before running a command.

## External preprocessing

### MNI SUV

Use `preprocessing_mni_suv.csv` when source and target are already MNI-space SUV images:

```bash
smartpet-prepare-external \
  --metadata-file examples/preprocessing_mni_suv.csv \
  --output-root data/prepared \
  --mni-reference resources/templates/csymT.nii.gz \
  --input-kind mni_suv
```

### Calibrated activity PET

Use either:

```text
external_activity_metadata_template.xlsx
preprocessing_raw_activity.csv
```

The XLSX template requires the optional Excel dependency:

```bash
python -m pip install '.[excel]'
```

The bundled workbook uses `Raw_Activity_Template` as the default data sheet. Use `--metadata-sheet SHEET_NAME` for another sheet in a multi-sheet workbook.

`preprocessing_raw_activity.csv` shows the compact `ADMIN` + `quantitative` case.

`preprocessing_raw_activity_none.csv` shows calibrated activity with **no decay correction**. For `NONE`, acquisition time is the beginning of the image integration interval and `*_image_duration_seconds` is required so SMART-PET can correct frame-average activity to administration time before SUVbw.

Raw/native activity preprocessing also requires ANTs. MNI-domain inputs do not.

Run `smartpet-prepare-external --help` for the complete column contract.

## Training and validation

`train_manifest.csv` and `validation_manifest.csv` use:

```text
subject_id,source_path,target_path
```

The source and target must contain SMART-PET normalized MNI PET. Keep subjects patient-disjoint across splits.

## Batch inference

`inference_manifest.csv` uses:

```text
subject_id,input_path,normalized_output,suv_output
```

Each row needs at least one output path.
