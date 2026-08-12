# Data preparation contract

## Model-facing manifest

Training and validation consume only:

```csv
subject_id,source_path,target_path
```

Dose, body weight, sex, age, scanner, tracer, count provenance, and timing fields are preprocessing/QC metadata only. They never enter the SMART-PET network.

## External input states

| `--input-kind` | Input requirement | Pipeline |
|---|---|---|
| `raw_activity` | calibrated activity-concentration scalar 3D NIfTI outside MNI | shared target-estimated ANTs registration → SUVbw → asinh |
| `mni_activity` | calibrated activity-concentration scalar 3D NIfTI in MNI | SUVbw → asinh |
| `mni_suv` | body-weight SUV scalar 3D NIfTI in MNI | asinh |
| `mni_suv_normalized` | already asinh-normalized scalar 3D MNI SUV | validate/stage |

Uncalibrated reconstructed counts, sinograms, list-mode files, or arbitrary scanner intensities cannot be converted to SUV from body weight and dose alone.

Dynamic 4D PET must first be converted into a scientifically documented static 3D image. SMART-PET does not choose or combine dynamic frames automatically.

## Metadata files

CSV is supported by the base installation. XLSX/XLSM is supported with:

```bash
python -m pip install '.[excel]'
```

Use:

```bash
smartpet-prepare-external --metadata-file metadata.xlsx ...
```

The bundled workbook uses `Raw_Activity_Template` as its data sheet, and SMART-PET selects that sheet automatically when present. For another multi-sheet workbook, select the intended sheet explicitly with `--metadata-sheet SHEET_NAME`.

The repository provides:

- `examples/external_activity_metadata_template.xlsx` — spreadsheet template with instructions and validation-oriented columns;
- `manifests/templates/external_activity_manifest.csv` — full CSV header template;
- `examples/preprocessing_raw_activity.csv` — compact CSV example.

The spreadsheet's data sheets contain only table rows. Explanatory notes are kept on the Instructions sheet so a normal Excel-to-CSV export cannot create fake subject rows.

## Body-weight SUV

For calibrated activity-concentration images:

```text
SUVbw = activity concentration [Bq/mL] × body weight [g]
        --------------------------------------------------
        effective dose matched to image reference [Bq]
```

Body weight and net administered activity are required for `raw_activity` and `mni_activity`. Sex and age are optional cohort/QC fields and are not used in SUVbw.

## Required activity columns

| Column | Meaning |
|---|---|
| `subject_id` | unique paired study identifier |
| `source_image_path`, `target_image_path` | absolute paths or paths relative to `--data-root` |
| `weight_kg` | body weight used for SUVbw |
| `source_net_injected_dose_mbq`, `target_net_injected_dose_mbq` | actual net administered activity after residual correction |
| `source_activity_unit`, `target_activity_unit` | `Bq/mL`, `kBq/mL`, or `MBq/mL` |
| `source_decay_reference`, `target_decay_reference` | `ADMIN`, `START`, or `NONE` |
| `source_count_scaling`, `target_count_scaling` | `quantitative` or `count_scaled` |
| `source_count_fraction`, `target_count_fraction` | retained fraction in `(0,1]` |

Use the **actual administered activity** for both source and target when both reconstructions come from the same injection. Do not enter 10% of the injected activity merely because the source is called D10.

## Decay-reference handling

PET voxel values and the SUV denominator must refer to the same temporal reference.

### `ADMIN`

The image is already decay-corrected to administration time. The net injected activity at administration is used directly.

No timing fields are required solely for SUV conversion.

### `START`

The image is already decay-corrected to acquisition start. SMART-PET decays the administered activity from injection to acquisition start.

Required:

- `source_injection_datetime` / `target_injection_datetime`;
- `source_acquisition_datetime` / `target_acquisition_datetime`;
- `radionuclide_half_life_seconds`.

### `NONE`

The calibrated image has **no decay correction** and represents the average physical activity concentration over the image acquisition interval.

SMART-PET first scales the voxel values to administration time using the frame-average radioactive-decay factor, then uses the administration-time injected activity for SUVbw.

Required for each `NONE` image:

- injection datetime;
- acquisition-start datetime;
- radionuclide half-life;
- `source_image_duration_seconds` or `target_image_duration_seconds`.

The mathematical contract for an interval `[t0,t1]` after injection is:

```text
mean_decay = [exp(-λ t0) - exp(-λ t1)] / [λ (t1 - t0)]
activity_ADMIN = activity_uncorrected_average / mean_decay
λ = ln(2) / half_life
```

`NONE` must only be used when the reconstruction values genuinely represent a calibrated frame-average activity concentration with no decay correction. If scanner/reconstruction scaling is unknown, do not guess.

The preprocessing QC report records the applied activity correction factor and the final SUV denominator.

## Count-scaling handling

Low-count PET can be produced under different reconstruction scaling conventions.

### Quantitatively calibrated

```text
source_count_scaling = quantitative
source_count_fraction = 0.10
```

The lower-count reconstruction remains in calibrated activity concentration. The full decay-matched administered dose remains the SUV denominator. `source_count_fraction` is provenance only.

### Proportionally count-scaled

```text
source_count_scaling = count_scaled
source_count_fraction = 0.10
```

Voxel values are proportional to retained counts. SMART-PET multiplies the decay-matched dose by the retained fraction before SUV conversion.

Do not choose `count_scaled` solely because a file is named “10%”. Confirm reconstruction scaling first.

## Optional count-protocol provenance

When the source was constructed from selected list-mode intervals, these fields can validate duration-derived fractions:

| Column | Meaning |
|---|---|
| `source_sampling_scheme` | `random_noncontiguous`, `contiguous`, `full_window`, or `unknown` |
| `source_chunk_duration_seconds` | duration of each selected source chunk |
| `source_number_of_chunks` | number of selected chunks |
| `source_total_duration_seconds` | total retained source duration |
| `target_total_duration_seconds` | full reference duration |
| `selection_window_start_minutes` | selection-window start |
| `selection_window_end_minutes` | selection-window end |

When any of these fields are supplied, the software validates the internally implied duration fraction. These fields describe **count-selection provenance** and are separate from `source_image_duration_seconds` / `target_image_duration_seconds`, which describe the temporal integration interval needed for `decay_reference=NONE`.

A 10% example over a 20-minute target window is:

```text
4 random noncontiguous chunks × 30 s = 120 s
full reference window                 = 1,200 s
retained fraction                     = 0.10
```

Example CSV fields:

```csv
source_count_scaling,target_count_scaling,source_count_fraction,target_count_fraction,source_sampling_scheme,source_chunk_duration_seconds,source_number_of_chunks,source_total_duration_seconds,target_total_duration_seconds,selection_window_start_minutes,selection_window_end_minutes
count_scaled,quantitative,0.10,1.0,random_noncontiguous,30,4,120,1200,40,60
```

## Paired native-space registration

For `raw_activity`, source and target must have identical native shape and physical-space affine and both must be scalar 3D images.

SMART-PET estimates **one** target-to-MNI transform from the higher-SNR target and applies the same forward transform stack to both target and source. The source never estimates an independent nonlinear warp.

The default `--transform-type s` means **SyN**. Registration provenance records both the ANTs code and a human-readable transform label.

Required external commands:

```text
antsRegistrationSyNQuick.sh
antsApplyTransforms
```

Alliance/Narval:

```bash
module load ants/2.6.5
```

Other systems should follow the official ANTsX installation instructions. See [External preprocessing environment](PREPROCESSING_ENVIRONMENT.md).

## NIfTI orientation and canonical MNI geometry

SMART-PET uses canonical RAS internally. The MNI reference does not need to be stored as RAS on disk. For example, public `csymT.nii.gz` is stored as LAS; SMART-PET reorients voxel data and affine together to canonical RAS.

Geometry checks are made against the canonicalized MNI reference. Do not edit only NIfTI headers to force an orientation label.

## Outputs

```text
prepared_dataset/
  mni/source/ and mni/target/
  mni/registration/<subject>_shared_transform.json
  suv/source/ and suv/target/
  normalized/source/ and normalized/target/
  manifests/pairs_normalized.csv
  qc/preprocessing_report.csv
  qc/preprocessing_summary.json
  work/
```

Successful timestamped workspaces are removed by default. Failed runs retain their workspace for diagnosis. Use `--keep-work` to retain successful intermediates deliberately.
