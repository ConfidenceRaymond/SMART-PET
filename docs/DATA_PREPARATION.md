# Data preparation contract

## Model-facing manifest

Training and validation consume only:

```csv
subject_id,source_path,target_path
```

Dose, body weight, sex, age, scanner, tracer, and timing fields are preprocessing/QC metadata only. They never enter the SMART-PET network.

## Internal ULDP/RECAP data

The bundled internal manifests select D10 as source and D1 as target. The referenced images are already MNI-registered, converted to SUV, count-scale corrected under the ULDP reconstruction convention, and normalized with `asinh(max(SUV,0)/1.0)`. Do not process them again.

## External input states

| `--input-kind` | Input requirement | Pipeline |
|---|---|---|
| `raw_activity` | calibrated activity-concentration NIfTI outside MNI | ANTs registration → SUVbw → asinh |
| `mni_activity` | calibrated activity-concentration NIfTI in MNI | SUVbw → asinh |
| `mni_suv` | body-weight SUV NIfTI in MNI | asinh |
| `mni_suv_normalized` | already asinh-normalized MNI SUV | validate/stage |

Uncalibrated reconstructed counts, sinograms, list-mode files, or arbitrary scanner intensities cannot be converted to SUV from body weight and dose alone.

## Body-weight SUV

For calibrated activity-concentration images:

```text
SUVbw = activity concentration [Bq/mL] × body weight [g]
        --------------------------------------------------
        effective dose matched to image reference [Bq]
```

Body weight and effective dose are required for `raw_activity` and `mni_activity`. Sex and age are optional cohort/QC fields; they are not used in SUVbw. Height and sex would be needed only for alternative normalizations such as lean-body-mass SUV, which this repository does not calculate.

## External activity CSV

Required activity columns:

| Column | Meaning |
|---|---|
| `subject_id` | unique paired study identifier |
| `source_image_path`, `target_image_path` | absolute paths or paths relative to `--data-root` |
| `weight_kg` | body weight used for SUVbw |
| `source_net_injected_dose_mbq`, `target_net_injected_dose_mbq` | net administered activity at injection after residual correction |
| `source_activity_unit`, `target_activity_unit` | `Bq/mL`, `kBq/mL`, or `MBq/mL` |
| `source_decay_reference`, `target_decay_reference` | `ADMIN` or `START` |
| `source_count_scaling`, `target_count_scaling` | `quantitative` or `count_scaled` |
| `source_count_fraction`, `target_count_fraction` | retained fraction in `(0,1]` |

Optional but required when a decay reference is `START`:

| Column | Meaning |
|---|---|
| `source_injection_datetime`, `target_injection_datetime` | ISO-8601 administration timestamps |
| `source_acquisition_datetime`, `target_acquisition_datetime` | ISO-8601 acquisition-start timestamps |
| `radionuclide_half_life_seconds` | half-life used to decay activity to acquisition start |

Optional QC columns include `sex` and `age_years`.

See `manifests/templates/external_activity_manifest.csv`, `manifests/examples/external_activity_pairs.csv`, and `manifests/examples/udunna_random_30s_chunks.csv`.

## Decay-reference handling

PET voxel values and injected activity must refer to the same time:

- `ADMIN`: image values are decay-corrected to administration time. The net injected activity is used directly.
- `START`: image values are decay-corrected to acquisition start. The pipeline decays the net injected activity from injection time to acquisition start using the supplied half-life.

The QC report records both `dose_at_image_reference_mbq` and the final `suv_denominator_mbq` for source and target.

## Count-decimation handling

Low-count PET can be produced in two materially different ways:

### Quantitatively calibrated

```text
source_count_scaling = quantitative
source_count_fraction = 0.10
```

The lower-count reconstruction remains in calibrated activity concentration. The full decay-matched administered dose remains the SUV denominator. The fraction is retained for provenance only.

### Proportionally count-scaled

```text
source_count_scaling = count_scaled
source_count_fraction = 0.10
```

Voxel values are proportional to retained counts. The pipeline multiplies the decay-matched dose by `0.10` before SUV conversion. This reproduces the historical ULDP D10/D1 convention.

Do not choose `count_scaled` solely because an image is called “10%.” Confirm the reconstruction scaling first.

### Duration-derived count provenance

When the source was produced from selected list-mode frames, provide these optional-but-strongly-recommended CSV fields:

| Column | Meaning |
|---|---|
| `source_sampling_scheme` | `random_noncontiguous`, `contiguous`, `full_window`, or `unknown` |
| `source_chunk_duration_seconds` | duration of each selected source chunk |
| `source_number_of_chunks` | number of selected chunks |
| `source_total_duration_seconds` | total retained source duration |
| `target_total_duration_seconds` | full reference duration |
| `selection_window_start_minutes` | start of the list-mode selection window after injection |
| `selection_window_end_minutes` | end of the list-mode selection window after injection |

When any duration-provenance field is supplied, the pipeline requires enough information to validate the protocol. It enforces:

```text
source_chunk_duration_seconds × source_number_of_chunks
    = source_total_duration_seconds

source_total_duration_seconds / target_total_duration_seconds
    = source_count_fraction

(selection_window_end_minutes − selection_window_start_minutes) × 60
    = target_total_duration_seconds
```

The target defines the full reference duration, so `target_count_fraction` must be `1.0` when these fields are used.

### Udunna 40–60 minute random-chunk protocol

```text
4 random noncontiguous chunks × 30 s = 120 s
full 40–60 min reference             = 20 min = 1,200 s
retained count fraction              = 120 / 1,200 = 0.10
```

Use the same actual net injected activity for source and target. Do not enter 10% of the administered activity as the source injected dose. The software derives the fractional SUV denominator from `source_count_fraction` only when `source_count_scaling=count_scaled`.

Example fields:

```csv
source_count_scaling,target_count_scaling,source_count_fraction,target_count_fraction,source_sampling_scheme,source_chunk_duration_seconds,source_number_of_chunks,source_total_duration_seconds,target_total_duration_seconds,selection_window_start_minutes,selection_window_end_minutes
count_scaled,quantitative,0.10,1.0,random_noncontiguous,30,4,120,1200,40,60
```

Other valid examples for the same 1,200-second reference:

| Selection | Source duration | Fraction |
|---|---:|---:|
| `4 × 30 s` | 120 s | 0.100 |
| `3 × 30 s` | 90 s | 0.075 |
| `2 × 30 s` | 60 s | 0.050 |

See `COUNT_DECIMATION_POLICY.md` for the full decision policy and quantitative-versus-count-scaled checks.

## Outputs

```text
prepared_dataset/
  mni/source/ and mni/target/
  suv/source/ and suv/target/
  normalized/source/ and normalized/target/
  manifests/pairs_normalized.csv
  qc/preprocessing_report.csv
  qc/preprocessing_summary.json
  work/
```

The final model-facing CSV contains only `subject_id,source_path,target_path`.

## Work directory

All registration transforms and temporary files are created below the selected `work/` directory. Successful runs delete their timestamped run workspace by default; failed runs retain it for diagnosis. Use `--keep-work` to retain successful intermediates deliberately.


## NIfTI orientation and MNI geometry

SMART-PET uses a canonical RAS internal NIfTI representation.

The supplied MNI reference does not need to be stored as RAS on disk.
For example, an MNI reference may be stored in LAS orientation. SMART-PET
canonicalizes the reference and every validated MNI-space PET volume using
NiBabel's orientation-aware canonicalization before training, inference, or
downstream preprocessing.

Consequently:

- registration software may write an MNI-aligned image using the original
  reference's storage orientation;
- SMART-PET subsequently canonicalizes that image to RAS;
- SUV and normalized PET outputs produced by SMART-PET are stored using the
  canonical RAS geometry;
- voxel data are reoriented together with the affine; changing only a NIfTI
  header is not permitted.

Geometry comparisons for SMART-PET inputs must therefore be made against the
canonicalized MNI reference rather than against the raw on-disk affine of a
non-canonical reference.

The required contract is spatial equivalence after canonicalization, not
byte-for-byte equality with the original reference header.


## Paired PET registration contract

For `raw_activity` inputs, SMART-PET treats the low-count source and
full-count target as a spatially paired PET acquisition.

The source and target must therefore have the same native NIfTI geometry
before MNI registration:

- identical voxel-grid shape;
- matching physical-space affine within numerical tolerance.

SMART-PET estimates **one** subject-to-MNI transform from the higher-SNR
full-count target image. That same forward transform stack is then applied
to both the target and low-count source.

The low-count source does not estimate an independent nonlinear transform.
This preserves source-target voxel correspondence and prevents differential
registration deformation from being introduced into a supervised training
pair.

For raw-activity preprocessing, a registration provenance JSON is written
for each subject. It records:

- `registration_driver = target`;
- `strategy = target_estimated_shared_transform`;
- transform type;
- interpolation;
- native-pair geometry validation;
- forward transform filenames.

Existing raw-activity MNI outputs that do not contain this shared-transform
provenance contract are not silently reused. Regeneration requires `--force`.

Inputs already supplied in an MNI input domain (`mni_activity`, `mni_suv`,
or `mni_suv_normalized`) are not spatially re-registered by this pathway.
