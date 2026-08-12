# Changelog

## 0.3.2

- Added a one-command isolated environment setup path and an Alliance-specific activation helper.
- Added pinned public-asset download/verification profiles and a complete repository checksum manifest.
- Named and documented the exact public MNI reference and fixed brain mask throughout the user workflow.
- Changed normal-user installation guidance from editable to non-editable installation.
- Added direct XLSX/XLSM external metadata support through the optional `excel` dependency.
- Added explicit Excel-sheet selection and safe default selection of the bundled `Raw_Activity_Template` sheet.
- Added a public Excel metadata template designed so explanatory notes cannot export as fake subject rows.
- Expanded `smartpet-prepare-external --help` with required fields, accepted units, count-scaling modes, decay semantics, and 3D requirements.
- Added explicit `decay_reference=NONE` support for calibrated frame-average activity images with no decay correction, using documented frame-average decay correction to administration time.
- Added explicit dynamic-4D rejection guidance and clearer scalar-3D error messages.
- Added human-readable registration transform labels alongside ANTs transform codes.
- Hardened Alliance module isolation by removing module-provided Python prefix paths before SMART-PET Python execution.
- Updated public Narval launchers to use the repository-local `.venv` by default instead of a user-specific legacy environment, and made the primary training/fine-tuning configs point to the public MNI resource path.
- Corrected stale v0.3.0/development-candidate wording across the data, model-card, release, and reproducibility documentation.
- Documented the clean-room external-user validation scope and limitations.
- Hardened the release-development dependency contract so `setuptools>=69` and `wheel` are installed and preflighted before no-build-isolation validation.

## 0.3.1

- Released the G0.01-parent architecture/training contract and public inference/fine-tuning artifacts.
- Added shared target-estimated paired PET registration for external native-space activity data.
- Added canonical MNI orientation handling and release-grade checkpoint/inference-weight auditing.
- Added full-state DDP continuation, fine-tuning initialization, and public model provenance.

## 0.3.0

- Licensed source, documentation, and distributed model weights under CC BY-NC-SA 4.0; commercial use is prohibited.
- Added strict JSON configuration with command-line overrides.
- Added configurable attention levels and checkpoint recording.
- Separated exact resume from weight-initialized fine-tuning.
- Added reusable inference engine and batch-inference CSV workflow.
- Added fixed-mask SUV evaluation CLI.
- Added portable single-GPU, DDP, and Narval launch examples.
- Removed institution-specific preprocessing, manifests, logs, caches, and temporary bundles from the public source tree.
- Documented differences from the 2024 published implementation.
