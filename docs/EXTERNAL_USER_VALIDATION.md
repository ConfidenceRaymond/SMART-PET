# External-user clean-room validation

SMART-PET v0.3.1 was exercised from a fresh public clone in an isolated environment without using the development repository, private manifests, private checkpoints, or private MNI paths.

## What was validated

The clean-room workflow verified:

- package installation and all public CLI entry points;
- public G0.01-parent inference-weight checksum and artifact audit;
- public `csymT.nii.gz` MNI-reference checksum and geometry;
- public fixed brain-mask checksum;
- raw calibrated activity preprocessing using ANTs 2.6.5;
- one target-estimated shared transform applied to both paired PET images;
- MNI canonicalization from storage orientation to canonical RAS;
- exact voxelwise SUVbw conversion after canonicalization;
- exact voxelwise `asinh(SUV)` normalization;
- whole-volume G0.01-parent inference on an A100 GPU;
- public inference audit for geometry, finite values, and non-negative output;
- public fixed-brain-mask evaluation.

This exercise validates the software pathway and its contracts. A single clean-room case is **not** evidence of cross-domain clinical performance or a replacement for cohort-level external validation.

## Validated Narval execution environment

The clean-room GPU path completed with:

- Python 3.11.4;
- PyTorch 2.6.0+computecanada;
- CUDA runtime 12.2;
- NVIDIA A100-SXM4-40GB;
- ANTs 2.6.5 for native-space registration.

The preprocessing constraint profile retained in `requirements/preprocessing-tested.txt` pins the Python packages used in the dedicated preprocessing validation environment.

## Issues discovered and corrected

The clean-room pass identified user-facing gaps that are addressed in the current correction set:

- isolated environment creation was not prominent in the README;
- normal users were shown an editable install;
- tested PyTorch/CUDA/ANTs information was not surfaced clearly;
- the actual public MNI-reference and brain-mask filenames/hashes were not named consistently;
- public asset download and integrity checking were manual;
- the mutable Drive checksum file could be incomplete relative to v0.3.1 assets;
- activity metadata help did not list required columns, units, or decay semantics;
- XLSX metadata input was not supported directly;
- there was no public spreadsheet template designed for safe Excel-to-table use;
- scalar 3D requirements and 4D dynamic-PET handling needed clearer guidance;
- calibrated activity with no decay correction could not be represented truthfully;
- ANTs acquisition/setup was underspecified for external users;
- Alliance ANTs modules could expose system Python packages into a virtual environment;
- registration provenance used the terse transform code `s` without a human-readable `SyN` label;
- several documentation pages retained stale v0.3.0 or development-candidate wording.

The complete finding-to-correction traceability table is retained in [`EXTERNAL_USER_CORRECTION_LOG.md`](EXTERNAL_USER_CORRECTION_LOG.md).

The source release remains research software and must still be validated on each new deployment environment and dataset.
