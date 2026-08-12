# External-user correction log

This log preserves the user-facing findings from the post-release v0.3.1 clean-room exercise and maps each finding to the current correction set. It is an engineering traceability record, not a clinical-performance claim.

| ID | Clean-room finding | Correction status |
|---|---|---|
| UX-001 | Install guidance did not start from an isolated environment or verify the resulting installation. | **Corrected in source:** `scripts/setup_environment.sh`, README quick start. |
| UX-002 | README used a generic MNI-reference placeholder instead of naming the public reference. | **Corrected in source:** `resources/templates/csymT.nii.gz` is named throughout user-facing workflows. |
| UX-003 | Tested PyTorch/CUDA configuration was not surfaced. | **Corrected in docs:** validated Narval Python/PyTorch/CUDA/A100 environment is recorded. |
| UX-004 | The preprocessing-environment document was not surfaced from installation guidance. | **Corrected in README.** |
| UX-005 | Preprocessing documentation implied the requirements file pinned the Python interpreter. | **Corrected:** Python version is documented separately; the constraints file is explicitly Python-package-only. |
| UX-006 | No one-command environment/bootstrap path was available. | **Corrected in source:** `scripts/setup_environment.sh`. |
| UX-007 | Lower-bound dependencies alone did not reproduce the tested preprocessing package stack. | **Corrected:** setup uses `requirements/preprocessing-tested.txt` as a constraint profile; platform-specific PyTorch remains separate. |
| UX-008 | Normal users were directed to an editable install. | **Corrected:** non-editable install is the normal manual path; editable install is labeled development-only. |
| UX-009 | Parent-model checksum/audit was not prominent in quick start. | **Corrected:** exact SHA-256 and `smartpet-audit-weights` command are in README/public-assets docs. |
| UX-010 | Data/reproducibility docs did not consistently name/hash the exact MNI reference. | **Corrected:** MNI reference and fixed mask are named and hashed consistently. |
| UX-011 | `docs/DATA.md` retained stale v0.3.0 wording. | **Corrected.** |
| UX-012 | `docs/DATA.md` said registration pipelines were excluded although v0.3.1 exposes `raw_activity` registration. | **Corrected:** native-space ANTs registration and MNI-domain paths are separated explicitly. |
| UX-013 | Public Drive display title still identified the mirror as v0.3.0. | **External mirror action required:** rename the Drive folder to the current asset release name. |
| UX-014 | Public assets required manual browser acquisition and integrity checking. | **Corrected in source:** `smartpet-download-assets` supports verified `inference`, `finetune`, and `all` profiles; no silent download occurs during `pip install`. |
| UX-015 | Mirror `SHA256SUMS.txt` did not cover all v0.3.1 runtime assets. | **Repository corrected; external mirror action required:** replace the mirror checksum file with `docs/PUBLIC_ASSET_SHA256_v0.3.1.txt`. |
| UX-016 | External metadata could arrive as XLSX while the CLI accepted CSV only. | **Corrected in source:** optional `excel` extra, XLSX/XLSM parsing, sheet selection, and workbook template. |
| UX-017 | Dynamic 4D PET users were not told that the model/preprocessing contract is scalar 3D. | **Corrected:** CLI/docs/errors state that 4D PET must first be converted to a scientifically documented static 3D image. |
| UX-018 | `smartpet-prepare-external --help` did not expose required activity metadata, accepted units, decay semantics, or count-scaling semantics. | **Corrected in CLI help.** |
| UX-019 / PREP-001 | `ADMIN`/`START` could not truthfully represent calibrated activity with no decay correction. | **Corrected in source:** `NONE` is supported for calibrated frame-average activity when timing, half-life, and image duration are supplied; activity is explicitly corrected to administration time before SUVbw. |
| UX-020 | Public/repository workflow lacked a dedicated external-user spreadsheet template. | **Corrected in repository:** `examples/external_activity_metadata_template.xlsx`; **external mirror action required:** copy the workbook into the public asset folder. |
| UX-021 | Explanatory notes placed below an example table could export as fake subject rows. | **Corrected:** explanatory text is confined to the workbook Instructions sheet; data sheets contain only table rows. |
| UX-022 | A fresh Python environment did not contain ANTs, blocking `raw_activity`. | **Corrected in docs/setup contract:** ANTs is explicitly an external system dependency used only by native-space registration. |
| UX-023 | External users were not told how to obtain ANTs. | **Corrected:** Alliance uses `module load ants/2.6.5`; other systems are directed to the official ANTsX installation instructions. |
| UX-024 / ENV-025 | Loading the Alliance ANTs module exposed a system SciPy incompatible with the virtual-environment NumPy. | **Corrected in launch helpers:** module-provided Python prefixes are removed and user site packages are disabled before SMART-PET Python execution. |
| DATA-057 | Registration provenance reported terse transform code `s` even though it represented SyN. | **Corrected:** provenance/QC retain the machine code and add the human-readable `SyN` label. |

## Clean-room validation checkpoints retained

The correction work also preserves the successful external-user checkpoints that motivated the fixes:

- fresh public clone and isolated environment;
- public parent-weight, MNI-reference, and brain-mask SHA-256 verification;
- A100 CUDA execution;
- ANTs 2.6.5 native-space preprocessing;
- target-estimated shared registration transform for a paired PET input;
- canonical LAS-to-RAS handling;
- exact voxelwise SUV conversion after canonicalization;
- exact voxelwise `asinh(SUV)` normalization;
- parent inference and public inference audit;
- fixed-brain-mask evaluation.

The single clean-room case is a software-path validation only and is not a cohort-level external-performance result.
