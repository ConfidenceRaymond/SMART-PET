# External preprocessing environment

## Recommended setup

From a fresh clone:

```bash
bash scripts/setup_environment.sh --with-excel --assets inference
source .venv/bin/activate
```

On Alliance Canada / Narval, create the environment on an **internet-enabled login node**:

```bash
bash scripts/setup_environment.sh --alliance --with-excel
source scripts/activate_alliance.sh
```

The setup script creates an isolated environment, installs SMART-PET with the preprocessing constraint file, optionally installs XLSX and asset-download support, and performs basic package/CLI checks.

Alliance compute nodes may not have outbound internet. Keep environment creation and asset acquisition separate on HPC: download assets on an internet-connected machine or permitted login node, copy them to `resources/`, and use `smartpet-download-assets --verify-only` on the cluster.

## Validated environments

Two environment records are relevant:

### Clean-room v0.3.1 external-user execution

The public workflow completed on Narval with:

- Python 3.11.4;
- PyTorch 2.6.0+computecanada;
- CUDA runtime 12.2;
- NVIDIA A100-SXM4-40GB;
- ANTs 2.6.5.

### Dedicated preprocessing package profile

The external preprocessing pathway has also been validated with:

- Python 3.11.5;
- NumPy 2.2.2;
- pandas 2.2.3;
- NiBabel 5.3.2;
- ANTs 2.6.5.

The **Python package constraints** are recorded in:

```text
requirements/preprocessing-tested.txt
```

The Python interpreter version is documented here; a requirements file does not pin the interpreter itself.

PyTorch is intentionally not pinned in that constraints file because the correct build depends on the platform, accelerator, and package index. On Narval, `torch==2.6.0` resolves through the Alliance wheelhouse to the Compute Canada build used in clean-room validation.

## ANTs

ANTs is an external command-line dependency used only by `raw_activity` preprocessing.

Required executables:

```text
antsRegistrationSyNQuick.sh
antsApplyTransforms
```

### Alliance Canada / Narval

```bash
module load ants/2.6.5
```

Then verify:

```bash
command -v antsRegistrationSyNQuick.sh
command -v antsApplyTransforms
antsRegistration --version
```

### Other systems

Install ANTs from the official ANTsX project:

- project: `https://github.com/ANTsX/ANTs`
- releases: `https://github.com/ANTsX/ANTs/releases`

Use the official binary/source installation instructions for the operating system and ensure the ANTs `bin` directory is on `PATH`.

Do not treat ANTs as a normal `pip` dependency of SMART-PET.

MNI-domain input kinds (`mni_activity`, `mni_suv`, `mni_suv_normalized`) do not run ANTs registration.

## Alliance Python-path isolation

Module-based HPC systems can expose Python packages from compiled software stacks. During the clean-room test, loading the ANTs module exposed a system SciPy that warned about incompatibility with the NumPy installed in the SMART-PET virtual environment.

ANTs itself is used by SMART-PET as external executables, so those module-provided Python package paths are unnecessary. The Alliance activation helper therefore uses:

```bash
module load ants/2.6.5
source .venv/bin/activate
unset PYTHONPATH
unset EBPYTHONPREFIXES
export PYTHONNOUSERSITE=1
```

Use:

```bash
source scripts/activate_alliance.sh
```

before `raw_activity` preprocessing on Alliance.

## MNI orientation

SMART-PET canonicalizes the supplied MNI reference and validated PET volumes to RAS with NiBabel's orientation-aware canonicalization.

The public `templates/csymT.nii.gz` is stored as LAS on disk but represents the same physical MNI space. SMART-PET reorients voxel data together with the affine and then validates canonical shape, affine, and spacing.

Do **not** alter only a NIfTI affine header to make an image appear RAS. Orientation canonicalization is not anatomical registration.
