# External preprocessing environment

## Known-good validated environment

The SMART-PET external preprocessing pathway has been validated using:

- Python 3.11.5
- NumPy 2.2.2
- pandas 2.2.3
- NiBabel 5.3.2
- ANTs 2.6.5

Exact Python versions used for the validation are recorded in:

    requirements/preprocessing-tested.txt

ANTs is an external system dependency used by the `raw_activity` pathway for
registration to the supplied MNI reference.

## Environment isolation

SMART-PET preprocessing must not inherit arbitrary system Python package paths.

On module-based HPC systems, compiled extensions from system modules may be
ABI-incompatible with the NumPy version installed in the user's Python
environment.

A safe Alliance Canada setup is:

    module --force purge
    module load ants/2.6.5
    source /path/to/python/environment/bin/activate
    unset PYTHONPATH
    export PYTHONNOUSERSITE=1

The SMART-PET preprocessing launcher sets its own source-tree PYTHONPATH and
does not append an inherited PYTHONPATH.

## MNI orientation

SMART-PET canonicalizes its MNI reference and PET volumes to RAS using
NiBabel's orientation-aware canonicalization.

A reference stored on disk as LAS therefore remains the same physical MNI
space but is represented internally and in downstream SMART-PET outputs as
canonical RAS.

This is intentional and must not be corrected by editing NIfTI affine headers
alone.


## ANTs requirement for raw-activity pairs

`raw_activity` preprocessing requires both:

- `antsRegistrationSyNQuick.sh`
- `antsApplyTransforms`

Registration is estimated once from the full-count target to the configured
MNI reference. The resulting forward transform stack is applied identically
to the target and low-count source.

The tested external ANTs version for the Narval preprocessing environment is
ANTs 2.6.5.

MNI-domain inputs do not require ANTs registration.
