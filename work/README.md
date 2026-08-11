# Work directory

All temporary, disposable, or machine-specific SMART-PET files belong under this directory.

Recommended subdirectories:

```text
work/
  audits/              generated validation and inventory reports
  cache/               pytest, Python bytecode, and other local caches
  config/              private machine-specific environment files
  logs/                non-SLURM logs
  slurm_logs/          SLURM stdout/stderr
  staging/             downloaded or extracted patch/release bundles
  temporary_scripts/   disposable one-off launch and audit scripts
  tmp/                 temporary preprocessing/build files
```

Repository scripts create these directories as needed. Nothing under `work/` is required for source control or scientific reproducibility, except this README.

Safe cleanup:

```bash
find work -mindepth 1 -maxdepth 1 ! -name README.md -exec rm -rf {} +
```

Training outputs, checkpoints, prepared datasets, and final manifests are **not** temporary and should be written to explicit persistent paths, normally under `/scratch`.

Repository shell entry points export:

```bash
TMPDIR="$SMARTPET_WORK_ROOT/tmp"
PYTHONPYCACHEPREFIX="$SMARTPET_WORK_ROOT/cache/pycache"
```

Pytest is configured to use `work/cache/pytest`. Do not allow `__pycache__`, `.pytest_cache`, extracted archives, or scratch reports to accumulate under `src/`, `tests/`, or the repository root.
