#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
WORK_ROOT="${SMARTPET_WORK_ROOT:-$ROOT/work}"
export PYTHONPYCACHEPREFIX="$WORK_ROOT/cache/pycache"
mkdir -p "$PYTHONPYCACHEPREFIX"

# Test fixtures must remain outside the source repository. Several tests
# intentionally create temporary NIfTI volumes and checkpoint files.
VALIDATION_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/smartpet-release-validation.XXXXXX")
trap 'rm -rf "$VALIDATION_TMP"' EXIT
export TMPDIR="$VALIDATION_TMP"
python -m compileall -q src tests
for script in scripts/*.sh scripts/slurm/*.sh; do bash -n "$script"; done
python - <<'PYCHECK'
required = ("numpy", "nibabel", "pandas", "torch", "tqdm", "matplotlib", "pytest", "ruff")
missing = []
for name in required:
    try:
        __import__(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit(f"[STOP] Missing required validation packages: {missing}")
PYCHECK
python -m pip install --no-build-isolation --no-deps -e .
pytest --basetemp "$VALIDATION_TMP/pytest"
python -m ruff check .
python - <<'PYSCAN'
from pathlib import Path
patterns = (
    "/home/" + "ray" + "02",
    "/scratch/" + "ray" + "02",
    "/lustre" + "07",
    "Anonymous" + "_ANO",
)
violations = []
for path in Path(".").rglob("*"):
    if not path.is_file() or any(part in {".git", "work"} for part in path.parts):
        continue
    if path.as_posix() == "scripts/validate_release.sh":
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for pattern in patterns:
        if pattern in text:
            violations.append((str(path), pattern))
if violations:
    raise SystemExit(f"[STOP] Private or machine-specific content found: {violations}")
PYSCAN
if find . \
  -path './.git' -prune -o \
  -path './work' -prune -o \
  -type f \
  \( -name '*.nii' -o -name '*.nii.gz' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) \
  -print -quit | grep -q .; then
  echo "[STOP] Data or checkpoint file found in source repository" >&2
  exit 1
fi
echo "[PASS] SMART-PET release validation complete"
