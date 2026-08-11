#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

mkdir -p \
  work/audits \
  work/cache \
  work/logs \
  work/slurm_logs \
  work/staging \
  work/tmp \
  work/config

export SMARTPET_WORK_ROOT="${SMARTPET_WORK_ROOT:-$ROOT/work}"
export TMPDIR="${SMARTPET_WORK_ROOT}/tmp"
export PYTHONPYCACHEPREFIX="${SMARTPET_WORK_ROOT}/cache/pycache"

mkdir -p \
  "$SMARTPET_WORK_ROOT" \
  "$TMPDIR" \
  "$PYTHONPYCACHEPREFIX"

if [[ -n "${SMARTPET_VENV_ACTIVATE:-}" ]]; then
  if [[ ! -f "$SMARTPET_VENV_ACTIVATE" ]]; then
    echo "[ERROR] SMARTPET_VENV_ACTIVATE does not exist: $SMARTPET_VENV_ACTIVATE" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$SMARTPET_VENV_ACTIVATE"
fi

python -m pip install -e '.[dev]'

PYTHONNOUSERSITE=1 \
PYTHONPATH="$ROOT/src" \
python -m pytest -ra tests

mkdir -p "$SMARTPET_WORK_ROOT/cache/ruff"

RUFF_CACHE_DIR="$SMARTPET_WORK_ROOT/cache/ruff" \
ruff check src tests

while IFS= read -r -d '' file; do
    bash -n "$file"
done < <(
    find scripts \
      -type f \
      -name '*.sh' \
      -print0
)

cat > "$SMARTPET_WORK_ROOT/audits/bootstrap_summary.txt" <<EOF
SMART-PET bootstrap completed successfully.
Repository: $ROOT
Python: $(python --version 2>&1)
Work root: $SMARTPET_WORK_ROOT
Tests and Ruff validation completed successfully.
No Git commit and no SLURM job were created.
EOF

echo "[OK] SMART-PET bootstrap completed at $ROOT"
echo "Work directory: $SMARTPET_WORK_ROOT"
echo "No commit and no SLURM job were created."
