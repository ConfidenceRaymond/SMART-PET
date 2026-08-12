#!/usr/bin/env bash
# Source this file from the SMART-PET repository on Alliance Canada systems:
#   source scripts/activate_alliance.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "[ERROR] Source this script instead of executing it:" >&2
  echo "        source scripts/activate_alliance.sh" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SMARTPET_VENV:-$ROOT/.venv}"

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "[ERROR] SMART-PET environment not found: $VENV" >&2
  return 1
fi
if ! type module >/dev/null 2>&1; then
  echo "[ERROR] Environment-modules command is unavailable on this system." >&2
  return 1
fi

module load ants/2.6.5
# shellcheck disable=SC1090
source "$VENV/bin/activate"
unset PYTHONPATH || true
unset EBPYTHONPREFIXES || true
export PYTHONNOUSERSITE=1

command -v antsRegistrationSyNQuick.sh >/dev/null || return 1
command -v antsApplyTransforms >/dev/null || return 1

echo "[OK] SMART-PET environment: $VENV"
echo "[OK] ANTs: $(antsRegistration --version 2>&1 | head -1)"
