#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_ROOT="${SMARTPET_WORK_ROOT:-${ROOT}/work}"
mkdir -p "${WORK_ROOT}/tmp" "${WORK_ROOT}/logs"
export TMPDIR="${WORK_ROOT}/tmp"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/cache/pycache"

: "${SMARTPET_EXTERNAL_METADATA_CSV:?Set SMARTPET_EXTERNAL_METADATA_CSV}"
: "${SMARTPET_EXTERNAL_DATA_ROOT:?Set SMARTPET_EXTERNAL_DATA_ROOT}"
: "${SMARTPET_EXTERNAL_OUTPUT_ROOT:?Set SMARTPET_EXTERNAL_OUTPUT_ROOT}"
: "${SMARTPET_MNI_REFERENCE:?Set SMARTPET_MNI_REFERENCE}"

INPUT_KIND="${SMARTPET_EXTERNAL_INPUT_KIND:-raw_activity}"

PYTHONNOUSERSITE=1 PYTHONPATH="${ROOT}/src" python -m smartpet.cli.prepare_external \
  --metadata-csv "${SMARTPET_EXTERNAL_METADATA_CSV}" \
  --data-root "${SMARTPET_EXTERNAL_DATA_ROOT}" \
  --output-root "${SMARTPET_EXTERNAL_OUTPUT_ROOT}" \
  --mni-reference "${SMARTPET_MNI_REFERENCE}" \
  --input-kind "${INPUT_KIND}" \
  --asinh-scale "${SMARTPET_ASINH_SCALE:-1.0}" \
  --threads "${SMARTPET_PREPROCESS_THREADS:-4}" \
  --work-dir "${SMARTPET_EXTERNAL_OUTPUT_ROOT}/work" \
  "$@"
