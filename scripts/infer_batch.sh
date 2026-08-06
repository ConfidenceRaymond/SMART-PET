#!/usr/bin/env bash
set -euo pipefail
: "${SMARTPET_CHECKPOINT:?Set SMARTPET_CHECKPOINT}"
: "${SMARTPET_MNI_REFERENCE:?Set SMARTPET_MNI_REFERENCE}"
MANIFEST=${1:-examples/inference_manifest.csv}
smartpet-infer-batch \
  --manifest "$MANIFEST" \
  --checkpoint "$SMARTPET_CHECKPOINT" \
  --mni-reference "$SMARTPET_MNI_REFERENCE" \
  --input-domain "${SMARTPET_INPUT_DOMAIN:-suv}" \
  --amp-dtype auto
