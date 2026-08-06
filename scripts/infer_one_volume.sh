#!/usr/bin/env bash
set -euo pipefail
: "${SMARTPET_CHECKPOINT:?Set SMARTPET_CHECKPOINT}"
: "${SMARTPET_INPUT:?Set SMARTPET_INPUT}"
: "${SMARTPET_MNI_REFERENCE:?Set SMARTPET_MNI_REFERENCE}"
: "${SMARTPET_SUV_OUTPUT:?Set SMARTPET_SUV_OUTPUT}"
smartpet-infer \
  --checkpoint "$SMARTPET_CHECKPOINT" \
  --input "$SMARTPET_INPUT" \
  --mni-reference "$SMARTPET_MNI_REFERENCE" \
  --input-domain "${SMARTPET_INPUT_DOMAIN:-suv}" \
  --suv-output "$SMARTPET_SUV_OUTPUT" \
  --amp-dtype auto
