#!/usr/bin/env bash
set -euo pipefail
: "${SMARTPET_INIT_CHECKPOINT:?Set SMARTPET_INIT_CHECKPOINT}"
CONFIG=${1:-configs/finetune.json}
shift || true
smartpet-train --config "$CONFIG" --backend single --init-checkpoint "$SMARTPET_INIT_CHECKPOINT" "$@"
