#!/usr/bin/env bash
set -euo pipefail
: "${SMARTPET_INIT_CHECKPOINT:?Set SMARTPET_INIT_CHECKPOINT}"
CONFIG=${1:-configs/finetune.json}
NPROC=${SMARTPET_NPROC_PER_NODE:-2}
shift || true
torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m smartpet.cli.train --config "$CONFIG" --backend ddp --init-checkpoint "$SMARTPET_INIT_CHECKPOINT" "$@"
