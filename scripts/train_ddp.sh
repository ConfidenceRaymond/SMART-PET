#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/train_from_scratch.json}
NPROC=${SMARTPET_NPROC_PER_NODE:-2}
shift || true
torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" -m smartpet.cli.train --config "$CONFIG" --backend ddp "$@"
