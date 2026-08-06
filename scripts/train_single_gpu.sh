#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/train_from_scratch.json}
shift || true
smartpet-train --config "$CONFIG" --backend single "$@"
