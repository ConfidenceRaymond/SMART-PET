#!/bin/bash
#SBATCH --job-name=SMARTPET_FINETUNE
#SBATCH --account=rpp-uanazodo_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
ROOT=${SMARTPET_ROOT:-$SLURM_SUBMIT_DIR}
cd "$ROOT"
VENV="${SMARTPET_VENV:-$ROOT/.venv}"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "[ERROR] SMART-PET environment not found: $VENV" >&2
  echo "Run scripts/setup_environment.sh first or set SMARTPET_VENV." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
unset PYTHONPATH || true
unset EBPYTHONPREFIXES || true
export PYTHONNOUSERSITE=1
: "${SMARTPET_INIT_CHECKPOINT:?Set SMARTPET_INIT_CHECKPOINT}"
export SMARTPET_NPROC_PER_NODE=${SMARTPET_NPROC_PER_NODE:-2}
bash scripts/finetune_ddp.sh "${SMARTPET_CONFIG:-configs/finetune.json}" "$@"
