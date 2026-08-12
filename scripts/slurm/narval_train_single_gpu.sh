#!/bin/bash
#SBATCH --job-name=SMARTPET_TRAIN_1GPU
#SBATCH --account=rpp-uanazodo_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
bash scripts/train_single_gpu.sh "${SMARTPET_CONFIG:-configs/train_from_scratch.json}"
