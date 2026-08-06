#!/bin/bash
#SBATCH --job-name=SMARTPET_INFER
#SBATCH --account=rpp-uanazodo_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
ROOT=${SMARTPET_ROOT:-$SLURM_SUBMIT_DIR}
cd "$ROOT"
source "${SMARTPET_VENV:-/home/$USER/pytorch}/bin/activate"
export PYTHONPATH="$ROOT/src"
bash scripts/infer_one_volume.sh
