#!/bin/bash
#SBATCH --job-name=SMARTPET_TRAIN_DDP
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
source "${SMARTPET_VENV:-/home/$USER/pytorch}/bin/activate"
export PYTHONPATH="$ROOT/src"
export SMARTPET_NPROC_PER_NODE=${SMARTPET_NPROC_PER_NODE:-2}
bash scripts/train_ddp.sh "${SMARTPET_CONFIG:-configs/train_from_scratch.json}"
