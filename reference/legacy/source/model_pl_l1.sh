#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=4 #this only affects MPI job
#SBATCH --time=22:15:00
#SBATCH -p compute_full_node


module load anaconda3
source activate python_38
python /gpfs/fs0/scratch/u/uanazodo/uanazodo/Ray/Best_Experimet/model_pl_l1/train16_8.py

