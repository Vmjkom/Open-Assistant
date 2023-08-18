#!/bin/bash
#SBATCH --job-name=pytest_modelTraining
#SBATCH --account=project_462007628
#SBATCH -p gputest
#SBATCH -c 4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH -t 00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load python-data

export OMP_NUMBER_THREADS=$SLURM_CPUS_PER_TASK


srun python3 pytest ..