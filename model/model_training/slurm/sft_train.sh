#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --account=project_462000119
#SBATCH -p small-g
#SBATCH -c 7
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err


rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err

export DATA_PATH=/scratch/project_462000241/ville/oa_data
export CACHE=/scratch/project_462000241/ville/cache
export MODEL_PATH=/scratch/project_462000241/ville/oa_models
export LOGS=/pfs/lustrep2/scratch/project_462000241/ville/logs


module purge
module load LUMI/22.08 partition/G
module load rocm
module use /appl/local/csc/modulefiles
module load pytorch

cd ..
srun python3 trainer_sft.py --configs debug webgpt_dataset_only \
                                --cache_dir $CACHE \
                                --output_dir $MODEL_PATH/debug_pythia70M \
                                --log_dir $LOGS \
                                --show_dataset_stats \