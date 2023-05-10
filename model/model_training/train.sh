#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --account=project_462000241
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

export DATA_PATH=/pfs/lustrep2/projappl/project_462000185/ville/Open-Assistant/model/.cache
export MODEL_PATH=/pfs/lustrep2/projappl/project_462000185/ville/Open-Assistant/model/.saved_models

module purge
module use /appl/local/csc/modulefiles
module load pytorch

srun $PROJAPPL/ville/Open-Assistant/model/model_training/trainer_sft.py --configs finnish_gpt oasst_only \
--cache_dir $DATA_PATH --output_dir $MODEL_PATH/finnish_gpt_small_sft