#!/bin/bash
#SBATCH --job-name=debug_sft_train
#SBATCH --account=project_462000241
#SBATCH -p gputest
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

srun -l python3 trainer_sft.py --configs gpt3-finnish-xl oasst_finnish \
        --cache_dir $CACHE \
        --output_dir $MODEL_PATH/debug/$SLURM_JOB_NAME \
        --log_dir $LOGS \
        --local_rank $LOCAL_RANK \
        --report_to tensorboard