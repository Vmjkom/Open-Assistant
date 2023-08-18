#!/bin/bash
#SBATCH --job-name=debug_sft_train
#SBATCH --account=project_2007628
#SBATCH -p gputest
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH -t 0:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err


#LOGGING
rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err

#
export CACHE=/scratch/project_2007628/$USER/cache
export DATA_PATH=/scratch/project_2007628/villekom/data
export MODEL_PATH=/scratch/project_2007628/villekom/oa_models

#DISTRIBUTED
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

#DEBUG
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
export TORCH_DISTRIBUTED_DEBUG=DETAIL

#MODULES
module purge
module load python-data

srun -l python3 -m torch.distributed.run --standalone --nproc-per-node=$SLURM_GPUS_ON_NODE trainer_sft.py \
        --configs gpt3-finnish-small oasst_finnish \
        --cache_dir $CACHE \
        --output_dir $MODEL_PATH/$SLURM_JOB_NAME \
        --log_dir ./logs \
        --local_rank $LOCAL_RANK \