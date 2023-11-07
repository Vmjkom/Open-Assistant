#!/bin/bash
#SBATCH --job-name=debug_sft_train
#SBATCH --account=project_2007628
#SBATCH -p gputest
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:2
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

#DEBUG LOGGING
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
export TORCH_DISTRIBUTED_DEBUG=INFO

#MODULES 
module purge
export PATH="/projappl/project_2007628/villekom/OA_tykky/bin:$PATH"
#module load python-data

export OMP_NUM_THREADS=1

#VENV
#source /projappl/project_2007628/villekom/Open-Assistant/model/model_training/.venv/bin/activate
#export PYTHONPATH=/projappl/project_2007628/villekom/Open-Assistant/model/model_training/python3.10/site-packages/

srun -l python3.10 -m torch.distributed.run --standalone --nproc-per-node=$SLURM_GPUS_ON_NODE trainer_sft.py \
        --configs gpt3-finnish-small \
        --cache_dir $CACHE \
        --output_dir $MODEL_PATH/$SLURM_JOB_NAME \
        --log_dir ./logs \
        --local_rank $LOCAL_RANK \