#!/bin/bash
#SBATCH --job-name=debug_rl_small
#SBATCH --account=project_462000241
#SBATCH -p dev-g
#SBATCH -c 4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err


echo "Job name" $SLURM_JOB_NAME

module --force purge
module load LUMI/22.08 partition/G 
module load rocm/5.2.3

module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load aws-ofi-rccl/rocm-5.2.3
module use /appl/local/csc/modulefiles
module load pytorch

export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions
export CACHE=/scratch/project_462000241/$USER/cache
export MODEL_PATH=/scratch/project_462000241/$USER/oa_models/rl
export LOGS=/pfs/lustrep2/scratch/project_462000241/$USER/logs

#Compile deepspeed with torch backend to only mi250x
export PYTORCH_ROCM_ARCH=gfx90a

#Distributed variables
#export MASTER_PORT=1356
#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

#DEBUG
#CUDA_LAUNCH_BLOCKING=1
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

export TOKENIZERS_PARALLELISM=true

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python3 -m torch.distributed.run --standalone --nproc-per-node=$SLURM_GPUS_ON_NODE \
        trainer_rl_lumi.py --configs defaults_rlhf finnish_gpt_small_rlhf \
        --cache_dir $CACHE \
        --log_dir $LOGS \
        --output_dir $MODEL_PATH/$SLURM_JOB_NAME \
        --local_rank $LOCAL_RANK \