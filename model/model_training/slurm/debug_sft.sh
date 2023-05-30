#!/bin/bash
#SBATCH --job-name=sft_debug
#SBATCH --account=project_462000119
#SBATCH -p dev-g
#SBATCH -c 7
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=50G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err


rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err


module --force purge
module load LUMI/22.08 partition/G 
module load rocm/5.2.3

module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load aws-ofi-rccl/rocm-5.2.3
module use /appl/local/csc/modulefiles
module load pytorch

export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions

export DATA_PATH=/scratch/project_462000241/ville/oa_data
export CACHE=/scratch/project_462000241/ville/cache
export MODEL_PATH=/scratch/project_462000241/ville/oa_models
export LOGS=/scratch/project_462000241/ville/logs

#Distributed variables
#export MASTER_PORT=6317
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

#DEBUG
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export INST_SCRATCH=/scratch/project_462000241/ville
export INST_PROJAPPL=/projappl/project_462000241/ville

export TOKENIZERS_PARALLELISM=true

#export OMP_NUM_THREADS=1

export MAX_JOBS=4


srun -l python3 trainer_sft.py --configs debug oasst_only \
        --cache_dir $CACHE \
        --output_dir $MODEL_PATH/debug \
        --log_dir $LOGS \
        --show_dataset_stats \
        --local_rank $LOCAL_RANK \
        --deepspeed \
        --report_to tensorboard
