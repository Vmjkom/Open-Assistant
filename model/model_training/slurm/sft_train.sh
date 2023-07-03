#!/bin/bash
#SBATCH --job-name=sft_xl_1e-5_10epoch_withqa
#SBATCH --account=project_462000241
#SBATCH -p standard-g
#SBATCH -c 4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH -t 24:00:00
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

export CACHE=/scratch/project_462000241/$USER/cache
export MODEL_PATH=/scratch/project_462000241/$USER/oa_models
export LOGS=/scratch/project_462000241/$USER/logs

export TOKENIZERS_PARALLELISM=true

#Distributed variables
#export MASTER_PORT=6317
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))

#LOGGING
export TORCH_DISTRIBUTED_DEBUG=OFF
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -l trainer_sft.py --configs gpt3-finnish-xl \
        --cache_dir $CACHE \
        --output_dir $MODEL_PATH/sft/$SLURM_JOB_NAME \
        --log_dir $LOGS \
        --local_rank $LOCAL_RANK \
        --deepspeed \
        --report_to tensorboard