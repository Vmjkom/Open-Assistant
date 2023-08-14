#!/bin/bash
#SBATCH --job-name=rm_train
#SBATCH --account=project_462000241
#SBATCH -p dev-g
#SBATCH -c 4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

mkdir -p logs
rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err

export CACHE=/scratch/project_462000241/$USER/cache
export MODEL_PATH=/scratch/project_462000241/$USER/oa_models/rm
export LOGS=/pfs/lustrep2/scratch/project_462000241/$USER/logs

module --force purge
module load LUMI/22.08 partition/G 
module load rocm/5.2.3

module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load aws-ofi-rccl/rocm-5.2.3
module use /appl/local/csc/modulefiles
module load pytorch


srun python3 trainer_rm.py --configs oasst-finnish-gpt-rm defaults_rm \
                                --cache_dir $CACHE \
                                --output_dir $MODEL_PATH/finnish_gpt_small_rm \
                                --log_dir $LOGS \
                                --show_dataset_stats \