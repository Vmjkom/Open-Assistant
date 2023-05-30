#!/bin/bash
#SBATCH --account=project_462000241
#SBATCH -gpus 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -p small-g
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH -t 01:00:00

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./