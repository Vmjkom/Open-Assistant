#!/bin/bash
#SBATCH --job-name=OA_setup
#SBATCH --account=project_462000241
#SBATCH -N 1
#SBATCH -p small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH -t 02:00:00
#SBATCH -o %x.out
#SBATCH -e %x.err

cd model_training
mkdir logs #Logs for the slurm job
module --force purge
module use /appl/local/csc/modulefiles
module load pytorch
rm -rf $HOME/.local #Remove your python userbase

export PYTHONUSERBASE=/projappl/project_462000241/$USER/python_userspace
pip install --upgrade
export PATH=$PATH:$PYTHONUSERBASE/bin

#Install apex with ninja, without cloning
pip install -v --install-option="--cpp_ext" --install-option="--cuda_ext" 'git+https://github.com/ROCmSoftwarePlatform/apex.git'

pip install -e .
pip install -e ../oasst-data/

#Gotta install ds into userspace as writing permissions are needed for hip compiling the cuda stuff
pip install --upgrade deepspeed
