#!/bin/bash
#SBATCH --job-name=OA_setup
#SBATCH --account=project_462000241
#SBATCH --nodes=1
#SBATCH --partition=dev-g
#SBATCH --gpus-per-node=1
#SBATCH -c 56
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH -t 02:00:00
#SBATCH -o %x.out
#SBATCH -e %x.err


module --force purge
module use /appl/local/csc/modulefiles #This path is enabled by default, consider adding this to .bashrc
module load pytorch
rm -rf $HOME/.local #Remove possible previous python user space
python3 -m venv oa_venv --system-site-packages
source oa_venv/bin/activate
pip install pip --upgrade

#These are relative paths to the model training dir
pip install -e .. #Install pyproject.toml from /model directory
pip install -e ../../oasst-data/ #Same thing for oasst data dir
