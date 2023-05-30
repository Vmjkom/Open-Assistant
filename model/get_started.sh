#!/bin/bash
git clone git@github.com:Vmjkom/Open-Assistant.git
git checkout Lumi

module --force purge
module use /appl/local/csc/modulefiles
module load pytorch
rm -rf $HOME/.local #Remove your python userbase

cd Open-Assistant/model

pip install -e .
pip install -e ../oasst-data/

./install_apex.sh

export CACHE_PATH=<your_path_here>
#This is the output dir for the to be trained sft model
export MODEL_PATH=<your_path_here>

python trainer_sft.py --configs finnish_gpt --cache_dir $CACHE_PATH --output_dir $MODEL_PATH/sft_model