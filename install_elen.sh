#!/bin/bash 
#SBATCH -J install

# create conda environment
# conda create -n elen python=3.9 pip
# conda activate elen


# install torch/cuda dependencies
#TODO try old shi conda channel way of installing torch
#TORCH="1.13.0"
#CUDA="cu116"
#
#pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
#pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#pip install torch-geometric
#
#pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
#exit 0
# install other dependencies
pip install pytorch-lightning 

# TODO check stuff from edn repo - e3nn e.g. then try run  code
# install e3nn
pip install git+ssh://git@github.com/drorlab/e3nn_edn.git

# install atom3d
pip install atom3d

# install ELEN #TODO change to project.toml?
pip install -e .
