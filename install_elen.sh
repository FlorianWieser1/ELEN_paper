#!/bin/bash 

# create conda environment
conda create -n elen_inference python=3.9 pip
source activate elen_inference

# install torch/cuda dependencies
# choose correct torch+cuda versions to match your setup
TORCH="2.1.2"
CUDA="cu121"
pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html

# install other dependencies
pip install pytorch-lightning==1.9.0
pip install tabulate wandb seaborn atom3d

# install e3nn
pip install git+https://github.com/drorlab/e3nn_edn.git

# install pyrosetta
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'


# install ELEN #TODO change to project.toml?
pip install -e .
