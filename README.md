❯ cat elen_readme_update.md
# ELEN - Equivariant Loop Evaluation Network

## Overview

In contemporary protein design, advanced machine learning algorithms generate novel amino acid sequences with unprecedented functionalities, such as enzymes with enhanced catalytic rates or new reactivities. These designed sequences are typically assessed *in silico* using deep learning-based protein structure prediction methods. While these models excel at predicting well-ordered regions, accurately modeling **loop regions**—which are often flexible and crucial for protein function—remains a significant challenge.

To address this, we introduce the **Equivariant Loop Evaluation Network (ELEN)**, a local model quality assessment (MQA) method tailored specifically for evaluating the quality of protein loops.

<p align="center">
  <img src="images/ELEN_scheme_new.svg" alt="ELEN Model Overview" width="400" />
</p>

## Key Features

- ELEN predicts three distinct residue-level quality metrics by comparing predicted loops to reference crystal structures:
  - Local Distance Difference Test (lDDT)  
  - Contact Area Difference Score (CAD-score)  
  - Root Mean Squared Deviation (RMSD)  

- Operates as an all-atom model and uses **3D equivariant group convolutions** to capture the local geometric environment of each atom.

- Incorporates sequence embeddings from large protein language models such as Meta's **Evolutionary Scale Model 2 (ESM-2)** and **SaProt**, enhancing sequence and evolutionary context awareness.

- Integrates per-residue physicochemical features including solvent accessible surface area (SASA), Rosetta energy terms, and hydrogen bond counts.

- Demonstrates competitive or superior accuracy compared to state-of-the-art MQA methods on the Continuous Automated Model EvaluatiOn (CAMEO) benchmark dataset.

- Primarily designed for loop quality assessment but shows promise for broader residue-level quality evaluation tasks, including identifying flexible or disordered regions and assessing effects of single-residue mutations.

<p align="center">
  <img src="images/af2_xtal_overlayed_docday.png" alt="ELEN Model Overview" width="600" />
</p>

## Installation Instructions

Follow these steps to install ELEN and its dependencies:

### 1. Create and Activate Conda Environment

```bash
conda create -n elen_inference python=3.9 pip
conda activate elen_inference
```

### 2. Install PyTorch and CUDA Dependencies

Adjust `TORCH` and `CUDA` variables based on your system:

```bash
TORCH="2.1.2"
CUDA="cu121"

pip install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
```

### 3. Install Other Dependencies

```bash
pip install pytorch-lightning==1.9.0
pip install tabulate wandb seaborn atom3d
```

### 4. Install e3nn

```bash
pip install git+https://github.com/drorlab/e3nn_edn.git
```

### 5. Install PyRosetta

```bash
conda install pyrosetta
```

### 6. Install ELEN

```bash
pip install -e .
```

## Status and Availability

This project is currently in active development and pending publication. The source code, trained models, and datasets will be made publicly available upon official release.

## Contact

For inquiries, collaborations, or further information, please contact Florian Wieser.

---

*This repository serves as a public overview of the ELEN project. Detailed implementations will be shared once published.*

