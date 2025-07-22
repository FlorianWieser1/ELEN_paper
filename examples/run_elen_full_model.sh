#!/bin/bash
#SBATCH -J ELEN
#SBATCH -o ELEN_%j.log
#SBATCH -e ELEN_%e.err
# Run ELEN full model inference on example data
# Usage: ./run_elen_full_example.sh
# Description: Will run the full ELEN model (including Rosetta and LLM features)

# activate conda environment
source activate elen_test

# Defaults
INPUT_DIR="input_ELEN_full_model"
OUTPUT_DIR="output_ELEN_full_model"

python ../elen/inference/run_elen_inference.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --feature_mode "full" \
    --saprot_embeddings_file "input_saprot_only/saprot_650M.h5" \
    --overwrite
