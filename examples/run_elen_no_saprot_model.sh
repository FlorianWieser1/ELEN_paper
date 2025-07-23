#!/bin/bash
#SBATCH -J ELEN
#SBATCH -o ELEN_%j.log
#SBATCH -e ELEN_%e.err
# Run ELEN full model inference on example data
# Usage: ./run_elen_full_example.sh
# Description: Will run the ELEN model without LLM features but with Rosetta features

# activate conda environment
source activate elen_test

# Defaults
INPUT_DIR="input_ELEN"
OUTPUT_DIR="output_ELEN_no_saprot"

python ../elen/inference/run_elen_inference.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
	--pocket_type "RP" \
    --elen_models_dir "../models" \
    --feature_mode "no_saprot" \
    --overwrite
