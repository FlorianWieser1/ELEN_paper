#!/bin/bash
#SBATCH -J add_to_lmdb
#SBATCH -o add_to_lmdb.log
#SBATCH -e add_to_lmdb.err
set -e

INPUT_DIR=$1
PATH_LABELS=$2

PATH_DP="/home/florian_wieser/projects/ELEN/elen/data_preparation"

# ----------------------------
# Step 1: Normalize the labels
# ----------------------------
echo "Normalizing labels..."
$PATH_DP/normalize_labels_json.py $PATH_LABELS normalized_labels.json scales.json

# ----------------------------
# Step 2: Prepare output folder
# ----------------------------
OUTPUT_DIR=$INPUT_DIR"_labelled"
rm -rf $OUTPUT_DIR

echo "Creating new folder structure in $OUTPUT_DIR..."
# Create the base folder
mkdir -p "$OUTPUT_DIR"

# Copy all files from LP_20 except the lmdbs folder
rsync -av --exclude='lmdbs' LP_20/ "$OUTPUT_DIR/"

# Create a new lmdbs folder inside the labelled directory
mkdir -p "$OUTPUT_DIR/lmdbs"

# --------------------------------------------
# Step 3: Process each LMDB dataset split
# --------------------------------------------
for SPLIT in train val test; do
    INPUT_LMDB="LP_20/lmdbs/${SPLIT}"
    OUTPUT_LMDB="$OUTPUT_DIR/lmdbs/${SPLIT}"
    echo "Processing ${SPLIT} LMDB..."
    $PATH_DP/add_labels_to_lmdb.py "$INPUT_LMDB" "$OUTPUT_LMDB" normalized_labels.json
done

mv normalized_labels.json scales.json $OUTPUT_DIR
rm -rf $OUTPUT_DIR/labels.json

echo "All done! The new labelled dataset is in the folder: $OUTPUT_DIR"

