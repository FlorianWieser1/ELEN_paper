#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J get_res_features
#SBATCH -o get_res_features.log
#SBATCH -e get_res_features.err
"""
This script extracts various features from PDB models by calling the corresponding functions 
from the module 'elen.data_prepation.utils_features'.

It supports:
  - Residue-level features via calculate_residue_features
  - Atom-level features via calculate_atom_features
  - ESM embeddings (JSON and HDF5) via calculate_esm_json and calculate_esm_hdf5
  - SaProt embeddings (HDF5) via calculate_saprot_hdf5
"""

import os
import sys
import glob
import shutil
import logging
import argparse
import subprocess
from Bio.PDB import PDBParser

from elen.shared_utils.utils_features import (
    calculate_residue_features,
    calculate_atom_features,
    #calculate_esm_json,
    #calculate_esm_hdf5,
    #calculate_saprot_hdf5
)

### HELPERS ###################################################################

    
###############################################################################
def main(args):
    # Remove output directory if overwrite flag is set.
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    
    # Call residue-level feature extraction if enabled.
    pdb_paths_cleaned = glob.glob(os.path.join(args.inpath_models, "*.pdb"))
    if not pdb_paths_cleaned:
        logging.info("No PDB files found in the input directory.")
        sys.exit(1)

    if args.residue_features:
        logging.info("Starting residue-level feature extraction...")
        calculate_residue_features(pdb_paths_cleaned, args.outpath, args.path_discarded)
        
    # Call atom-level feature extraction if enabled.
    #if args.atom_features:
    #    logging.info("Starting atom-level feature extraction...")
    #    calculate_atom_features(pdb_paths_cleaned, args.outpath, args.path_discarded)
        
###############################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-AF3_LiMD-dp-pipeline-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(
        description="Extract features from PDB models using functions from elen.data_prepation.utils_features."
    )
    parser.add_argument('--inpath_models', default="model",
                        help="Input directory containing model PDB structures")
    parser.add_argument('--outpath', default="extracted_features",
                        help="Output directory for extracted features")
    parser.add_argument("--residue_features", action="store_true", default=False,
                        help="Calculate residue-level features")
    parser.add_argument("--atom_features", action="store_true", default=False,
                        help="Calculate atom-level features")
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing output directory')
    parser.add_argument("--path_discarded", type=str, default=f"discarded", help="Output directory for failed PDB files.")
    args = parser.parse_args()
    main(args)
