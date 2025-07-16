#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J dp_pipeline
#SBATCH -o dp_pipeline.log
#SBATCH -e dp_pipeline.err
# TODO further debug filtering STEP  - add check if for each EL_AF3_model a respective identifier is found in labels json hdf5
# TODO find way to filter bad examples, i.e. in the end get of each step identifiers of discarded dirs, make a set, remove them from all train_val_test mattering files
# TODO make os.path.exists statements depend on actual produced file
# TODO refactor the 4 subcodefiles
# TODO upscale for multidir - make wrapper functions, and waiters and number of dirs an argument
import os
import sys
import glob
import json
import shutil
import logging
import argparse
import subprocess
import warnings
from Bio import BiopythonDeprecationWarning
import h5py  # for filtering in SaProt HDF5
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from elen.shared_utils.utils_others import func_timer
from elen.config import PATH_DP_SCRIPTS
from elen.config import PATH_ROSETTA_TOOLS

### HELPERS ###################################################################

def rosetta_clean(path_pdb, outpath):
    """
    Cleans a PDB file using Rosetta's clean_pdb.py and places the result in outpath.
    """
    subprocess.run(
        [
            f"{PATH_ROSETTA_TOOLS}/clean_pdb.py",
            path_pdb,
            "--allchains",
            "X"
        ],
        stdout=subprocess.DEVNULL
    )
    fname_pdb = os.path.basename(path_pdb)
    outpath_pdb = os.path.join(outpath, fname_pdb.replace("_X.pdb", ".pdb"))
    shutil.move(fname_pdb.replace(".pdb", "_X.pdb"), outpath_pdb)
    os.remove(fname_pdb.replace(".pdb", "_X.fasta"))

### PROTOCOLS #################################################################

def run_harmonization(args, outpath_harmonization, path_discarded):
    subprocess.run([
        f"{PATH_DP_SCRIPTS}/harmonize_pdb_to_af3.py", 
        "--inpath_AF", args.inpath_AF,
        "--inpath_natives", args.inpath_natives,
        "--inpath_MD", args.inpath_MD,
        "--outpath", outpath_harmonization,
        "--path_discarded", path_discarded
    ], check=True)

def run_TM_score_filtering(cutoff, inpath, path_discarded):
    subprocess.run([
        f"{PATH_DP_SCRIPTS}/filter_AF3_by_TM.py", 
        "--inpath_AF", os.path.join(inpath, "AF3_models"),
        "--inpath_natives", os.path.join(inpath, "natives"),
        "--inpath_MD", os.path.join(inpath, "MD_frames"),
        "--cutoff", str(cutoff),
        "--outpath", path_discarded,
        "--path_discarded", path_discarded
    ], check=True)

def run_loop_extraction(inpath, outpath_extraction, path_discarded):
    subprocess.run([
        f"{PATH_DP_SCRIPTS}/extract_LP_AF3_LiMD.py", 
        "--inpath_AF", os.path.join(inpath, "AF3_models"),
        "--inpath_natives", os.path.join(inpath, "natives"),
        "--inpath_MD", os.path.join(inpath, "MD_frames"),
        "--outpath", outpath_extraction,
        "--path_discarded", path_discarded
    ], check=True)

def run_label_calculation(args, inpath_harmonized, inpath_loop, outpath_labels, path_discarded):
    command = [
        f"{PATH_DP_SCRIPTS}/calculate_labels_AF3_LiMD.py", 
        "--inpath_AF", os.path.join(inpath_harmonized, "AF3_models"),
        "--inpath_natives", os.path.join(inpath_harmonized, "natives"),
        "--inpath_MD", os.path.join(inpath_harmonized, "MD_frames"),
        "--inpath_loop", os.path.join(inpath_loop, "EL_AF3_models"),
        "--inpath_MD_simulations", args.inpath_MD_simulations,
        "--outpath", outpath_labels,
        "--path_discarded", path_discarded
    ]
    if args.plot_hist:
        command.append("--plot_hist")
    subprocess.run(command, check=True)
    
def run_calculate_residue_features(inpath, outpath_residue_features, path_discarded):
    subprocess.run([
        f"{PATH_DP_SCRIPTS}/compute_residue_features.py", 
        "--inpath_models", str(inpath),
        "--outpath", str(outpath_residue_features),
        "--residue_features",
        "--path_discarded", path_discarded
    ], check=True)
   
def run_calculate_saprot_embeddings(inpath_models, outpath_residue_features, path_discarded):
    saprot_model = "saprot_650M"
    subprocess.run([
        f"{PATH_DP_SCRIPTS}/compute_SaProt_embeddings.py",
        "--inpath_models", inpath_models,
        "--outpath", outpath_residue_features,
        "--saprot_model", saprot_model,
        "--path_discarded", path_discarded])

def run_create_dataset(args, inpath_extracted_loops, inpath_labels, outpath_dataset, outpath_residue_features):
    """
    Uses train_val_test.py to create the final dataset folder with:
     - extracted loops
     - labels.json
     - residue_features.json
     - SaProt embeddings, etc.
    """
    command = [
        f"{PATH_DP_SCRIPTS}/train_val_test.py", 
        "--inpath", os.path.join(inpath_extracted_loops, "EL_AF3_models"),
        "--outpath", outpath_dataset,
        "--make_dataset",
        "--max_sized_dataset",
        "--path_labels", f"{inpath_labels}/labels.json"
    ]
    if args.plot_hist:
        command.append("--plot_hist")
    subprocess.run(command, check=True)
    
    # Copy necessary files into the dataset folder
    shutil.copy(f"{outpath_residue_features}/residue_features.json", outpath_dataset)
    shutil.copy(f"{outpath_residue_features}/saprot_650M.h5", outpath_dataset)
    shutil.copy(f"{inpath_labels}/labels.json", outpath_dataset)

### NEW STEP: FILTERING DISCARDED IDENTIFIERS #################################

def filter_discarded(path_discarded,
                     path_extracted_loops,
                     path_labels_json,
                     path_residue_features_json,
                     path_saprot_h5):
    """
    1) Collects all IDs from 'path_discarded' directory by looking at the filenames.
       (Here, we assume the first 4 characters of the filename are the unique PDB code.)
    2) For each PDB file in the extracted loops directory (EL_AF3_models), checks that a corresponding
       entry exists in labels.json, residue_features.json, and the saprot_650M.h5 file based on the identifier.
       If any required entry is missing, the PDB file is moved to the discarded folder and its identifier is recorded.
    3) Removes any matching entries from:
         - labels.json,
         - residue_features.json,
         - saprot_650M.h5 (in HDF5).
    """

    # --- Gather Discarded IDs ------------------------------------------------
    discarded_ids = set()
    if os.path.isdir(path_discarded):
        for fname in os.listdir(path_discarded):
            if fname.endswith(".pdb"):
                pdb_id = fname[:4].lower()
                discarded_ids.add(pdb_id)
    logging.info(f"Initial discarded IDs from discarded folder: {discarded_ids}")

    # --- Check each extracted loop file for corresponding entries ----------------
    loop_dir = os.path.join(path_extracted_loops, "EL_AF3_models")
    if os.path.isdir(loop_dir):
        # Load labels data
        labels_data = {}
        if os.path.exists(path_labels_json):
            with open(path_labels_json, "r") as f:
                labels_data = json.load(f)
        # Load residue features data
        residue_features_data = {}
        if os.path.exists(path_residue_features_json):
            with open(path_residue_features_json, "r") as f:
                residue_features_data = json.load(f)
        # Load SAPROT HDF5 keys
        saprot_keys = set()
        if os.path.exists(path_saprot_h5):
            with h5py.File(path_saprot_h5, "r") as hf:
                saprot_keys = set(hf.keys())
                
        for pdb_file in os.listdir(loop_dir):
            if not pdb_file.endswith(".pdb"):
                continue
            file_id = pdb_file[:4].lower()
            has_label = any(key.lower().startswith(file_id) for key in labels_data)
            has_feature = any(key.lower().startswith(file_id) for key in residue_features_data)
            has_saprot = any(key.lower().startswith(file_id) for key in saprot_keys)
            if file_id in discarded_ids or not (has_label and has_feature and has_saprot):
                source_path = os.path.join(loop_dir, pdb_file)
                target_path = os.path.join(path_discarded, pdb_file)
                shutil.move(source_path, target_path)
                if file_id in discarded_ids:
                    reason = "identifier already discarded"
                else:
                    missing = []
                    if not has_label:
                        missing.append("label")
                    if not has_feature:
                        missing.append("residue_feature")
                    if not has_saprot:
                        missing.append("saprot")
                    reason = "missing required entries: " + ", ".join(missing)
                logging.info(f"Moved {pdb_file} to discarded folder because {reason}.")
                discarded_ids.add(file_id)

    # --- Remove entries from labels.json -------------------------------------
    if os.path.exists(path_labels_json):
        with open(path_labels_json, "r") as f:
            labels = json.load(f)
        filtered_labels = {}
        for key, value in labels.items():
            key_id = key[:4].lower()
            if key_id not in discarded_ids:
                filtered_labels[key] = value
        with open(path_labels_json, "w") as f:
            json.dump(filtered_labels, f, indent=2)
    
    # --- Remove entries from residue_features.json ---------------------------
    if os.path.exists(path_residue_features_json):
        with open(path_residue_features_json, "r") as f:
            residue_features = json.load(f)
        filtered_features = {}
        for key, value in residue_features.items():
            key_id = key[:4].lower()
            if key_id not in discarded_ids:
                filtered_features[key] = value
        with open(path_residue_features_json, "w") as f:
            json.dump(filtered_features, f, indent=2)

    # --- Remove groups from saprot_650M.h5 -----------------------------------
    if os.path.exists(path_saprot_h5):
        with h5py.File(path_saprot_h5, "r+") as hf:
            all_groups = list(hf.keys())
            for group_name in all_groups:
                group_id = group_name[:4].lower()
                if group_id in discarded_ids:
                    del hf[group_name]

###############################################################################
@func_timer
def main(args):
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    path_discarded = os.path.join(args.outpath, "discarded")
    os.makedirs(path_discarded, exist_ok=True)
    
    # Step 1: Run harmonization 
    outpath_harmonization = os.path.join(args.outpath, "harmonized")
    if not os.path.exists(outpath_harmonization):
        logging.info("STEP 1: HARMONIZATION")
        run_harmonization(args, outpath_harmonization, path_discarded) 
   
    # Step 2: Filter AF3 models via TM-score 
    logging.info("STEP 2: FILTERING AF3 MODELS REGARDING TM-SCORE")
    run_TM_score_filtering(0.6, outpath_harmonization, path_discarded)    
  
    # Step 3: Run loop extraction
    outpath_extraction = os.path.join(args.outpath, "extracted_loops")
    if not os.path.exists(outpath_extraction):
        logging.info("STEP 3: EXTRACTING LOOPS")
        run_loop_extraction(outpath_harmonization, outpath_extraction, path_discarded) 
   
    # Step 4: Run label calculation
    outpath_labels = os.path.join(args.outpath, "labels")
    if not os.path.exists(outpath_labels):
        logging.info("STEP 4: CALCULATING LABELS")
        run_label_calculation(args, outpath_harmonization, outpath_extraction, outpath_labels, path_discarded) 
           
    # Step 5: Rosetta clean .pdbs for Rosetta    
    outpath_cleaned = os.path.join(args.outpath, "cleaned")
    if not os.path.exists(outpath_cleaned):         
        os.makedirs(outpath_cleaned, exist_ok=True)
        logging.info("STEP 5: CLEANING AF3 .PDB FILES WITH ROSETTA")
        for path_pdb in glob.glob(os.path.join(outpath_harmonization, "AF3_models", "*.pdb")):
            rosetta_clean(path_pdb, outpath_cleaned)
    
    # Step 6: Calculate residue features
    outpath_residue_features = os.path.join(args.outpath, "residue_features")
    os.makedirs(outpath_residue_features, exist_ok=True)

    if not os.path.exists(os.path.join(outpath_residue_features, "residue_features.json")):
        logging.info("STEP 6: CALCULATING RESIDUE FEATURES")
        run_calculate_residue_features(outpath_cleaned, outpath_residue_features, path_discarded)

    # Step 7: Calculate SaProt embeddings
    if not os.path.exists(os.path.join(outpath_residue_features, "saprot_650M.h5")):
        logging.info("STEP 7: CALCULATING SAPROT SEQUENCE EMBEDDINGS")
        run_calculate_saprot_embeddings(outpath_cleaned, outpath_residue_features, path_discarded)

    # --- NEW STEP 8: Filter out discarded entries before dataset creation ----
    logging.info("STEP 8: FILTERING DISCARDED IDENTIFIERS FROM FINAL FILES")
    filter_discarded(
        path_discarded=path_discarded,
        path_extracted_loops=outpath_extraction,
        path_labels_json=os.path.join(outpath_labels, "labels.json"),
        path_residue_features_json=os.path.join(outpath_residue_features, "residue_features.json"),
        path_saprot_h5=os.path.join(outpath_residue_features, "saprot_650M.h5")
    )

    # Step 9: train_val_test - create final dataset
    dirname_dataset = f"DS_{os.path.basename(args.outpath)}"
    outpath_dataset = os.path.join(args.outpath, dirname_dataset)
    if not os.path.exists(outpath_dataset):
        logging.info("STEP 9: CREATING DATASET")
        run_create_dataset(args, outpath_extraction, outpath_labels, outpath_dataset, outpath_residue_features)
    
    logging.info("Done.")

###############################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-AF3_LiMD-dp-pipeline-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser()
    DEFAULT_PATH = "/home/florian_wieser/projects/ELEN/elen_training/data_preparation/AF3_LiMD/AF_LiMD_200/harmonized/fix"
    parser.add_argument('--inpath_AF', default=f"{DEFAULT_PATH}/AF3_models")
    parser.add_argument('--inpath_natives', default=f"{DEFAULT_PATH}/natives")
    parser.add_argument('--inpath_MD', default=f"{DEFAULT_PATH}/MD_frames")
    parser.add_argument("--inpath_MD_simulations", type=str, default=f"{DEFAULT_PATH}/MD_simulations",
                        help="Input directory for MD simulation folders.")
    parser.add_argument('--outpath', default=f"{DEFAULT_PATH}/DS_prep_test")
    parser.add_argument('--plot_hist', action='store_true', default=False, help='Plot histograms of labels and dataset.')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing output.')
    args = parser.parse_args()
    main(args)