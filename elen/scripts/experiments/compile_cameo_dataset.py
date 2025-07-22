#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J cameo1
#SBATCH -o cameo1.log
#SBATCH -e cameo1.err
# Instructions:
# Download raw Data from CAMEO and extract .tar.gz
# rename quality_estimation to dataset name and move contents of subfolders (2024.23.12)
# to main folder
import os
import sys
import glob
import shutil
import subprocess
import logging
import argparse as ap
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.SeqUtils import seq1
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)

DICT_METHODS = {
    "server23": "Baseline",
    "server40": "DeepUMQA",
    "server43": "DeepUMQA2",
    "server47": "MEGA-Assessment",
    "server28": "ModFOLD7_lDDT",
    "server39": "ModFOLD8",
    "server44": "ModFOLD9",
    "server45": "ModFOLD9_pure",
    "server8": "ProQ2",
    "server32": "ProQ3",
    "server33": "ProQ3D",
    "server31": "ProQ3D_LDDT",
    "server20": "QMEAN_3",
    "server29": "QMEANDisCo_3",
    "server15": "VoroMQA_sw5",
    "server17": "VoroMQA_v2",
    "server46": "ZJUT-GraphCPLMQA",
    "server60": "ZJUT-MultiViewMQA",
    "server24": "QMEANDISCO_beta",
    "server4": "Qmean_7.11",
    "server18": "ModFOLD6",
    "server19": "QMEANDisCo 2",
    "server1": "Dfire_v1.1",
    "server3": "Naive_PSIBlast",
    "server2": "Prosa2003",
    "server0": "Verify3d_smoothed",
    "server16": "EQuant 2",
    "server7": "ModFOLD4",
    "server41": "Atom_ProteinQA",
    "server21": "ModFOLD6",
}

###############################################################################
def main(args, logger):
    path_natives = os.path.join(args.outpath, "natives")
    path_models = os.path.join(args.outpath, "models")
    path_methods = os.path.join(args.outpath, "methods")
   
    if args.prepare:
        # make outpath directory
        if args.overwrite and os.path.exists(args.outpath):
            shutil.rmtree(args.outpath)
        os.makedirs(args.outpath, exist_ok=True)
        os.makedirs(path_natives, exist_ok=True)
        os.makedirs(path_models, exist_ok=True)
        os.makedirs(path_methods, exist_ok=True)
        
        # pool .pdb files per target
        # there are multiple models per target and multiple MQA assessment groups per model
        logger.info(f"Gathering data.")
        for path_dir_target in glob.glob(os.path.join(args.dataset, f"????_?_*_?")):
            id_target = os.path.basename(path_dir_target)[:6]
            id_model = os.path.basename(path_dir_target)
            logger.info(f"Processing {id_target}")
            shutil.copy(os.path.join(path_dir_target, "target.pdb"), os.path.join(path_natives, f"{id_model[:6]}_native.pdb"))
            shutil.copy(os.path.join(path_dir_target, "model.pdb"), os.path.join(path_models, f"{id_model}_m1.pdb"))
            shutil.copy(os.path.join(path_dir_target, "lddt.json"), os.path.join(path_models, f"{id_model}_lddt.json"))
            for path_dir_server in glob.glob(os.path.join(path_dir_target, "servers", "server*")):
                id_server = os.path.basename(path_dir_server)
                id_method = DICT_METHODS[id_server].replace(" ", "_")
                path_dir_method =  os.path.join(args.outpath, "methods", id_method)
                os.makedirs(path_dir_method, exist_ok=True)
                shutil.copy(os.path.join(path_dir_server, "qe_pred-1.pdb"), os.path.join(path_methods, id_method, f"{id_model}_{id_method}.pdb"))
                shutil.copy(os.path.join(path_dir_server, "scores", "qelddt.json"), os.path.join(path_methods, id_method, f"{id_model}_{id_method}_lddt.json"))
    if args.harmonize:
        # Check if there are corresponding files existing for all methods
        identifiers_to_remove = set()
        for path_model in glob.glob(os.path.join(path_models, "*.pdb")):
            fname_model = os.path.basename(path_model)
            identifier = fname_model.replace("_m1.pdb", "")
            identifier_native = identifier[:6]
            
            # check native
            if not os.path.exists(os.path.join(path_natives, f"{identifier_native}_native.pdb")):
                print(f"Warning: No native .pdb found for {identifier}.")
                identifiers_to_remove.add(identifier_native)
            # check methods
            for path_dir_method in glob.glob(os.path.join(path_methods, "*")):
                method = os.path.basename(path_dir_method)
                if not os.path.exists(os.path.join(path_dir_method, f"{identifier}_{method}.pdb")):
                    print(f"Warning: No method .pdb found for {method} {identifier}.")
                    identifiers_to_remove.add(identifier_native)
                if not os.path.exists(os.path.join(path_dir_method, f"{identifier}_{method}_lddt.json")):
                    print(f"Warning: No method .lddt found for {method} {identifier}.")
                    identifiers_to_remove.add(identifier_native)
        remove_models(args.outpath, identifiers_to_remove)
    print("Done.")

def remove_models(outpath, identifiers_to_remove):
    print(f"Warning: {identifiers_to_remove} do not have files for each method. Removing.")
    for identifier in identifiers_to_remove:
        pdbs_to_remove = glob.glob(os.path.join(outpath, f"natives/{identifier}*"))
        pdbs_to_remove.extend(glob.glob(os.path.join(outpath, f"methods/*/{identifier}*")))
        
        for path_pdb in pdbs_to_remove:
            os.remove(path_pdb)
            
###############################################################################                
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("cameo")
    
    parser = ap.ArgumentParser()
    default_path = "/home/florian_wieser/testbox/cameo"
    parser.add_argument("--dataset", type=str, default=f"{default_path}/cameo_test")
    parser.add_argument("--outpath", type=str, default=f"{default_path}/out")
    parser.add_argument("--prepare", action="store_true", default=False)
    parser.add_argument("--harmonize", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing run.")
    args = parser.parse_args()
    main(args, logger)