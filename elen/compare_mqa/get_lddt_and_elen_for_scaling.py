#!/usr/bin/env python3
#SBATCH -J scale
#SBATCH -o scale.log
#SBATCH -e scale.err
#SBATCH --gres=gpu:1
import os
import re
import sys
import glob
import subprocess
import argparse
import pandas as pd
from elen.config import PATH_SOFTWARE, PATH_INFERENCE, PATH_ELEN_MODELS
PATH_LDDT = os.path.join(PATH_SOFTWARE, "lddt-linux", "lddt")

def calc_per_residue_lddt(lddt_bin, model_pdb, native_pdb):
    """
    Returns a list of per-residue lDDT values, one per residue in model_pdb.
    """
    process = subprocess.Popen([lddt_bin, model_pdb, native_pdb],
                              stdout=subprocess.PIPE, text=True)
    stdout, _ = process.communicate()
    lddt_list = []
    for line in stdout.split('\n'):
        if 'Yes' in line:
            parts = line.split()
            if len(parts) > 4:
                lddt = float(parts[4])
                lddt_list.append(lddt)
    return lddt_list

def run_elen_on_model(args):
    """
    Calls ELEN model on given pdb file, writes result to out_csv.
    Here we assume ELEN writes a CSV with columns: resid, elen_score
    Replace the subprocess call as needed for your setup.
    """
    # Example: subprocess.run([elen_bin, '--inpath', model_pdb, '--outpath', out_csv])
    # For this script, we'll assume elen_bin writes the required CSV directly.
    print(f"args.inpath_af_models: {args.inpath_af_models}")
    subprocess.run([
        f"{PATH_INFERENCE}/ELEN_inference.py",
        "--inpath", args.inpath_af_models,
        "--outpath", f"out_elen",
        "--path_elen_models", PATH_ELEN_MODELS,
        "--ss_frag_size", str(4),
        "--nr_residues", str(28),
        "--loop_max_size", str(10),
        "--pocket_type", "RP",
        "--elen_score_to_pdb", "lddt_cad",
        "--overwrite", 
        "--elen_models"] + args.elen_models, check=True
        )

###############################################################################        
def main(args):
    # run ELEN 
    run_elen_on_model(args)
    df_elen = pd.read_csv("out_elen/elen_results_jwwrx159/elen_scores_RP.csv")
    df_elen['fname_pdb'] = df_elen['fname_pdb'].str.replace(r'_[A-Za-z0-9](?=\.pdb$)', '', regex=True)
    # CHATGPT CHANGE FROM HERE
    native_files = sorted(glob.glob(os.path.join(args.inpath_natives, "*.pdb")))
    af_files = sorted(glob.glob(os.path.join(args.inpath_af_models, "*.pdb")))
    # Map basenames to full paths for matching
    natives_dict = {os.path.basename(f)[:6]: f for f in native_files}
    af_dict = {os.path.basename(f)[:6]: f for f in af_files}
    all_rows = []
    dict_lddt = {}
    for key in set(natives_dict):
        native_pdb = natives_dict[key]
        af_pdb = af_dict[key]
        lddt_list = calc_per_residue_lddt(PATH_LDDT, af_pdb, native_pdb)
        if not lddt_list:
            print(f"Warning: No lDDT results for {af_pdb}")
            continue
        dict_lddt[key] = lddt_list
        
        af_pdb_base = os.path.basename(af_pdb)
        elen_rows = df_elen[df_elen['fname_pdb'] == af_pdb_base].reset_index(drop=True)
        elen_scores = elen_rows['ELEN_score'].tolist()
        res_ids = elen_rows['res_id'].tolist()  # If you want actual residue numbering
        for i, (lddt, elen, res_id) in enumerate(zip(lddt_list, elen_scores, res_ids)):
            row = {
                "key": key,
                "af_pdb": af_pdb_base,
                "res_id": res_id,
                "lddt_gt": lddt,
                "elen_score": elen
            }
            all_rows.append(row)
    # Save all data to CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath_natives', type=str, required=True, help='Path to native PDBs')
    parser.add_argument('--inpath_af_models', type=str, required=True, help='Path to AF models')
    parser.add_argument('--elen_models', nargs='+', default=["jwwrx159"])
    parser.add_argument('--out_csv', type=str, default='per_residue_lddt_vs_elen.csv', help='Output CSV filename')
    args = parser.parse_args()
    main(args)
