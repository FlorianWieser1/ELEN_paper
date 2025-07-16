#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J per-residue_labels
#SBATCH -o per-residue_labels.log
#SBATCH -e per-residue_labels.err
import os
import re
import sys
import glob
import json
import shutil
import argparse
import subprocess
import pandas as pd
import pkg_resources
import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from elen.shared_utils.utils_pdb import get_residue_ids
from elen.shared_utils.utils_others import func_timer
from elen.config import PATH_SOFTWARE, PATH_ROSETTA_BIN
PATH_XML = pkg_resources.resource_filename('elen.data_preparation', 'calc_rmsd.xml')

### HELPERS ###################################################################
def rename_model_chain(path_native, path_model):
    command = f"awk '$1 == \"ATOM\" || $1 == \"HETATM\" {{print $5}}' {path_native} | sort | uniq"
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    chain_native = result.stdout.strip().split('\n')[0]
    fname_model = os.path.basename(path_model)
    chain_native = fname_model[5]
    command = f"/home/florian_wieser/anaconda3/bin/pdb_chain -{chain_native} {path_model} >> {path_model}.tmp"
    subprocess.run(command, shell=True)
    shutil.move(f"{path_model}.tmp", path_model)

def write_to_label_dict(path_model, label_list, label_dict, label_key):
    fname_model = os.path.basename(path_model)
    if fname_model in label_dict:
        label_dict[fname_model].update({label_key: label_list})
    else:
        label_dict[fname_model] = {label_key: label_list}

        
### PER-RESIDUE LABEL METHODS: rmsd, lddt, CAD-score
@func_timer
def calc_rmsd_labels_with_Rosetta(path_native, path_model, path_output, label_dict):
    subprocess.run([f"{PATH_ROSETTA_BIN}/rosetta_scripts.linuxgccrelease",
                        "-in:file:native", path_native,
                        "-in:file:s", path_model,
                        "-parser:protocol", PATH_XML,
                        "-nstruct", "1",
                        "-out:path:all", path_output,
                        "--overwrite"])
    
    df = pd.read_csv(os.path.join(path_output, "score.sc"), delim_whitespace=True, skiprows=1)
    os.remove(os.path.join(path_output, "score.sc"))       
    cols = df.filter(regex='^res_rmsd_')
    sorted_cols = sorted(cols.columns, key=lambda x: int(x.split('_')[-1]))
    df = df[sorted_cols]
    rmsd_list = []
    for _, row in df.iterrows():
        rmsd_list = row[sorted_cols].tolist()
    write_to_label_dict(path_model, rmsd_list, label_dict, 'rmsd')

@func_timer
def calc_lddt_labels(path_native, path_model, label_dict):
    process = subprocess.Popen([f"{PATH_SOFTWARE}/lddt-linux/lddt",
                              path_model, 
                              path_native],
                              stdout=subprocess.PIPE, text=True)
    stdout, _ = process.communicate()
    lddt_list = []
    for line in stdout.split('\n'):
        if 'Yes' in line:
            parts = line.split()
            if len(parts) > 4:
                lddt = float(parts[4])
                lddt_list.append(lddt)
    write_to_label_dict(path_model, lddt_list, label_dict, 'lddt')

@func_timer
def calc_CAD_score_labels(args, path_native, path_model, path_output, label_dict):
    subprocess.run([args.path_cad,
                        "--input-target", path_native,
                        "--input-model", path_model,
                        "--output-residue-scores", f"{path_output}/CAD_score.sc"])

    df = pd.read_csv(os.path.join(path_output, "CAD_score.sc"), delim_whitespace=True, names=["CAD_rest", "CAD_score"], header=None)
    CAD_list = df['CAD_score'].tolist()
    write_to_label_dict(path_model, CAD_list, label_dict, 'CAD')

    
############################################################################################################
@func_timer
def main(args):
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    
    label_dict_chains = {} 
    for path_native in glob.glob(os.path.join(args.inpath_natives, "*.pdb")):
        fname_native = os.path.basename(path_native)
        identifier = f"{fname_native[:10]}*"
        for path_model in glob.glob(os.path.join(args.inpath_models, identifier)):
            print(f"Processing model {os.path.basename(path_model)}", flush=True)
            rename_model_chain(path_native, path_model)
            calc_rmsd_labels_with_Rosetta(path_native, path_model, args.outpath, label_dict_chains)
            calc_lddt_labels(path_native, path_model, label_dict_chains)
            calc_CAD_score_labels(args, path_native, path_model, args.outpath, label_dict_chains)
            res_ids_list = get_residue_ids(path_native)
            write_to_label_dict(path_model, res_ids_list, label_dict_chains, 'res_id')
    print(label_dict_chains)
    with open(os.path.join(args.outpath, 'labels.json'), 'w') as json_file:
        json.dump(label_dict_chains, json_file, indent=4)
    print(f"Done.")
    """
        #with open(os.path.join(args.outpath, 'per-residue_labels_chain.json'), 'w') as json_file:
        #json.dump(label_dict_chains, json_file, indent=4)
        
        #with open(os.path.join(args.outpath, 'per-residue_labels_chain.json'), 'r') as json_file:
        #json_data = json.load(json_file)
    label_dict_loops = {}
    for path_pdb in glob.glob(os.path.join(args.inpath_loops, "*.pdb")):
        resnum_list = get_residue_ids(path_pdb)
        pattern = re.compile(r'(m[12345]).*?(\.pdb)')
        fname_pdb = os.path.basename(path_pdb)
        fname_pdb_json = re.sub(pattern, r'\1\2', fname_pdb)
        if fname_pdb == fname_pdb_json:
            fname_pdb_json = fname_pdb[:11] + ".pdb"
        #print(fname_pdb_json)
        if fname_pdb_json in label_dict_chains:
            for label in label_dict_chains[fname_pdb_json].keys():
                label_list = label_dict_chains[fname_pdb_json][label]
                label_list_filtered = [label_list[i - 1] for i in resnum_list]
                write_to_label_dict(path_pdb, label_list_filtered, label_dict_loops, label)
                write_to_label_dict(path_pdb, resnum_list, label_dict_loops, "res_id")
        else:
            print("no")
            
    with open(os.path.join(args.outpath, 'labels.json'), 'w') as json_file:
        json.dump(label_dict_loops, json_file, indent=4)
    print(f"Done.")
    """

###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    DEFAULT_PATH="/home/florian_wieser/testbox/get_PDBvsAF/calc_labels_py"
    parser.add_argument('--inpath_natives', default=f"{DEFAULT_PATH}/natives")
    parser.add_argument('--inpath_models', default=f"{DEFAULT_PATH}/AF_models")
    parser.add_argument('--inpath_loops', default=f"{DEFAULT_PATH}/extracted_residue_pockets")
    parser.add_argument('--outpath', default=f"{DEFAULT_PATH}/labels")
    parser.add_argument('--path_cad', default="/home/florian_wieser/software/CAD/voronota_1.27.3834/voronota-cadscore")
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing output')
    args = parser.parse_args()
    main(args)

