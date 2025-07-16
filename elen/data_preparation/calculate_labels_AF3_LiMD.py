#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J get_labels
#SBATCH -o get_labels.log
#SBATCH -e get_labels.err
# Usage: --plot_hist, --id_to_plot can be savely run afterwards again
import os
import re
import sys
import glob
import json
import shutil
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
import MDAnalysis as mda
import warnings

from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
from Bio.PDB import PDBParser, PDBIO, Select
from elen.data_preparation.utils_plot import plot_labels_bar_rmsd, plot_labels_bar_lddt
from elen.data_preparation.utils_plot import plot_labels_violin

from elen.shared_utils.utils_others import func_timer, discard_pdb
from elen.config import PATH_SOFTWARE, PATH_ROSETTA_BIN

from importlib.resources import files
PATH_XML = str(files('elen.data_preparation').joinpath('calc_rmsd.xml'))

# -----------------------------------------------------------------------------
### HELPERS ###################################################################
def filter_labels_by_res_ids(loop_res_ids, whole_res_ids, whole_labels):
    """
    Filter whole protein labels to retain only those corresponding to the residue IDs in loop_res_ids.
    """
    filtered_labels = []
    for res_id, label in zip(whole_res_ids, whole_labels):
        if res_id in loop_res_ids:
            filtered_labels.append(label)
    return filtered_labels


def get_residue_ids(path_pdb: str) -> list:
    """
    Parse a PDB file to extract a list of residue numbers.
    """
    parser = PDBParser(QUIET=True)  # Quiet mode to suppress warnings
    resnum_list = []
    
    try:
        structure = parser.get_structure("structure", path_pdb)
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Ensure the residue ID is correctly accessed
                    if residue.id[0] == ' ':
                        res_chain_id = f"{chain.get_id()}_{residue.id[1]}"
                        resnum_list.append(res_chain_id)
        return resnum_list
    except FileNotFoundError:
        logging.error(f"Error: The file {path_pdb} does not exist.")
        return []
    except Exception as e:
        logging.error(f"An error occurred while parsing the PDB file: {e}")
        return []
    

class ChainSelect(Select):
    """
    A custom Select class that accepts only the specified chain(s).
    """
    def __init__(self, chain_id):
        super().__init__()
        self.chain_id = chain_id

    def accept_chain(self, chain):
        if chain.id == self.chain_id:
            return 1
        return 0


def split_pdb_into_chains(input_pdb_path, output_dir=None):
    """
    Splits a multi-chain PDB file into separate files, each containing a single chain.
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_pdb_path)
    if not output_dir:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    parser = PDBParser(QUIET=True)
    structure_id = os.path.splitext(os.path.basename(input_pdb_path))[0]
    structure = parser.get_structure(structure_id, input_pdb_path)
    io = PDBIO()
    chain_ids = set()
    for model in structure:
        for chain in model:
            chain_ids.add(chain.id)

    output_files = []
    for chain_id in chain_ids:
        chain_file_name = f"{structure_id}_{chain_id}_split.pdb"
        chain_file_path = os.path.join(output_dir, chain_file_name)
        io.set_structure(structure)
        io.save(chain_file_path, select=ChainSelect(chain_id))
        output_files.append(chain_file_path)
        logging.debug(f"Saved chain {chain_id} to {chain_file_path}")
    return output_files


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

        
### PER-RESIDUE LABEL METHODS: RMSD, LDDT, CAD-score
def calc_rmsd_labels_with_Rosetta(path_native, path_model, path_output, label_dict, json_tag):
    try:
        subprocess.run([f"{PATH_ROSETTA_BIN}/rosetta_scripts.linuxgccrelease",
                        "-in:file:native", path_native,
                        "-in:file:s", path_model,
                        "-parser:protocol", PATH_XML,
                        "-nstruct", "1",
                        "-out:path:all", path_output,
                        "-ignore_zero_occupancy", "false",
                        "--overwrite"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        score_file = os.path.join(path_output, "score.sc")
        df = pd.read_csv(score_file, sep='\s+', skiprows=1)
        os.remove(score_file)
        cols = df.filter(regex='^res_rmsd_')
        sorted_cols = sorted(cols.columns, key=lambda x: int(x.split('_')[-1]))
        df = df[sorted_cols]
        rmsd_list = []
        for _, row in df.iterrows():
            rmsd_list = row[sorted_cols].tolist()
        return rmsd_list
    except Exception as e:
        logging.error(f"Error in calc_rmsd_labels_with_Rosetta for {os.path.basename(path_model)}: {e}")
        raise

def calc_lddt_labels(path_native, path_model, label_dict, json_tag):
    try:
        process = subprocess.Popen([f"{PATH_SOFTWARE}/lddt-linux/lddt",
                                    path_model, 
                                    path_native],
                                    stdout=subprocess.PIPE, text=True)
        stdout, _ = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"lddt process failed with return code {process.returncode}")
        lddt_list = []
        for line in stdout.split('\n'):
            if 'Yes' in line:
                parts = line.split()
                if len(parts) > 4:
                    lddt = float(parts[4])
                    lddt_list.append(lddt)
        return lddt_list
    except Exception as e:
        logging.error(f"Error in calc_lddt_labels for {os.path.basename(path_model)}: {e}")
        raise
    
def calc_lddt_labels_full_protein(path_ref, path_model):
    outpath_ref = os.path.dirname(path_ref)
    paths_split_ref = split_pdb_into_chains(path_ref, outpath_ref)
    outpath_model = os.path.dirname(path_model)
    paths_split_model = split_pdb_into_chains(path_model, outpath_model)
    lddt_list = []
    for chain_ref, chain_model in zip(paths_split_ref, paths_split_model):
        try:
            process = subprocess.Popen([f"{PATH_SOFTWARE}/lddt-linux/lddt",
                                        chain_model, 
                                        chain_ref],
                                        stdout=subprocess.PIPE, text=True)
            stdout, _ = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"lddt process failed with return code {process.returncode}")
            lddt_list_chain = []
            for line in stdout.split('\n'):
                if 'Yes' in line:
                    parts = line.split()
                    if len(parts) > 4:
                        lddt = float(parts[4])
                        lddt_list_chain.append(lddt)
            lddt_list.extend(lddt_list_chain)
        except Exception as e:
            logging.error(f"Error in calc_lddt_labels for {os.path.basename(path_model)}: {e}")
            raise
    return lddt_list

def extract_loop_pattern(filename):
    match = re.search(r'_\d+_(EH|HE|HH|EE)\.pdb$', filename)
    return match.group(0) if match else None

def remove_hetatom_entries(input_file, output_file):
    """
    Removes all lines that start with 'HETATM' from a PDB file.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if not line.startswith("HETATM"):
                outfile.write(line)
                
def remove_hetatoms_from_loop_pdbs(inpath):
    logging.info(f"Removing hetatoms from dir {os.path.dirname(inpath)}.")
    normalized_inpath = os.path.normpath(inpath)
    outpath = os.path.join(args.outpath, f"{os.path.basename(normalized_inpath)}_wo_het_tmp")
    os.makedirs(outpath, exist_ok=True)
    for path_pdb in glob.glob(os.path.join(inpath, f"*.pdb")):
        fname_pdb = os.path.basename(path_pdb)
        path_pdb_wo_het = os.path.join(outpath, fname_pdb)
        remove_hetatom_entries(path_pdb, path_pdb_wo_het)
    return outpath

def calculate_avg_md_rmsd(ref_file, traj_file, selection="protein and name CA"):
    """
    Compute the average RMSD per residue over all frames in a trajectory.
    """
    u = mda.Universe(ref_file, traj_file)
    residues = u.select_atoms(selection).residues
    res_rmsd = {res.resid: [] for res in residues}
    u.trajectory[0]
    ref_positions = {}
    for res in residues:
        ref_positions[res.resid] = res.atoms.positions.copy()
    for ts in u.trajectory:
        for res in residues:
            current_positions = res.atoms.positions
            if current_positions.shape[0] != ref_positions[res.resid].shape[0]:
                continue
            diff = current_positions - ref_positions[res.resid]
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            res_rmsd[res.resid].append(rmsd)
    avg_rmsd_list = [np.mean(res_rmsd[res.resid]) for res in residues]
    return avg_rmsd_list

############################################################################################################
@func_timer
def main(args):
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    outpath_labels_final = os.path.join(args.outpath, 'labels.json')
    if not os.path.exists(outpath_labels_final):
        path_whole_protein_labels = os.path.join(args.outpath, f"whole_protein_labels.json") 
        if not os.path.exists(path_whole_protein_labels):
            logging.info("Step 1: Calculating distance metrics (rmsd and lddt) on whole protein structures.") 
            path_af3_wo_het = remove_hetatoms_from_loop_pdbs(args.inpath_AF)
            path_natives_wo_het = remove_hetatoms_from_loop_pdbs(args.inpath_natives)
            path_md_wo_het = remove_hetatoms_from_loop_pdbs(args.inpath_MD)
            
            label_dict = {} 
            for path_af3 in glob.glob(os.path.join(path_af3_wo_het, f"*.pdb")):
                fname_af3 = os.path.basename(path_af3)
                logging.info(f"Processing {fname_af3}.")
                try:
                    identifier = fname_af3[:4]
                    native_matches = glob.glob(os.path.join(path_natives_wo_het, f"{identifier}*.pdb"))
                    md_matches = glob.glob(os.path.join(path_md_wo_het, f"{identifier}*.pdb"))
                    if not native_matches or not md_matches:
                        logging.error(f"Error: Matching native or MD file not found for {path_af3}. Skipping.")
                        af_orig = os.path.join(args.inpath_AF, fname_af3)
                        if os.path.exists(af_orig):
                            discard_pdb(af_orig, args.path_discarded, "label calculation", f"Error: Matching native or MD file not found for {path_af3}. Skipping.")
                        continue
                    path_native = native_matches[0]
                    path_md = md_matches[0]

                    logging.info(f"Calculating rmsd of {fname_af3}.")
                    rmsd_native = calc_rmsd_labels_with_Rosetta(path_af3, path_native, args.outpath, label_dict, "rmsd_nat")
                    rmsd_md = calc_rmsd_labels_with_Rosetta(path_af3, path_md, args.outpath, label_dict, "rmsd_md")
                    
                    logging.info(f"Calculating average MD rmsd of {fname_af3}.")
                    path_md_gro = glob.glob(os.path.join(args.inpath_MD_simulations, f"{identifier}", f"{identifier}*_solv_ions.gro"))[0]
                    path_md_traj = glob.glob(os.path.join(args.inpath_MD_simulations, f"{identifier}", f"md.xtc"))[0]
                    rmsd_avg_md = calculate_avg_md_rmsd(path_md_gro, path_md_traj)
                    
                    logging.info(f"Calculating lddt of {fname_af3}.")
                    lddt_native = calc_lddt_labels_full_protein(path_af3, path_native)
                    lddt_md = calc_lddt_labels_full_protein(path_af3, path_md)

                    residue_ids = get_residue_ids(path_af3)       
                    if len(residue_ids) == len(rmsd_native) == len(lddt_native) == len(rmsd_md) == len(lddt_md) == len(rmsd_avg_md):
                        write_to_label_dict(fname_af3, rmsd_native, label_dict, "rmsd_nat")
                        write_to_label_dict(fname_af3, lddt_native, label_dict, "lddt_nat")
                        write_to_label_dict(fname_af3, rmsd_md, label_dict, "rmsd_md")
                        write_to_label_dict(fname_af3, lddt_md, label_dict, "lddt_md")
                        write_to_label_dict(fname_af3, rmsd_avg_md, label_dict, "rmsd_avg_md")
                        write_to_label_dict(fname_af3, residue_ids, label_dict, "res_id")
                    else:
                        logging.warning(f"Warning: length of labels don't match for {path_af3}.")
                        logging.warning(f"len rmsd_nat, rmsd_md, rmsd_avg_md, lddt_nat, lddt_md, res_id: {len(rmsd_native), len(rmsd_md), len(rmsd_avg_md), len(lddt_native), len(lddt_md), len(residue_ids)}")
                        continue
                except Exception as e:
                    logging.error(f"Error processing file {path_af3}: {e}. Skipping this example.")
                    fname_af3 = os.path.basename(path_af3)
                    af_orig = os.path.join(args.inpath_AF, fname_af3)
                    if os.path.exists(af_orig):
                        discard_pdb(af_orig, args.path_discarded, "label calculation", e)
                    try:
                        identifier = fname_af3[:4]
                        native_matches_orig = glob.glob(os.path.join(args.inpath_natives, f"{identifier}*.pdb"))
                        if native_matches_orig:
                            native_orig = native_matches_orig[0]
                            discard_pdb(native_orig, args.path_discarded, "label calculation", e)

                    except Exception as ex:
                        logging.error(f"Could not copy native file for {fname_af3}: {ex}")
                    try:
                        identifier = fname_af3[:4]
                        md_matches_orig = glob.glob(os.path.join(args.inpath_MD, f"{identifier}*.pdb"))
                        if md_matches_orig:
                            md_orig = md_matches_orig[0]
                            discard_pdb(md_orig, args.path_discarded, "label calculation", e)

                    except Exception as ex:
                        logging.error(f"Could not copy MD file for {fname_af3}: {ex}")
                    continue

            with open(path_whole_protein_labels, 'w') as json_file:
                json.dump(label_dict, json_file, indent=4, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else o)

            logging.info("Done calculating per-residue labels.")
        else:
            logging.info(f"Step 1: Skipped... {path_whole_protein_labels} found.") 

        with open(path_whole_protein_labels, 'r') as json_file:
            dict_labels_whole_proteins = json.load(json_file)
            
        logging.info(f"Step 2: Creating loop labels by filtering {path_whole_protein_labels}.")
        dict_labels_loops = {}
        print(os.path.join(args.inpath_loops, "*.pdb"))
        for path_loop_af3 in glob.glob(os.path.join(args.inpath_loops, "*.pdb")):
            try:
                fname_loop_af3 = os.path.basename(path_loop_af3)
                logging.info(f"Getting loop labels of {fname_loop_af3}.")
                identifier = fname_loop_af3[:4]
                fname_loop_json = f"{identifier}_af3.pdb"

                list_res_ids_loop = get_residue_ids(path_loop_af3)
                if fname_loop_json in dict_labels_whole_proteins:
                    for label in dict_labels_whole_proteins[fname_loop_json].keys():
                        list_labels_whole_proteins = dict_labels_whole_proteins[fname_loop_json][label]
                        list_res_ids_whole_proteins = dict_labels_whole_proteins[fname_loop_json]["res_id"]
                        list_labels_loop = filter_labels_by_res_ids(list_res_ids_loop, list_res_ids_whole_proteins, list_labels_whole_proteins)
                        if len(list_res_ids_loop) == len(list_labels_loop):
                            write_to_label_dict(fname_loop_af3, list_labels_loop, dict_labels_loops, label)
                            write_to_label_dict(fname_loop_af3, list_res_ids_loop, dict_labels_loops, "res_id")
                        else:
                            logging.warning(f"Warning: length of labels don't match for {path_loop_af3}.")
                            logging.warning(f"list_res_ids_loop vs list_labels_loop: {len(list_res_ids_loop)} vs {len(list_labels_loop)}")
                            discard_pdb(path_loop_af3, args.path_discarded, "label calculation", f"length of labels don't match for {path_loop_af3}.")
                            continue
                else:
                    logging.error(f"Error: {fname_loop_json} not found in whole protein labels dictionary. Skipping {path_loop_af3}.")
                    discard_pdb(path_loop_af3, args.path_discarded, "label calculation", f"{fname_loop_json} not found in whole protein labels dictionary. Skipping {path_loop_af3}.")
                    continue
            except Exception as e:
                logging.error(f"Error processing loop file {path_loop_af3}: {e}. Skipping this loop.")
                discard_pdb(path_loop_af3, args.path_discarded, "label calculation", e)
                continue
            
        with open(outpath_labels_final, 'w') as json_file:
            json.dump(dict_labels_loops, json_file, indent=4)
        if args.plot_hist:
            logging.info("Plotting violin plots for all labels in the dataset.")
            plot_labels_violin(outpath_labels_final, args.outpath)           
        logging.info("Done.")

    else:
        if args.plot_hist:
            logging.info("Plotting violin plots for all labels in the dataset.")
            plot_labels_violin(outpath_labels_final, args.outpath)           
        if args.id_to_plot:
            logging.info(f"Plotting per-residue barplot for {args.id_to_plot}.")
            plot_labels_bar_rmsd(outpath_labels_final, args.id_to_plot, args.outpath)
            plot_labels_bar_lddt(outpath_labels_final, args.id_to_plot, args.outpath)
        logging.info("Done.")


###############################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-calculate_labels_AF3_LiMD-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser()
    DEFAULT_PATH = "/home/florian_wieser/projects/ELEN/elen_training/data_preparation/AF3_LiMD/AF_LiMD_2"
    parser.add_argument('--inpath_AF', default=f"{DEFAULT_PATH}/EL_AF3_LiMD_2/harmonized/AF3_models")
    parser.add_argument('--inpath_natives', default=f"{DEFAULT_PATH}/EL_AF3_LiMD_2/harmonized/natives")
    parser.add_argument('--inpath_MD', default=f"{DEFAULT_PATH}/EL_AF3_LiMD_2/harmonized/MD_frames")
    parser.add_argument('--inpath_MD_simulations', default=f"{DEFAULT_PATH}/MD")
    parser.add_argument('--inpath_loops', default=f"{DEFAULT_PATH}/EL_AF3_LiMD_2/extracted_loops/EL_AF3_models")
    parser.add_argument('--outpath', default=f"{DEFAULT_PATH}/labels")
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing output')
    parser.add_argument('--id_to_plot', default=None, help='Identifier of protein to plot (e.g., "1ubq")')
    parser.add_argument('--plot_hist', action='store_true', help='Plot violin plots for all labels in the dataset')
    parser.add_argument("--path_discarded", type=str, default=f"discarded", help="Output directory for failed PDB files.")
    args = parser.parse_args()

    main(args)

