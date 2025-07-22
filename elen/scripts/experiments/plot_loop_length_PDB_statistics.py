#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J plot_loop_lenghts_statistics
#SBATCH -o plot_loop_lenghts_statistics.log
#SBATCH -e plot_loop_lenghts_statistics.err
import warnings
from Bio import BiopythonDeprecationWarning
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=PDBConstructionWarning)
from Bio.PDB import PDBParser, PDBIO
import os
import sys
import glob
import re
import matplotlib.pyplot as plt
import argparse as ap
import numpy as np
import subprocess
from pyrosetta import *
init("-mute all")

PROJECT_PATH = os.environ.get('PROJECT_PATH')
sys.path.append(f"{PROJECT_PATH}/geometricDL/edn/edn_multi_labels_pr")
from utils import load_from_json, dump_to_json
sys.path.append(f"{PROJECT_PATH}/geometricDL/ARES_PL_sweep/scripts")
from PDBvsAF_extract_loops import get_BioPython_DSSP, rosetta_numbering, print_ruler
PATH_DSSP="/home/florian_wieser/miniconda3/envs/elen_test/bin/mkdssp"


### HELPERS ###################################################################
def count_loop_lengths(s, length_dict):
    pattern = r"L+"
    matches = list(re.finditer(pattern, s))  # Convert iterator to list to handle entries
    # Skip the first and last matches by slicing the list [1:-1]
    for match in matches[1:-1]:  # Adjust to ignore the first and last occurrence
        l_length = len(match.group())
        if l_length in length_dict:
            length_dict[l_length] += 1
        else:
            length_dict[l_length] = 1
    sorted_length_dict = {key: length_dict[key] for key in sorted(length_dict)}
    return sorted_length_dict

def dump_chain(chain, fname_pdb, dirpath_natives_BIO):
    chain = rosetta_numbering(chain)  # let the residue numbering start with 1
    io = PDBIO()
    io.set_structure(chain)
    path_chain = os.path.join(dirpath_natives_BIO, f"{os.path.basename(fname_pdb)}_{chain.id}_native.pdb")
    io.save(path_chain)
    return path_chain


def plot_loop_length_statistics(dict_loop_lengths, outpath, logarithmic, threshold_sumup=40):
    #dict_loop_lengths.pop(1, None)
    # sum up all values in the dict bigger then a threshold
    sum_over_50 = sum(value for key, value in dict_loop_lengths.items() if int(key) > threshold_sumup)
    dict_loop_lengths[str(threshold_sumup)] = sum_over_50
    keys_to_remove = [key for key in dict_loop_lengths if int(key) > threshold_sumup]
    keys_to_remove = keys_to_remove + ['1']
    print(keys_to_remove)
    for key in keys_to_remove:
        del dict_loop_lengths[key]
    
    bins = np.arange(0.5, threshold_sumup + 1.5, 1)  # Shift bins to center on integers
    data = []
    for key, count in dict_loop_lengths.items():
        data.extend([int(key)] * count)
        
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins, color="seagreen", log=logarithmic, edgecolor='white', linewidth=0.5, zorder=2)
    plt.xlim(1, threshold_sumup)
    plt.xticks(np.arange(1, threshold_sumup, step=1), fontsize=9)
    #plt.bar(keys, values, color='seagreen', zorder=2) # zorder, bars before grid
    plt.xlabel('Loop lengths [1]')
    plt.ylabel('Frequency [1]')
    plt.title('Loop lengths of Protein Data Bank')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=1)
    plt.savefig(outpath, bbox_inches='tight')
   
###############################################################################
def main(args):
    path_json = args.outpath.replace(".png", ".json")   
    if not os.path.exists(path_json):
        dict_loop_lengths = {}
        pdb_files = glob.glob(os.path.join(args.inpath_natives, "*.pdb"))
        for idx, path_pdb in enumerate(pdb_files):
            print(f"Processing {os.path.basename(path_pdb)} ({idx + 1}/{len(pdb_files)})")
            pdb_parser = PDBParser()
            struct_native = pdb_parser.get_structure("protein", path_pdb)
            fname_pdb = path_pdb.replace(".pdb", "")
   
            dirpath_natives_BIO = "natives_split"
            os.makedirs(dirpath_natives_BIO, exist_ok=True)

            for chain in struct_native[0]:
                # dump chain as .pdb file in order to get ss and sequence
                path_chain = dump_chain(chain, fname_pdb, dirpath_natives_BIO)
                ss, sequence = get_BioPython_DSSP(path_chain, PATH_DSSP)  # get secondary structure
                #print_ruler(ss, sequence)
                dict_loop_lengths = count_loop_lengths(ss, dict_loop_lengths)
        dump_to_json(dict_loop_lengths, path_json)
    else:
        for logarithmic in [True, False]:
            print(f"Loading from {path_json}.")
            dict_loop_lengths = load_from_json(path_json)
            outpath = "loop_length_histogram_log.png" if logarithmic == True else "loop_length_histogram.png"
            print(dict_loop_lengths)
            plot_loop_length_statistics(dict_loop_lengths, outpath, logarithmic)
    path_final = "loop_length_histograms.png" 
    subprocess.run(['convert', "loop_length_histogram.png", "loop_length_histogram_log.png", '-append', path_final])
    print("Done.")

###############################################################################
if __name__ == "__main__":
    parser = ap.ArgumentParser()
    default_path = "/home/florian_wieser/software/ARES/geometricDL/shared_data/PISCES/20A20R90SI_PDB-sub/natives_pool"
    parser.add_argument("--inpath_natives", type=str, default=default_path)
    parser.add_argument("--outpath", type=str, default=f"loop_lenghts_statistics_histogram.png")
    parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite existing run")
    args = parser.parse_args()
    main(args)
