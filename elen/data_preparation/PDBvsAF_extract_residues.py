#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J extract_residues
#SBATCH -o extract_residues.log
#SBATCH -e extract_residues.err

# TODO 
# replace get SS
# replace get total residues
# take into account special first and last residues
# bake in 3 or 6 length fragments
# todo fix issue - if first or last residue it could be that there are not enough 
# resiudes to include, in this case, increase until max or discard entire structure?
# no fill until 22! to make predicion properly working
import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.DSSP import DSSP
import os
import sys
import glob
import shutil
import argparse as ap
import numpy as np
from pyrosetta import *
init("-mute all")
from elen.config import PATH_DSSP

# Filter Biopython warnings and set the filter to "ignore"
#warnings.filterwarnings("ignore", category=BiopythonWarning)

def get_BioPython_DSSP(fname, dssp_executable):
    """DSSP according to BioPython (using mkdssp executable)"""
    model = PDBParser().get_structure("new_protein", fname)[0]
    dssp = DSSP(model, fname, dssp=dssp_executable)
    # convert BioPython's dssp dictionary to sequence and secondary_structure strings, respectively
    sequence = "".join([dssp[res_id][1] for res_id in dssp.keys()])
    ss_orig = "".join([dssp[res_id][2] for res_id in dssp.keys()])
    # convert dssp nomenclature to simple one: H - helix, E - sheet, L - loop
    ss = (ss_orig.replace("B", "E").replace("G", "H").replace("I", "H").replace("T", "L").replace("-", "L")
          .replace("S", "L"))
    return ss, sequence


def get_residues_around_loop(res_id, struct, chain, nr_residues=28):
    residues_list = []
    for chain in struct[0]:
        for res in chain:
            residues_list.append(res)

    mid_residue_coords = np.array(list(struct.get_residues())[res_id - 1]["CA"].get_coord())
    residues_sorted_by_distance = sorted(residues_list, key=lambda residue: np.linalg.norm(residue["CA"].get_coord() - mid_residue_coords))
    # Keep only the first 'max_residues' residues
    residues_to_keep = residues_sorted_by_distance[:nr_residues]
    return residues_to_keep


def remove_all_but_loop(struct, residues_to_keep):
    chain_id = ""
    for chain in struct[0]:
        chain_id = chain.id
    # adjust residues_to_keep's residues' chain.id to match the one of struct
    for res in residues_to_keep:
        res.parent.id = chain_id

    # Remove any residues outside the 'residues_to_keep' set
    for chain in struct[0]:
        residues_to_remove = []
        for residue in chain:
            if residue not in residues_to_keep:
                residues_to_remove.append(residue)
        for residue in residues_to_remove:
            chain.detach_child(residue.id)

    return struct


def write_to_pdb(args, struct, fname, res_pos_native, res_pos_tensor, res_aa, res_ss):

    """write extracted loop to file and append metrics/info"""
    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    pdb_io.save(fname)

    path_pdb = os.path.join(args.outpath, "AF_models", fname)
    with open(fname, 'a') as file:
        file.write(f"res_pos_native {res_pos_native}\n")
        file.write(f"res_pos_tensor {res_pos_tensor}\n")
        file.write(f"res_aa {res_aa}\n")
        file.write(f"res_ss {res_ss}\n")
    if "_native_" in path_pdb:
        shutil.move(fname, os.path.join(args.outpath, "natives", fname))
    else:
        shutil.move(fname, path_pdb)
    return path_pdb 


def rosetta_numbering(chain):
    """convert residue numbering of chain to rosetta numbering (starting with 1)"""
    residue_idx = 1
    for res in chain:
        res.id = (" ", residue_idx, " ")
        residue_idx += 1
    return chain

def get_total_residues(struct):
    total_residues = 0
    for model in struct:
        for chain in model:
            for res in chain:
                total_residues += 1
    return total_residues

def get_res_position_in_extracted_tensor(residues_to_keep, res_id): 
    for index, residue in enumerate(residues_to_keep):
        # Assuming the residue information is formatted as a string and resseq is the unique identifier
        if res_id == residue.get_id()[1]:
            return index + 1
       
###############################################
def extract_loops(args, path_native):
    pdb_parser = PDBParser()
    struct_native = pdb_parser.get_structure("protein", path_native)
    fname_native = path_native.replace(".pdb", "")
    
    dirpath_natives_BIO="natives_split"
    os.makedirs(dirpath_natives_BIO, exist_ok=True)

    for chain in struct_native[0]:
        chain = rosetta_numbering(chain)  # let the residue numbering start with 1
        
        # dump chain as .pdb file in order to get ss and sequence
        io = PDBIO()
        io.set_structure(chain)
        fname_chain = f"{fname_native}_native.pdb"
        
        path_chain = os.path.join(dirpath_natives_BIO, f"{os.path.basename(fname_native)}_native.pdb")
        io.save(path_chain)
        ss, sequence = get_BioPython_DSSP(path_chain, PATH_DSSP)  # get secondary structure
        # set up list with .pdb files with loops to be extracted
        identifier = os.path.basename(fname_chain)[:6]
        path_models = glob.glob(f"{args.inpath_models}/{identifier}*")
        path_models.insert(0, path_chain) # add native to the beginning of the pathlist

        for res_id, (res_aa, res_ss) in enumerate(zip(sequence, ss), start=1):
            residues_to_keep = []
            for path_pdb in path_models: # iterate over list of .pdb files
                fname_final = f"{os.path.basename(path_pdb).replace('.pdb', '')}_{res_id}.pdb"
                if (os.path.exists(os.path.join(args.outpath, "AF_models", fname_final)) and os.path.exists(os.path.join(args.outpath, "natives", fname_final))):
                    print(f"Skipping {fname_final}")
                else:
                    print(f"Extracting residue pocket for res_id {res_id} of {path_pdb}.") 
                    chain_struct = pdb_parser.get_structure("protein", path_pdb)
                    for chain_native in chain_struct[0]:
                        tmp_struct = chain_struct.copy()
                        if "_native_" in fname_final:
                            residues_to_keep = get_residues_around_loop(res_id, tmp_struct, chain_native, args.nr_residues)
                            residues_to_keep = sorted(residues_to_keep, key=lambda x: x.get_id()[1])
                        
                        # to be aware which is the actual residue of interest in the final tensor:
                        res_position_in_extracted_tensor = get_res_position_in_extracted_tensor(residues_to_keep, res_id)
                        fin_struct = remove_all_but_loop(tmp_struct, residues_to_keep)

                        # calculate general metrics about the loop, to be appended to the .pdb file, when extracted
                        total_residues = get_total_residues(fin_struct)
                        if total_residues == args.nr_residues: #, f"Error number of extracted residues ({total_residues}) doesnt match nr_residues ({args.nr_residues})"
                            path_pdb = write_to_pdb(args, fin_struct, fname_final, res_id, res_position_in_extracted_tensor, res_aa, res_ss)
                        else:
                            print(f"Skipping {res_id}. total_residues != args.nr_residues")


##############################################################################################################
def main(args):
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    os.makedirs(os.path.join(args.outpath, "natives"), exist_ok=True)
    os.makedirs(os.path.join(args.outpath, "AF_models"), exist_ok=True)
    
    pdb_files = glob.glob(os.path.join(args.inpath_natives, "*.pdb"))
    for input_pdb in pdb_files:
        extract_loops(args, input_pdb)
    print("Done extracting.")
    """ 
    # check integrity of extracted loop couples (SI and length of natives vs AF models)
    for path_native in glob.glob(os.path.join(args.outpath, "natives/*.pdb")):
        identifier_PDB = os.path.basename(path_native)[:6]
        for _ in glob.glob(os.path.join(args.outpath, f"AF_models/{identifier_PDB}*")):
            _, seq_native = get_BioPython_DSSP(path_native, PATH_DSSP)
            _, seq_model = get_BioPython_DSSP(path_native, PATH_DSSP)
            sequence_identity = get_sequence_identity(seq_native, seq_model)
            print(f"Checking integrity of {os.path.basename(path_native)}: SI {sequence_identity}, length: {len(seq_native)} {len(seq_model)}")
            assert sequence_identity == 100, f"SEQUENCES DONT MATCH! {seq_native, seq_model}"
            assert len(seq_native) == len(seq_model), f"DIFFERENT LENGTH! {len(seq_native), len(seq_model)}"
    print("Done checking.")
    """

###############################################################################
if __name__ == "__main__":
    parser = ap.ArgumentParser()
    default_path = "/home/florian_wieser/testbox/extract_residues/dir_1"
    parser.add_argument("--inpath_natives", type=str, default=f"{default_path}/natives")
    parser.add_argument("--inpath_models", type=str, default=f"{default_path}/AF_models")
    parser.add_argument("--outpath", type=str, default=f"extracted_RP")
    parser.add_argument("--nr_residues", type=int, default=28)
    parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite existing run")
    args = parser.parse_args()
    main(args)

