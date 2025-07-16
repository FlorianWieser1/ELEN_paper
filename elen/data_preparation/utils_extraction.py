#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
import warnings
from Bio import BiopythonDeprecationWarning
from Bio.PDB.PDBExceptions import PDBConstructionWarning
# Ignore specific category warnings
warnings.simplefilter('ignore', BiopythonDeprecationWarning)
warnings.simplefilter('ignore', PDBConstructionWarning)
from Bio.PDB import PDBParser, PDBIO
import os
import sys
import shutil
import subprocess
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBIO, Select
from Bio import pairwise2
import numpy as np
import logging
from elen.inference.utils_inference import get_total_number_of_residues
from Bio.PDB import Structure, Model, Chain
from pyrosetta.toolbox.cleaning import cleanATOM

### HELPERS ###################################################################
def clean_pdb(path_pdb, outpath):
    """
    Cleans a PDB file by extracting ATOM and TER records using PyRosetta's cleanATOM.
    
    Args:
        path_pdb (str): The path to the original PDB file.
        outpath (str): The output directory where the cleaned PDB should be saved.

    Returns:
        str: The path to the cleaned PDB file in the output directory.
    """
    try:
        base_name = os.path.splitext(os.path.basename(path_pdb))[0]
        path_cleaned_pdb = os.path.join(outpath, f"{base_name}.pdb")
        cleanATOM(path_pdb, out_file=path_cleaned_pdb)
        return path_cleaned_pdb
    except Exception as e:
        print(f"An error during clean_pdb occurred: {e}")
        return None
    
    
def rosetta_numbering(path_pdb, chain):
    """Convert residue numbering of chain to rosetta numbering (starting with 1)."""
    residues = chain.get_residues()
    first_residue = next(residues)
    residue_id = first_residue.get_id()
    
    # Check if the sequence number is 0
    is_first_zero = (residue_id[1] != 1)
    if is_first_zero:
        for idx, res in enumerate(chain, start=1):
            res.id = (" ", idx, " ")
            
    # overwrite .pdb
    pdb_io = PDBIO()
    pdb_io.set_structure(chain)
    pdb_io.save(path_pdb)
    return path_pdb

def get_first_residues_id(chain):
    residues = chain.get_residues()
    first_residue = next(residues)
    residue_id = first_residue.get_id()
    first_residue_id = residue_id[1]
    return first_residue_id

def print_ruler(ss, sequence):
    chunk_size = 100
    for i in range(0, len(sequence), chunk_size):
        print(sequence[i : i+100])
        print(ss[i : i+100])
        print("1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890")
        print("1   5   10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90   95  100")


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


def get_sequence_identity(seq1, seq2):
    alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True, score_only=True)
    identity = (alignment / max(len(seq1), len(seq2))) * 100
    return identity


def get_loop_positions(ss, ss_frag, ss_frag_size, loop_max_size):
    if ss_frag == "helix":
        ss_opt_1, ss_opt_2 = "H", "H"
    elif ss_frag == "sheet":
        ss_opt_1, ss_opt_2 = "E", "E"
    else:
        ss_opt_1, ss_opt_2 = "H", "E"

    loop_positions = []
    i = 0
    while i < len(ss):
        # find SS fragment
        ss_counter = 0
        while i < len(ss) and (ss[i] == ss_opt_1 or ss[i] == ss_opt_2):
            ss_counter += 1
            i += 1
        if ss_counter >= ss_frag_size:  # check if SS fragment is large enough
            loop_counter = 0
            loop_start = i + 1
            # find loop fragment
            while i < len(ss):
                if ss[i] == "L":
                    loop_counter += 1
                    i += 1
                elif (ss[i] == ss_opt_1 or ss[i] == ss_opt_2) and (i + 1 < len(ss) and ss[i + 1] == "L"):
                    # allow single "E" or "H" within a loop
                    loop_counter += 1
                    i += 1
                else:
                    break
            if loop_counter >= 2 and loop_counter <= loop_max_size:
                loop_stop = i
                ss_counter = 0
                # find SS fragment after loop
                while i < len(ss) and (ss[i] == ss_opt_1 or ss[i] == ss_opt_2):
                    ss_counter += 1
                    i += 1
                if ss_counter >= ss_frag_size:
                    loop_positions.append((loop_start, loop_stop))
                    i = loop_stop  # move past the loop
                    continue
        i += 1
    return loop_positions


def get_residues_around_loop(loop_start, loop_stop, struct, chain_id, max_residues=28):
    """
    Keeps only the residues within the loop pocket radius in the structure, up to a maximum number of residues.
    Ensures all residues in the loop defined by loop_start to loop_stop are included.
    """
    # Get the specific chain
    chain = struct[0][chain_id.id]
    residues_list = list(chain.get_residues())
    # Ensure indices are valid (0-based indexing in Python)
    loop_start_index = loop_start - 1
    loop_stop_index = loop_stop - 1
    # Collect all residues in the loop
    loop_residues = residues_list[loop_start_index:loop_stop_index + 1]
    # Calculate midpoint of the loop and its coordinates
    mid_res_index = (loop_start_index + loop_stop_index) // 2
    mid_residue_coords = loop_residues[mid_res_index - loop_start_index]["CA"].get_coord()
    # Calculate distance for all residues to the loop midpoint
    all_residues_with_distances = [(res, np.linalg.norm(res["CA"].get_coord() - mid_residue_coords)) for res in residues_list]
    # Sort residues by distance to the midpoint (excluding the loop residues)
    sorted_residues_by_distance = sorted(
        [res for res in all_residues_with_distances if res[0] not in loop_residues],
        key=lambda x: x[1])
    # Number of additional residues we can include
    additional_res_needed = max_residues - len(loop_residues)
    # Combine loop residues with the closest other residues, not exceeding max_residues
    residues_to_keep = loop_residues + [res[0] for res in sorted_residues_by_distance[:additional_res_needed]]
    return residues_to_keep


def get_residues_around_residue(res_id, struct, chain, nr_residues=24):
    residues_list = []
    for chain in struct[0]:
        for res in chain:
            residues_list.append(res)
            
    residue_coords = np.array(list(struct.get_residues())[res_id - 1]["CA"].get_coord())
    residues_sorted_by_distance = sorted(residues_list, key=lambda residue: np.linalg.norm(residue["CA"].get_coord() - residue_coords))
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

class NonHydrogenSelect(Select):
    """ Class to exclude hydrogen atoms from being written to the PDB file. """
    def accept_atom(self, atom):
        return not atom.element.strip() == 'H'
    
def write_loop_to_pdb(struct, loop_type, outpath, loop_position, loop_position_target):
    """write extracted loop to file and append extraction info"""
    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    pdb_io.save(outpath, select=NonHydrogenSelect())
    with open(outpath, "a") as file:
        file.write(f"loop_type {loop_type}\n")
        file.write(f"loop_position {loop_position}\n")
        file.write(f"loop_position_target {loop_position_target}\n")
    logging.info(f"Extracted loop {loop_position_target} to {outpath}")


def write_residue_pocket_to_pdb(struct, outpath, res_id, residue_position_tensor):
    """write extracted loop to file and append extraction info"""
    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    pdb_io.save(outpath, select=NonHydrogenSelect())
    with open(outpath, "a") as file:
        file.write(f"residue_position_tensor {residue_position_tensor}\n")
    logging.info(f"Extracted residue pocket for residue {res_id} to {outpath}")

def extract_loops(path_pdb, outpath, ss_frag, ss_frag_size, loop_max_size, nr_residues):
    outpath_loops = f"{outpath}/extracted_loops"
    os.makedirs(outpath_loops, exist_ok=True)
    
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure("protein", path_pdb)
    for chain in structure[0]:
        path_pdb = rosetta_numbering(path_pdb, chain)
        ss, sequence = get_BioPython_DSSP(path_pdb, "/home/florian_wieser/miniconda3/envs/elen_test/bin/mkdssp")
        print_ruler(ss, sequence)
        loop_positions = get_loop_positions(ss, ss_frag, ss_frag_size, loop_max_size)
        for idx, loop in enumerate(loop_positions): # iterate of found loop positions
            residues_to_keep = []
            #structure = pdb_parser.get_structure("protein", path_pdb)
            # for chain in structure[0]:
            # loop extraction machinery - will extract all kind of EE, EH, HE, HH loops
            loop_type = f"{ss[loop[0] - 2]}{ss[loop[1]]}"
            structure_tmp = structure.copy()
            residues_to_keep = get_residues_around_loop(loop[0], loop[1], structure_tmp, chain, nr_residues)
            structure_final = remove_all_but_loop(structure_tmp, residues_to_keep)
            # calculate general metrics about the loop, to be appended to the .pdb file, when extracted
            # sort residues according to later res numbering    
            sorted_residues = sorted(residues_to_keep, key=lambda residue: residue.get_id()[1])
            for idx_ten, residue in enumerate(sorted_residues):
                if residue.get_id()[1] == loop[0]:
                    loop_start_ext = idx_ten
                if residue.get_id()[1] == loop[1]:
                    loop_stop_ext = idx_ten
            loop_position_target = f"{str(loop[0])} {str(loop[1])}"
            loop_position = f"{str(loop_start_ext)} {str(loop_stop_ext)}"
            fname = f"{os.path.basename(path_pdb)[:-4]}_{str(idx + 1)}_{loop_type}.pdb"
            outpath_pdb = os.path.join(outpath_loops, fname)
            write_loop_to_pdb(structure_final, loop_type, outpath_pdb, loop_position, loop_position_target)


class ResidueSelect(Select):
    def accept_residue(self, residue):
        # Only accept residues that are standard amino acids
        return residue.id[0] == ' '


def clean_structure(original_structure):
    filtered_structure = Structure.Structure('Filtered_PDB_ID')
    for model in original_structure:
        # Create a new Model object
        new_model = Model.Model(model.id)
        filtered_structure.add(new_model)
        for chain in model:
            new_chain = Chain.Chain(chain.id)
            new_model.add(new_chain)
            for residue in chain:
                if ResidueSelect().accept_residue(residue):
                    new_chain.add(residue)
    return filtered_structure


def extract_residues(path_pdb, outpath, nr_residues):
    outpath_loops = f"{outpath}/extracted_residues"
    os.makedirs(outpath_loops, exist_ok=True)
    pdb_parser = PDBParser()
    original_structure = pdb_parser.get_structure('PDB_ID', path_pdb)
    # Create a new Structure object to hold the filtered structure
    new_model = clean_structure(original_structure)
    
    for chain in new_model:
        path_pdb = rosetta_numbering(path_pdb, chain)
        total_residues = get_total_number_of_residues(path_pdb)
        
        for res_id in range(1, total_residues + 1):
            residues_to_keep = []
            structure_tmp = new_model.copy()
            residues_to_keep = get_residues_around_residue(res_id, structure_tmp, chain, nr_residues)
            structure_final = remove_all_but_loop(structure_tmp, residues_to_keep)
            sorted_residues = sorted(residues_to_keep, key=lambda residue: residue.get_id()[1])
            
            for idx_ten, residue in enumerate(sorted_residues):
                if residue.get_id()[1] == res_id:
                    residue_position_tensor = idx_ten
            fname = f"{os.path.basename(path_pdb)[:-4]}_{str(res_id)}.pdb"
            outpath_pdb = os.path.join(outpath_loops, fname)
            write_residue_pocket_to_pdb(structure_final, outpath_pdb, res_id, residue_position_tensor)


 
################################################################################
if __name__ == "__main__":
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument("--inpath", type=str, default=f"input_folder")
    parser.add_argument("--outpath", type=str, default=f"output_folder")
    # old settings ss_frag None ss_frag_size 6 loop_max_size 10 nr_residues 22
    # new settings ss_frag None ss_frag_size 4 loop_max_size 10 nr_residues 28
    parser.add_argument("--ss_frag", type=str, default="None", choices=["None", "helix", "sheet"],
                        help="defaults to extract any kind of loop, e.g. EHEHHELLLHEHEHEH")
    parser.add_argument("--ss_frag_size", type=int, default=2)
    parser.add_argument("--loop_max_size", type=int, default=12)
    parser.add_argument("--nr_residues", type=int, default=28)
    parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite existing run")

    args = parser.parse_args()
    if args.ss_frag == "None": args.ss_frag == None
    extract_loops(args.inpath, args.outpath, args.ss_frag, args.ss_frag_size, args.loop_max_size, args.nr_residues)
    extract_residues(args.inpath, args.outpath, args.nr_residues)
