#!/usr/bin/env python3
import warnings
import os
import sys
import shutil
import subprocess
import logging
from typing import List, Tuple, Optional
from Bio import BiopythonDeprecationWarning
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser, PDBIO, Select, Structure, Model, Chain
from Bio.PDB.DSSP import DSSP
from Bio import pairwise2
import numpy as np
from elen.inference.utils_inference import get_total_number_of_residues
from elen.config import PATH_DSSP
from pyrosetta.toolbox.cleaning import cleanATOM

# Suppress noisy warnings from BioPython
warnings.simplefilter('ignore', BiopythonDeprecationWarning)
warnings.simplefilter('ignore', PDBConstructionWarning)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

### HELPERS ###################################################################

def clean_pdb(path_pdb: str, outpath: str) -> Optional[str]:
    """
    Cleans a PDB file by extracting ATOM and TER records using PyRosetta's cleanATOM.

    Args:
        path_pdb (str): The path to the original PDB file.
        outpath (str): The output directory for the cleaned PDB.

    Returns:
        Optional[str]: The path to the cleaned PDB file, or None if cleaning failed.
    """
    try:
        base_name = os.path.splitext(os.path.basename(path_pdb))[0]
        path_cleaned_pdb = os.path.join(outpath, f"{base_name}.pdb")
        cleanATOM(path_pdb, out_file=path_cleaned_pdb)
        return path_cleaned_pdb
    except Exception as e:
        logging.error(f"Error in clean_pdb: {e}")
        return None

def rosetta_numbering(path_pdb: str, chain) -> str:
    """
    Convert residue numbering of chain to Rosetta numbering (starting with 1).

    Args:
        path_pdb (str): Path to PDB file (will be overwritten).
        chain: BioPython Chain object.

    Returns:
        str: Path to the renumbered PDB.
    """
    residues = chain.get_residues()
    first_residue = next(residues)
    residue_id = first_residue.get_id()
    is_first_zero = (residue_id[1] != 1)
    if is_first_zero:
        for idx, res in enumerate(chain, start=1):
            res.id = (" ", idx, " ")
    pdb_io = PDBIO()
    pdb_io.set_structure(chain)
    pdb_io.save(path_pdb)
    return path_pdb

def get_first_residues_id(chain) -> int:
    """
    Get the sequence number of the first residue in a chain.
    """
    residues = chain.get_residues()
    first_residue = next(residues)
    residue_id = first_residue.get_id()
    return residue_id[1]

def print_ruler(ss: str, sequence: str) -> None:
    """
    Print the sequence and secondary structure with ruler below for inspection.
    """
    chunk_size = 100
    for i in range(0, len(sequence), chunk_size):
        print(sequence[i:i+chunk_size])
        print(ss[i:i+chunk_size])
        print("1234567890" * 10)
        print("1   5   10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90   95  100")

def get_BioPython_DSSP(fname: str, dssp_executable: str) -> Tuple[str, str]:
    """
    Runs DSSP and returns simplified secondary structure and sequence.

    Args:
        fname (str): Path to the PDB file.
        dssp_executable (str): Path to the DSSP executable.

    Returns:
        Tuple[str, str]: (secondary_structure, sequence)
    """
    model = PDBParser().get_structure("new_protein", fname)[0]
    dssp = DSSP(model, fname, dssp=dssp_executable)
    sequence = "".join([dssp[res_id][1] for res_id in dssp.keys()])
    ss_orig = "".join([dssp[res_id][2] for res_id in dssp.keys()])
    # Map DSSP codes to H/E/L
    ss = (ss_orig.replace("B", "E").replace("G", "H").replace("I", "H")
                   .replace("T", "L").replace("-", "L").replace("S", "L"))
    return ss, sequence

def get_sequence_identity(seq1: str, seq2: str) -> float:
    """
    Returns sequence identity between two sequences in percent.
    """
    alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True, score_only=True)
    identity = (alignment / max(len(seq1), len(seq2))) * 100
    return identity

def get_loop_positions(
    ss: str, ss_frag: Optional[str], ss_frag_size: int, loop_max_size: int
) -> List[Tuple[int, int]]:
    """
    Find loop regions in secondary structure string matching criteria.

    Returns:
        List of (start, stop) positions for each loop (1-based indexing).
    """
    if ss_frag == "helix":
        ss_opt_1, ss_opt_2 = "H", "H"
    elif ss_frag == "sheet":
        ss_opt_1, ss_opt_2 = "E", "E"
    else:
        ss_opt_1, ss_opt_2 = "H", "E"

    loop_positions = []
    i = 0
    while i < len(ss):
        ss_counter = 0
        while i < len(ss) and (ss[i] == ss_opt_1 or ss[i] == ss_opt_2):
            ss_counter += 1
            i += 1
        if ss_counter >= ss_frag_size:
            loop_counter = 0
            loop_start = i + 1
            while i < len(ss):
                if ss[i] == "L":
                    loop_counter += 1
                    i += 1
                elif (ss[i] == ss_opt_1 or ss[i] == ss_opt_2) and (i + 1 < len(ss) and ss[i + 1] == "L"):
                    loop_counter += 1
                    i += 1
                else:
                    break
            if 2 <= loop_counter <= loop_max_size:
                loop_stop = i
                ss_counter = 0
                while i < len(ss) and (ss[i] == ss_opt_1 or ss[i] == ss_opt_2):
                    ss_counter += 1
                    i += 1
                if ss_counter >= ss_frag_size:
                    loop_positions.append((loop_start, loop_stop))
                    i = loop_stop
                    continue
        i += 1
    return loop_positions

def get_residues_around_loop(loop_start: int, loop_stop: int, struct, chain_id, max_residues: int = 28):
    """
    Returns residues around a loop region for extraction.
    """
    chain = struct[0][chain_id.id]
    residues_list = list(chain.get_residues())
    loop_start_index = loop_start - 1
    loop_stop_index = loop_stop - 1
    loop_residues = residues_list[loop_start_index:loop_stop_index + 1]
    mid_res_index = (loop_start_index + loop_stop_index) // 2
    mid_residue_coords = loop_residues[mid_res_index - loop_start_index]["CA"].get_coord()
    all_residues_with_distances = [
        (res, np.linalg.norm(res["CA"].get_coord() - mid_residue_coords))
        for res in residues_list
    ]
    sorted_residues_by_distance = sorted(
        [res for res in all_residues_with_distances if res[0] not in loop_residues],
        key=lambda x: x[1])
    additional_res_needed = max_residues - len(loop_residues)
    residues_to_keep = loop_residues + [res[0] for res in sorted_residues_by_distance[:additional_res_needed]]
    return residues_to_keep

def get_residues_around_residue(res_id: int, struct, chain, nr_residues: int = 24):
    """
    Returns residues around a given residue index.
    """
    residues_list = [res for chain in struct[0] for res in chain]
    residue_coords = np.array(list(struct.get_residues())[res_id - 1]["CA"].get_coord())
    residues_sorted_by_distance = sorted(
        residues_list, key=lambda residue: np.linalg.norm(residue["CA"].get_coord() - residue_coords)
    )
    residues_to_keep = residues_sorted_by_distance[:nr_residues]
    return residues_to_keep

def remove_all_but_loop(struct, residues_to_keep):
    """
    Removes all residues except those in residues_to_keep from the structure.
    """
    chain_id = ""
    for chain in struct[0]:
        chain_id = chain.id
    for res in residues_to_keep:
        res.parent.id = chain_id
    for chain in struct[0]:
        residues_to_remove = [residue for residue in chain if residue not in residues_to_keep]
        for residue in residues_to_remove:
            chain.detach_child(residue.id)
    return struct

class NonHydrogenSelect(Select):
    """Excludes hydrogen atoms when writing PDB."""
    def accept_atom(self, atom):
        return not atom.element.strip() == 'H'

def write_loop_to_pdb(struct, loop_type, outpath, loop_position, loop_position_target):
    """
    Save a loop structure to a file, including metadata.
    """
    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    pdb_io.save(outpath, select=NonHydrogenSelect())
    with open(outpath, "a") as file:
        file.write(f"loop_type {loop_type}\n")
        file.write(f"loop_position {loop_position}\n")
        file.write(f"loop_position_target {loop_position_target}\n")
    logging.info(f"Extracted loop {loop_position_target} to {outpath}")

def write_residue_pocket_to_pdb(struct, outpath, res_id, residue_position_tensor):
    """
    Save a residue pocket structure to a file, including metadata.
    """
    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    pdb_io.save(outpath, select=NonHydrogenSelect())
    with open(outpath, "a") as file:
        file.write(f"residue_position_tensor {residue_position_tensor}\n")
    logging.info(f"Extracted residue pocket for residue {res_id} to {outpath}")

def extract_loops(path_pdb, outpath, ss_frag, ss_frag_size, loop_max_size, nr_residues):
    """
    Extracts loop regions from a PDB and saves them in outpath.
    """
    outpath_loops = os.path.join(outpath, "extracted_loops")
    os.makedirs(outpath_loops, exist_ok=True)
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure("protein", path_pdb)
    for chain in structure[0]:
        path_pdb = rosetta_numbering(path_pdb, chain)
        ss, sequence = get_BioPython_DSSP(path_pdb, PATH_DSSP)
        print_ruler(ss, sequence)
        loop_positions = get_loop_positions(ss, ss_frag, ss_frag_size, loop_max_size)
        for idx, loop in enumerate(loop_positions):
            loop_type = f"{ss[loop[0] - 2]}{ss[loop[1]]}"
            structure_tmp = structure.copy()
            residues_to_keep = get_residues_around_loop(loop[0], loop[1], structure_tmp, chain, nr_residues)
            structure_final = remove_all_but_loop(structure_tmp, residues_to_keep)
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
    """Select only standard amino acid residues."""
    def accept_residue(self, residue):
        return residue.id[0] == ' '

def clean_structure(original_structure):
    """
    Returns a new structure with only standard residues.
    """
    filtered_structure = Structure.Structure('Filtered_PDB_ID')
    for model in original_structure:
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
    """
    Extract residue pockets around each residue and save them in outpath.
    """
    outpath_res = os.path.join(outpath, "extracted_residues")
    os.makedirs(outpath_res, exist_ok=True)
    pdb_parser = PDBParser()
    original_structure = pdb_parser.get_structure('PDB_ID', path_pdb)
    new_model = clean_structure(original_structure)
    for chain in new_model:
        path_pdb = rosetta_numbering(path_pdb, chain)
        total_residues = get_total_number_of_residues(path_pdb)
        for res_id in range(1, total_residues + 1):
            structure_tmp = new_model.copy()
            residues_to_keep = get_residues_around_residue(res_id, structure_tmp, chain, nr_residues)
            structure_final = remove_all_but_loop(structure_tmp, residues_to_keep)
            sorted_residues = sorted(residues_to_keep, key=lambda residue: residue.get_id()[1])
            for idx_ten, residue in enumerate(sorted_residues):
                if residue.get_id()[1] == res_id:
                    residue_position_tensor = idx_ten
            fname = f"{os.path.basename(path_pdb)[:-4]}_{str(res_id)}.pdb"
            outpath_pdb = os.path.join(outpath_res, fname)
            write_residue_pocket_to_pdb(structure_final, outpath_pdb, res_id, residue_position_tensor)

################################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract loops and residue pockets from PDB files.")
    parser.add_argument("--inpath", type=str, default="input_folder", help="Input PDB file")
    parser.add_argument("--outpath", type=str, default="output_folder", help="Output folder")
    parser.add_argument("--ss_frag", type=str, default=None, choices=[None, "helix", "sheet"],
                        help="Secondary structure type (helix/sheet/None for any)")
    parser.add_argument("--ss_frag_size", type=int, default=2, help="Min. secondary structure fragment size")
    parser.add_argument("--loop_max_size", type=int, default=12, help="Max loop size")
    parser.add_argument("--nr_residues", type=int, default=28, help="Total residues in a pocket/loop")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output")
    args = parser.parse_args()

    extract_loops(args.inpath, args.outpath, args.ss_frag, args.ss_frag_size, args.loop_max_size, args.nr_residues)
    extract_residues(args.inpath, args.outpath, args.nr_residues)

if __name__ == "__main__":
    main()
