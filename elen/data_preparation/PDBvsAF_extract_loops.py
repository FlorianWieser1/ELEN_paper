#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J extract_loops
#SBATCH -o 2_extract_loops.log
#SBATCH -e 2_extract_loops.err
import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.DSSP import DSSP
from Bio.PDB.SASA import ShrakeRupley
import os
import sys
import glob
import shutil
import argparse as ap
import numpy as np
from pyrosetta import *
from pyrosetta.io import pose_from_pdb
import pyrosetta.rosetta.core.pose
init("-mute all")
from elen.config import PATH_DSSP
from elen.data_preparation.utils_extraction import get_sequence_identity


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
        while ss[i] == ss_opt_1 or ss[i] == ss_opt_2:
            ss_counter += 1
            i += 1
        loop_start = i + 1
        if ss_counter >= ss_frag_size:  # check if SS fragment is at least 4
            loop_counter = 0
            # find loop fragment
            while ss[i] == "L" and i < len(ss) - 1:
                loop_counter += 1
                i += 1
            if loop_counter >= 2 and loop_counter <= loop_max_size:  # check if loop has good size
                loop_stop = i
                ss_counter = 0
                # find SS fragment after loop
                while ss[i] == ss_opt_1 or ss[i] == ss_opt_2:
                    ss_counter += 1
                    i += 1
                if ss_counter >= ss_frag_size:
                    loop_positions.append((loop_start, loop_stop))
                    i = loop_stop - 2  # return to start of last SS fragment
                    continue
        i += 1
    return loop_positions


def get_residues_around_loop(loop_start, loop_stop, struct, chain_id, max_residues=24):
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
    all_residues_with_distances = [
        (res, np.linalg.norm(res["CA"].get_coord() - mid_residue_coords))
        for res in residues_list]
    # Sort residues by distance to the midpoint (excluding the loop residues)
    sorted_residues_by_distance = sorted(
        [res for res in all_residues_with_distances if res[0] not in loop_residues],
        key=lambda x: x[1])
    # Number of additional residues we can include
    additional_res_needed = max_residues - len(loop_residues)
    # Combine loop residues with the closest other residues, not exceeding max_residues
    residues_to_keep = loop_residues + [res[0] for res in sorted_residues_by_distance[:additional_res_needed]]
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


def extract_residue_stretch(args, struct, chain, start_residue, end_residue):
    """extracts loop residue stretch from protein and returns it as BioPython structure"""
    # fill loop residues
    residues_to_keep = [ (' ', res, ' ') for res in range(start_residue, end_residue + 1, 1)]
    
    # attach residues at both sides until max_residues reached
    inc = 0
    while len(residues_to_keep) < args.nr_residues:
        residues_to_keep.append((' ', start_residue - inc - 1, ' '))
        if len(residues_to_keep) == args.nr_residues:
            continue
        residues_to_keep.append((' ', end_residue + inc + 1, ' '))
        inc += 1
    
    # remove all the other residues than the selected ones
    for chain in struct[0]:
        residues_to_remove = []
        for residue in chain:
            if residue.id not in residues_to_keep:
                residues_to_remove.append(residue)
        for residue in residues_to_remove:
            chain.detach_child(residue.id)
    return struct


def write_to_pdb(args, struct, loop_type, loop_length, total_residues, loop_sequence, sasa, fname, status, loop_position, loop_position_target):
    """write extracted loop to file and append metrics/info"""
    path_pdb = os.path.join(args.outpath, "AF_models", fname + ".pdb")
    fname = fname + ".pdb"
    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    pdb_io.save(fname)
    if "_native_" not in path_pdb:
        with open(fname, "a") as file:
            file.write(f"loop_length {loop_length}\n")
            file.write(f"total_residues {total_residues}\n")
            file.write(f"loop_type {loop_type}\n")
            file.write(f"loop_sequence {loop_sequence}\n")
            file.write(f"sasa {str(sasa)[:8]}\n")
            file.write(f"surf_stat {str(status)}\n")
            file.write(f"loop_position {loop_position}\n")
            file.write(f"loop_position_target {loop_position_target}\n")
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

def get_bio_SASA(loop, chain_struct, chain_id):
    """compute SASA of loop residues and return the average"""
    sr = ShrakeRupley()
    sasa = sr.compute(chain_struct, level="R")
    sasa_list = [chain_struct[0][chain_id]
                 [res].sasa for res in range(loop[0], loop[1] + 1)]
    return sum(sasa_list) / len(sasa_list)

def is_loop_on_surface(loop, chain_struct, chain_id):
    sasa_total = {"ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0, "GLN": 225.0,
                  "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
                  "MET": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0, "THR": 172.0, "TRP": 285.0,
                  "TYR": 263.0, "VAL": 174.0}  # from Tien MZ "Maximum allowed solvent accessibility.."
    sr = ShrakeRupley()  # probe radius = 1.4
    sasa = sr.compute(chain_struct, level="R")  # @level: R - residue
    sasa_list = []
    residue_id_list = []
    for res in range(loop[0] - 0, loop[1] + 1 + 0):
        sasa_list.append(chain_struct[0][chain_id][res].sasa)
        residue_id_list.append(chain_struct[0][chain_id][res].resname)
    status_list = []
    for sasa, res_type in zip(sasa_list, residue_id_list):
        relative_sasa = sasa / sasa_total[res_type]
        if sasa / sasa_total[res_type] > 0.2:
            status_list.append(1)
        else:
            status_list.append(0)
    if (sum(status_list) / len(status_list)) >= 0.5:
        status = "exposed"
    else:
        status = "buried"
    return status

def get_total_residues(struct):
    total_residues = 0
    for model in struct:
        for chain in model:
            for res in chain:
                total_residues += 1
    return total_residues

def print_ruler(ss, sequence):
    print(sequence)
    print(ss)
    print("123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890")
    print("1   5   10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90")

    
#######################################
def extract_loops(args, path_native, logger):
    pdb_parser = PDBParser()
    struct_native = pdb_parser.get_structure("protein", path_native)
    fname_native = path_native.replace(".pdb", "")
   
    # hotfix: make tmp folder for processed native structs
    #dirpath_natives_BIO = os.path.join(args.outpath, "natives_tmp")
    dirpath_natives_BIO = "natives_split"
    #if os.path.exists(dirpath_natives_BIO):
    #    shutil.rmtree(dirpath_natives_BIO)
    os.makedirs(dirpath_natives_BIO, exist_ok=True)

    for chain in struct_native[0]:
        chain = rosetta_numbering(chain)  # let the residue numbering start with 1
        # dump chain as .pdb file in order to get ss and sequence
        io = PDBIO()
        io.set_structure(chain)
        fname_chain = f"{fname_native}_{chain.id}_native.pdb"
        
        path_chain = os.path.join(dirpath_natives_BIO, f"{os.path.basename(fname_native)}_{chain.id}_native.pdb")
        io.save(path_chain)
        ss, sequence = get_BioPython_DSSP(path_chain, PATH_DSSP)  # get secondary structure
        print_ruler(ss, sequence)
        loop_positions = get_loop_positions(ss, args.ss_frag, args.ss_frag_size, args.loop_max_size)
        logger.info(f"Extracting loops for {path_native}.")
        logger.debug(f"{loop_positions}") 
        logger.debug(f"with setting ss_frag {args.ss_frag} ss_frag_size {args.ss_frag_size} loop_max_size {args.loop_max_size}")
                    
        # set up list with .pdb files with loops to be extracted
        identifier = os.path.basename(fname_chain)[:6]
        path_models = glob.glob(f"{args.inpath_models}/{identifier}*")
        path_models.insert(0, path_chain) # add native to the beginning of the pathlist
        
        for idx, loop in enumerate(loop_positions): # iterate of found loop positions
            residues_to_keep = []
            for path_pdb in path_models: # iterate over list of .pdb files
                logger.debug(f"Extracting loop {loop} of {path_pdb}.") 
                chain_struct = pdb_parser.get_structure("protein", path_pdb)
                #print(f"chain_struct: {chain_struct}")
                #for chain in chain_struct:
                #    print(f"chain: {chain}")
                #    for residue in chain:
                #        print(f"residue: {residue}")
                #for chain_native in next(chain_struct.get_models()):
                ##model = list(chain_struct.get_models())[0]
                for model in chain_struct:
                    for chain_native in model:
                        # loop extraction machinery - will extract all kind of HH, EH, EE, HE, EE loops
                        loop_type = f"{ss[loop[0] - 2]}{ss[loop[1]]}"
                        fname_final = f"{os.path.basename(path_pdb)[:-4]}_{str(idx+1)}_{loop_type}"
                        
                        tmp_struct = chain_struct.copy()
                        # two modes: either extract loop stretch or loop pocket
                        if args.loop_mode == "loop_pocket":
                            # residues to keep is set initially with native structure
                            if "_native_" in fname_final:
                                residues_to_keep = get_residues_around_loop(
                                    loop[0], loop[1], tmp_struct, chain_native, args.nr_residues)
                                fin_struct = remove_all_but_loop(tmp_struct, residues_to_keep)
                            else:
                                fin_struct = remove_all_but_loop(tmp_struct, residues_to_keep)
                        elif args.loop_mode == "loop_stretch":
                            fin_struct = extract_residue_stretch(args, tmp_struct, chain_native, loop[0], loop[1])
                            
                        # calculate general metrics about the loop, to be appended to the .pdb file, when extracted
                        # sort residues according to later res numbering    
                        sorted_residues = sorted(residues_to_keep, key=lambda residue: residue.get_id()[1])
                        for idx_ten, residue in enumerate(sorted_residues):
                            if residue.get_id()[1] == loop[0]:
                                loop_start_ext = idx_ten
                            if residue.get_id()[1] == loop[1]:
                                loop_stop_ext = idx_ten
                            
                        
                        loop_length = loop[1] - loop[0] + 1
                        total_residues = get_total_residues(fin_struct)
                        sasa = get_bio_SASA(loop, chain_struct, chain_native.id)
                        surface_status = is_loop_on_surface(loop, chain_struct, chain_native.id)
                        loop_sequence = sequence[loop[0] - 2: loop[1] + 1]
                        
                        loop_position_target = f"{str(loop[0])} {str(loop[1])}"
                        loop_position = f"{str(loop_start_ext)} {str(loop_stop_ext)}"
                  
                        path_pdb = write_to_pdb(args, fin_struct, loop_type, loop_length, total_residues,
                                     loop_sequence, sasa, fname_final, surface_status, loop_position, loop_position_target)
        #os.remove(path_chain)


##############################################################################################################
def main(args, logger):
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    os.makedirs(os.path.join(args.outpath, "natives"), exist_ok=True)
    os.makedirs(os.path.join(args.outpath, "AF_models"), exist_ok=True)

    pdb_files = glob.glob(os.path.join(args.inpath_natives, "*.pdb"))
    for input_pdb in pdb_files:
        extract_loops(args, input_pdb, logger)
        """
        try:
            extract_loops(args, input_pdb)
        except FileNotFoundError:
            print(f"Issue in extract_loops. Skipping {input_pdb}")
        """
    for tmp_file in glob.glob("*tmp*.pdb"):
        os.remove(tmp_file)
    print("Done.")

    # check integrity of extracted loop couples (SI and length of natives vs AF models)
    for path_native in glob.glob(os.path.join(args.outpath, "natives/*.pdb")):
        identifier_PDB = os.path.basename(path_native)[:6]
        identifier_loop = os.path.basename(path_native)[-9:]
        for path_model in glob.glob(os.path.join(args.outpath, f"AF_models/{identifier_PDB}*{identifier_loop}")):
            _, seq_native = get_BioPython_DSSP(path_native, PATH_DSSP)
            _, seq_model = get_BioPython_DSSP(path_native, PATH_DSSP)
            sequence_identity = get_sequence_identity(seq_native, seq_model)
            assert sequence_identity == 100, f"SEQUENCES DONT MATCH! {seq_native, seq_model}"
            assert len(seq_native) == len(seq_model), f"DIFFERENT LENGTH! {len(seq_native), len(seq_model)}"

################################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PDBvsAF_extract_loops")
    
    parser = ap.ArgumentParser()
    default_path = "/home/florian_wieser/software/ARES/geometricDL/edn/ELEN_testing/cameo_1bug"
    parser.add_argument("--inpath_natives", type=str, default=f"{default_path}/natives")
    parser.add_argument("--inpath_models", type=str, default=f"{default_path}/AF_predictions")
    parser.add_argument("--outpath", type=str, default=f"{default_path}/extracted_loops")
    # old settings ss_frag None ss_frag_size 6 loop_max_size 10 nr_residues 22
    parser.add_argument("--ss_frag", type=str, default="None", choices=["None", "helix", "sheet"],
                        help="defaults to extract any kind of loop, e.g. EHEHHELLLHEHEHEH")
    parser.add_argument("--ss_frag_size", type=int, default=4)
    parser.add_argument("--loop_max_size", type=int, default=6)
    parser.add_argument("--loop_mode", type=str, default="loop_pocket", choices=["loop_stretch", "loop_pocket"])
    parser.add_argument("--nr_residues", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite existing run")

    args = parser.parse_args()
    if args.ss_frag == "None": args.ss_frag = None
    main(args, logger)
