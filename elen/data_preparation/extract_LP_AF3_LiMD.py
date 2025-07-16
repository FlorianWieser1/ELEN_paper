#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J extract_loops
#SBATCH -o extract_loops_AF3_LiMD.log
#SBATCH -e extract_loops_AF3_LiMD.err

# TODO fix extracted LP are not fixed in length!!
import warnings
import logging
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
from elen.shared_utils.utils_others import func_timer, discard_pdb

###############################################################################
def get_BioPython_DSSP(fname, dssp_executable):
    """DSSP according to BioPython (using mkdssp executable). Returns secondary structure and sequence."""
    model = PDBParser().get_structure("new_protein", fname)[0]
    dssp = DSSP(model, fname, dssp=dssp_executable)
    # Convert BioPython's dssp dictionary to sequence and secondary_structure strings, respectively
    sequence = "".join([dssp[res_id][1] for res_id in dssp.keys()])
    ss_orig = "".join([dssp[res_id][2] for res_id in dssp.keys()])
    # Convert dssp nomenclature to simple one: H - helix, E - sheet, L - loop
    ss = (ss_orig.replace("B", "E")
               .replace("G", "H")
               .replace("I", "H")
               .replace("T", "L")
               .replace("-", "L")
               .replace("S", "L"))
    return ss, sequence

###############################################################################
def get_loop_positions(ss, ss_frag, ss_frag_size, loop_max_size):
    """
    Identify loop positions based on upstream and downstream secondary structure.
    If ss_frag is None, accept either H or E as flanking structure.
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

        loop_start = i + 1  # 1-based
        if ss_counter >= ss_frag_size:
            loop_counter = 0
            while i < len(ss) and ss[i] == "L":
                loop_counter += 1
                i += 1
            if 2 <= loop_counter <= loop_max_size:
                loop_stop = i
                ss_counter = 0
                while i < len(ss) and (ss[i] == ss_opt_1 or ss[i] == ss_opt_2):
                    ss_counter += 1
                    i += 1
                if ss_counter >= ss_frag_size:
                    loop_positions.append((loop_start, loop_stop))
                    i = loop_stop - 2
                    continue
        i += 1

    return loop_positions

###############################################################################
def rosetta_numbering(chain):
    """Reassign residue numbers sequentially starting at 1."""
    residue_idx = 1
    for res in chain:
        res.id = (" ", residue_idx, " ")
        residue_idx += 1
    return chain

###############################################################################
def get_atom_coords(res):
    """
    Return a representative coordinate for a residue.
    If 'CA' exists, return its coordinate; otherwise return the centroid.
    """
    if "CA" in res:
        return res["CA"].get_coord()
    else:
        coords = [atom.get_coord() for atom in res]
        return np.mean(coords, axis=0)

from Bio.PDB.Polypeptide import is_aa
from math import dist
def get_residues_around_loop(loop_start, loop_stop, struct, chain_id, max_residues=24):
    """
    Identify the loop pocket residues:
      1) The loop residues (protein only) from the designated chain.
      2) Additional protein residues (from that chain) based on proximity.
      3) All non-water HETATM residues, metals, ligands, etc. from all chains
         if they fall within a distance threshold from the loop’s midpoint.
         
    This updated function guarantees that the number of standard amino acids
    (protein residues) returned is fixed to max_residues (if available), while
    any ligand or non–protein residues meeting the criteria are added on top.
    """
    model = struct[0]
    chain = model[chain_id]
    # Get all standard amino acid residues from the designated chain.
    protein_residues = [res for res in chain if is_aa(res, standard=True)]
    if loop_stop > len(protein_residues):
        loop_stop = len(protein_residues)
    loop_residues = protein_residues[loop_start - 1:loop_stop]

    if not loop_residues:
        return []

    # Determine the midpoint of the loop based on the central residue.
    mid_res_index = (len(loop_residues) - 1) // 2
    midpoint_res = loop_residues[mid_res_index]
    if "CA" not in midpoint_res:
        # If the midpoint residue has no CA, return at most max_residues from loop_residues.
        protein_to_keep = loop_residues[:max_residues]
        return protein_to_keep

    mid_residue_coords = midpoint_res["CA"].get_coord()

    # --- Select protein (standard amino acid) residues ---
    # Case 1: More (or equal) loop residues than needed -- trim by proximity.
    if len(loop_residues) >= max_residues:
        sorted_loop = sorted(loop_residues,
                             key=lambda r: dist(r["CA"].get_coord(), mid_residue_coords))
        protein_to_keep = sorted_loop[:max_residues]
        # Use the farthest distance among the chosen protein residues as threshold.
        distance_threshold = max(dist(r["CA"].get_coord(), mid_residue_coords)
                                 for r in protein_to_keep)
    else:
        # Case 2: Fewer loop residues -- add extra protein residues from the same chain.
        protein_to_keep = list(loop_residues)
        remaining_protein = [r for r in protein_residues if r not in loop_residues]
        dist_list = []
        for res in remaining_protein:
            if "CA" in res:
                d = dist(res["CA"].get_coord(), mid_residue_coords)
                dist_list.append((res, d))
        dist_list.sort(key=lambda x: x[1])
        additional_res_needed = max_residues - len(protein_to_keep)
        chosen_protein = [tup[0] for tup in dist_list[:additional_res_needed]]
        protein_to_keep.extend(chosen_protein)
        # Compute the threshold as the maximum distance among the selected protein residues.
        distance_threshold = max(dist(r["CA"].get_coord(), mid_residue_coords)
                                 for r in protein_to_keep)

    # --- Add ligand and other non–water residues from all chains ---
    ligand_residues = set()
    extra_margin = 5.0  # Extra margin for non–protein residues.
    for ch in model:
        for res in ch:
            # Skip if this residue is already selected as a protein residue.
            if res in protein_to_keep:
                continue
            # Skip water molecules.
            if res.get_resname().strip() in ("HOH", "WAT"):
                continue
            try:
                coords = [atom.get_coord() for atom in res]
                c = np.mean(coords, axis=0) if coords else None
            except Exception:
                continue
            if c is None:
                continue
            d = dist(c, mid_residue_coords)
            # Determine the appropriate distance threshold.
            if is_aa(res, standard=True):
                threshold = distance_threshold
            else:
                threshold = distance_threshold + extra_margin
            if d <= threshold:
                # Only add non–protein (e.g. ligand) residues so as not to exceed max_residues for proteins.
                if not is_aa(res, standard=True):
                    ligand_residues.add(res)

    # Combine exactly max_residues of protein residues with any ligand residues.
    residues_to_keep = list(protein_to_keep) + list(ligand_residues)
    return residues_to_keep

def fet_residues_around_loop(loop_start, loop_stop, struct, chain_id, max_residues=24):
    """
    Identify the loop pocket residues:
      1) The loop residues (protein only) from the designated chain.
      2) Additional protein residues (from that chain) based on proximity.
      3) All non-water HETATM residues, metals, ligands, etc. from all chains
         if they fall within a distance threshold from the loop’s midpoint.
    """
    model = struct[0]
    chain = model[chain_id]
    protein_residues = [res for res in chain if is_aa(res, standard=True)]
    if loop_stop > len(protein_residues):
        loop_stop = len(protein_residues)
    loop_residues = protein_residues[loop_start - 1:loop_stop]

    if not loop_residues:
        return []

    mid_res_index = (len(loop_residues) - 1) // 2
    midpoint_res = loop_residues[mid_res_index]
    if "CA" not in midpoint_res:
        return loop_residues
    mid_residue_coords = midpoint_res["CA"].get_coord()
    residues_to_keep = set(loop_residues)

    # Add additional protein residues from the same chain.
    remaining_protein = [r for r in protein_residues if r not in loop_residues]
    dist_list = []
    for res in remaining_protein:
        if "CA" in res:
            d = dist(res["CA"].get_coord(), mid_residue_coords)
            dist_list.append((res, d))
    dist_list.sort(key=lambda x: x[1])
    print(f"len(loop_residues): {len(loop_residues)}")
    print(f"loop_residues: {loop_residues}")
    additional_res_needed = max_residues - len(loop_residues)
    if additional_res_needed < 0:
        additional_res_needed = 0
    chosen_protein = [tup[0] for tup in dist_list[:additional_res_needed]]
    residues_to_keep.update(chosen_protein)

    if chosen_protein:
        distance_threshold = dist_list[:additional_res_needed][-1][1]
    else:
        loop_dists = [dist(r["CA"].get_coord(), mid_residue_coords)
                      for r in loop_residues if "CA" in r]
        distance_threshold = max(loop_dists) if loop_dists else 0.0

    # Extra margin for non–protein residues.
    extra_margin = 5.0

    # Now iterate over all chains to add any non-water extra residues.
    for ch in model:
        for res in ch:
            if res in residues_to_keep:
                continue
            if res.get_resname().strip() in ("HOH", "WAT"):
                continue
            try:
                coords = [atom.get_coord() for atom in res]
                c = np.mean(coords, axis=0) if coords else None
            except Exception:
                continue
            if c is None:
                continue
            d = dist(c, mid_residue_coords)
            # Use a larger threshold if the residue is not a standard amino acid.
            if is_aa(res, standard=True):
                threshold = distance_threshold
            else:
                threshold = distance_threshold + extra_margin
            if d <= threshold:
                residues_to_keep.add(res)

    return list(residues_to_keep)

###############################################################################
def remove_all_but_loop(struct, residues_to_keep):
    """
    Remove any residues outside 'residues_to_keep' from the structure.
    To allow for differences in chain/residue numbering across models (e.g. for ligands),
    we compare unique identifiers (chain id, residue number, insertion code, and residue name).
    """
    keep_ids = set()
    for res in residues_to_keep:
        keep_ids.add((res.get_parent().id, res.id[1], res.id[2], res.get_resname()))
    for chain in struct[0]:
        to_remove = []
        for res in chain:
            identifier = (chain.id, res.id[1], res.id[2], res.get_resname())
            if identifier not in keep_ids:
                to_remove.append(res)
        for res in to_remove:
            chain.detach_child(res.id)
    return struct

###############################################################################
def write_to_pdb(struct,
                 loop_type,
                 loop_length,
                 total_residues,
                 loop_sequence,
                 sasa,
                 outpath,
                 status,
                 loop_position,
                 loop_position_target):
    """Write extracted loop structure to a PDB file and append metadata."""
    pdb_io = PDBIO()
    pdb_io.set_structure(struct)
    pdb_io.save(outpath)
    with open(outpath, "a") as file:
        file.write(f"loop_length {loop_length}\n")
        file.write(f"total_residues {total_residues}\n")
        file.write(f"loop_type {loop_type}\n")
        file.write(f"loop_sequence {loop_sequence}\n")
        file.write(f"sasa {str(sasa)[:8]}\n")
        file.write(f"surf_stat {str(status)}\n")
        file.write(f"loop_position {loop_position}\n")
        file.write(f"loop_position_target {loop_position_target}\n")
    logging.info(f"Extracted loop to {outpath}.")

def print_chain(chain_struct):
    for chain in chain_struct[0]:
        print(f"chain: {chain}")
        for residue in chain:
            print(f"residue: {residue}")

###############################################################################
def get_bio_SASA(loop, chain_struct, chain_id):
    """
    Compute the average SASA of the specified loop range in chain_struct (residues loop[0] to loop[1] inclusive).
    """
    sr = ShrakeRupley()
    sr.compute(chain_struct, level="R")
    sasa_vals = []
    for res_idx in range(loop[0], loop[1] + 1):
        try:
            residue = chain_struct[0][chain_id][(' ', res_idx, ' ')]
            if is_aa(residue, standard=True):
                sasa_vals.append(chain_struct[0][chain_id][res_idx].sasa)
        except KeyError:
            continue
        if not is_aa(residue):
            continue
    return sum(sasa_vals) / len(sasa_vals) if sasa_vals else 0.0

###############################################################################
def is_loop_on_surface(loop, chain_struct, chain_id):
    """
    Decide if a loop is 'exposed' or 'buried' based on the average relative SASA of its residues.
    """
    sasa_total = {
        "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
        "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
        "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
        "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0
    }
    sr = ShrakeRupley()
    sr.compute(chain_struct, level="R")
    sasa_list = []
    residue_id_list = []
    for res_idx in range(loop[0], loop[1] + 1):
        try:
            residue = chain_struct[0][chain_id][(' ', res_idx, ' ')]
            if is_aa(residue, standard=True):
                sasa_list.append(residue.sasa)
                residue_id_list.append(residue.resname)
        except KeyError:
            continue
        if not is_aa(residue):
            continue
    if not sasa_list:
        return "buried"
    status_list = []
    for sasa_val, res_type in zip(sasa_list, residue_id_list):
        if res_type in sasa_total:
            relative_sasa = sasa_val / sasa_total[res_type]
        else:
            relative_sasa = 0.0
        status_list.append(1 if relative_sasa > 0.2 else 0)
    fraction_exposed = sum(status_list) / len(status_list)
    return "exposed" if fraction_exposed >= 0.5 else "buried"

###############################################################################
def get_total_residues(struct):
    """Count total residues in the structure."""
    count = 0
    for model in struct:
        for chain in model:
            for res in chain:
                if res.get_id()[0] == ' ':
                    count += 1
    return count

###############################################################################
def print_ruler(ss, sequence):
    """Print sequence, secondary structure, and a numeric ruler."""
    print(sequence)
    print(ss)
    print("123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890")
    print("1   5   10   15   20   25   30   35   40   45   50   55   60   65   70   75   80   85   90")

###############################################################################
def extract_loops(args, path_BioPython_tmp, path_af, outpath_af3, outpath_natives, outpath_md, discarded_base):
    """
    Main routine for extracting loops from a native PDB file and from a set of AF and MD files that share the same prefix.
    In case of any errors during extraction, the respective AF, native and MD files are copied to the discarded folder.
    """
    try:
        pdb_parser = PDBParser()
        struct_native = pdb_parser.get_structure("protein", path_af)
        fname_native = path_af.replace(".pdb", "")

        # --- Select a protein chain from the native structure ---
        from Bio.PDB.Polypeptide import is_aa
        protein_chain = None
        for chain in struct_native[0]:
            if any(is_aa(res, standard=True) for res in chain):
                chain = rosetta_numbering(chain)
                protein_chain = chain
                break
        if protein_chain is None:
            logging.error("No protein chain found in native structure.")
            discard_pdb(path_af, args.path_discarded, "loop extraction", "no chain found in native structure")

            return

        # Save a temporary PDB using only the protein chain.
        io = PDBIO()
        io.set_structure(protein_chain)
        fname_chain = f"{fname_native}_for_BIOpySS.pdb"
        path_af_BIOpySS = os.path.join(path_BioPython_tmp, f"{os.path.basename(fname_native)}_native.pdb")
        io.save(path_af_BIOpySS)

        # Compute secondary structure and sequence.
        ss, sequence = get_BioPython_DSSP(path_af_BIOpySS, PATH_DSSP)
        print_ruler(ss, sequence)
        loop_positions = get_loop_positions(ss, args.ss_frag, args.ss_frag_size, args.loop_max_size)
        logging.info(f"Extracting loops for {path_af}.")
        logging.debug(f"Loop positions: {loop_positions}")
        logging.debug(f"with setting ss_frag {args.ss_frag} ss_frag_size {args.ss_frag_size} loop_max_size {args.loop_max_size}")

        # Store the chosen protein chain id.
        protein_chain_id = protein_chain.id

        # Gather all models: AF, native, MD.
        identifier = os.path.basename(fname_chain)[:4]
        path_native_files = glob.glob(f"{args.inpath_natives}/{identifier}*")
        path_md = glob.glob(f"{args.inpath_MD}/{identifier}*")
        # We expect one AF (the input file), one native and one MD file.
        path_models = [path_af] + path_native_files + path_md
        assert len(path_models) == 3, "Error: Either AF model, MD frame or native structure not found."

        # For each loop found.
        for idx, loop in enumerate(loop_positions):
            residues_to_keep_native = []  # This will store the pocket computed from the AF model.
            # Process each model.
            for path_pdb in path_models:
                fname_pdb = os.path.basename(path_pdb)
                if "af3.pdb" in fname_pdb and "gro.pdb" not in fname_pdb:
                    pdb_type = "AF"
                elif "gro.pdb" in fname_pdb:
                    pdb_type = "MD"
                else:
                    pdb_type = "native"
                chain_struct = pdb_parser.get_structure("protein", path_pdb)
                try:
                    chain_native = chain_struct[0][protein_chain_id]
                except KeyError:
                    logging.error(f"Chain {protein_chain_id} not found in model {path_pdb}.")
                    discard_pdb(path_pdb, args.path_discarded, "loop extraction", f"Chain {protein_chain_id} not found in model {path_pdb}.")

                    continue

                # Determine loop type.
                up_ss = ss[loop[0] - 2] if (loop[0] - 2) >= 0 else "X"
                down_ss = ss[loop[1]] if loop[1] < len(ss) else "X"
                loop_type = f"{up_ss}{down_ss}"
                fname_final = f"{fname_pdb[:-4]}_{str(idx+1)}_{loop_type}.pdb"
                tmp_struct = chain_struct.copy()

                if pdb_type == "AF":
                    residues_to_keep_native = get_residues_around_loop(loop[0], loop[1], tmp_struct, protein_chain_id, args.nr_residues)
                    fin_struct = remove_all_but_loop(tmp_struct, residues_to_keep_native)
                    path_final = os.path.join(outpath_af3, fname_final)
                elif pdb_type in ("native", "MD"):
                    fin_struct = remove_all_but_loop(tmp_struct, residues_to_keep_native)
                    if pdb_type == "native":
                        path_final = os.path.join(outpath_natives, fname_final)
                    else:
                        path_final = os.path.join(outpath_md, fname_final)

                sorted_residues = sorted(residues_to_keep_native, key=lambda r: r.get_id()[1])
                loop_start_ext = loop_stop_ext = None
                for idx_ten, residue in enumerate(sorted_residues):
                    if residue.get_id()[1] == loop[0]:
                        loop_start_ext = idx_ten
                    if residue.get_id()[1] == loop[1]:
                        loop_stop_ext = idx_ten
                loop_length = loop[1] - loop[0] + 1
                total_residues = get_total_residues(fin_struct)
                sasa = get_bio_SASA(loop, chain_struct, protein_chain_id)
                surface_status = is_loop_on_surface(loop, chain_struct, protein_chain_id)
                loop_sequence = sequence[loop[0] - 1: loop[1]]
                # Updated to include the chain id in the loop position variables.
                loop_position_target = f"{protein_chain_id}:{str(loop[0])} {protein_chain_id}:{str(loop[1])}"
                loop_position = f"{protein_chain_id}:{str(loop_start_ext)} {protein_chain_id}:{str(loop_stop_ext)}"
                write_to_pdb(fin_struct,
                             loop_type,
                             loop_length,
                             total_residues,
                             loop_sequence,
                             sasa,
                             path_final,
                             surface_status,
                             loop_position,
                             loop_position_target)
    except Exception as e:
        logging.error(f"Error during loop extraction for {path_af}: {e}")
        # In case of error, copy the problematic AF file and any corresponding native and MD files to discarded.
        identifier = os.path.basename(path_af)[:4]
        discard_pdb(path_af, args.path_discarded, "loop extraction", e)
        for nf in glob.glob(os.path.join(args.inpath_natives, f"{identifier}*.pdb")):
            discard_pdb(nf, args.path_discarded, "loop extraction", e)
        for mf in glob.glob(os.path.join(args.inpath_MD, f"{identifier}*.pdb")):
            discard_pdb(nf, args.path_discarded, "loop extraction", e)
        # Optionally, if any loop file was already written, copy it as well.
        return

import re
def extract_loop_pattern(filename):
    match = re.search(r'_\d+_(EH|HE|HH|EE)\.pdb$', filename)
    return match.group(0) if match else None

###############################################################################
@func_timer
def main(args):
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    outpath_native = os.path.join(args.outpath, "EL_natives")
    os.makedirs(outpath_native, exist_ok=True)
    outpath_af3 = os.path.join(args.outpath, "EL_AF3_models")
    os.makedirs(outpath_af3, exist_ok=True)
    outpath_md = os.path.join(args.outpath, "EL_MD_frames")
    os.makedirs(outpath_md, exist_ok=True)
    path_BioPython_tmp = os.path.join(args.outpath, "af_for_BIOpySS")
    os.makedirs(path_BioPython_tmp, exist_ok=True)
    
    pdb_files = glob.glob(os.path.join(args.inpath_AF, "*.pdb"))
    for input_pdb in pdb_files:
        try:
            extract_loops(args, path_BioPython_tmp, input_pdb, outpath_af3, outpath_native, outpath_md, args.path_discarded)
        except Exception as e:
            logging.error(f"Error processing {input_pdb}: {e}")
            identifier = os.path.basename(input_pdb)[:4]
            discard_pdb(input_pdb, args.path_discarded, "loop extraction", e)
            for nf in glob.glob(os.path.join(args.inpath_natives, f"{identifier}*.pdb")):
                discard_pdb(nf, args.path_discarded, "loop extraction", e)
            for mf in glob.glob(os.path.join(args.inpath_MD, f"{identifier}*.pdb")):
                discard_pdb(mf, args.path_discarded, "loop extraction", e)
            continue
        
    # clean up temporary files
    for tmp_file in glob.glob("*tmp*.pdb"):
        try:
            os.remove(tmp_file)
        except Exception:
            pass
    shutil.rmtree(path_BioPython_tmp)
    
    # final check via sequence identity 
    for path_af3 in glob.glob(os.path.join(outpath_af3, f"*.pdb")):
        try:
            logging.info(f"Checking {path_af3}.")
            fname_af3 = os.path.basename(path_af3) 
            identifier = fname_af3[:4]
            loop_pattern = extract_loop_pattern(fname_af3)
            native_matches = glob.glob(os.path.join(outpath_native, f"{identifier}*{loop_pattern}"))
            md_matches = glob.glob(os.path.join(outpath_md, f"{identifier}*{loop_pattern}"))
            if not native_matches or not md_matches:
                logging.error("Native or MD file not found for final check!")
                discard_pdb(path_af3, args.path_discarded, "loop extraction", "Native or MD file not found for final check!")
                continue
            path_native = native_matches[0]
            path_md = md_matches[0]
            _, seq_af3 = get_BioPython_DSSP(path_af3, PATH_DSSP)
            _, seq_native = get_BioPython_DSSP(path_native, PATH_DSSP)
            _, seq_md = get_BioPython_DSSP(path_md, PATH_DSSP)
            seq_af3 = seq_af3.replace("X", "")
            seq_native = seq_native.replace("X", "")
            seq_md = seq_md.replace("X", "")
            sequence_identity_native = get_sequence_identity(seq_af3, seq_native)
            sequence_identity_md = get_sequence_identity(seq_af3, seq_md)
            pdb_parser = PDBParser()
            struct_af3 = pdb_parser.get_structure("protein", path_af3)
            totres_af3 = get_total_residues(struct_af3)   
            struct_native = pdb_parser.get_structure("protein", path_native)
            totres_native = get_total_residues(struct_native)   
            struct_md = pdb_parser.get_structure("protein", path_md)
            totres_md = get_total_residues(struct_md)   
            if totres_af3 != totres_native or totres_af3 != totres_md: 
                logging.warning(f"Warning: Number of residues of {path_af3} doesn't match those of native or MD loop. Discarding.")
                logging.info(f"SI AF3 to native and MD: {sequence_identity_native, sequence_identity_md}")    
                logging.info(f"nr residues AF3, native, MD: {totres_af3, totres_native, totres_md}")    
                discard_pdb(path_af3, args.path_discarded, "loop extraction", "missmatch length extracted loops")
                discard_pdb(path_native, args.path_discarded, "loop extraction", "missmatch length extracted loops")
                discard_pdb(path_md, args.path_discarded, "loop extraction", "missmatch length extracted loops")
        except Exception as e:
            logging.error(f"Error in final check for {path_af3}: {e}")
            discard_pdb(path_af3, args.path_discarded, "loop extraction", e)
            continue
    print("Done.")
    
###############################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-extract_loops_MD-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    parser = ap.ArgumentParser()
    default_path = "/home/florian_wieser/projects/ELEN/elen_training/data_preparation/AF3_LiMD"
    parser.add_argument("--inpath_natives", type=str, default=f"{default_path}/natives")
    parser.add_argument("--inpath_AF", type=str, default=f"{default_path}/AF3_models")
    parser.add_argument("--inpath_MD", type=str, default=f"{default_path}/MD_frames")
    parser.add_argument("--outpath", type=str, default=f"{default_path}/extracted_loops")
    parser.add_argument("--ss_frag", type=str, default="None", choices=["None", "helix", "sheet"],
                        help="If 'None', extract loops flanked by either helix or sheet. Otherwise force helix or sheet.")
    parser.add_argument("--ss_frag_size", type=int, default=8)
    parser.add_argument("--loop_max_size", type=int, default=10)
    parser.add_argument("--loop_mode", type=str, default="loop_pocket", choices=["loop_stretch", "loop_pocket"])
    parser.add_argument("--nr_residues", type=int, default=28)
    parser.add_argument("--path_discarded", type=str, default=f"discarded", help="Output directory for failed PDB files.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output folder")
    args = parser.parse_args()
    if args.ss_frag == "None":
        args.ss_frag = None
    main(args)