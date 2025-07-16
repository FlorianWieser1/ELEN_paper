#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
# TODO continue with MD script, debug
# TODO finish implement, then test and prepare test dataset
# TODO fix missmatch movement
# TODO filter salt ligands

import os
import sys
import glob
import shutil
import logging
import argparse as ap
from Bio import PDB
import os
from Bio.PDB import PDBParser, PDBIO, Select

class ChainSelect(Select):
    """
    A custom Select class that accepts only the specified chain(s).
    """
    def __init__(self, chain_id):
        super().__init__()
        self.chain_id = chain_id

    def accept_chain(self, chain):
        # Return True (1) only if the chain ID matches
        if chain.id == self.chain_id:
            return 1
        return 0


def split_pdb_into_chains(input_pdb_path, output_dir=None):
    """
    Splits a multi-chain PDB file into separate files, 
    each containing a single chain from the original PDB.

    Parameters
    ----------
    input_pdb_path : str
        Path to the input PDB file.
    output_dir : str, optional
        Directory where the split chain files will be saved.
        If None, files will be saved in the same directory as input_pdb_path.

    Returns
    -------
    list of str
        A list of the output file paths that were created.
    """

    # If no output directory is specified, use the directory of the input file
    if output_dir is None:
        output_dir = os.path.dirname(input_pdb_path)
    if not output_dir:
        output_dir = "."  # current directory

    os.makedirs(output_dir, exist_ok=True)

    # Initialize parser and load structure
    parser = PDBParser(QUIET=True)
    structure_id = os.path.splitext(os.path.basename(input_pdb_path))[0]
    structure = parser.get_structure(structure_id, input_pdb_path)

    # Prepare a PDBIO object for writing
    io = PDBIO()

    # Collect unique chain IDs across all models
    chain_ids = set()
    for model in structure:
        for chain in model:
            chain_ids.add(chain.id)

    # For each chain, select and save
    output_files = []
    for chain_id in chain_ids:
        chain_file_name = f"{structure_id}_chain_{chain_id}.pdb"
        chain_file_path = os.path.join(output_dir, chain_file_name)

        # Set the structure and use our ChainSelect to filter
        io.set_structure(structure)
        io.save(chain_file_path, select=ChainSelect(chain_id))

        output_files.append(chain_file_path)
        print(f"Saved chain {chain_id} to {chain_file_path}")

    return output_files


if __name__ == "__main__":
    # Example usage:
    input_pdb = "example.pdb"
    output_directory = "chains"
    split_pdb_into_chains(input_pdb, output_directory)

def process_directory(inpath, outpath):
    logging.info(f"Processing directory {inpath}.")
    for path_pdb in glob.glob(os.path.join(inpath, "*.pdb")):
        outpath_pdb = os.path.join(outpath, os.path.basename(path_pdb))
        renumber_and_clean_pdb_chains(path_pdb, outpath_pdb)

def renumber_and_clean_pdb_chains(path_pdb, outpath_pdb):
    """
    Processes a PDB file by removing water molecules, renumbering residues,
    renaming chains consecutively starting from 'A', and writing the cleaned structure
    to a new PDB file.
    """
    logging.info(f"Processing .pdb {path_pdb}.")
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('PDB', path_pdb)
    except Exception as e:
        logging.error(f"Failed to parse {path_pdb}: {e}")
        return

    chain_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    # Create a new structure and model for cleaned data
    new_structure = PDB.Structure.Structure(structure.id)
    new_model = PDB.Model.Model(0)
    new_structure.add(new_model)

    chain_index = 0

    # Assuming the input has a single model (structure[0])
    for old_chain in structure[0]:
        if chain_index >= len(chain_letters):
            logging.warning(f"Exceeded chain letters for {path_pdb}. Additional chains will not be renamed.")
            break  # Avoid exceeding available chain letters

        # Create a new chain with a new sequential ID
        new_chain_id = chain_letters[chain_index]
        new_chain = PDB.Chain.Chain(new_chain_id)
        chain_index += 1

        # Filter out water residues and renumber remaining residues
        non_water_residues = [res for res in old_chain if res.get_resname() != 'HOH']
        for i, old_residue in enumerate(non_water_residues, start=1):
            # Create a new residue with renumbered ID, same resname and segid
            new_residue = PDB.Residue.Residue((' ', i, ' '), 
                                              old_residue.get_resname(), 
                                              old_residue.segid)
            # Copy atoms from the old residue to the new residue
            for atom in old_residue:
                new_atom = atom.copy()
                new_residue.add(new_atom)

            new_chain.add(new_residue)

        new_model.add(new_chain)

    # Write out the entire modified structure to a single output file
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    try:
        io.save(outpath_pdb)
        logging.info(f"Saved cleaned structure to {outpath_pdb}")
    except Exception as e:
        logging.error(f"Failed to save {outpath_pdb}: {e}")
        
        
def harmonize_structure(ref_file, target_file, output_file):
    """
    Harmonizes the target PDB structure to match the reference structure in terms
    of atom order, types, and hierarchy. Non-matching atoms are omitted.
    """
    parser = PDB.PDBParser(QUIET=True)
    
    try:
        # Parse reference and target structures
        ref_struct = parser.get_structure('ref', ref_file)
        target_struct = parser.get_structure('target', target_file)
    except Exception as e:
        logging.error(f"Error parsing PDB files: {e}")
        return
    
    # List atoms from reference in desired order
    ref_atoms = list(ref_struct.get_atoms())
    
    # Build a pool of atoms from the target structure keyed by (resname, atom name, element)
    target_pool = {}
    for atom in target_struct.get_atoms():
        key = (atom.parent.get_resname(), atom.get_name(), atom.element)
        target_pool.setdefault(key, []).append(atom)
    
    # Filter and reorder target atoms based on the reference order
    ordered_atoms = []
    for ref_atom in ref_atoms:
        key = (ref_atom.parent.get_resname(), ref_atom.get_name(), ref_atom.element)
        if key in target_pool and target_pool[key]:
            ordered_atoms.append(target_pool[key].pop(0))
        # If an atom in reference is not found in target, it's skipped.
    
    # Create a new structure for the harmonized target using reference hierarchy
    new_struct = PDB.Structure.Structure('harmonized')
    # Assuming single-model structures for simplicity
    new_model = PDB.Model.Model(0)
    new_struct.add(new_model)
    
    # Iterate over chains and residues in the reference structure to recreate hierarchy
    for ref_chain in ref_struct[0]:
        new_chain = PDB.Chain.Chain(ref_chain.id)
        new_model.add(new_chain)
        
        for ref_residue in ref_chain:
            # Create a new residue with same id, resname, and segid as in reference
            new_residue = PDB.Residue.Residue(ref_residue.id, ref_residue.resname, ref_residue.segid)
            new_chain.add(new_residue)
            
            # For each atom in the reference residue, assign corresponding atom from ordered_atoms
            for ref_atom in ref_residue:
                # Only proceed if we still have atoms in our ordered list
                if not ordered_atoms:
                    break
                # Pop the next atom which should correspond to the current ref_atom
                candidate_atom = ordered_atoms.pop(0)
                # Check if this atom name already exists in the current residue
                if candidate_atom.get_name() in new_residue.child_dict:
                    continue  # Skip adding duplicate atoms
                new_residue.add(candidate_atom)
    
    # Write out the harmonized target structure
    io = PDB.PDBIO()
    io.set_structure(new_struct)
    try:
        io.save(output_file)
        logging.info(f"Harmonized structure saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save harmonized structure to {output_file}: {e}")


def check_atom_counts_and_discard(path_pdb, path_harmonized, outpath_discarded):
    """
    Checks that the number of atoms across processed natives, AF3_models, and MD_frames
    PDB files match for each identifier. If they do not match, moves the related files
    to the 'discarded' directory.
    """
    # Create discarded directories
    # Extract identifiers assuming the first four characters of the filename
    print(f"path_pdb: {path_pdb}")
    parser = PDB.PDBParser(QUIET=True)
    struct_pdb = parser.get_structure('af', path_pdb)
    struct_harmonized = parser.get_structure('native', path_harmonized)
    pdb_atoms = sum(1 for _ in struct_pdb.get_atoms())
    harmonized_atoms = sum(1 for _ in struct_harmonized.get_atoms())
    if pdb_atoms == harmonized_atoms:
        logging.info(f"Atom counts match for identifier {path_pdb} atoms.")
    else:
        logging.warning(f"Atom counts mismatch for identifier {path_harmonized}: native={pdb_atoms}, af3/md={harmonized_atoms}. Moving to discarded.")
        shutil.move(path_harmonized, outpath_discarded)

###############################################################################
def main(args):
    # Define output paths
    outpath_af = args.inpath_AF + "_clean_renumbered"
    outpath_natives = args.inpath_natives + "_clean_renumbered"
    outpath_md = args.inpath_MD + "_clean_renumbered"
    outpath_discarded = "discarded"
    
    # Create output directories if they do not exist
    os.makedirs(outpath_af, exist_ok=True)
    os.makedirs(outpath_natives, exist_ok=True)
    os.makedirs(outpath_md, exist_ok=True)
    os.makedirs(outpath_discarded, exist_ok=True)
    
    # Handle overwrite option
    if args.overwrite:
        for path in [outpath_af, outpath_natives, outpath_md, "discarded"]:
            if os.path.exists(path):
                shutil.rmtree(path)
                logging.info(f"Removed existing directory {path}")
            os.makedirs(path, exist_ok=True)
    
    # Renumbering residues and removing water
    process_directory(args.inpath_AF, outpath_af)    
    process_directory(args.inpath_natives, outpath_natives)    
    process_directory(args.inpath_MD, outpath_md)
    
    # Harmonize .pdbs regarding number, order, name of atoms  
    for path_pdb in glob.glob(os.path.join(outpath_af, "*.pdb")):
        logging.info(f"Harmonizing {path_pdb}.")
        identifier = os.path.basename(path_pdb)[:4]
        path_native_candidates = glob.glob(os.path.join(outpath_natives, f"{identifier}*.pdb"))
        path_md_candidates = glob.glob(os.path.join(outpath_md, f"{identifier}*.pdb"))
        if not path_native_candidates or not path_md_candidates:
            logging.warning(f"No matching native PDB or MD frame found for {identifier}. Skipping harmonization.")
            continue
        path_native = path_native_candidates[0]
        path_md = path_md_candidates[0]
        output_harmonized_af = path_pdb.replace(".pdb", "") + "_final.pdb"
        output_harmonized_md = path_md.replace(".pdb", "") + "_final.pdb"
        harmonize_structure(path_native, path_pdb, output_harmonized_af)
        harmonize_structure(path_native, path_md, output_harmonized_md)
        check_atom_counts_and_discard(path_native, output_harmonized_af, outpath_discarded)
        check_atom_counts_and_discard(path_native, output_harmonized_md, outpath_discarded)
        
    logging.info("Done.")

###############################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-split_into_chains-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    parser = ap.ArgumentParser(description="Process and harmonize PDB files by renumbering residues, removing water molecules, renaming chains, and ensuring atom consistency across datasets.")
    default_path = "/home/florian_wieser/projects/ELEN/elen_training/data_preparation/AF3_LiMD"
    parser.add_argument("--inpath_natives", type=str, default=f"{default_path}/natives", help="Input directory for native PDB files.")
    parser.add_argument("--inpath_AF", type=str, default=f"{default_path}/AF3_models", help="Input directory for AF3 model PDB files.")
    parser.add_argument("--inpath_MD", type=str, default=f"{default_path}/MD_frames", help="Input directory for MD frame PDB files.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output folders.")
    args = parser.parse_args()
    main(args)