#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J extract_FP
#SBATCH -o extract_FP.log
#SBATCH -e extract_FP.err
#TODO extract loop pockets including any other atom except water, extract them for AF, MD, and native, 
#TODO assure all have the same number of atoms in the same order

import os
import sys
import glob
import shutil
import numpy as np
import argparse
from Bio.PDB import PDBParser, PDBIO, Select

### HELPER FUNCTIONS ###
def get_ligand_residues(structure):
    """
    Identify ligand residues in a PDB structure.

    Args:
    structure (Bio.PDB.Structure): The PDB structure from which to extract ligand residues.

    Returns:
    list: A list of ligand residues (non-standard residues).
    """
    ligand_residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ' and residue.id[0] != 'W':
                    ligand_residues.append(residue)
    return ligand_residues

def get_protein_residues(structure):
    """
    Extract protein residues from a structure.

    Args:
    structure (Bio.PDB.Structure): The PDB structure to process.

    Returns:
    list: A list of protein residues containing CA atoms.
    """
    protein_residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ' and 'CA' in residue:
                    protein_residues.append(residue)
    return protein_residues

def get_geometric_center(atoms):
    """
    Calculate the geometric center of a collection of atoms.

    Args:
    atoms (list): A list of Bio.PDB.Atom objects.

    Returns:
    ndarray: The geometric center of the atoms.
    """
    coords = np.array([atom.get_coord() for atom in atoms])
    return coords.mean(axis=0)

def filter_closest_residues(ligand_residue, geometric_center, protein_residues, nr_residues=40):
    """
    Select the closest residues to a ligand based on geometric center.

    Args:
    ligand_residue (Bio.PDB.Residue): The ligand residue.
    geometric_center (ndarray): The geometric center of the ligand.
    protein_residues (list): A list of protein residues.
    nr_residues (int): Number of residues to select.

    Returns:
    list: A list of the closest residues including the ligand residue.
    """
    distances = [(residue, np.linalg.norm(residue['CA'].get_coord() - geometric_center)) for residue in protein_residues]
    distances.sort(key=lambda x: x[1])
    closest_residues = [residue for residue, _ in distances[:nr_residues]]
    return [ligand_residue] + closest_residues

class ResidueSelect(Select):
    """
    Custom selection for saving specific residues to a PDB file.
    """
    def __init__(self, residues_to_keep):
        self.residues_to_keep = residues_to_keep

    def accept_residue(self, residue):
        return residue in self.residues_to_keep

###############################################################################
def main(args):
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    os.makedirs(os.path.join(args.outpath, "EL_natives"), exist_ok=True)
    os.makedirs(os.path.join(args.outpath, "EL_AF3_models"), exist_ok=True)
    os.makedirs(os.path.join(args.outpath, "EL_MD_frames"), exist_ok=True)
    
    parser = PDBParser(QUIET=True)
    for path_af3 in glob.glob(os.path.join(args.inpath_AF, "*.pdb")):
        structure = parser.get_structure('protein', path_af3)
        ligand_residues = get_ligand_residues(structure)
        if ligand_residues:
            for ligand_residue in ligand_residues:
                geometric_center = get_geometric_center(ligand_residue)
                protein_residues = get_protein_residues(structure)
                residues_to_keep = filter_closest_residues(ligand_residue, geometric_center, protein_residues, args.nr_residues)
                process_pdb_files(path_af3, args, residues_to_keep, parser)
        else:
            print("No ligands detected in the PDB file.")

def process_pdb_files(path_pdb_native, args, residues_to_keep, parser):
    """
    Process and save PDB files based on residue selections.
    """
    path_pdbs = [path_pdb_native] + glob.glob(os.path.join(args.inpath_AF, f"{os.path.basename(path_pdb_native).replace('.pdb', '')}*.pdb"))
    for idx, path_pdb in enumerate(path_pdbs):
        structure = parser.get_structure('protein', path_pdb)
        io = PDBIO()
        io.set_structure(structure)
        output_dir = "natives" if idx == 0 else "models"
        output_tag = "nat" if idx == 0 else "mod"
        path_output = os.path.join(args.outpath, output_dir, f'{os.path.basename(path_pdb_native).replace(".pdb", "")}_{output_tag}.pdb')
        io.save(path_output, ResidueSelect(residues_to_keep))
        print(f"Saved FP to {path_output}")

###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract fingerprint regions from PDB files.")
    parser.add_argument("--inpath_natives", type=str, required=True, help="Input path for native PDB files.")
    parser.add_argument("--inpath_AF", type=str, required=True, help="Input path for model PDB files.")
    parser.add_argument("--inpath_MD", type=str, required=True, help="Input path for MD frames.")
    parser.add_argument("--outpath", type=str, default="EL_AF3_LiMD", help="Output path for extracted loop pockets.")
    parser.add_argument("--nr_residues", type=int, default=40, help="Number of closest residues to include in the loop pocket.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in the output directory.")
    args = parser.parse_args()
    main(args)