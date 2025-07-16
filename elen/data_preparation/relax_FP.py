#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J relax_FP
#SBATCH -o relax_FP.log
#SBATCH -e relax_FP.err
#SBATCH --tasks-per-node 1
#SBATCH --nodes 1
#TODO make relax intentionally worse to get higher label values 
import os
import sys
import glob
import shutil
import subprocess
import argparse
from collections import Counter
from elen.config import PATH_ROSETTA_BIN
from elen.inference.utils_inference import split_into_chain
from Bio.PDB import PDBParser

###############################################################################
def gather_ligands(pdb_file):
    """
    Identifies ligands in a PDB file, excluding water molecules.

    Args:
    pdb_file (str): Path to the PDB file.

    Returns:
    dict: Ligands with their counts.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    ligands = Counter()
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ' and residue.get_resname() not in ['HOH', 'WAT']:
                    ligands[residue.get_resname()] += 1
    return ligands

def write_ligand_statistics(ligands, output_file):
    """
    Writes ligand statistics to a specified output file.

    Args:
    ligands (dict): Dictionary of ligand counts.
    output_file (str): Path to the output file.
    """
    with open(output_file, 'w') as file:
        if os.path.exists(output_file):
            file.write(f"ligand,count\n")
        for ligand, count in sorted(ligands.items()):
            file.write(f"{ligand},{count}\n")
            
def run_relax(nstruct, path_pdb, outpath):
    """
    Executes the Rosetta relaxation protocol on a given PDB file.

    Args:
    nstruct (int): Number of structures to generate.
    path_pdb (str): Path to the input PDB file.
    """
    if args.mpi:
        subprocess.run([
            "mpirun", f"{PATH_ROSETTA_BIN}/relax.mpi.linuxgccrelease",
            "--in:file:s", path_pdb,
            "--relax:constrain_relax_to_start_coords",
            "-nstruct", str(nstruct),
            "-out:path:all", outpath,
            "--overwrite"
        ])
    else:
        subprocess.run([
            f"{PATH_ROSETTA_BIN}/relax.linuxgccrelease",
            "--in:file:s", path_pdb,
            "--relax:constrain_relax_to_start_coords",
            "-nstruct", str(nstruct),
            "-out:path:all", outpath,
            "--overwrite"
        ])

###############################################################################
def main(args):
    total_ligands = Counter()
    dir_filtered = "natives_filtered"
    dir_split = "natives_split"
    dir_relaxed = "models_relaxed"
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)

    if args.filter_for_ligands: 
        outpath_filtered = os.path.join(args.outpath, dir_filtered)
        os.makedirs(outpath_filtered, exist_ok=True)
        
        for path_pdb in glob.glob(os.path.join(args.inpath, '*.pdb')):
            ligands = gather_ligands(path_pdb)
            if ligands:
                print(f"Found ligand(s) in {path_pdb}: {', '.join(ligands.keys())}.")
                path_pdb_final = os.path.join(outpath_filtered, os.path.basename(path_pdb))
                shutil.copy(path_pdb, path_pdb_final)
                total_ligands.update(ligands)
        
        # Write combined ligand statistics
        write_ligand_statistics(total_ligands, os.path.join(args.outpath, "ligand_statistics.txt"))

    if args.split_into_chains: 
        outpath_split = os.path.join(args.outpath, dir_split)
        os.makedirs(outpath_split, exist_ok=True)
        for path_pdb in glob.glob(os.path.join(args.outpath, dir_filtered, '*.pdb')):
            print(f"Splitting chains for {os.path.basename(path_pdb)}.")
            split_into_chain(path_pdb, outpath_split)

    if args.relax: 
        outpath_relaxed = os.path.join(args.outpath, dir_relaxed)
        os.makedirs(outpath_relaxed, exist_ok=True)
        for path_pdb in glob.glob(os.path.join(args.outpath, dir_split, '*.pdb')):
            print(f"Relaxing {os.path.basename(path_pdb)}.")
            run_relax(args.nstruct, path_pdb, outpath_relaxed)

###############################################################################
if __name__ == "__main__":
    #tag = ""
    tag = "_test"
    parser = argparse.ArgumentParser(description="Script to process PDB files for ligands, split, and relax structures.")
    parser.add_argument("--inpath", type=str, default=f"input_natives{tag}", help="Input directory path containing PDB files.")
    parser.add_argument("--outpath", type=str, default=f"output_relaxed{tag}", help="Output directory path for relaxed PDB files.")
    parser.add_argument("--nstruct", type=int, default=3, help="Number of structures to generate per PDB file.")
    parser.add_argument("--filter_for_ligands", action="store_true", help="Filter input directory for .pdb files containing ligands.")
    parser.add_argument("--split_into_chains", action="store_true", help="Split filtered .pdbs into single chains.")
    parser.add_argument("--relax", action="store_true", help="Run relax with --nstruct trajetories.")
    parser.add_argument("--mpi", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()
    main(args)