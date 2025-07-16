#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J gromacs
#SBATCH -o gromacs.log
#SBATCH -e gromacs.err
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 4
#SBATCH --gres gpu:1

# TODO Gustav
# multiplicity in ligand parameterization
# rmsd label from final struct to native or averaged over all frames?
# TODO single metall ligands like Zn won't be extracted to the final .pdb
# TODO refactor variable names
# TODO get into multiplicity (Gustav)
# TODO get averaged rmsf values
# TODO make video better
# Description: Runs (automatically) the MD protocol of Kurniawan et. al
# features:
# RMSD to initial structure normalized to 1 and inverted

import os
import sys
import glob
import time
import shutil
import logging
import subprocess
import argparse as ap
from rdkit import Chem
import warnings
from Bio import BiopythonDeprecationWarning
from Bio.PDB import PDBParser, Superimposer

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

# Path to the GROMACS executable
PATH_GMX = '/home/thaidaev/programmi/gromacs-2023.3/bin/gmx'
PATH_GMX_MPI = '/home/thaidaev/programmi/gromacs-2023.3/bin/gmx_mpi'
PATH_ACPYPE = "/home/florian_wieser/miniconda3/envs/elen_test/bin/acpype"
PATH_REDUCE = "/usr/local/amber22/bin/reduce"
PATH_MDP = "/home/florian_wieser/projects/ELEN/elen/data_preparation/Kurniawan_protocol"
os.environ["PATH"] = "/home/florian_wieser/miniconda3/envs/elen_test/bin:" + os.environ["PATH"]

from elen.data_preparation.gromacs_utils import (
    run_command, get_pdb_from_gro, elapsed
)
from elen.data_preparation.gromacs_video import create_md_video, create_md_video2

### MD ########################################################################
def remove_h_and_water(input_pdb, output_pdb):
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            if line.startswith(('ATOM', 'HETATM')):
                # Extract element from columns 77-78 (0-based index 76:78)
                element = line[76:78].strip()
                # Extract residue name from columns 18-20 (0-based 17:20)
                res_name = line[17:20].strip()
                # Check if it's a hydrogen by element
                if element == 'H':
                    continue  # Skip this line
                # Check if it's a water molecule
                if res_name in ['HOH', 'WAT']:
                    continue  # Skip this line
                # Check if the element is not a two-character symbol and atom name starts with H
                if len(element) != 2:
                    # Extract atom name from columns 13-16 (0-based 12:16)
                    atom_name = line[12:16].strip()
                    if atom_name and atom_name[0] == 'H':
                        continue  # Skip this line
                # If passed all checks, write the line
                outfile.write(line)
            else:
                # Write non-ATOM/HETATM lines directly
                outfile.write(line)
    return output_pdb    


def compute_rmsd(pdb1, pdb2):
    """
    Compute RMSD between two PDB files using Biopython's Superimposer.

    :param pdb1: Path to the first PDB file (reference).
    :param pdb2: Path to the second PDB file (to be superimposed).
    :return: RMSD value in angstroms.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure1 = parser.get_structure('struct1', pdb1)
        structure2 = parser.get_structure('struct2', pdb2)
    except Exception as e:
        logging.error(f"Error parsing PDB files for RMSD computation: {e}")
        raise

    # Extract all atoms; ensure both structures have the same number of atoms
    atoms1 = [atom for atom in structure1.get_atoms()]
    atoms2 = [atom for atom in structure2.get_atoms()]
    print(f"len(atoms1): {len(atoms1)}") 
    print(f"len(atoms2): {len(atoms2)}") 
    if len(atoms1) != len(atoms2):
        logging.error("Structures have different number of atoms. Cannot compute RMSD.")
        raise ValueError("Structures have different number of atoms.")

    # Optional: Filter atoms to include only CA or backbone if desired
    # For now, use all atoms
    sup = Superimposer()
    try:
        sup.set_atoms(atoms1, atoms2)
        sup.apply(structure2.get_atoms())
    except Exception as e:
        logging.error(f"Error during RMSD superimposition: {e}")
        raise

    return sup.rms


def check_rmsd(initial_pdb, final_pdb, threshold, too_high_dir, failed_log):
    """
    Compute RMSD between initial and final PDB files. If RMSD exceeds the threshold,
    move the final PDB to the too_high_dir and log the event.

    :param initial_pdb: Path to the initial PDB file.
    :param final_pdb: Path to the final PDB file.
    :param threshold: RMSD threshold in angstroms.
    :param too_high_dir: Directory to move PDBs with high RMSD.
    :param failed_log: Path to the failed_pdbs log file.
    :return: RMSD value.
    """
    try:
        rmsd = compute_rmsd(initial_pdb, final_pdb)
        logging.info(f"Computed RMSD between {initial_pdb} and {final_pdb}: {rmsd:.3f} Å")
        if rmsd > threshold:
            try:
                destination = os.path.join(too_high_dir, os.path.basename(final_pdb))
                if os.path.exists(destination):
                    os.remove(destination)  # remove existing file to avoid error
                shutil.move(final_pdb, too_high_dir)
                logging.warning(f"RMSD {rmsd:.3f} Å exceeds threshold {threshold} Å. Moved {final_pdb} to {too_high_dir}.")
                with open(failed_log, 'a') as fl:
                    fl.write(f"{final_pdb} (RMSD: {rmsd:.3f} Å)\n")
            except Exception as e:
                logging.error(f"Error moving file {final_pdb} to {too_high_dir}: {e}")
        return rmsd
    except Exception as e:
        logging.error(f"Failed to compute RMSD for {final_pdb}: {e}")
        with open(failed_log, 'a') as fl:
            fl.write(f"{final_pdb} (RMSD computation failed)\n")
        return None

###############################################################################

def main(args):
    path_final_pdbs = os.path.join(args.inpath_gromacs_out, "pdbs_last_frame")
    too_high_dir = os.path.join(args.inpath_gromacs_out, "too_high_rmsd")
    failed_log = os.path.join(args.inpath_gromacs_out, "failed_pdbs.ls")
    
    # Ensure necessary directories exist
    os.makedirs(path_final_pdbs, exist_ok=True)
    os.makedirs(too_high_dir, exist_ok=True)
    
    if args.get_pdb:
        tag = os.path.basename(os.getcwd())
        fname_out = f"{tag}_{args.get_pdb}.pdb"
        try:
            get_pdb_from_gro(args.get_pdb, fname_out, args, ".")
        except Exception as e:
            logging.error(f"Error extracting pdb: {e}")
        return 0
    elif args.get_mkv:
        try:
            create_md_video("topol.tpr", "md.trr")
            create_md_video2("topol.tpr", "md.trr")
        except Exception as e:
            logging.error(f"Error creating md video: {e}")
        return 0

    for path_md in glob.glob(os.path.join(args.inpath_gromacs_out, f"????")):
        logging.info(f"---------- Processing {path_md} ----------")
        tag = os.path.basename(path_md)
        total_start = time.perf_counter()  # Start total timing
        
        try:
            # Step 8: Production MD run
            step_start = time.perf_counter()
            md_tpr = os.path.join(path_md, "md.tpr")
            if not os.path.exists(md_tpr):
                if args.mode == "gpu":
                    run_command(
                        f"{PATH_GMX} grompp -f {os.path.join(PATH_MDP, 'md.mdp')} -c npt.gro -p topol.top -o md.tpr -maxwarn 100",
                        args.mute_gro, cwd=path_md
                    )
                    run_command(
                        f"{PATH_GMX_MPI} mdrun -deffnm md -ntomp 4 -pin on -update gpu -pme gpu",
                        args.mute_gro, cwd=path_md
                    )
                elif args.mode == "dev":
                    run_command(
                        f"{PATH_GMX} grompp -f {os.path.join(PATH_MDP, 'md_dev.mdp')} -c npt.gro -p topol.top -o md.tpr -maxwarn 100",
                        args.mute_gro, cwd=path_md
                    )
                    run_command(
                        f"{PATH_GMX} mdrun -deffnm md -ntomp 1 -pin on",
                        args.mute_gro, cwd=path_md
                    )
            logging.info(f"Step 8 (Production MD) took {elapsed(step_start)}")

            # Extract last frame only if simulation was successful
            path_pdb_final = os.path.join(path_final_pdbs, f"{tag}_lf_gro.pdb")
            
            if not os.path.exists(path_pdb_final):
                try:
                    fname_out = f"{tag}_lf_gro.pdb"
                    get_pdb_from_gro("md", fname_out, args, path_md)
                    temp_final_pdb = os.path.join(path_md, f"{tag}_lf_gro.pdb")
                    shutil.move(temp_final_pdb, path_final_pdbs)
                    logging.info(f"Simulation successful - moved {tag}_lf_gro.pdb to {path_final_pdbs}.")
                    logging.info(f"Total simulation time: {elapsed(total_start)}\n")
                except Exception as e:
                    logging.error(f"Error extracting final pdb for {tag}: {e}")
                    continue  # Skip to next MD folder

            # Remove waters and hydrogens
            path_pdb_final_wo_H = path_pdb_final.replace(".pdb", "_wo_H.pdb")
            if not os.path.exists(path_pdb_final_wo_H):
                try:
                    remove_h_and_water(path_pdb_final, path_pdb_final_wo_H)
                except Exception as e:
                    logging.error(f"Error removing hydrogens/waters for {tag}: {e}")

            # Compute RMSD between initial and final PDB
            try:
                initial_pdb_list = glob.glob(os.path.join(args.inpath, f"{tag}*.pdb"))
                if not initial_pdb_list:
                    logging.warning(f"No initial pdb found for tag {tag}. Skipping RMSD computation.")
                    continue
                initial_pdb = initial_pdb_list[0]
                final_pdb = path_pdb_final_wo_H
                if os.path.exists(final_pdb):
                    rmsd = check_rmsd(
                        initial_pdb=initial_pdb,
                        final_pdb=final_pdb,
                        threshold=2.5,
                        too_high_dir=too_high_dir,
                        failed_log=failed_log
                    )
                    print(f"rmsd: {rmsd}")
                    if rmsd is None:
                        logging.info(f"{tag}: Error during rmsd calculation.")
                    elif rmsd > 2.5:
                        logging.info(f"{tag}: RMSD {rmsd:.3f} Å exceeds threshold. Moved to 'too_high_rmsd'.")
                    else:
                        logging.info(f"{tag}: RMSD {rmsd:.3f} Å.")
                else:
                    logging.warning(f"Final PDB {final_pdb} does not exist. Skipping RMSD computation.")
            except Exception as e:
                logging.error(f"Error computing RMSD for {tag}: {e}")
        except Exception as e:
            logging.error(f"Error processing {path_md}: {e}")
            continue

    logging.info("Done.") 

###############################################################################

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='ELEN-MD-%(levelname)s(%(asctime)s): %(message)s', 
        datefmt='%y-%m-%d %H:%M:%S'
    )
    parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inpath", type=str, default="input", help="Path that contains input .pdb files.")
    parser.add_argument("--inpath_gromacs_out", type=str, default="1ubq", help="Path that contains all the outpath folders from gromacs.py main script.")
    parser.add_argument("--get_pdb", type=str, default=None, help="Will try to extract a .pdb file from a GROMACS input file (.gro).")
    parser.add_argument("--get_mkv", type=str, default=None, help="Will try to render a .mkv video from a GROMACS input file (.gro).")
    parser.add_argument("--mode", type=str, default="mpi", choices=['gpu', 'mpi', 'dev'])
    parser.add_argument("--mute_gro", action="store_true", default=False, help="Mute stdout of GROMACS.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()
    main(args)
