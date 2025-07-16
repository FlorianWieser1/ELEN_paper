#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J gromacs
#SBATCH -o gromacs.log
#SBATCH -e gromacs.err
##SBATCH --ntasks 1 
##SBATCH --cpus-per-task 4

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
    run_command, get_pdb_from_gro,
    contains_ligand, extract_ligands,
    extract_atomtypes_section, append_line_to_file, append_after_matching_line,
    elapsed, compute_net_charge_and_electron_count,
    remove_lines_in_place, count_hetatm_types
)
from elen.data_preparation.gromacs_video import create_md_video, create_md_video2

### MD ########################################################################

def prepare_complex(path_ligand, tag, outpath_md, args):
    fname_ligand = os.path.basename(path_ligand)
    
    logging.debug(f"Reducing ligand {fname_ligand}.")
    path_reduced = path_ligand.replace(".pdb", "_H.pdb")
    fname_reduced = os.path.basename(path_reduced)
    run_command(f"{PATH_REDUCE} {fname_ligand} > {fname_reduced}", args.mute_gro, cwd=outpath_md)
    
    logging.debug(f"Computing net charge of {fname_reduced}.") 
    net_charge, total_electrons = compute_net_charge_and_electron_count(path_reduced)
    
    # Define charge adjustment strategy
    max_charge_steps = 5 
    if net_charge < 0:
        charge_steps = [net_charge + i for i in range(0, max_charge_steps)]
    elif net_charge > 0:
        charge_steps = [net_charge - i for i in range(0, max_charge_steps)]
    else:
        charge_steps = [0] + [i for i in range(1, max_charge_steps)]
    
    parameterized = False
    for charge in charge_steps:
        # Determine spin multiplicity based on electron count
        try:
            mol = Chem.MolFromPDBFile(path_reduced, removeHs=False)
            if mol is None:
                raise ValueError("RDKit failed to parse the PDB file.")
            electron_count = sum([atom.GetAtomicNum() for atom in mol.GetAtoms()]) + charge
            if electron_count % 2 == 0:
                multiplicity = 1  # Closed-shell
            else:
                multiplicity = 2  # Radical
        except Exception as e:
            logging.error(f"Error determining multiplicity: {e}")
            multiplicity = 1  # Default to closed-shell
        
        logging.debug(f"Attempting parameterization with charge {charge}, multiplicity {multiplicity}.")
        try:
            path_ligand_parameterized = os.path.join(
                outpath_md, f"{fname_ligand.replace('.pdb', '')}_H.acpype", f"{fname_ligand.replace('.pdb', '')}_H_NEW.pdb"
            )
            run_command(f"{PATH_ACPYPE} -i {fname_reduced} -c bcc -n {charge} -m {multiplicity}", args.mute_gro, cwd=outpath_md)
            # Check if parameterization succeeded by verifying output files
            if os.path.exists(path_ligand_parameterized):
                logging.debug(f"Parameterization succeeded with charge {charge}.")
                parameterized = True
                break
            else:
                logging.debug(f"Parameterization failed with charge {charge}.")
        except subprocess.CalledProcessError:
            logging.debug(f"Parameterization failed with charge {charge}.")
            continue  # Try next charge
    if not parameterized:
        raise Exception(f"Parameterization failed for {fname_ligand} after trying charges {charge_steps}.")
    

    tag_ligand = fname_ligand.replace('.pdb', '')
    path_ligand_parameterized = os.path.join(f"{tag_ligand}_H.acpype", f"{tag_ligand}_H_NEW.pdb") 
    
    logging.debug("Merging protein with ligand.") 
    run_command(f"cat {path_ligand_parameterized} >> {tag}_ligands.pdb", args.mute_gro, cwd=outpath_md) # write ligand(s) to file for later merging

    # Add ligand infos to topology file topol.top
    logging.debug("Extracting atomtypes record.") 
    # Extract [ atomtypes ] section into new file
    path_ligand_itp = os.path.join(outpath_md, path_ligand_parameterized.replace("_NEW.pdb", "_GMX.itp"))
    fname_ligand_atomtypes = f"{tag_ligand}_atomtypes.itp"
    path_ligand_atomtypes_itp = os.path.join(outpath_md, fname_ligand_atomtypes)
    extract_atomtypes_section(path_ligand_itp, path_ligand_atomtypes_itp)
     
    logging.debug("Registering ligand in topol.top.") 
    # Include ligand.gro and atomtypes into topology file and register ligand in [ molecules ] section
    append_after_matching_line(
        os.path.join(outpath_md, "topol.top"), 
        '#include "amber99sb.ff/forcefield.itp"', 
        f'#include "{fname_ligand_atomtypes}"'
    )
    append_after_matching_line(
        os.path.join(outpath_md, "topol.top"), 
        '; Include water topology', 
        f'#include "{path_ligand_parameterized.replace("_NEW.pdb", "_GMX.itp")}"'
    )
    append_line_to_file(os.path.join(outpath_md, "topol.top"), f"{fname_reduced.replace('.pdb', '')}\t 1")


def run_MD(tag, outpath_md, args):
    # Step 3: Define the simulation box
    step_start = time.perf_counter()
    box_gro = f"{outpath_md}/{tag}_box.gro"
    if not os.path.exists(box_gro):
        run_command(
            f"{PATH_GMX} editconf -f {tag}_final.gro -o {tag}_box.gro -c -d 1.0 -bt cubic",
            args.mute_gro, cwd=outpath_md
        )
    logging.info(f"Step 3 (Box definition) took {elapsed(step_start)}")
    
    # Step 4: Solvate protein
    step_start = time.perf_counter()
    solv_gro = f"{outpath_md}/{tag}_solv.gro"
    if not os.path.exists(solv_gro):
        run_command(
            f"{PATH_GMX} solvate -cp {tag}_box.gro -cs spc216.gro -o {tag}_solv.gro -p topol.top", 
            args.mute_gro, cwd=outpath_md
        )
    logging.info(f"Step 4 (Solvation) took {elapsed(step_start)}")

    # Step 5: Add ions to neutralize the system
    step_start = time.perf_counter()
    solv_ions_gro = f"{outpath_md}/{tag}_solv_ions.gro"
    if not os.path.exists(solv_ions_gro):
        run_command(
            f"{PATH_GMX} grompp -f {os.path.join(PATH_MDP, 'ions.mdp')} -c {tag}_solv.gro -p topol.top -o ions.tpr -maxwarn 100", 
            args.mute_gro, cwd=outpath_md
        )
        run_command(
            f"echo 'SOL' | {PATH_GMX} genion -s ions.tpr -o {tag}_solv_ions.gro -p topol.top -pname NA -nname CL -neutral", 
            args.mute_gro, cwd=outpath_md
        )
    logging.info(f"Step 5 (Ion addition) took {elapsed(step_start)}")

    # Step 6: Energy minimization
    step_start = time.perf_counter()
    em_gro = f"{outpath_md}/em.gro"
    if not os.path.exists(em_gro):
        run_command(
            f"{PATH_GMX} grompp -f {os.path.join(PATH_MDP, 'minim.mdp')} -c {tag}_solv_ions.gro -p topol.top -o em.tpr -maxwarn 100", 
            args.mute_gro, cwd=outpath_md
        )
        run_command(
            f"{PATH_GMX_MPI} mdrun -v -deffnm em",
            args.mute_gro, cwd=outpath_md
        )
    logging.info(f"Step 6 (Energy minimization) took {elapsed(step_start)}")
    
    # Step 7: Equilibration
    # NVT    
    step_start = time.perf_counter()
    nvt_gro = f"{outpath_md}/nvt.gro"
    if not os.path.exists(nvt_gro):
        run_command(
            f"{PATH_GMX} grompp -f {os.path.join(PATH_MDP, 'nvt.mdp')} -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 100",
            args.mute_gro, cwd=outpath_md
        )
        run_command(
            f"{PATH_GMX_MPI} mdrun -deffnm nvt",
            args.mute_gro, cwd=outpath_md
        )
    logging.info(f"Step 7a (NVT Equilibration) took {elapsed(step_start)}")

    # NPT
    step_start = time.perf_counter()
    npt_gro = f"{outpath_md}/npt.gro"
    if not os.path.exists(npt_gro):
        run_command(
            f"{PATH_GMX} grompp -f {os.path.join(PATH_MDP, 'npt.mdp')} -c nvt.gro -r nvt.gro -p topol.top -o npt.tpr -maxwarn 100",
            args.mute_gro, cwd=outpath_md
        )
        run_command(
            f"{PATH_GMX_MPI} mdrun -deffnm npt",
            args.mute_gro, cwd=outpath_md
        )
    logging.info(f"Step 7b (NPT Equilibration) took {elapsed(step_start)}")

    # Step 8: Production MD run
    step_start = time.perf_counter()
    md_tpr = f"{outpath_md}/md.tpr"
    if not os.path.exists(md_tpr):
        if args.mode == "gpu":
            logging.info("Skipping production with gpu to be run with a separate script.")
        elif args.mode == "mpi":
            run_command(
                f"{PATH_GMX} grompp -f {os.path.join(PATH_MDP, 'md.mdp')} -c npt.gro -p topol.top -o md.tpr -maxwarn 100", 
                args.mute_gro, cwd=outpath_md
            )
            run_command(
                f"{PATH_GMX_MPI} mdrun -deffnm md -ntomp 8 -pin on",
                args.mute_gro, cwd=outpath_md
            )
        elif args.mode == "dev":
            run_command(
                f"{PATH_GMX} grompp -f {os.path.join(PATH_MDP, 'md_dev.mdp')} -c npt.gro -p topol.top -o md.tpr -maxwarn 100",
                args.mute_gro, cwd=outpath_md
            )
            run_command(
                f"{PATH_GMX} mdrun -deffnm md -ntomp 1 -pin on",
                args.mute_gro, cwd=outpath_md
            )
    logging.info(f"Step 8 (Production MD) took {elapsed(step_start)}")


def preprocess_protein(fname_protein, tag, outpath_md, args):
    # Preprocess protein 
    fname_pdb_prepro = f"{tag}_protein_preprocessed.pdb"
    step_start = time.perf_counter()
    run_command(
        f"{PATH_GMX} pdb2gmx -f {fname_protein} -o {fname_pdb_prepro} -p topol.top -water spc -ff amber99sb -ignh", 
        args.mute_gro, cwd=outpath_md
    )
    logging.debug(f"Step 1 (Preprocessing protein) took {elapsed(step_start)}")
    return fname_pdb_prepro


def convert_mse_to_met(pdb_file, output_file):
    """
    Convert MSE residues to MET in the PDB file.
    This involves renaming residue name from MSE to MET and replacing SE atom names with SD.

    :param pdb_file: Path to the input PDB file containing MSE residues.
    :param output_file: Path to the output PDB file with MSE converted to MET.
    """
    try:
        with open(pdb_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # Process only ATOM and HETATM lines
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # Extract residue name (columns 18-20, 0-based index 17:20)
                    res_name = line[17:20]
                    # Check if residue is MSE
                    if res_name.strip() == "MSE":
                        # Replace residue name with MET
                        new_res_name = "MET"
                        # Ensure the new residue name occupies columns 18-20
                        line = line[:17] + new_res_name.ljust(3) + line[20:]
                        if "HETATM" in line:
                            line = line.replace("HETATM", "ATOM  ")
                        # Extract atom name (columns 13-16, 0-based index 12:16)
                        atom_name = line[12:16]
                        # Check if atom name contains SE
                        if "SE" in atom_name.strip():
                            # Replace SE with SD, preserving formatting
                            new_atom_name = atom_name.replace("SE", "SD")
                            # Ensure the new atom name occupies columns 13-16
                            line = line[:12] + new_atom_name.ljust(4) + line[16:]
                # Write the (possibly modified) line to the output file
                outfile.write(line)
    except Exception as e:
        logging.error(f"Error converting MSE to MET in {pdb_file}: {e}")
        raise
   
 
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
            shutil.move(final_pdb, too_high_dir)
            logging.warning(f"RMSD {rmsd:.3f} Å exceeds threshold {threshold} Å. Moved {final_pdb} to {too_high_dir}.")
            with open(failed_log, 'a') as fl:
                fl.write(f"{final_pdb} (RMSD: {rmsd:.3f} Å)\n")
        return rmsd
    except Exception as e:
        logging.error(f"Failed to compute RMSD for {final_pdb}: {e}")
        with open(failed_log, 'a') as fl:
            fl.write(f"{final_pdb} (RMSD computation failed)\n")
        return None

###############################################################################

def main(args):
    path_final_pdbs = os.path.join(args.outpath, "pdbs_last_frame")
    too_high_dir = os.path.join(args.outpath, "too_high_rmsd")
    failed_log = os.path.join(args.outpath, "failed_pdbs.ls")
    
    if args.get_pdb:
        tag = os.path.basename(os.getcwd())
        fname_out = f"{tag}_{args.get_pdb}.pdb"
        get_pdb_from_gro(args.get_pdb, fname_out, args, ".")
        return 0
    elif args.get_mkv:
        create_md_video("topol.tpr", "md.trr")
        create_md_video2("topol.tpr", "md.trr")
        return 0

    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    os.makedirs(path_final_pdbs, exist_ok=True)
    os.makedirs(too_high_dir, exist_ok=True)
        
    for path_pdb in glob.glob(os.path.join(args.inpath, f"*.pdb")):
        logging.info(f"---------- Processing {path_pdb} ----------")
        total_start = time.perf_counter()  # Start total timing
       
        fname_pdb = os.path.basename(path_pdb)
        tag = fname_pdb[:4]  # Assuming first 4 characters are the tag
        path_pdb = os.path.abspath(path_pdb)
        outpath_md = os.path.join(args.outpath, tag)
        os.makedirs(outpath_md, exist_ok=True)
        try:
            # Convert MSE to MET
            path_pdb_MET = os.path.join(outpath_md, f"{tag}_MET.pdb")
            convert_mse_to_met(path_pdb, path_pdb_MET) 
            # Check if ligand is present
            ligands = contains_ligand(path_pdb_MET)
            path_gro_final = os.path.join(outpath_md, f"{tag}_final.gro")
            if not os.path.exists(path_gro_final):
                if ligands:
                    logging.debug(f"Found ligands: {ligands} in {fname_pdb}.")
                    step_start = time.perf_counter()  # start total timing
                    path_protein, paths_ligands, path_ions = extract_ligands(path_pdb_MET, ligands, tag, outpath_md)
                    fname_protein = os.path.basename(path_protein)
                    fname_protein_prepro = preprocess_protein(fname_protein, tag, outpath_md, args)
                    try:
                        for path_ligand in paths_ligands:
                            fname_ligand = os.path.basename(path_ligand)
                            logging.debug(f"Preparing protein-ligand complex of {fname_protein} - {fname_ligand}")
                            prepare_complex(path_ligand, tag, outpath_md, args)
                        path_protein_prepro = os.path.join(outpath_md, fname_protein_prepro)
                        path_all_ligands = os.path.join(outpath_md, f"{tag}_ligands.pdb")
                        
                        remove_lines_in_place(path_protein_prepro, "ENDMDL")
                        if path_ions:
                            remove_lines_in_place(path_all_ligands, "END")
                            append_line_to_file(path_all_ligands, "TER")
                            append_line_to_file(path_ions, "END")
                            
                        run_command(
                            f"cat {fname_protein_prepro} {tag}_ligands.pdb {tag}_ions.pdb > {tag}_final.pdb", 
                            args.mute_gro, cwd=outpath_md
                        )
                        run_command(
                            f"{PATH_GMX} editconf -f {tag}_final.pdb -o {tag}_final.gro", 
                            args.mute_gro, cwd=outpath_md
                        )
                        dict_ion_count = count_hetatm_types(path_ions)
                        for ion, count in dict_ion_count.items():
                            append_line_to_file(os.path.join(outpath_md, "topol.top"), f"{ion}\t{count}")
                        logging.info(f"Step 2 (complex preparation) took {elapsed(step_start)}")
       
                    except Exception as e:
                        logging.error(f"Simulation failed for {path_pdb}. Skipping file.")
                        logging.exception(e)
                        with open(failed_log, 'a') as fl:
                            fl.write(f"{path_pdb}\n")
                        continue
                else:
                    logging.debug(f"Step 2 (Complex preparation) skipped for {fname_pdb} (no ligands found).")
                    fname_protein_prepro = preprocess_protein(os.path.basename(path_pdb_MET), tag, outpath_md, args)
                    run_command(
                        f"{PATH_GMX} editconf -f {fname_protein_prepro} -o {tag}_final.gro", 
                        args.mute_gro, cwd=outpath_md
                    )
                
            # Run MD simulation protocol
            run_MD(tag, outpath_md, args)
            if args.mode != 'gpu': 
                # Extract last frame only if simulation was successful
                path_pdb_final = os.path.join(path_final_pdbs, f"{tag}_lf_gro.pdb")
                
                if not os.path.exists(path_pdb_final):
                    fname_out = f"{tag}_lf_gro.pdb"
                    get_pdb_from_gro("md", fname_out, args, outpath_md)
                    temp_final_pdb = os.path.join(outpath_md, f"{tag}_lf_gro.pdb")
                    shutil.move(temp_final_pdb, path_final_pdbs)
                    logging.info(f"Simulation successful - moved {tag}_lf_gro.pdb to {path_final_pdbs}.")
                    logging.info(f"Total simulation time: {elapsed(total_start)}\n")
                
                # TODO remove waters 
                path_pdb_final_wo_H = path_pdb_final.replace(".pdb", "_wo_H.pdb")
                if not os.path.exists(path_pdb_final_wo_H):
                    remove_h_and_water(path_pdb_final, path_pdb_final_wo_H)
                    
                # Compute RMSD between initial and final PDB
                initial_pdb = path_pdb
                final_pdb = path_pdb_final_wo_H
                if os.path.exists(final_pdb):
                    rmsd = check_rmsd(
                        initial_pdb=initial_pdb,
                        final_pdb=final_pdb,
                        threshold=2.5,
                        too_high_dir=too_high_dir,
                        failed_log=failed_log
                    )
                    if rmsd is not None and rmsd > 2.5:
                        logging.info(f"{fname_pdb}: RMSD {rmsd:.3f} Å exceeds threshold. Moved to 'too_high_rmsd'.")
                    else:
                        logging.info(f"{fname_pdb}: RMSD {rmsd:.3f} Å within acceptable range.")
                else:
                    logging.warning(f"Final PDB {final_pdb} does not exist. Skipping RMSD computation.")
        except Exception as e:
            logging.error(f"Simulation failed for {path_pdb}. Skipping file.")
            logging.exception(e)
            with open(failed_log, 'a') as fl:
                fl.write(f"{path_pdb}\n")
            continue  # Skip to the next file on error
    logging.info(f"Done.") 

###############################################################################

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='ELEN-MD-%(levelname)s(%(asctime)s): %(message)s', 
        datefmt='%y-%m-%d %H:%M:%S'
    )
    parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inpath", type=str, default="1ubq.pdb")
    parser.add_argument("--outpath", type=str, default="MD_out")
    parser.add_argument("--get_pdb", type=str, default=None, help="Will try to extract a .pdb file from a GROMACS input file (.gro).")
    parser.add_argument("--get_mkv", type=str, default=None, help="Will try to render a .mkv video from a GROMACS input file (.gro).")
    parser.add_argument("--mode", type=str, default="mpi", choices=['gpu', 'mpi', 'dev'])
    parser.add_argument("--mute_gro", action="store_true", default=False, help="Mute stdout of GROMACS.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()
    main(args)