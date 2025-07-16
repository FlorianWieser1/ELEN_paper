import os
import re
import sys
import time
import logging
import subprocess
from rdkit import Chem

# ==============================
# Configuration Constants
# ==============================

# Paths to executables
PATH_GMX = '/home/thaidaev/programmi/gromacs-2023.3/bin/gmx'
PATH_GMX_MPI = '/home/thaidaev/programmi/gromacs-2023.3/bin/gmx_mpi'
PATH_ACPYPE = "/home/florian_wieser/miniconda3/envs/elen_test/bin/acpype"
PATH_REDUCE = "/usr/local/amber22/bin/reduce"
PATH_MDP = "/home/florian_wieser/projects/ELEN/elen/data_preparation/Kurniawan_protocol"
# ==============================
# Data Definitions
# ==============================

MOD_AA = {"MSE"}
WATERS = {"HOH", "WAT"}
IONS = {"CL", "ZN", "NA", "MG", "K", "CA", "FE", "CU", "MN", "NI", "CO", "CD", "PB", "SR", "BA"}
AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

# ==============================
# Helper Functions
# ==============================

def elapsed(step_start):
    """
    Calculates the elapsed time since step_start and returns a formatted string in hh:mm:ss.

    :param step_start: The start time captured using time.perf_counter().
    :return: Formatted elapsed time as a string in "hh:mm:ss".
    """
    elapsed_time = time.perf_counter() - step_start
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    elapsed_formatted = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    return elapsed_formatted


def run_command(command, mute_gro, cwd=None):
    """
    Executes a shell command and handles its output based on the mute_gro flag.

    :param command: The shell command to execute.
    :param mute_gro: If False, prints stdout and stderr.
    :param cwd: The working directory to execute the command in.
    :return: CompletedProcess instance.
    :raises subprocess.CalledProcessError: If the command execution fails.
    """
    logging.debug(f"Running command: {command} | CWD: {cwd}")
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if not mute_gro:
        if result.stdout:
            logging.info(result.stdout)
        if result.stderr:
            logging.warning(result.stderr)
    return result


def get_pdb_from_simulation_step(input_file, output_file, mode, mute_gro, cwd):
    """
    Generates a PDB file from a simulation step based on the specified mode.

    :param input_file: Path to the input file.
    :param output_file: Path to the output PDB file.
    :param mode: Operation mode ("whole_system" or "protein").
    :param mute_gro: If False, outputs command logs.
    :param cwd: The working directory to execute the command in.
    """
    if os.path.exists(output_file):
        os.remove(output_file)
        logging.debug(f"Removed existing output file: {output_file}")

    if mode == "whole_system":
        command = f"{PATH_GMX} editconf -f {input_file} -o {output_file}"
    elif mode == "protein":
        command = f"echo \"Protein\" | {PATH_GMX} trjconv -f {input_file} -o {output_file}"
    else:
        logging.error(f"Unknown mode '{mode}' specified.")
        raise ValueError(f"Unsupported mode: {mode}")

    logging.debug(f"Executing get_pdb_from_simulation_step with mode '{mode}'.")
    run_command(command, mute_gro, cwd=cwd)


def count_hetatm_types(file_path):
    """
    Counts each HETATM type in a PDB file and returns a dictionary with the counts.

    :param file_path: The path to the PDB file.
    :return: A dictionary with element symbols as keys and their counts as values.
    """
    counts = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('HETATM'):
                    # Element symbol is typically in columns 77-78
                    element = line[76:78].strip().upper()
                    if element:
                        counts[element] = counts.get(element, 0) + 1
        logging.debug(f"HETATM counts for {file_path}: {counts}")
        return counts
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return {}
    except Exception as e:
        logging.error(f"An error occurred while counting HETATM types in {file_path}: {e}")
        return {}


# ==============================
# Ligand Handling Functions
# ==============================

def contains_ligand(pdb_file):
    """
    Checks if the given PDB file contains ligand atoms (non-water, non-metal HETATM records).

    :param pdb_file: Path to the PDB file to be analyzed.
    :return: A set of ligand identifiers combining residue name and chain ID, or None if no ligands are found.
    """
    skip_residues = MOD_AA.union(WATERS).union(IONS)
    ligands = set()

    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("HETATM"):
                    residue = line[17:20].strip()
                    chain = line[21].strip()
                    if residue not in skip_residues:
                        ligand_id = f"{residue}_{chain}" if chain else f"{residue}_?"
                        ligands.add(ligand_id)
        if ligands:
            logging.info(f"Ligands found in {pdb_file}: {ligands}")
            return ligands
    except IOError as e:
        logging.error(f"Error reading file {pdb_file}: {e}")

    logging.info(f"No ligands found in {pdb_file}.")
    return None


def extract_ligands(path_pdb, ligands, tag, outpath):
    """
    Extracts HETATM records for each ligand per chain from a PDB file and writes them to separate ligand-only files.
    Simultaneously, removes the ligand lines from the original PDB and writes the
    protein structure without the ligands to a new protein-only PDB file. Additionally, extracts ion records.

    :param path_pdb: Path to the input PDB file.
    :param ligands: Set of ligand residue names (e.g., {"NAD", "CDP"}).
    :param tag: Tag for naming output files.
    :param outpath: Directory where the output files will be saved.
    :return: Tuple containing (protein_filename, ligand_filenames, ions_filename) if successful, else (None, None, None).
    """
    skip_residues = WATERS.union(IONS)
    ligand_files = {}
    ligand_filenames = []

    os.makedirs(outpath, exist_ok=True)
    protein_outpath = os.path.join(outpath, f"{tag}_protein.pdb")
    ions_outpath = os.path.join(outpath, f"{tag}_ions.pdb")

    try:
        with open(protein_outpath, 'w') as protein_outfile, \
             open(ions_outpath, 'w') as ion_outfile, \
             open(path_pdb, 'r') as infile:

            for line in infile:
                if line.startswith("HETATM"):
                    residue = line[17:20].strip()
                    chain = line[21].strip()
                    ligand_id = f"{residue}_{chain}" if chain else f"{residue}_?"

                    if ligand_id in ligands:
                        if ligand_id not in ligand_files:
                            ligand_outpath = os.path.join(outpath, f"{tag}_{ligand_id}.pdb")
                            ligand_files[ligand_id] = open(ligand_outpath, 'w')
                            ligand_filenames.append(ligand_outpath)
                            logging.info(f"Created ligand file: {ligand_outpath}")
                        ligand_files[ligand_id].write(line)
                    elif residue in IONS:
                        ion_outfile.write(line)
                    elif residue not in skip_residues:
                        logging.warning(f"Unrecognized HETATM residue '{residue}' in {path_pdb}.")
                elif line.startswith("ATOM"):
                    residue = line[17:20].strip()
                    if residue in AMINO_ACIDS:
                        protein_outfile.write(line)

        # Close all ligand files
        for f in ligand_files.values():
            f.close()

        logging.debug(f"Protein file written to: {protein_outpath}")
        logging.debug(f"Ions file written to: {ions_outpath}")
        logging.debug(f"Ligand files written: {ligand_filenames}")

        return protein_outpath, ligand_filenames, ions_outpath

    except IOError as e:
        logging.error(f"Error processing file {path_pdb}: {e}")
        # Ensure all opened ligand files are closed in case of an error
        for f in ligand_files.values():
            try:
                f.close()
            except Exception:
                pass
        return None, None, None


# ==============================
# Topology Manipulation Functions
# ==============================

def extract_atomtypes_section(input_file, output_file):
    """
    Extracts the [ atomtypes ] section from the input_file, saves it to output_file,
    and removes it from the input_file.

    :param input_file: Path to the original file.
    :param output_file: Path to save the extracted [ atomtypes ] section.
    """
    in_atomtypes = False
    extracted_lines = []
    remaining_lines = []

    try:
        with open(input_file, 'r') as infile:
            for line in infile:
                stripped = line.strip()
                if stripped.startswith('[') and stripped.endswith(']'):
                    if in_atomtypes:
                        in_atomtypes = False
                    if stripped.lower() == '[ atomtypes ]':
                        in_atomtypes = True
                        extracted_lines.append(line)
                        continue
                if in_atomtypes:
                    if stripped.startswith(';') or stripped == '':
                        extracted_lines.append(line)
                    else:
                        extracted_lines.append(line)
                    continue
                else:
                    remaining_lines.append(line)

        with open(output_file, 'w') as outfile:
            outfile.writelines(extracted_lines)
        logging.info(f"Extracted [ atomtypes ] section to {output_file}.")

        with open(input_file, 'w') as infile:
            infile.writelines(remaining_lines)
        logging.info(f"Removed [ atomtypes ] section from {input_file}.")

    except IOError as e:
        logging.error(f"IOError during atomtypes extraction: {e}")


def append_after_matching_line(file_path, match_line, new_line):
    """
    Appends `new_line` after the next free line following `match_line` in the file.
    If no free line is found after `match_line`, appends `new_line` at the end.

    :param file_path: Path to the text file.
    :param match_line: The line to match (exact match).
    :param new_line: The line to append.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        match_indices = [i for i, line in enumerate(lines) if line.strip() == match_line]

        if not match_indices:
            logging.warning(f"Match line '{match_line}' not found in {file_path}. No action taken.")
            return

        match_index = match_indices[0]
        insert_index = match_index + 1

        while insert_index < len(lines):
            if lines[insert_index].strip() == "":
                break
            insert_index += 1

        if insert_index < len(lines):
            lines.insert(insert_index, new_line + '\n')
            logging.info(f"Appended '{new_line}' after '{match_line}' in {file_path}.")
        else:
            lines.append(new_line + '\n')
            logging.info(f"Appended '{new_line}' at the end of {file_path}.")

        with open(file_path, 'w') as file:
            file.writelines(lines)

    except IOError as e:
        logging.error(f"IOError while appending line to {file_path}: {e}")


def append_line_to_file(file_path, line_to_append):
    """
    Appends a string line to the end of a file.

    :param file_path: The path to the file.
    :param line_to_append: The line to append to the file.
    """
    try:
        with open(file_path, "a") as file:
            file.write(line_to_append + "\n")
        logging.info(f"Appended line to {file_path}: {line_to_append}")
    except IOError as e:
        logging.error(f"An error occurred while appending to the file {file_path}: {e}")


def remove_lines_in_place(file_path, string_to_match):
    """
    Removes lines containing the specified string from the file in place.

    :param file_path: Path to the file to be modified.
    :param string_to_match: The string to search for in each line. Lines containing this string will be removed.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        filtered_lines = [line for line in lines if string_to_match not in line]

        with open(file_path, 'w') as file:
            file.writelines(filtered_lines)

        logging.debug(f"Removed all lines containing '{string_to_match}' from '{file_path}'.")
    except FileNotFoundError:
        logging.error(f"The file '{file_path}' does not exist.")
    except IOError as e:
        logging.error(f"IOError occurred while modifying '{file_path}': {e}")


def compute_net_charge_and_electron_count(pdb_file):
    """
    Computes the net charge and total number of electrons of a molecule from its PDB file using RDKit.

    :param pdb_file: Path to the PDB file.
    :return: Tuple containing (formal_charge, total_electrons).
    :raises ValueError: If the PDB file cannot be parsed.
    """
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    if mol is None:
        logging.error(f"Cannot parse PDB file {pdb_file}")
        raise ValueError(f"Cannot parse PDB file {pdb_file}")

    formal_charge = Chem.GetFormalCharge(mol)
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    total_electrons = sum(atomic_numbers) + formal_charge  # Assuming all valence electrons are accounted for

    logging.debug(f"Computed net charge: {formal_charge}, total electrons: {total_electrons} for {pdb_file}")
    return formal_charge, total_electrons


# ==============================
# PDB Extraction Functions
# ==============================

def get_group_number(group_name, cwd):
    """
    Runs 'gmx make_ndx' to parse the topology file and retrieve the group number for the specified group name.

    :param group_name: Name of the group to find (e.g., 'Protein').
    :param cwd: The working directory to execute the command in.
    :return: Group number as integer if found, else None.
    """
    
    cmd = [PATH_GMX, 'make_ndx', '-f', "topol.tpr", '-o', 'tmp']

    try:
        process = subprocess.run(
            cmd,
            input='q\n',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=cwd
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running make_ndx: {e.stderr}")
        return None

    group_pattern = re.compile(r'^\s*(\d+)\s+(.+?)\s*:\s*\d+\s+atoms$')

    for line in process.stdout.splitlines():
        match = group_pattern.match(line)
        if match:
            num, name = match.groups()
            if name.strip().lower() == group_name.strip().lower():
                logging.debug(f"Found group '{group_name}' with number {num}.")
                return int(num)
    logging.warning(f"Group '{group_name}' not found in make_ndx output.")
    return None


def get_topol_tpr(path_input, protocol, args, cwd):
    """
    Prepares the topology file using GROMACS grompp.

    :param path_input: Path to the input GRO file.
    :param protocol: Protocol name corresponding to the MDP file.
    :param args: Namespace containing additional arguments (expects 'mute_gro').
    :param cwd: The working directory to execute the command in.
    """
    mdp_path = os.path.join(PATH_MDP, protocol)
    command = (
        f"{PATH_GMX} grompp -f {mdp_path} -c {path_input} -p topol.top -o topol.tpr --maxwarn 100"
    )
    logging.debug(f"Preparing topology with command: {command}")
    run_command(command, args.mute_gro, cwd=cwd)


def create_ndx_group(list_group_numbers, list_group_names, cwd):
    """
    Creates a custom index group combining 'Protein' and 'Other'.

    :param grnr_protein: Group number for Protein.
    :param grnr_other: Group number for Other.
    :param cwd: The working directory to execute the command in.
    :return: Filename of the created index group.
    :raises subprocess.CalledProcessError: If the command execution fails.
    """
    
    if len(list_group_numbers) >= 1:
        command_select_groups = str(list_group_numbers[0])
        for grnr in list_group_numbers[1:]:
            command_select_groups += " | " + str(grnr)
        
        command_name = "name "
        for gr_name in list_group_names:
            command_name += gr_name + "_and_"
        command_name = command_name[:-5] 
        
        fname_ndx = command_name + ".ndx"
        make_ndx_commands = f"{command_select_groups}\n{command_name}\nq\n"

    try:
        subprocess.run(
            [PATH_GMX, "make_ndx", "-f", "topol.tpr", "-o", fname_ndx],
            input=make_ndx_commands,
            text=True,
            capture_output=True,
            check=True,
            cwd=cwd
        )
        logging.info(f"Created combined index group file: {fname_ndx}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error creating combined index group in {fname_ndx}: {e.stderr}")
        raise

    return fname_ndx


def extract_pdb(input_gro, group, fname_ndx, output_pdb, args, cwd):
    """
    Extracts specified group into a PDB file.
    :param input_gro: Path to the input GRO file (e.g., md.gro).
    :param group: Group name to extract (e.g., "Other" or "Ion").
    :param fname_ndx: Path to the index file containing the group.
    :param output_pdb: Path to the output PDB file.
    :param args: Namespace containing additional arguments (expects 'mute_gro').
    :param cwd: The working directory to execute the command in.
    """
    trjconv_commands = f"{group}\n"
    print(f"fname_ndx: {fname_ndx}")
    if fname_ndx is not None:
        command = [PATH_GMX, "trjconv", "-f", input_gro, "-o", output_pdb, "-s", "topol.tpr", "-n", fname_ndx]
    else:
        command = [PATH_GMX, "trjconv", "-f", input_gro, "-o", output_pdb, "-s", "topol.tpr"]

    try:
        process = subprocess.run(
            command,
            input=trjconv_commands,
            text=True,
            capture_output=True,
            check=True,
            cwd=cwd
        )
        if not args.mute_gro:
            logging.info(f"Extracted PDB saved to {output_pdb}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting PDB from {input_gro}: {e.stderr}")
        raise


def get_pdb_from_gro(protocol_step, fname_pdb, args, outpath_md):
    """
    Extracts a PDB file from a GRO file based on the protocol step.

    :param protocol_step: Protocol step name corresponding to the GRO and MDP files.
    :param fname_pdb: Filename for the output PDB.
    :param args: Namespace containing additional arguments (expects 'mute_gro').
    :param outpath_md: Output directory for the MD files.
    """
    fname_gro = f"{protocol_step}.gro"
    fname_mdp = f"{protocol_step}.mdp"

    logging.debug(f"Generating topology for protocol step '{protocol_step}'.")
    get_topol_tpr(fname_gro, fname_mdp, args, outpath_md)

    grnr_protein = get_group_number("Protein", outpath_md)
    grnr_other = get_group_number("Other", outpath_md)
    grnr_ions = get_group_number("Ion", outpath_md)
    logging.debug(f"Group numbers - Protein: {grnr_protein}, Other: {grnr_other}, Ion: {grnr_ions}")
    print(f"grnr_other: {grnr_other}") 
    if grnr_other is not None:
        fname_ndx = create_ndx_group([grnr_protein, grnr_other], ["Protein", "Other"], outpath_md)
        extract_pdb(fname_gro, "Protein_Other", fname_ndx, fname_pdb, args, outpath_md)

    else: 
        fname_ndx  = None
        extract_pdb(fname_gro, "Protein", fname_ndx, fname_pdb, args, outpath_md)

    logging.info(f"Extracted PDB file: {fname_pdb}")