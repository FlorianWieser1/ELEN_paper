from Bio.PDB import PDBParser

def get_residue_ids(path_pdb: str) -> list:
    """
    Parse a PDB file to extract a list of residue numbers.
    
    Parameters:
        path_pdb (str): Path to the PDB file.
    
    Returns:
        list: A list of integers representing the residue numbers in the PDB structure.
    """
    parser = PDBParser(QUIET=True)  # Quiet mode to suppress warnings
    resnum_list = []
    try:
        structure = parser.get_structure("structure", path_pdb)
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Ensure the residue ID is correctly accessed
                    if residue.id[0] == ' ':
                        resnum_list.append(residue.id[1])
    except FileNotFoundError:
        print(f"Error: The file {path_pdb} does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred while parsing the PDB file: {e}")
        return []
    
    return resnum_list


def discard_pdb(path_pdb, path_discarded, step, error, log_file="discarded_pdb.log"):
    """
    Safely move a PDB file to the discarded directory and log the action.
    
    Args:
        path_pdb (str): Path to the PDB file.
        path_discarded (str): Path to the discarded directory.
        step (str): The step at which the file was discarded.
        error (str): The error message associated with the discard.
        log_file (str): Path to the logfile (default: "discarded_pdb.log").
    """

    # Ensure the discarded directory exists
    os.makedirs(path_discarded, exist_ok=True)
    
    dest = os.path.join(path_discarded, os.path.basename(path_pdb))
    
    # Check if the destination file already exists
    if os.path.exists(dest):
        message = (f"File {dest} already exists. Skipping move for {path_pdb}. "
                   f"Step: {step}, Error: {error}\n")
    else:
        try:
            shutil.move(path_pdb, path_discarded)
            message = (f"Moved {path_pdb} to {path_discarded}. "
                       f"Step: {step}, Error: {error}\n")
        except Exception as move_error:
            message = (f"Error moving {path_pdb} to {path_discarded}: {move_error}. "
                       f"Step: {step}, Error: {error}\n")
    
    # Log the discard event
    with open(log_file, "a") as log:
        log.write(message)