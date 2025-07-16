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