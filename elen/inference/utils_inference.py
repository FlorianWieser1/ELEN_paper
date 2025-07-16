import os
import sys
import glob
import h5py
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from Bio.PDB import PDBParser, PDBIO
from types import SimpleNamespace
from torch_geometric.data import Batch
from elen.shared_utils.utils_pdb import get_residue_ids
from elen.shared_utils.utils_io import load_from_json

def convert_numpy(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError

def find_pdb_files(input_dir):
    """
    Search for PDB files in the specified directory.
    Exits with an error if no PDB files are found.
    
    Args:
    input_dir (str): The directory to search for PDB files.
    
    Returns:
    list: A list of paths to PDB files.
    """
    path_pattern = os.path.join(input_dir, "*.pdb")
    paths_pdbs = glob.glob(path_pattern)
    logging.info(f"Detected input file(s): {paths_pdbs}")
    if not paths_pdbs:
        logging.error("No input files found in the directory.")
        sys.exit(1)  # Exit with a non-zero code to indicate an error
    return paths_pdbs


def load_and_modify_hyperparameters(checkpoint: dict[str, any], input_dir: str, use_labels: bool, outpath: str) -> SimpleNamespace:
    """
    Loads hyperparameters from a model checkpoint, modifies them based on provided arguments, 
    and returns a namespace object with updated hyperparameters.

    Args:
    checkpoint (Dict[str, Any]): A dictionary containing the checkpoint data from which hyperparameters are extracted.
    input_dir (str): The directory for input data.
    use_labels (bool): Flag indicating whether to use labels.
    outpath (str): The output directory for results.

    Returns:
    SimpleNamespace: An object containing the updated hyperparameters.
    """
    # Extract hyperparameters from checkpoint and convert to SimpleNamespace for easy attribute access
    hparams_ckpt = checkpoint['hyper_parameters']
    hparams_ckpt = SimpleNamespace(**hparams_ckpt)

    # Modify hyperparameters based on the function arguments
    hparams_ckpt.one_hot_encoding = "standard"
    hparams_ckpt.skip_connections = False
    hparams_ckpt.test_dir = input_dir
    hparams_ckpt.use_labels = use_labels
    hparams_ckpt.outpath = outpath
    hparams_ckpt.activation = "relu"
    return hparams_ckpt


def flatten_predictions(dict_pred: dict[str, dict[str, dict[str, list[float]]]]) -> list[dict[str, str]]:
    """
    Flattens a nested dictionary of predictions into a list of dictionaries.

    Args:
        dict_pred (Dict[str, Dict[str, Dict[str, List[float]]]]): A dictionary with metrics as keys, and another
             dictionary with filenames as keys and a list of prediction values as values.
    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing a flattened structure of the prediction data.
    """
    rows = []
    for metric in dict_pred:
        for filename in dict_pred[metric]:
            for i, pred_value in enumerate(dict_pred[metric][filename]['pred']):
                rows.append({'metric': metric, 'filename': filename, 'index': i, 'pred': pred_value})
    return rows


def get_loop_position_from_file(file_path: str) -> tuple:
    """
    Extracts the start and stop positions of a loop from a given file.
    
    The function expects the file to contain a line starting with 'loop_position_target'
    followed by start and stop positions.

    Args:
        file_path (str): Path to the file containing loop position data.

    Returns:
        tuple: A tuple containing the start and stop positions as integers.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If 'loop_position_target' is not found or the positions are not integers.
        Exception: For other unforeseen issues.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        start, stop = None, None
        for line in lines:
            if line.strip().startswith('loop_position_target'):
                _, start, stop = line.split()
                start, stop = int(start), int(stop)
                break
        
        if start is None or stop is None:
            raise ValueError("Loop positions 'start' and 'stop' not found in the file.")

        return start, stop
    except FileNotFoundError:
        raise FileNotFoundError(f"No file found at the specified path: {file_path}")
    except ValueError as ve:
        raise ValueError(f"Data format error in the file: {ve}")
    except Exception as e:
        raise Exception(f"An error occurred while processing the file: {e}")

def split_into_chain(path_pdb: str, outpath: str) -> None:
    """
    Splits a PDB file into separate files for each chain,
    preserving original residue IDs even if MSE/CSE gets renamed.

    - Reads the PDB file using Biopython
    - For each chain in each model:
      * Renames MSE -> MET or CSE -> CYS
      * Forces the 'hetero-flag' to ' ' and keeps the same sequence number,
        preserving residue numbering
      * Writes each chain to its own PDB file
    
    Args:
        path_pdb (str): The file path to the input PDB file.
        outpath (str): The directory where the output files will be saved.

    Raises:
        FileNotFoundError: If the input PDB file does not exist.
        Exception: If parsing or writing fails.
    """
    if not os.path.exists(path_pdb):
        raise FileNotFoundError(f"The file {path_pdb} does not exist.")
    
    os.makedirs(outpath, exist_ok=True)

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("Protein_ID", path_pdb)
        
        # For each model and chain
        for model in structure:
            for chain in model:
                
                for residue in chain.get_residues():
                    old_id = residue.id    # (hetero_flag, res_seq, i_code)
                    
                    if residue.resname.strip() == "MSE":
                        residue.resname = "MET"
                        # Force the residue to be treated as a standard residue.
                        residue.id = (' ', old_id[1], old_id[2])
                    elif residue.resname.strip() == "CSE":
                        residue.resname = "CYS"
                        residue.id = (' ', old_id[1], old_id[2])

                io = PDBIO()
                io.set_structure(chain)
                
                fname_base = os.path.splitext(os.path.basename(path_pdb))[0]
                chain_id = chain.id.strip() if chain.id.strip() else 'A'
                fname_pdb_split = f"{fname_base}_{chain_id}.pdb"
                path_pdb_split = os.path.join(outpath, fname_pdb_split)

                io.save(path_pdb_split)
    
    except Exception as e:
        raise Exception(f"An error occurred while processing the file {path_pdb}: {e}")


def process_loop_residue_data(path_loops: str, elen_score: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes PDB files to extract residue numbers and loop positions, merges predictions,
    and filters data. Skips files whose residue list lengths differ from the reference.

    Args:
        path_loops (str): Directory containing PDB files.
        elen_score (str): The metric name to filter predictions on.
        df (pd.DataFrame): DataFrame containing predicted scores.

    Returns:
        pd.DataFrame: A DataFrame containing filtered predictions for residues
                      within specified loop regions.
    """
    #print(df)
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    paths_loops = glob.glob(os.path.join(path_loops, "*.pdb"))
    dict_resnums, dict_positions = {}, {}

    # Keep track of invalid files (length mismatch) for logging
    invalid_files = []

    # This will be set by the first valid file we encounter
    reference_length = None

    # Process each PDB file
    for path_loop in paths_loops:
        fname_loop = os.path.basename(path_loop)
        
        resnum_list = get_residue_ids(path_loop)  # e.g. [1, 2, 3, ...]
        loop_start, loop_stop = get_loop_position_from_file(path_loop)
        
        # Determine the length of the current residue list
        current_length = len(resnum_list)

        # If this is our first valid file, establish the reference length
        if reference_length is None:
            reference_length = current_length

        # Check if this file's residue list length matches the reference
        if current_length != reference_length:
            logging.warning(f"Skipping {fname_loop} - length mismatch: "
                            f"expected {reference_length}, got {current_length}")
            invalid_files.append(fname_loop)
            continue

        # Store valid entries
        dict_resnums[fname_loop] = resnum_list
        dict_positions[fname_loop] = {'loop_start': loop_start, 'loop_stop': loop_stop}
    
    # Log any invalid files
    if invalid_files:
        logging.warning(f"Files skipped due to length mismatch: {invalid_files}")

    # If everything is invalid, return empty DataFrame early
    if not dict_resnums:
        logging.info("No valid PDB files to process after length filtering.")
        return pd.DataFrame(columns=['filename', 'res_id', 'pred'])

    # Create DataFrames for residue numbers and positions
    df_resnums = pd.DataFrame(dict_resnums)
    df_positions = pd.DataFrame(dict_positions).T
    
    # Merging and processing predictions
    df_lddt = df[df['metric'] == elen_score].copy()
    
    # Reshape df_resnums so it's "tidy": columns -> [index, filename, res_id]
    df_resnums = df_resnums.reset_index()
    melted_df = df_resnums.melt(
        id_vars=['index'], var_name='filename', value_name='res_id'
    )
    df_lddt['index_mod'] = df_lddt.groupby('filename').cumcount()

    # Merge on (filename, index) = (filename, index_mod)
    merged_df = pd.merge(
        melted_df,
        df_lddt,
        left_on=['filename', 'index'],
        right_on=['filename', 'index_mod'],
        how='inner'
    )

    final_df = merged_df[['filename', 'res_id', 'pred']]
    final_df.loc[:, 'res_id'] = final_df['res_id'].astype(int)

    # Set data types for position columns
    df_positions['loop_start'] = df_positions['loop_start'].astype(int)
    df_positions['loop_stop']   = df_positions['loop_stop'].astype(int)

    # Filter results based on loop positions
    filtered_df = pd.DataFrame(columns=final_df.columns)
    for idx, row in df_positions.iterrows():
        loop_start, loop_stop = row['loop_start'], row['loop_stop']
        mask = (
            (final_df['filename'] == idx) &
            (final_df['res_id'] >= loop_start) &
            (final_df['res_id'] <= loop_stop)
        )
        filtered_rows = final_df[mask]
        filtered_df = pd.concat([filtered_df, filtered_rows], ignore_index=True)
    logging.info("Final predictions processed.")
    
    # Extract loop_id and truncate filename
    filtered_df["loop_id"] = (filtered_df["filename"].str.extract(r".*?_(\d+)_[HE]{2}\.pdb$")[0].astype(int))
    filtered_df['fname_pdb'] = filtered_df['filename'].str.replace(r'_A_\d+_[HE]{2}\.pdb$', '_A.pdb', regex=True)

    # Compute average pred per loop_id
    #filtered_df['avg_per_loop'] = filtered_df.groupby('loop_id')['pred'].transform('mean')
    filtered_df['avg_per_loop'] = filtered_df.groupby(['fname_pdb','loop_id'])['pred'].transform('mean')

    # Compute average pred per truncated filename
    filtered_df['avg_per_chain'] = filtered_df.groupby('fname_pdb')['pred'].transform('mean')

    # Reorder columns for readability
    filtered_df = filtered_df[['fname_pdb', 'loop_id', 'res_id', 'pred', 'avg_per_loop', 'avg_per_chain']]
    filtered_df = filtered_df.sort_values(by=["fname_pdb", "loop_id", "res_id"])
    filtered_df = filtered_df.rename(columns={'pred': 'ELEN_score'})
    
    # round scores to 3 meaningful digits
    filtered_df[['ELEN_score', 'avg_per_loop', 'avg_per_chain']] = \
    filtered_df[['ELEN_score', 'avg_per_loop', 'avg_per_chain']].round(3)
    return filtered_df

def process_residue_data(path_extracted: str, elen_score: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts rows from a DataFrame based on the residue positions specified in separate files,
    each linked to the DataFrame via a 'filename' column.

    Each file contains a line starting with 'residue_position_tensor' followed by the index
    of the residue of interest. This function filters the DataFrame to only include rows
    where the 'index' column matches this residue position, and the 'metric' column value
    is 'score_lddt_cad' (or whatever is passed in elen_score).

    Args:
        df (pd.DataFrame): DataFrame containing at least the columns 
                           'filename', 'index', 'metric', and 'pred'.
        path_extracted (str): Path to the directory containing the files 
                              referenced in the DataFrame.
        elen_score (str): The name of the metric to filter on (e.g. "score_lddt_cad").

    Returns:
        pd.DataFrame: Filtered DataFrame based on the criteria described, 
                      potentially empty if no matching conditions are found 
                      or files are missing.

    Raises:
        FileNotFoundError: If a file listed in the DataFrame's 'filename' column 
                           cannot be found.
    """

    # Step 1: Build a dictionary {filename: residue_position}
    positions = {}
    unique_fnames = df['filename'].unique()

    for fname in unique_fnames:
        file_path = os.path.join(path_extracted, fname)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        # Read the file once, find the line with 'residue_position_tensor'
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('residue_position_tensor'):
                    parts = line.split()
                    # The second token should be the position, convert to int
                    positions[fname] = int(parts[1])
                    break
        # If for some reason the file didn't have the expected line,
        # you might decide to store None or skip it.
        # positions[fname] = None

    # Step 2: Use a vectorized approach to filter the DataFrame
    # 2a) Map each row's filename to the corresponding residue position
    df['residue_position_tensor'] = df['filename'].map(positions)
    # 2b) Keep only rows where 'index' == 'residue_position_tensor'
    df = df[df['index'] == df['residue_position_tensor']]
    # 2c) Further filter on the metric
    df = df[df['metric'] == elen_score]
    
    # Now perform the final transformations.
    # (Everything here is identical to your original logic.)
    # Extract integer ID from 'filename' (assuming it ends with e.g. "_1234.pdb")
    df['res_id'] = df['filename'].str.extract(r'(\d+)\.pdb$').astype(int)

    # Drop the 'index' and 'residue_position_tensor' columns
    df.drop(columns=['index', 'residue_position_tensor'], inplace=True)

    # Keep only the needed columns in a specific order
    df = df[['metric', 'filename', 'res_id', 'pred']]

    # Replace 'filename' like "_1234.pdb" -> ".pdb"
    df['filename'] = df['filename'].str.replace(r'_(\d+)\.pdb$', '.pdb', regex=True)

    # Group by filename and compute mean
    df['avg_per_chain'] = df.groupby('filename')['pred'].transform('mean')

    # Round your numeric columns
    df['pred'] = df['pred'].round(3)
    df['avg_per_chain'] = df['avg_per_chain'].round(3)
    # Rename columns
    df = df.rename(columns={
        'filename': 'fname_pdb',
        'pred': 'ELEN_score'
    })

    # Reset index and ensure 'res_id' is int
    df = df.reset_index(drop=True)
    df['res_id'] = df['res_id'].astype(int)

    # Sort for convenience
    df = df.sort_values(by=['fname_pdb', 'res_id'])
    return df

def get_total_number_of_residues(path_pdb: str) -> int:
    """
    Calculate the total number of residues in a protein structure from a PDB file.
    
    Args:
    path_pdb (str): The file path to the PDB file containing the protein structure.
    
    Returns:
    int: The total number of residues across all models and chains in the structure.
    
    Raises:
    FileNotFoundError: If the PDB file cannot be found at the specified path.
    ValueError: If the PDB file cannot be parsed.
    """
    parser = PDBParser(QUIET=True)  # Suppress warnings from the PDBParser
    try:
        structure = parser.get_structure("protein", path_pdb)
    except IOError:
        raise FileNotFoundError(f"Unable to find or open the file at {path_pdb}")
    except ValueError:
        raise ValueError(f"Failed to parse PDB file at {path_pdb}")
    total_residues = 0
    for model in structure:
        for chain in model:
            total_residues += len(list(chain.get_residues()))
    return total_residues

def write_elen_scores_to_pdb(path_pdb: str, res_id_pred_dict: dict[int, float], outpath: str) -> None:
    """
    Modify the B-factor column of a PDB file to include ELEN quality scores and save the updated PDB.

    Args:
    path_pdb (str): Path to the input PDB file.
    res_id_pred_dict (Dict[int, float]): A dictionary mapping residue IDs to ELEN quality scores.
    outpath (str): Output directory where the modified PDB file will be saved.

    Raises:
    FileNotFoundError: If the input PDB file is not found.
    IOError: If there are issues reading or writing the PDB file.
    KeyError: If a residue ID from the PDB file does not exist in the res_id_pred_dict.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", path_pdb)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {path_pdb} does not exist.")
    except IOError:
        raise IOError(f"Could not read the file {path_pdb}.")

    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.id[1]
                try:
                    #new_bfactor = res_id_pred_dict[res_id]
                    new_bfactor = res_id_pred_dict.get(res_id, np.nan)

                except KeyError:
                    raise KeyError(f"Residue ID {res_id} not found in the prediction dictionary.")
                for atom in residue:
                    atom.set_bfactor(new_bfactor)

    io = PDBIO()
    io.set_structure(structure)
    try:
        path_output_pdb = os.path.join(outpath, f"{os.path.basename(path_pdb).replace('.pdb', '')}_elen_scored_tmp.pdb")
        io.save(path_output_pdb)
        return path_output_pdb
    except IOError:
        raise IOError(f"Could not write to the file {path_output_pdb}.")
    
def merge_residue_numbers(original_pdb, cleaned_pdb, outpath_pdb):
    """
    Replace residue numbering in the cleaned PDB with the original residue numbering.
    
    Parameters:
        original_pdb (str): Path to the original PDB file.
        cleaned_pdb (str): Path to the cleaned PDB file.
        output_pdb (str): Path to write the merged PDB file.
    """
    # 1. Extract the original residue numbers. 
    #    This should return a list of residue numbers in the order they appear.
    original_res_ids = get_residue_ids(original_pdb)
    
    with open(cleaned_pdb, 'r') as infile:
        lines = infile.readlines()
    
    new_lines = []
    current_res_index = 0
    previous_clean_residue = None  # To detect residue changes in the cleaned file
    
    for line in lines:
        # Only modify lines that contain atomic coordinate data.
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # Extract the current residue number from the cleaned line.
            # (PDB residue numbers are usually in columns 23-26; note Python indexing starts at 0)
            clean_residue = line[22:26]
            
            # Check if this line belongs to a new residue.
            if clean_residue != previous_clean_residue:
                previous_clean_residue = clean_residue
                # Use the original residue number if available.
                if current_res_index < len(original_res_ids):
                    orig_res = original_res_ids[current_res_index]
                else:
                    # If for some reason the mapping is exhausted, fall back to the cleaned value.
                    orig_res = clean_residue
                current_res_index += 1
            
            # Replace the residue number field (columns 22-26) with the original number.
            # We right-align the number in a 4-character field.
            new_line = line[:22] + f"{orig_res:>4}" + line[26:]
            new_lines.append(new_line)
        else:
            new_lines.append(line)
            
    with open(outpath_pdb, 'w') as outfile:
        outfile.writelines(new_lines) 
        
        
def process_pdb_files_LP(df_predictions, outpath, path_pdbs_prepared, path_original_split, pocket_type):
    """
    Processes multiple PDB files by updating their B-factor column based on a given DataFrame,
    then saves the modified PDB files.

    Args:
        df_predictions (pd.DataFrame): DataFrame containing columns ['fname_pdb', 'res_id', 'ELEN_score'].
        outpath (str): Base output directory for saving processed PDB files.
        path_pdbs_prepared (str): Directory containing prepared PDB files.
        path_original_split (str): Directory containing original split PDB files for residue renumbering.
        args: Additional arguments, particularly 'pocket_type'.

    Description:
    The function searches for all PDB files in `path_pdbs_prepared`, computes the total number 
    of residues for each PDB, and merges this data with `df_predictions` to fill missing 
    predictions with a default value of 0.0. Each PDB file is then updated with these 
    predictions and saved back to the output directory.

    If the corresponding original PDB file for renumbering is not found, the function catches
    the exception, logs a warning, and continues without crashing.
    """

    # Ensure the output directory exists
    os.makedirs(outpath, exist_ok=True)

    # Gather all prepared PDB files
    paths_pdbs = glob.glob(os.path.join(path_pdbs_prepared, "*.pdb"))
    
    for path_pdb in paths_pdbs:
        base_filename = os.path.splitext(os.path.basename(path_pdb))[0]  # e.g., '4uos_A' or 'some.file.with.dots'
        
        # Determine how many residues this PDB has
        total_residues = get_total_number_of_residues(path_pdb)
        # Create a simple DataFrame of all residue IDs
        res_df = pd.DataFrame({'res_id': range(1, total_residues + 1)})

        # Filter predictions that start with the same base filename
        df_filtered = df_predictions[df_predictions['fname_pdb'].str.startswith(base_filename)]
        if df_filtered.empty:
            logging.info(f"[INFO] No matching predictions for '{base_filename}'; filling all with 0.0.")
            # We'll simply fill everything with 0.0
            merged_df = res_df.copy()
            merged_df['ELEN_score'] = 0.0
        else:
            # Make sure 'res_id' is integer for merging
            df_filtered = df_filtered[['res_id', 'ELEN_score']].copy()
            df_filtered['res_id'] = df_filtered['res_id'].astype(int)

            # Merge to fill missing residues with 0.0
            merged_df = res_df.merge(df_filtered, on='res_id', how='left')
            merged_df['ELEN_score'] = merged_df['ELEN_score'].fillna(0.0)

        # Create dictionary: residue_id -> ELEN_score
        res_id_pred_dict = dict(zip(merged_df['res_id'], merged_df['ELEN_score']))

        # Write updated scores into a temporary PDB file
        path_pdb_elen_scored = write_elen_scores_to_pdb(path_pdb, res_id_pred_dict, outpath)
        
        # Attempt to merge old residue numbering from original PDB
        try:
            path_pdb_orig = glob.glob(os.path.join(path_original_split, f"{base_filename}.pdb"))[0]
        except IndexError:
            logging.warning(
                f"[WARNING] No original PDB found for '{base_filename}' in '{path_original_split}'. "
                "Skipping renumbering."
            )
            # If you want to clean up the temporary file, uncomment below:
            # if os.path.exists(path_pdb_elen_scored):
            #     os.remove(path_pdb_elen_scored)
            continue

        # Final output path, replacing the temp suffix with the pocket type
        outpath_pdb = path_pdb_elen_scored.replace("_elen_scored_tmp.pdb", f"_{pocket_type}_elen_scored.pdb")
        
        merge_residue_numbers(path_pdb_orig, path_pdb_elen_scored, outpath_pdb)
        
        # Remove the temporary file
        if os.path.exists(path_pdb_elen_scored):
            os.remove(path_pdb_elen_scored)

        
def process_pdb_files_RP(df_predictions, outpath, path_pdbs_prepared, path_original_split, pocket_type):
    """
    Processes PDB files by writing ELEN scores into the PDB,
    merging residue numbers, and removing intermediate files.

    :param df_predictions: Pandas DataFrame with columns ['fname_pdb', 'res_id', 'ELEN_score']
    :param outpath: Path to the output directory
    :param path_pdbs_prepared: Directory containing the prepared PDB files
    :param path_original_split: Directory containing the original PDB files
    :param args: Arguments object with at least 'pocket_type' attribute
    """
    # Ensure the output directory exists
    os.makedirs(outpath, exist_ok=True)

    paths_pdbs = glob.glob(os.path.join(path_pdbs_prepared, "*.pdb"))
    for path_pdb in paths_pdbs:
        # Safely extract the base filename (minus .pdb)
        base_filename = os.path.splitext(os.path.basename(path_pdb))[0]

        # Filter predictions that match the current PDB filename
        filtered_df = df_predictions[df_predictions['fname_pdb'].str.startswith(base_filename)]
        if filtered_df.empty:
            print(f"[WARNING] No predictions found for {base_filename}. Skipping scoring.")
            continue

        # Build a dict for residue_id -> ELEN_score
        res_id_pred_dict = filtered_df.set_index('res_id')['ELEN_score'].to_dict()

        # Write ELEN scores to a temporary PDB
        path_pdb_elen_scored = write_elen_scores_to_pdb(path_pdb, res_id_pred_dict, outpath)

        # Attempt to find the original PDB file in 'path_original_split'
        pdb_orig_matches = glob.glob(os.path.join(path_original_split, f"{base_filename}.pdb"))
        if not pdb_orig_matches:
            print(f"[WARNING] No original PDB file found for '{base_filename}'. Skipping merge.")
            # Clean up the temporary file if desired
            if os.path.exists(path_pdb_elen_scored):
                os.remove(path_pdb_elen_scored)
            continue

        # Merge old residue numbering using the first match
        path_pdb_orig = pdb_orig_matches[0]
        outpath_pdb = path_pdb_elen_scored.replace(
            "_elen_scored_tmp.pdb", f"_{pocket_type}_elen_scored.pdb"
        )

        # Merge residue numbers and clean up
        merge_residue_numbers(path_pdb_orig, path_pdb_elen_scored, outpath_pdb)
        os.remove(path_pdb_elen_scored)


def add_combined_scores_to_dict(dict_pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhances the input dictionary with combined scores based on the weights of different metrics.

    Parameters:
        dict_pred (dict): A dictionary containing the predictions for metrics 'lddt', 'rmsd', and 'CAD'.

    Returns:
        Dict[str, Any]: The updated dictionary with new keys 'score_all' and 'score_lddt_cad' holding the combined scores.
    """
    max_rmsd = find_maximum_pred_rmsd(dict_pred)
    weights_all = {'lddt': 0.33, 'cad': 0.33, 'rmsd': 0.34}
    weights_lddt_cad = {'lddt': 0.5, 'cad': 0.5}

    dict_pred['all'] = {}
    dict_pred['lddt_cad'] = {}

    for fname_pdb, _ in dict_pred['lddt'].items():
        array_lddt = np.array(dict_pred['lddt'][fname_pdb]['pred'])
        array_rmsd = np.array(dict_pred['rmsd'][fname_pdb]['pred'])
        array_cad = np.array(dict_pred['CAD'][fname_pdb]['pred'])

        norm_rmsd = invert_scores(normalize_rmsd(array_rmsd, max_rmsd))

        score_all = weights_all['lddt'] * array_lddt + weights_all['cad'] * array_cad + weights_all['rmsd'] * norm_rmsd
        score_lddt_cad = weights_lddt_cad['lddt'] * array_lddt + weights_lddt_cad['cad'] * array_cad

        dict_pred['all'][fname_pdb] = {'pred': list(score_all)}
        dict_pred['lddt_cad'][fname_pdb] = {'pred': list(score_lddt_cad)}

    return dict_pred


def normalize_rmsd(rmsd: np.ndarray, max_rmsd: float) -> np.ndarray:
    """Normalizes RMSD values to the range [0, 1] using Min-Max scaling."""
    return rmsd / max_rmsd


def invert_scores(score: np.ndarray) -> np.ndarray:
    """Inverts scores for metrics where a lower value is better, transforming to a range [0, 1]."""
    return 1 - score


def find_maximum_pred_rmsd(data):
    """Finds the maximum RMSD value in the prediction data to use for normalization."""
    max_pred = float('-inf')  # Initialize to the smallest possible float
    for _, protein_values in data['rmsd'].items():
        for _, pdb_values in protein_values.items():
            current_max_pred = max(pdb_values)
            if current_max_pred > max_pred:
                max_pred = current_max_pred
    return max_pred

def custom_collate_fn(batch):
    filtered_batch = [data for data in batch if data is not None]
    if len(filtered_batch) == 0:
        raise ValueError("All datapoints in batch were None!")
    return Batch.from_data_list(filtered_batch)

def check_features_exist(path_extracted, paths_pdbs, path_features, saprot_features_path):
    """
    Check if for each .pdb in paths_pdbs, all required features exist in the features directory.
    This function verifies the presence of 'atom_features.json', 'residue_features.json', and a record
    in the saprot features (saprot_650M.h5) for each pdb file. For any pdb missing features, it removes
    the matching .pdb files in path_extracted (based on the shared identifier at the beginning of the filename)
    and prints how many and which files were removed.
    """

    residue_features_path = os.path.join(path_features, "residue_features.json")

    if not os.path.exists(residue_features_path):
        raise FileNotFoundError(f"Residue features file not found: {residue_features_path}")
    if not os.path.exists(saprot_features_path):
        raise FileNotFoundError(f"Saprot features file not found: {saprot_features_path}")

    residue_features = load_from_json(residue_features_path)

    with h5py.File(saprot_features_path, 'r') as f:
        saprot_keys = list(f.keys())

    missing = set()
    for pdb in paths_pdbs:
        pdb_basename = os.path.basename(pdb)
        if pdb_basename not in residue_features:
            missing.add(pdb)
            logging.info(f"residue_features not found for {pdb_basename}.")
        if pdb_basename not in saprot_keys:
            missing.add(pdb)
            logging.info(f"saprot embedding not found for {pdb_basename}.")

    #print(f"path_pdbs sample: {paths_pdbs[0] if paths_pdbs else 'No pdbs provided'}")
    if len(missing) > 0:
        logging.warning(f"Missing feature files for the following pdbs: {missing}")

    # Get all .pdb files in path_extracted
    paths_extracted = glob.glob(os.path.join(path_extracted, "*.pdb"))
    removed_files = []

    # Determine missing identifiers from the missing pdb files (identifier = first two underscore-separated parts)
    missing_identifiers = set()
    for missing_path in missing:
        missing_basename = os.path.basename(missing_path)
        parts = missing_basename.split('_')
        if len(parts) >= 2:
            identifier = "_".join(parts[:2])
            missing_identifiers.add(identifier)

    # Remove matching .pdb files from path_extracted
    for extracted_file in paths_extracted:
        extracted_basename = os.path.basename(extracted_file)
        parts = extracted_basename.split('_')
        if len(parts) >= 2:
            identifier = "_".join(parts[:2])
            if identifier in missing_identifiers:
                try:
                    os.remove(extracted_file)
                    removed_files.append(extracted_file)
                    logging.info(f"Removed {extracted_file} due to missing features.")
                except Exception as e:
                    logging.error(f"Error removing {extracted_file}: {str(e)}")

    if len(removed_files) > 0:
        print(f"Removed {len(removed_files)} files from {path_extracted}:")
        for file in removed_files:
            print(f" - {file}")
