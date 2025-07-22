import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from Bio.PDB import PDBParser, PDBIO
from types import SimpleNamespace
from elen.shared_utils.utils_pdb import get_residue_ids

# ------- Logging ---------
logger = logging.getLogger(__name__)

def convert_numpy(obj):
    """Convert a numpy array/object to a Python native list."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError("Object is not a numpy array.")

def find_pdb_files(input_dir: str) -> list[str]:
    """Find all .pdb files in a directory. Exit if none found."""
    path_pattern = os.path.join(input_dir, "*.pdb")
    paths_pdbs = glob.glob(path_pattern)
    logger.info(f"Detected input file(s): {paths_pdbs}")
    if not paths_pdbs:
        logger.error("No input files found in the directory.")
        sys.exit(1)
    return paths_pdbs

def load_and_modify_hyperparameters(
    checkpoint: dict, 
    input_dir: str, 
    use_labels: bool, 
    outpath: str
) -> SimpleNamespace:
    """
    Loads hyperparameters from a checkpoint and modifies them for inference.
    """
    hparams_ckpt = SimpleNamespace(**checkpoint['hyper_parameters'])
    hparams_ckpt.one_hot_encoding = "standard"
    hparams_ckpt.skip_connections = False
    hparams_ckpt.test_dir = input_dir
    hparams_ckpt.use_labels = use_labels
    hparams_ckpt.outpath = outpath
    hparams_ckpt.activation = "relu"
    return hparams_ckpt

def flatten_predictions(
    dict_pred: dict[str, dict[str, dict[str, list[float]]]]
) -> list[dict[str, str]]:
    """
    Flatten nested prediction dicts to list of dicts for DataFrame construction.
    """
    return [
        {'metric': metric, 'filename': filename, 'index': i, 'pred': pred_value}
        for metric in dict_pred
        for filename in dict_pred[metric]
        for i, pred_value in enumerate(dict_pred[metric][filename]['pred'])
    ]

def get_loop_position_from_file(file_path: str) -> tuple[int, int]:
    """
    Extracts the start and stop positions of a loop from a given file.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip().startswith('loop_position_target'):
                    _, start, stop = line.split()
                    return int(start), int(stop)
        raise ValueError("Loop positions 'start' and 'stop' not found in the file.")
    except FileNotFoundError:
        raise FileNotFoundError(f"No file found at the specified path: {file_path}")
    except Exception as e:
        raise Exception(f"Error in get_loop_position_from_file: {e}")

def split_into_chain(path_pdb: str, outpath: str) -> None:
    """
    Split a PDB file into separate files per chain, fixing MSE/CSE and residue numbering.
    """
    if not os.path.exists(path_pdb):
        raise FileNotFoundError(f"The file {path_pdb} does not exist.")
    os.makedirs(outpath, exist_ok=True)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("Protein_ID", path_pdb)
    for model in structure:
        for chain in model:
            for residue in chain.get_residues():
                old_id = residue.id
                if residue.resname.strip() == "MSE":
                    residue.resname = "MET"
                    residue.id = (' ', old_id[1], old_id[2])
                elif residue.resname.strip() == "CSE":
                    residue.resname = "CYS"
                    residue.id = (' ', old_id[1], old_id[2])
            io = PDBIO()
            io.set_structure(chain)
            fname_base = os.path.splitext(os.path.basename(path_pdb))[0]
            chain_id = chain.id.strip() or 'A'
            fname_pdb_split = f"{fname_base}_{chain_id}.pdb"
            path_pdb_split = os.path.join(outpath, fname_pdb_split)
            io.save(path_pdb_split)

def process_loop_residue_data(
    path_loops: str, 
    elen_score: str, 
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process loop PDBs: match predictions, filter, and aggregate scores for loops.
    """
    paths_loops = glob.glob(os.path.join(path_loops, "*.pdb"))
    dict_resnums, dict_positions = {}, {}
    invalid_files = []
    reference_length = None
    for path_loop in paths_loops:
        fname_loop = os.path.basename(path_loop)
        resnum_list = get_residue_ids(path_loop)
        loop_start, loop_stop = get_loop_position_from_file(path_loop)
        current_length = len(resnum_list)
        if reference_length is None:
            reference_length = current_length
        if current_length != reference_length:
            logger.warning(f"Skipping {fname_loop} - length mismatch: expected {reference_length}, got {current_length}")
            invalid_files.append(fname_loop)
            continue
        dict_resnums[fname_loop] = resnum_list
        dict_positions[fname_loop] = {'loop_start': loop_start, 'loop_stop': loop_stop}
    if not dict_resnums:
        logger.info("No valid PDB files to process after length filtering.")
        return pd.DataFrame(columns=['filename', 'res_id', 'pred'])
    df_resnums = pd.DataFrame(dict_resnums).reset_index()
    melted_df = df_resnums.melt(id_vars=['index'], var_name='filename', value_name='res_id')
    df_lddt = df[df['metric'] == elen_score].copy()
    df_lddt['index_mod'] = df_lddt.groupby('filename').cumcount()
    merged_df = pd.merge(melted_df, df_lddt, left_on=['filename', 'index'], right_on=['filename', 'index_mod'], how='inner')
    final_df = merged_df[['filename', 'res_id', 'pred']]
    final_df['res_id'] = final_df['res_id'].astype(int)
    df_positions = pd.DataFrame(dict_positions).T.astype({'loop_start': int, 'loop_stop': int})
    # Filter for loop regions
    filtered_df = pd.concat([
        final_df[(final_df['filename'] == idx) & 
                 (final_df['res_id'] >= row['loop_start']) & 
                 (final_df['res_id'] <= row['loop_stop'])]
        for idx, row in df_positions.iterrows()
    ], ignore_index=True)
    # Add loop_id and output file names
    filtered_df["loop_id"] = filtered_df["filename"].str.extract(r".*?_(\d+)_[HE]{2}\.pdb$")[0].astype(int)
    filtered_df['fname_pdb'] = filtered_df['filename'].str.replace(r'_A_\d+_[HE]{2}\.pdb$', '_A.pdb', regex=True)
    # Aggregation
    filtered_df['avg_per_loop'] = filtered_df.groupby(['fname_pdb', 'loop_id'])['pred'].transform('mean')
    filtered_df['avg_per_chain'] = filtered_df.groupby('fname_pdb')['pred'].transform('mean')
    filtered_df = filtered_df[['fname_pdb', 'loop_id', 'res_id', 'pred', 'avg_per_loop', 'avg_per_chain']]
    filtered_df = filtered_df.sort_values(by=["fname_pdb", "loop_id", "res_id"]).rename(columns={'pred': 'ELEN_score'})
    filtered_df[['ELEN_score', 'avg_per_loop', 'avg_per_chain']] = filtered_df[['ELEN_score', 'avg_per_loop', 'avg_per_chain']].round(3)
    return filtered_df

def process_residue_data(
    path_extracted: str, 
    elen_score: str, 
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extracts rows from DataFrame for residue positions listed in corresponding files.
    """
    positions = {}
    for fname in df['filename'].unique():
        file_path = os.path.join(path_extracted, fname)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip().startswith('residue_position_tensor'):
                    positions[fname] = int(line.split()[1])
                    break
    df['residue_position_tensor'] = df['filename'].map(positions)
    df = df[(df['index'] == df['residue_position_tensor']) & (df['metric'] == elen_score)]
    df['res_id'] = df['filename'].str.extract(r'(\d+)\.pdb$').astype(int)
    df = df.drop(columns=['index', 'residue_position_tensor'])
    df['filename'] = df['filename'].str.replace(r'_(\d+)\.pdb$', '.pdb', regex=True)
    df['avg_per_chain'] = df.groupby('filename')['pred'].transform('mean')
    df = df.rename(columns={'filename': 'fname_pdb', 'pred': 'ELEN_score'}).reset_index(drop=True)
    df['res_id'] = df['res_id'].astype(int)
    df[['ELEN_score', 'avg_per_chain']] = df[['ELEN_score', 'avg_per_chain']].round(3)
    df = df.sort_values(by=['fname_pdb', 'res_id'])
    df = df[['metric', 'fname_pdb', 'res_id', 'ELEN_score', 'avg_per_chain']]
    return df

def get_total_number_of_residues(path_pdb: str) -> int:
    """Count total residues in a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", path_pdb)
    return sum(len(list(chain.get_residues())) for model in structure for chain in model)

def write_elen_scores_to_pdb(
    path_pdb: str, 
    res_id_pred_dict: dict[int, float], 
    outpath: str
) -> str:
    """
    Overwrite B-factor column in a PDB file with ELEN scores. Returns output path.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", path_pdb)
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.id[1]
                new_bfactor = res_id_pred_dict.get(res_id, np.nan)
                for atom in residue:
                    atom.set_bfactor(new_bfactor)
    io = PDBIO()
    io.set_structure(structure)
    path_output_pdb = os.path.join(
        outpath, 
        f"{os.path.basename(path_pdb).replace('.pdb', '')}_elen_scored_tmp.pdb"
    )
    io.save(path_output_pdb)
    return path_output_pdb

def merge_residue_numbers(
    original_pdb: str, 
    cleaned_pdb: str, 
    outpath_pdb: str
):
    """
    Replace residue numbering in the cleaned PDB with the original numbers.
    """
    original_res_ids = get_residue_ids(original_pdb)
    with open(cleaned_pdb, 'r') as infile:
        lines = infile.readlines()
    new_lines = []
    current_res_index = 0
    previous_clean_residue = None
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            clean_residue = line[22:26]
            if clean_residue != previous_clean_residue:
                previous_clean_residue = clean_residue
                orig_res = original_res_ids[current_res_index] if current_res_index < len(original_res_ids) else clean_residue
                current_res_index += 1
            new_line = line[:22] + f"{orig_res:>4}" + line[26:]
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    with open(outpath_pdb, 'w') as outfile:
        outfile.writelines(new_lines)

def process_pdb_files(
    df_predictions: pd.DataFrame, 
    outpath: str, 
    path_pdbs_prepared: str, 
    path_original_split: str, 
    pocket_type: str,
    fill_missing: bool = True
):
    """
    Generalized processing for writing ELEN scores and merging residue numbers.
    Handles both loop and residue-predicted cases.
    """
    os.makedirs(outpath, exist_ok=True)
    paths_pdbs = glob.glob(os.path.join(path_pdbs_prepared, "*.pdb"))
    for path_pdb in paths_pdbs:
        base_filename = os.path.splitext(os.path.basename(path_pdb))[0]
        total_residues = get_total_number_of_residues(path_pdb)
        res_df = pd.DataFrame({'res_id': range(1, total_residues + 1)})
        df_filtered = df_predictions[df_predictions['fname_pdb'].str.startswith(base_filename)]
        if df_filtered.empty and fill_missing:
            logger.info(f"No predictions for '{base_filename}'; filling with 0.0.")
            merged_df = res_df.copy()
            merged_df['ELEN_score'] = 0.0
        elif not df_filtered.empty:
            df_filtered = df_filtered[['res_id', 'ELEN_score']].copy()
            df_filtered['res_id'] = df_filtered['res_id'].astype(int)
            merged_df = res_df.merge(df_filtered, on='res_id', how='left')
            merged_df['ELEN_score'] = merged_df['ELEN_score'].fillna(0.0)
        else:
            logger.warning(f"Skipping '{base_filename}' (no predictions).")
            continue
        res_id_pred_dict = dict(zip(merged_df['res_id'], merged_df['ELEN_score']))
        path_pdb_elen_scored = write_elen_scores_to_pdb(path_pdb, res_id_pred_dict, outpath)
        try:
            path_pdb_orig = glob.glob(os.path.join(path_original_split, f"{base_filename}.pdb"))[0]
        except IndexError:
            logger.warning(f"No original PDB found for '{base_filename}' in '{path_original_split}'. Skipping renumbering.")
            continue
        outpath_pdb = path_pdb_elen_scored.replace("_elen_scored_tmp.pdb", f"_{pocket_type}_elen_scored.pdb")
        merge_residue_numbers(path_pdb_orig, path_pdb_elen_scored, outpath_pdb)
        if os.path.exists(path_pdb_elen_scored):
            os.remove(path_pdb_elen_scored)

def add_combined_scores_to_dict(
    dict_pred: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Add combined scores to predictions dict, weighted averages of existing metrics.
    """
    max_rmsd = find_maximum_pred_rmsd(dict_pred)
    weights_all = {'lddt': 0.33, 'cad': 0.33, 'rmsd': 0.34}
    weights_lddt_cad = {'lddt': 0.5, 'cad': 0.5}
    dict_pred['all'], dict_pred['lddt_cad'] = {}, {}
    for fname_pdb in dict_pred['lddt']:
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
    """Normalize RMSD values to [0, 1]."""
    return rmsd / max_rmsd

def invert_scores(score: np.ndarray) -> np.ndarray:
    """Invert scores so higher is better."""
    return 1 - score

def find_maximum_pred_rmsd(data: dict) -> float:
    """Find max RMSD across all predictions."""
    max_pred = float('-inf')
    for protein_values in data['rmsd'].values():
        current_max_pred = max(protein_values['pred'])
        if current_max_pred > max_pred:
            max_pred = current_max_pred
    return max_pred

def custom_collate_fn(batch):
    """Custom collate for DataLoader, filtering out None."""
    filtered_batch = [data for data in batch if data is not None]
    if not filtered_batch:
        raise ValueError("All datapoints in batch were None!")
    return filtered_batch
