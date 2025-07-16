import os
import re
import sys
import glob
import json
import subprocess
import pandas as pd
from Bio.PDB import PDBParser, PDBIO
from typing import List, Any
from biopandas.pdb import PandasPdb
from collections import OrderedDict, defaultdict

# TODO refactor 313
# If you need these, uncomment if available in your environment
from biopandas.pdb import PandasPdb
from elen.shared_utils.shared_utils import func_timer
from elen.shared_utils.utils_pdb import get_residue_ids
from elen.config import PATH_INFERENCE, PATH_ELEN_MODELS
from elen.shared_utils.constants import AA_THREE_TO_ONE

################################################################################
# Helpers

def fast_get_bfactors_from_pdb(path_pdb: str) -> List[float]:
    """
    Quickly extract unique B-factors per residue from a raw PDB file.
    
    The PDB format expects the B-factor in columns 61-66 (1-based indexing).
    This function uses the chain identifier (column 22), residue sequence number 
    (columns 23-26), and insertion code (column 27) to uniquely identify each residue.
    
    Parameters:
        path_pdb (str): The path to the PDB file.
        
    Returns:
        List[float]: A list of B-factors, one per unique residue.
        
    Raises:
        FileNotFoundError: If the specified PDB file does not exist.
    """
    bfactors = []
    seen_residues = set()
    
    if not os.path.isfile(path_pdb):
        raise FileNotFoundError(f"No PDB file found at {path_pdb}.")
    
    with open(path_pdb, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                # Extract residue identifiers:
                # - chain: column 22 (index 21)
                # - residue sequence number: columns 23-26 (index 22:26)
                # - insertion code: column 27 (index 26)
                chain = line[21].strip()
                resseq = line[22:26].strip()
                insertion = line[26].strip()
                residue_id = (chain, resseq, insertion)
                
                if residue_id not in seen_residues:
                    seen_residues.add(residue_id)
                    b_str = line[60:66].strip()
                    bfactors.append(float(b_str))
    return bfactors

# old method
def get_perres_lddt_from_pdb(path_pdb: str) -> List[float]:
    try:
        pdb = PandasPdb().read_pdb(path_pdb)
    except FileNotFoundError:
        raise FileNotFoundError(f"No PDB file found at {path_pdb}.")
    
    # Group by residue_number, take the first atom's b_factor
    try:
        res_lddts = [residue.iloc[0]['b_factor'] 
                     for _, residue in pdb.df['ATOM'].groupby(['residue_number'])]
    except KeyError:
        raise ValueError("B-factor column is missing or not in the expected format.")
    return res_lddts

def read_lddt_json(filepath):
    """
    Reads a JSON file containing local LDDT scores and returns an ordered dictionary
    mapping residue numbers (rnum) to their corresponding local_lddt scores and single-letter sequences.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        OrderedDict: Mapping of residue numbers to dictionaries containing local lDDT scores and single-letter amino acid codes.
    """
    with open(filepath, 'r') as file:
        data = json.load(file)

    rnum_to_info = OrderedDict()

    for residue in data['results']['local_scores']:
        rnum = residue['rnum']
        local_lddt = residue['local_lddt']
        rname_three = residue['rname']
        rname_one = AA_THREE_TO_ONE.get(rname_three, 'X')  # 'X' for unknown amino acids

        if local_lddt >= 0:  # Exclude residues with invalid scores
            rnum_to_info[rnum] = {'lddt': local_lddt, 'aa': rname_one}

    return rnum_to_info

############################################################################
# 1) LOOP POCKETS (LP)
def get_mqa_data_inference_LP(args: Any) -> pd.DataFrame:
    """
    Retrieves and processes MQA data for loop modeling, but now supports multiple ELEN models.

    For each identifier:
      - If a matching ground truth file or any of the method files is not found, the entire
        identifier is skipped.
      - Each ELEN model found in `elen_results_*` is loaded, and its scores are stored
        as a separate column (e.g., ELEN_j0ribitb, ELEN_jwwrx159, ...).
      - Residues are intersected across GT and methods, then merged with any available ELEN scores.
      - A final DataFrame is returned with one row per residue, including:
            identifier, resid, GT, <method columns>, <ELEN_*> columns, loop_id
        where loop_id is computed (for convenience) from the first ELEN model encountered.
      - Non-loop residues (loop_id == 0) are filtered out in the return value.

    Parameters
    ----------
    args : Any
        Contains:
         - args.outpath
         - args.inpath_models
         - args.inpath_methods
         - args.methods
         - Additional fields needed by `run_elen_inference_LP` if it prepares the folders.

    Returns
    -------
    pd.DataFrame
        A DataFrame of loop-only residues with GT, methods, and columns for each ELEN model.
    """

    # -------------------------------------------------------------------------
    # 1) Run ELEN if needed (this is presumably your existing code or utility).
    # -------------------------------------------------------------------------
    outpath_elen = os.path.join(args.outpath, "elen")
    run_elen_inference(args, outpath_elen, "LP")

    # -------------------------------------------------------------------------
    # 2) Discover all ELEN result folders (for multiple models).
    #    e.g., elen_results_j0ribitb, elen_results_jwwrx159, etc.
    # -------------------------------------------------------------------------
    # Each folder has PDB files ending in *_elen_scored.pdb

    # We will collect all ELEN scores into a nested dict:
    # elen_scores[model_suffix][identifier][resid] = <score>
    elen_scores = defaultdict(lambda: defaultdict(dict))
    paths_elen_models = [os.path.join(outpath_elen, f"elen_results_{m}") for m in args.elen_models]
    for elen_model_path in paths_elen_models:
        model_name = os.path.basename(elen_model_path)         # e.g. "elen_results_jwwrx159"
        model_suffix = model_name.replace("elen_results_", "") # e.g. "jwwrx159"

        # For each ELEN-scored PDB in this folder, parse out the identifier and read the scores.
        for path_pdb_scored in glob.glob(os.path.join(elen_model_path, "*elen_scored.pdb")):
            base_pdb_name = os.path.basename(path_pdb_scored)
            # Example: 8W1M_A_36_1_m1_A_LP_elen_scored.pdb
            # We look for something up to "_m1"
            match = re.match(r"(.*?_m1)", base_pdb_name)
            if not match:
                continue
            # e.g., match.group(1) = '8W1M_A_36_1_m1'
            identifier = match.group(1).rstrip("_m1")  # '8W1M_A_36_1'

            # Retrieve the numeric bfactor scores from the PDB
            score_list = fast_get_bfactors_from_pdb(path_pdb_scored)
            resid_list = get_residue_ids(path_pdb_scored)

            # Store them in the nested dictionary
            for r, s in zip(resid_list, score_list):
                elen_scores[model_suffix][identifier][r] = s

    # -------------------------------------------------------------------------
    # 3) Identify which identifiers are valid (must have GT + all methods).
    #    If missing any method or GT, the entire identifier is skipped.
    # -------------------------------------------------------------------------
    valid_identifiers = []
    all_identifiers_in_elen = set()
    for model_suffix, id_dict in elen_scores.items():
        all_identifiers_in_elen.update(id_dict.keys())

    for identifier in sorted(all_identifiers_in_elen):
        # Check ground truth existence
        gt_files_json = glob.glob(os.path.join(str(args.inpath_models), f"{identifier}*lddt.json"))
        if not gt_files_json:
            # If there's no ground truth for this identifier, skip it entirely.
            continue

        # Check each method
        missing_method = False
        for method in args.methods:
            method_files = glob.glob(os.path.join(str(args.inpath_methods), method, f"{identifier}*.pdb"))
            if not method_files:
                missing_method = True
                break

        if not missing_method:
            valid_identifiers.append(identifier)

    # -------------------------------------------------------------------------
    # 4) Build a row-wise data structure for the final output.
    #    For each valid identifier, we retrieve:
    #      - GT
    #      - Each method's scores
    #      - All ELEN model scores for that identifier
    #    Then we intersect the residue sets (just like the original code).
    # -------------------------------------------------------------------------
    all_rows = []

    for identifier in valid_identifiers:
        # Load GT
        gt_files_json = glob.glob(os.path.join(str(args.inpath_models), f"{identifier}*lddt.json"))
        path_json_gt = gt_files_json[0]  # picking the first matching one
        dict_gt_raw = read_lddt_json(path_json_gt)
        dict_gt = {int(resid): entry['lddt'] for resid, entry in dict_gt_raw.items()}

        # Load each method
        method_dicts = {}
        for method in args.methods:
            method_files = glob.glob(os.path.join(str(args.inpath_methods), method, f"{identifier}*.pdb"))
            # Take the first match if multiple
            path_method = method_files[0]
            m_scores = fast_get_bfactors_from_pdb(path_method)
            m_resids = get_residue_ids(path_method)
            method_dicts[method] = dict(zip(m_resids, m_scores))

        # Determine the "common" residue set among GT + the methods.
        # (This is the intersection logic from your original snippet.)
        common_resids = set(dict_gt.keys())
        for method in args.methods:
            common_resids &= set(method_dicts[method].keys())

        # Additionally, we only keep residues that appear in at least one of the ELEN models
        # for this identifier. If you prefer strict intersection among all ELEN models,
        # adjust accordingly. For now, we do a union across all models, then intersect with
        # GT & methods.
        elen_union = set()
        for model_suffix in elen_scores.keys():
            if identifier in elen_scores[model_suffix]:
                elen_union |= set(elen_scores[model_suffix][identifier].keys())

        # final intersection:
        final_resids = common_resids & elen_union

        # For each residue in final_resids, build one row
        for resid in sorted(final_resids):
            row = {
                'identifier': identifier,
                'resid': resid,
                'GT': dict_gt[resid]
            }
            # Add each method's score
            for method in args.methods:
                row[method] = method_dicts[method][resid]

            # Add each ELEN model's score (if any for that residue)
            for model_suffix in elen_scores.keys():
                model_dict = elen_scores[model_suffix].get(identifier, {})
                if resid in model_dict:
                    row[f"ELEN_{model_suffix}"] = model_dict[resid]
                else:
                    row[f"ELEN_{model_suffix}"] = 0.0  # or None, if you prefer

            all_rows.append(row)

    # -------------------------------------------------------------------------
    # 5) Convert the collected rows to a DataFrame.
    # -------------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    if df.empty:
        # If nothing valid is found, just return an empty df
        return df

    # Sort to ensure correct ordering for loop ID calculation
    df.sort_values(by=["identifier", "resid"], inplace=True, ignore_index=True)

    # -------------------------------------------------------------------------
    # 6) Create a single loop_id column based on the *first* ELEN model we see.
    #    This mimics the original approach of: 
    #       df['loop_id'] = (df['ELEN'] != 0).astype(int).diff().fillna(0).eq(1).cumsum()
    #    We do it per-identifier so that IDs don't bleed into each other.
    # -------------------------------------------------------------------------
    elen_cols = [c for c in df.columns if c.startswith("ELEN_")]
    if elen_cols:
        # Pick the first ELEN column in alphabetical order
        elen_cols.sort()
        reference_elen = elen_cols[0]

        loop_ids = []
        current_loop_id = 0
        prev_val = 0
        prev_identifier = None

        for i, row in df.iterrows():
            if row['identifier'] != prev_identifier:
                # Reset for new structure
                current_loop_id = 0
                prev_val = 0
                prev_identifier = row['identifier']

            val = 1 if row[reference_elen] != 0 else 0
            if val == 1 and prev_val == 0:
                current_loop_id += 1
            loop_ids.append(current_loop_id if val == 1 else 0)
            prev_val = val

        df['loop_id'] = loop_ids
    else:
        # In case no ELEN columns exist
        df['loop_id'] = 0

    # -------------------------------------------------------------------------
    # 7) Filter out non-loop residues (those with loop_id == 0) and return.
    # -------------------------------------------------------------------------
    df_loops = df[df['loop_id'] != 0].copy()
    df_loops.reset_index(drop=True, inplace=True)
    return df_loops

############################################################################
# 2) RESIDUE POCKETS (RP)
@func_timer
def get_mqa_data_inference_RP(args):
    """
    Processes molecular quality data for Residue Pockets (RP), supporting multiple
    ELEN models. Each model's scores appear in columns: ELEN_<model_id>.

    Steps:
      1) Runs ELEN inference for "RP" if needed.
      2) Gathers all '*RP_elen_scored.pdb' files for each model in args.elen_models
         and extracts the B-factors (ELEN scores).
      3) Identifies valid identifiers (must have GT + each method's PDB).
      4) Builds a final DataFrame with columns:
          [identifier, resid, GT, <method columns>, ELEN_<model1>, ELEN_<model2>, ...]
      5) Drops rows with missing GT or method scores (dropna), then returns.
    """

    # 1) Run ELEN for Residue Pockets if necessary
    outpath_elen = os.path.join(args.outpath, "elen")
    run_elen_inference(args, outpath_elen, "RP")

    # Prepare a nested dict for storing:
    #   elen_scores[model_id][identifier][resid] = <float score>
    elen_scores = defaultdict(lambda: defaultdict(dict))

    # 2) Loop over all requested ELEN models in args.elen_models
    for model_id in args.elen_models:
        elen_model_path = os.path.join(outpath_elen, f"elen_results_{model_id}")
        if not os.path.isdir(elen_model_path):
            print(f"[Warning] No ELEN folder found for model '{model_id}': {elen_model_path} (skipping)")
            continue

        # Collect all RP-scored PDBs for this model
        scored_pdbs = glob.glob(os.path.join(elen_model_path, "*RP_elen_scored.pdb"))
        for path_pdb_scored in scored_pdbs:
            base_pdb = os.path.basename(path_pdb_scored)
            # Example: 8W1M_A_20_1_m1_A_RP_elen_scored.pdb
            # We'll look for up to "_m1"
            match = re.match(r"(.*?_m1)", base_pdb)
            if not match:
                continue
            # e.g. match.group(1) = "8W1M_A_20_1_m1"
            identifier = match.group(1).rstrip("_m1")  # => "8W1M_A_20_1"

            # Parse the ELEN scores from the PDB B-factors
            bfactors = fast_get_bfactors_from_pdb(path_pdb_scored)
            resids = get_residue_ids(path_pdb_scored)

            for r, s in zip(resids, bfactors):
                elen_scores[model_id][identifier][r] = s

    # 3) Figure out which identifiers appear in ANY ELEN model
    #    Then check if each has GT + each method file
    all_identifiers = set()
    for model_id, id_dict in elen_scores.items():
        all_identifiers.update(id_dict.keys())

    valid_identifiers = []
    for identifier in sorted(all_identifiers):
        # Check GT existence
        gt_files_json = glob.glob(os.path.join(str(args.inpath_models), f"{identifier}*lddt.json"))
        if not gt_files_json:
            print(f"[Skipping] GT JSON not found for {identifier}")
            continue

        # Check method files
        missing_method = False
        for method in args.methods:
            method_files = glob.glob(os.path.join(str(args.inpath_methods), method, f"{identifier}*.pdb"))
            if not method_files:
                print(f"[Skipping] Method '{method}' file not found for {identifier}")
                missing_method = True
                break
        if missing_method:
            continue

        valid_identifiers.append(identifier)

    # 4) Build rows for the final DataFrame
    rows = []

    for identifier in valid_identifiers:
        # Load GT
        gt_files_json = glob.glob(os.path.join(str(args.inpath_models), f"{identifier}*lddt.json"))
        path_json_gt = gt_files_json[0]
        dict_gt_raw = read_lddt_json(path_json_gt)
        dict_gt = {int(rid): entry['lddt'] for rid, entry in dict_gt_raw.items()}

        # Load each method
        method_dicts = {}
        for method in args.methods:
            method_files = glob.glob(os.path.join(str(args.inpath_methods), method, f"{identifier}*.pdb"))
            path_method = method_files[0]
            scores = fast_get_bfactors_from_pdb(path_method)
            resids = get_residue_ids(path_method)
            method_dicts[method] = dict(zip(resids, scores))

        # For the intersection logic, gather the union of all ELEN residues for this identifier
        elen_union = set()
        for model_id in args.elen_models:
            elen_union |= set(elen_scores[model_id][identifier].keys())

        # Our starting set is the intersection of GT + all methods
        common_resids = set(dict_gt.keys())
        for method in args.methods:
            common_resids &= set(method_dicts[method].keys())

        # Intersect with any ELEN residues so we only keep resids that appear in at least one ELEN column
        final_resids = common_resids & elen_union

        for resid in sorted(final_resids):
            row = {
                'identifier': identifier,
                'resid': resid,
                'GT': dict_gt.get(resid, None)
            }
            # Add method scores
            for method in args.methods:
                row[method] = method_dicts[method][resid]

            # Add each ELEN model's score
            for model_id in args.elen_models:
                val = elen_scores[model_id][identifier].get(resid, 0.0)
                row[f"ELEN_{model_id}"] = val

            rows.append(row)

    # 5) Make a DataFrame, drop any rows with missing GT or method scores, then return
    df = pd.DataFrame(rows)
    df_residues = df.dropna()  # ensure we only keep complete data
    return df_residues

def run_elen_inference(args, outpath_elen, pocket_type="RP"):
    """
    Runs the ELEN inference script for a given pocket type ('RP' or 'LP') 
    using the specified ELEN models, printing the output directly to the screen.

    Parameters
    ----------
    args : Namespace
        Should contain:
          - args.inpath_models
          - args.elen_models (list of model identifiers)
          - args.ss_frag_size
          - args.nr_residues
          - args.loop_max_size
          - (optionally more)
    outpath_elen : str
        The directory in which to run/store ELEN results.
    pocket_type : str, default="RP"
        The pocket type ("RP" or "LP", etc.)
    """
    os.makedirs(outpath_elen, exist_ok=True)

    subprocess.run([
        f"{PATH_INFERENCE}/ELEN_inference.py",
        "--inpath", args.inpath_models,
        "--outpath", outpath_elen,
        "--path_elen_models", PATH_ELEN_MODELS,
        "--ss_frag_size", str(args.ss_frag_size),
        "--nr_residues", str(args.nr_residues),
        "--loop_max_size", str(args.loop_max_size),
        "--pocket_type", pocket_type,
        "--elen_score_to_pdb", "lddt_cad",
        "--elen_models"
    ] + args.elen_models,
        check=True
    )

    print(f"[INFO] ELEN inference complete for pocket_type='{pocket_type}'. Output printed to screen.")