import os
import sys
import json
import shutil
import subprocess
import pkg_resources
import logging
import pandas as pd
from Bio.PDB import PDBParser, PDBIO
import pyrosetta
from pyrosetta.rosetta.core.scoring.hbonds import HBondSet
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from elen.shared_utils.utils_io import dump_dict_to_json
from elen.shared_utils.utils_pdb import discard_pdb, get_residue_ids

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

###############################################################################
#                               UTILITY HELPERS                               #
###############################################################################

def replace_na_with_global_mean(data):
    """Replace 'nA' entries with the global mean of all valid entries."""
    valid_entries = [float(x) for x in data if x != 'nA']
    global_mean = sum(valid_entries) / len(valid_entries) if valid_entries else 0.0
    return [global_mean if x == 'nA' else float(x) for x in data]

def replace_na_with_zero(data):
    """Replace 'nA' entries with zero."""
    return [0.0 if x == 'nA' else float(x) for x in data]

###############################################################################
#                         ATOM-LEVEL FEATURE HELPERS                          #
###############################################################################

def split_and_renumber_pdb_chains(path_pdb, outpath):
    """
    Split a PDB into separate chains, renumber residues from 1, and save each as a new PDB file.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB', path_pdb)
    for model in structure:
        for chain in model:
            io = PDBIO()
            for i, residue in enumerate(chain.get_residues(), start=1):
                if not residue.id == (' ', i, ' '):
                    residue.id = (' ', i, ' ')
            io.set_structure(chain)
            identifier = os.path.basename(path_pdb)[:4]
            output_file = f"{outpath}/{identifier}_{chain.id}.pdb"
            io.save(output_file)

def fetch_native(outpath_fetched_natives, fname_pdb_native, path_model):
    """
    Download a native structure from the RCSB PDB; if missing, move model to a special folder.
    """
    os.makedirs(outpath_fetched_natives, exist_ok=True)
    path_pdb_native = os.path.join(outpath_fetched_natives, fname_pdb_native)
    if not os.path.exists(path_pdb_native):
        download_success = subprocess.run(
            ["wget", "https://files.rcsb.org/download/" + fname_pdb_native]
        )
        if download_success.returncode == 8:
            notfound_dir = os.path.join(outpath_fetched_natives, "no_pdb_in_RSCB")
            os.makedirs(notfound_dir, exist_ok=True)
            shutil.move(path_model, notfound_dir)
        else:
            shutil.move(fname_pdb_native, outpath_fetched_natives)
    return path_pdb_native

def clean_native(path_pdb_native, fname_pdb_native):
    """
    Clean a native PDB file using Rosetta's clean_pdb.py, keeping all chains.
    """
    PATH_ROSETTA_TOOLS = "/home/florian_wieser/Rosetta_10-2022/main/tools/protein_tools/scripts/"
    subprocess.run(
        [f"{PATH_ROSETTA_TOOLS}/clean_pdb.py", path_pdb_native, "--allchains", "X"],
        stdout=subprocess.DEVNULL
    )
    os.remove(path_pdb_native)
    shutil.move(fname_pdb_native[:4] + "_X.pdb", path_pdb_native)
    os.remove(fname_pdb_native[:4] + "_X.fasta")

def update_atom_features_result_dict(fname_pdb, result_list, res_ids_list, atom_names_list, result_dict, dict_key):
    """
    Update the atom-level feature result dictionary in place.
    """
    if fname_pdb in result_dict:
        result_dict[fname_pdb].update({
            dict_key: result_list,
            "res_ids": res_ids_list,
            "atom_names": atom_names_list,
        })
    else:
        result_dict[fname_pdb] = {
            dict_key: result_list,
            "res_ids": res_ids_list,
            "atom_names": atom_names_list,
        }
    return result_dict

def get_bfactors_from_native(outpath_fetched_natives, path_pdb, result_dict):
    """
    Map B-factors from the native PDB structure to the provided model structure.
    Missing atoms are set to the global mean.
    """
    fname_pdb = os.path.basename(path_pdb)
    parser = PDBParser(QUIET=True)
    fname_pdb_native_split = fname_pdb[:6] + ".pdb"
    path_pdb_native_split = os.path.join(outpath_fetched_natives, fname_pdb_native_split)
    structure_1 = parser.get_structure("structure_1", path_pdb)
    structure_2 = parser.get_structure("structure_2", path_pdb_native_split)
    res_ids, atom_names, b_factors = [], [], []
    for residue_1 in structure_1.get_residues():
        for atom in residue_1:
            atom_name = atom.get_name()
            found = False
            for residue_2 in structure_2.get_residues():
                if residue_1.get_id() == residue_2.get_id():
                    for atom2 in residue_2:
                        if atom2.get_name() == atom_name:
                            res_ids.append(residue_1.get_id()[1])
                            atom_names.append(atom_name)
                            b_factors.append(atom2.get_bfactor())
                            found = True
                            break
            if not found:
                res_ids.append(residue_1.get_id()[1])
                atom_names.append(atom_name)
                b_factors.append('nA')
    b_factors = replace_na_with_global_mean(b_factors)
    update_atom_features_result_dict(fname_pdb, b_factors, res_ids, atom_names, result_dict, "b-factor")
    return result_dict

def calculate_bfactors(fname_pdb, outpath, path_pdb, result_dict):
    """
    Full B-factor computation pipeline: clean, split, map B-factors.
    """
    fname_pdb_native = fname_pdb[:4] + ".pdb"
    outpath_fetched_natives = os.path.join(outpath, "fetched-natives")
    path_pdb_native = os.path.join(outpath_fetched_natives, fname_pdb_native)
    clean_native(path_pdb_native, fname_pdb_native)
    split_and_renumber_pdb_chains(path_pdb_native, outpath_fetched_natives)
    result_dict = get_bfactors_from_native(outpath_fetched_natives, path_pdb, result_dict)
    return result_dict

def calculate_fake_bfactors(fname_pdb, outpath, path_pdb, result_dict):
    """
    Assign zero B-factors for all atoms (for e.g., synthetic structures).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB', path_pdb)
    res_ids, atom_names, zero_float_list = [], [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    res_ids.append(residue.get_id()[1])
                    atom_names.append(atom.get_name())
                    zero_float_list.append(0.0)
    update_atom_features_result_dict(fname_pdb, zero_float_list, res_ids, atom_names, result_dict, "b-factor")
    return result_dict

def calculate_hbonds(path_pdb, result_dict):
    """
    Count the number of hydrogen bonds per atom using PyRosetta.
    """
    parser = PDBParser(QUIET=True)
    structure_1 = parser.get_structure("structure_1", path_pdb)
    pose = pose_from_pdb(path_pdb)
    hbond_set = HBondSet(pose)
    res_ids, atom_names, hbonds = [], [], []
    for residue_1 in structure_1.get_residues():
        residue_index_biopy = residue_1.get_id()[1]
        for atom in residue_1:
            atom_name = atom.get_name()
            residue = pose.residue(residue_index_biopy)
            atom_index_pyro = residue.atom_index(atom_name)
            atom_id_pyro = pyrosetta.AtomID(atom_index_pyro, residue_index_biopy)
            nr_hbonds = len(hbond_set.atom_hbonds(atom_id_pyro))
            res_ids.append(residue_1.get_id()[1])
            atom_names.append(atom_name)
            hbonds.append(nr_hbonds)
    hbonds = replace_na_with_zero(hbonds)
    update_atom_features_result_dict(os.path.basename(path_pdb), hbonds, res_ids, atom_names, result_dict, "hbonds")
    return result_dict

def calculate_charges(path_pdb, outpath, result_dict):
    """
    Annotate per-atom partial charges using pdb2pqr and map them back.
    """
    path_pdb2pqr = os.path.join(outpath, "pdb2pqr")
    os.makedirs(path_pdb2pqr, exist_ok=True)
    path_pdb_charges = os.path.join(
        path_pdb2pqr,
        os.path.basename(path_pdb).replace('.pdb', '') + "_charges.pdb"
    )
    subprocess.run(
        ["/home/florian_wieser/miniconda3/envs/elen_test/bin/pdb2pqr", "--ff", "AMBER", path_pdb, path_pdb_charges],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    parser = PDBParser(QUIET=True)
    structure_model = parser.get_structure("structure_1", path_pdb)
    structure_charges = parser.get_structure("structure_1", path_pdb_charges)
    charges, res_ids, atom_names = [], [], []
    for residue_1 in structure_model.get_residues():
        chain_id = residue_1.get_parent().get_id()
        res_num = residue_1.get_id()[1]
        chain_resnum = f"{chain_id}_{res_num}"
        for atom in residue_1:
            atom_name = atom.get_name()
            found = False
            for residue_2 in structure_charges.get_residues():
                if residue_1.get_id() == residue_2.get_id():
                    for atom2 in residue_2:
                        if atom2.get_name() == atom_name:
                            res_ids.append(chain_resnum)
                            atom_names.append(atom_name)
                            charges.append(atom2.get_occupancy())
                            found = True
                            break
                if found:
                    break
            if not found:
                charges.append('nA')
                res_ids.append(chain_resnum)
                atom_names.append(atom_name)
    charges = replace_na_with_global_mean(charges)
    update_atom_features_result_dict(
        os.path.basename(path_pdb), charges, res_ids, atom_names, result_dict, "charges"
    )
    return result_dict

def calculate_atom_features(paths_pdbs, outpath, path_discarded):
    """
    Main entry: Calculates atom features for a list of PDB files.
    """
    result_dict = {}
    for path_pdb in paths_pdbs:
        try:
            result_dict = calculate_charges(path_pdb, outpath, result_dict)
            path_out = os.path.join(outpath, 'atom_features.json')
            dump_dict_to_json(result_dict, path_out)
        except Exception as e:
            logging.warning(f"Error processing {path_pdb}: {e}")
            discard_pdb(path_pdb, path_discarded, "atom features calculation", e)
    return result_dict

###############################################################################
#                        RESIDUE-LEVEL FEATURE HELPERS                        #
###############################################################################

def update_residue_feature_result_dict(fname_pdb, result_list, result_dict, dict_key):
    """
    Update residue-level feature results in the result dictionary.
    """
    if fname_pdb in result_dict:
        result_dict[fname_pdb].update({dict_key: result_list})
    else:
        result_dict[fname_pdb] = {dict_key: result_list}
    return result_dict

def process_metric(df, result_dict, fname_pdb, regex_pattern, dict_key):
    """
    Extract a per-residue metric from a dataframe using regex for column selection.
    """
    cols = df.filter(regex=regex_pattern)
    sorted_cols = sorted(cols.columns, key=lambda x: int(x.split('_')[-1]))
    for _, row in df.iterrows():
        value_list = row[sorted_cols].tolist()
        update_residue_feature_result_dict(fname_pdb, value_list, result_dict, dict_key)
    return result_dict

def process_string_metric(df, result_dict, metric, fname_pdb, dict_key):
    """
    Process string metrics (e.g., sequence, DSSP) into a per-residue list.
    """
    metric_string = df[metric].iloc[0]
    character_list = list(metric_string)
    update_residue_feature_result_dict(fname_pdb, character_list, result_dict, dict_key)
    return result_dict

def calculate_residue_features_pyrosetta_rs(paths_pdb, outpath, path_discarded):
    """
    Runs RosettaScripts SimpleMetrics XML via PyRosetta using XmlObjects,
    extracting residue features from the pose, and writes residue_features.json.
    """
    pyrosetta.init("-mute all", silent=True)
    path_xml = pkg_resources.resource_filename('elen.shared_utils.resources', 'metrics_for_residue_features.xml')
    result_dict = {}

    with open(path_xml, 'r') as f:
        xml_string = f.read()
    xml_obj = XmlObjects.create_from_string(xml_string)
    mover = xml_obj.get_mover("simple_metric")

    per_res_metrics = [
        ("hbonds_", "hbonds"),
        ("res_energy_", "energies"),
        ("res_sap_score_", "sap-score"),
        ("res_sasa_", "sasa")
    ]
    string_metrics = [
        ("secondary_structure", "secondary_structure"),
        ("sequence", "sequence"),
    ]

    for path_pdb in paths_pdb:
        fname_pdb = os.path.basename(path_pdb)
        try:
            pose = pose_from_pdb(path_pdb)
            mover.apply(pose)
            n_res = pose.total_residue()
            res_ids = get_residue_ids(path_pdb)
            feature_dict = {}
            for metric_key, dict_key in per_res_metrics:
                values = [pose.scores.get(f"{metric_key}{i}", 0.0) for i in range(1, n_res+1)]
                feature_dict[dict_key] = values
            for metric_key, dict_key in string_metrics:
                val = pose.scores.get(metric_key, "")
                feature_dict[dict_key] = list(val) if isinstance(val, str) else val
            feature_dict["res_id"] = res_ids
            result_dict[fname_pdb] = feature_dict
        except Exception as e:
            logging.warning(f"Error processing {fname_pdb}: {e}")
            if path_discarded:
                discard_pdb(path_pdb, path_discarded, "residue features calculation", e)
            continue

    path_out = os.path.join(outpath, 'residue_features.json')
    dump_dict_to_json(result_dict, path_out)

def calculate_residue_features(paths_pdb, outpath, path_discarded):
    """
    Runs RosettaScripts externally and extracts per-residue features from the score file.
    """
    PATH_ROSETTA_BIN = "/home/florian_wieser/Rosetta_10-2022/main/source/bin/"
    path_xml = pkg_resources.resource_filename('elen.shared_utils.resources', 'metrics_for_residue_features.xml')
    result_dict = {}
    for path_pdb in paths_pdb:
        fname_pdb = os.path.basename(path_pdb)
        try:
            subprocess.run([
                f"{PATH_ROSETTA_BIN}/rosetta_scripts.linuxgccrelease",
                "-in:file:s", path_pdb,
                "-parser:protocol", path_xml,
                "-nstruct", "1",
                "-out:path:all", outpath,
                "--overwrite"
            ], stdout=subprocess.DEVNULL, check=True)

            df = pd.read_csv(os.path.join(outpath, "score.sc"), sep='\s+', skiprows=1)
            result_dict = process_metric(df, result_dict, fname_pdb, "^hbonds_", "hbonds")
            result_dict = process_metric(df, result_dict, fname_pdb, "^res_energy_", "energies")
            result_dict = process_metric(df, result_dict, fname_pdb, "^res_sap_score", "sap-score")
            result_dict = process_metric(df, result_dict, fname_pdb, "^res_sasa_", "sasa")
            process_string_metric(df, result_dict, "secondary_structure", fname_pdb, "secondary_structure")
            process_string_metric(df, result_dict, "sequence", fname_pdb, "sequence")
            res_ids = get_residue_ids(path_pdb)
            update_residue_feature_result_dict(fname_pdb, res_ids, result_dict, "res_id")
            os.remove(os.path.join(outpath, "score.sc"))
        except Exception as e:
            logging.warning(f"Error processing {fname_pdb}: {e}")
            discard_pdb(path_pdb, path_discarded, "residue features calculation", e)
            continue

    path_out = os.path.join(outpath, 'residue_features.json')
    dump_dict_to_json(result_dict, path_out)

def add_features_to_df(dict_features):
    """
    Convert the per-structure feature dictionary to a pandas DataFrame for ML.
    """
    feature_rows = []
    for fname, features in dict_features.items():
        n_res = len(features["res_id"])
        for i in range(n_res):
            row = {
                "fname_pdb": fname,
                "res_id": features["res_id"][i],
                "hbonds": features["hbonds"][i],
                "energies": features["energies"][i],
                "sap-score": features["sap-score"][i],
                "sasa": features["sasa"][i],
                "secondary_structure": features["secondary_structure"][i],
                "sequence": features["sequence"][i],
            }
            feature_rows.append(row)
    df_features = pd.DataFrame(feature_rows)
    return df_features
