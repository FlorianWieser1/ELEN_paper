import os
import sys
import json
import shutil
import h5py
import subprocess
import pkg_resources
import pandas as pd
from Bio.PDB import PDBParser
from Bio import PDB
import pyrosetta
from pyrosetta.rosetta.core.scoring.hbonds import HBondSet
from elen.shared_utils.utils_io import dump_dict_to_json
from elen.shared_utils.utils_others import discard_pdb

pyrosetta.init()

# Rest of your code
def update_atom_features_result_dict(fname_pdb, result_list, res_ids_list, atom_names_list, result_dict, dict_key):
    if fname_pdb in result_dict:
        result_dict[fname_pdb].update({dict_key: result_list,
                                  "res_ids": res_ids_list,
                                  "atom_names": atom_names_list})
    else:
        result_dict[fname_pdb] = {dict_key: result_list,
                                  "res_ids": res_ids_list,
                                  "atom_names": atom_names_list}
    return result_dict

def replace_na_with_global_mean(data):
    valid_entries = [float(x) for x in data if x != 'nA']
    if valid_entries:
        global_mean = sum(valid_entries) / len(valid_entries)
    else:
        global_mean = 0  # Default mean value if there are no valid entries
    return [global_mean if x == 'nA' else float(x) for x in data]

def replace_na_with_zero(data):
    # Replace 'nA' with 0 in the list
    return [0 if x == 'nA' else float(x) for x in data]


### HELPERS b-factor ##############################################################################
def split_and_renumber_pdb_chains(path_pdb, outpath):
    parser = PDB.PDBParser()
    structure = parser.get_structure('PDB', path_pdb)
    
    for model in structure:
        for chain in model:
            io = PDB.PDBIO()
            # Renumber residues starting from 1
            for i, residue in enumerate(chain.get_residues(), start=1):
                if not residue.id == (' ', i, ' '):
                    residue.id = (' ', i, ' ')
                    
            # Set the structure for output to contain only the current chain
            io.set_structure(chain)
            # Define output file path
            identifier = os.path.basename(path_pdb)[:4]
            output_file = f"{outpath}/{identifier}_{chain.id}.pdb"
            # Save the modified chain
            io.save(output_file)


def fetch_native(outpath_fetched_natives, fname_pdb_native, path_model):
    os.makedirs(outpath_fetched_natives, exist_ok=True)
    path_pdb_native = os.path.join(outpath_fetched_natives, fname_pdb_native) 
    
    if not os.path.exists(path_pdb_native):
        download_success = subprocess.run(["wget", "https://files.rcsb.org/download/" + fname_pdb_native])
        if download_success.returncode == 8:
            os.makedirs(os.path.join(outpath_fetched_natives, "no_pdb_in_RSCB"), exist_ok=True) 
            shutil.move(path_model, os.path.join(outpath_fetched_natives, "no_pdb_in_RSCB"))
        else:
            shutil.move(fname_pdb_native, outpath_fetched_natives)
    return path_pdb_native



def clean_native(path_pdb_native, fname_pdb_native):
    PATH_ROSETTA_TOOLS = "/home/florian_wieser/Rosetta_10-2022/main/tools/protein_tools/scripts/"
    subprocess.run([f"{PATH_ROSETTA_TOOLS}/clean_pdb.py", path_pdb_native, "--allchains", "X"], stdout=subprocess.DEVNULL)
    os.remove(path_pdb_native)
    shutil.move(fname_pdb_native[:4] + "_X.pdb", path_pdb_native)
    os.remove(fname_pdb_native[:4] + "_X.fasta")


def get_bfactors_from_native(outpath_fetched_natives, path_pdb, result_dict):
    # Load the PDB data into structures
    fname_pdb = os.path.basename(path_pdb)
    parser = PDBParser(QUIET=True)
    fname_pdb_native_split = fname_pdb[:6] + ".pdb"
    path_pdb_native_split = os.path.join(outpath_fetched_natives, fname_pdb_native_split) 
    
    structure_1 = parser.get_structure("structure_1", path_pdb)
    structure_2 = parser.get_structure("structure_2", path_pdb_native_split)
    
     # Loop over atoms in the first structure
    res_ids = []
    atom_names = []
    b_factors = []
    for residue_1 in structure_1.get_residues():
        for atom in residue_1:
            atom_name = atom.get_name()
            # Check if this atom exists in the second structure and store its b-factor
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
    fname_pdb_native = fname_pdb[:4] + ".pdb"
    outpath_fetched_natives = os.path.join(outpath, "fetched-natives")
    path_pdb_native = os.path.join(outpath_fetched_natives, fname_pdb_native) 
    clean_native(path_pdb_native, fname_pdb_native)
    # split .pdb into single chains and renumber residues to begin with 1
    split_and_renumber_pdb_chains(path_pdb_native, outpath_fetched_natives)
    # obtain b-factors from native .pdb, nA if no matching atom was found 
    result_dict = get_bfactors_from_native(outpath_fetched_natives, path_pdb, result_dict)
    return result_dict   


def calculate_fake_bfactors(fname_pdb, outpath, path_pdb, result_dict):
    parser = PDBParser()
    structure = parser.get_structure('PDB', path_pdb)
    res_ids = []
    atom_names = []
    zero_float_list = []
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
    # Load the PDB data into structures
    parser = PDBParser(QUIET=True)
    structure_1 = parser.get_structure("structure_1", path_pdb)
    pose = pyrosetta.pose_from_pdb(path_pdb)
    hbond_set = HBondSet(pose)#, calculate_derivative=False)
  
     # Loop over atoms in the first structure
    res_ids = []
    atom_names = []
    hbonds = []
    for residue_1 in structure_1.get_residues():
        residue_index_biopy = residue_1.get_id()[1]
        for atom in residue_1:
            atom_name = atom.get_name()
            residue = pose.residue(residue_index_biopy)
            atom_index_pyro = residue.atom_index(atom_name)  # PyRosetta provides this method
            atom_id_pyro = pyrosetta.AtomID(atom_index_pyro, residue_index_biopy)
            nr_hbonds = len(hbond_set.atom_hbonds(atom_id_pyro))
            res_ids.append(residue_1.get_id()[1])                
            atom_names.append(atom_name)
            hbonds.append(nr_hbonds)
    hbonds = replace_na_with_zero(hbonds)
    update_atom_features_result_dict(os.path.basename(path_pdb), hbonds, res_ids, atom_names, result_dict, "hbonds")
    return result_dict

import os
import subprocess
from Bio.PDB import PDBParser
def calculate_charges(path_pdb, outpath, result_dict):
    path_pdb2pqr = os.path.join(outpath, "pdb2pqr")
    os.makedirs(path_pdb2pqr, exist_ok=True)
    
    # run pdb2pqr to produce .pdb with charges
    path_pdb_charges = os.path.join(
        path_pdb2pqr, os.path.basename(path_pdb).replace('.pdb', '') + "_charges.pdb"
    )
    subprocess.run(
        ["/home/florian_wieser/miniconda3/envs/elen_test/bin/pdb2pqr", "--ff", "AMBER", path_pdb, path_pdb_charges],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # match resulting charges onto model.pdb
    parser = PDBParser(QUIET=True)
    structure_model = parser.get_structure("structure_1", path_pdb)
    structure_charges = parser.get_structure("structure_1", path_pdb_charges)
    charges = []
    res_ids = []
    atom_names = []
    
    for residue_1 in structure_model.get_residues():
        # Get the chain id and residue number in chain_resnum format
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

def falculate_charges(path_pdb, outpath, result_dict):
    path_pdb2pqr = os.path.join(outpath, "pdb2pqr")
    os.makedirs(path_pdb2pqr, exist_ok=True)
    
    # run pdb2pqr to produce .pdb with charges
    path_pdb_charges = os.path.join(path_pdb2pqr, os.path.basename(path_pdb).replace('.pdb', '') + "_charges.pdb")
    subprocess.run(["/home/florian_wieser/miniconda3/envs/elen_test/bin/pdb2pqr", "--ff", "AMBER", path_pdb, path_pdb_charges], 
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # match resulting charges onto model.pdb
    parser = PDBParser(QUIET=True)
    structure_model = parser.get_structure("structure_1", path_pdb)
    structure_charges = parser.get_structure("structure_1", path_pdb_charges)
    charges = []
    res_ids = []
    atom_names = []
    for residue_1 in structure_model.get_residues():
        for atom in residue_1:
            atom_name = atom.get_name()
            # Check if this atom exists in the second structure and store its b-factor
            found = False
            for residue_2 in structure_charges.get_residues():
                if residue_1.get_id() == residue_2.get_id():
                    for atom2 in residue_2:
                        if atom2.get_name() == atom_name:
                            res_ids.append(residue_1.get_id()[1])                
                            atom_names.append(atom_name)
                            charges.append(atom2.get_occupancy())
                            found = True
                            break
            if not found:
                charges.append('nA')
                res_ids.append(residue_1.get_id()[1])                
                atom_names.append(atom_name)

    charges = replace_na_with_global_mean(charges)
    update_atom_features_result_dict(os.path.basename(path_pdb), charges, res_ids, atom_names, result_dict, "charges")
    return result_dict

def calculate_atom_features(paths_pdbs, outpath, path_discarded):
    result_dict = {}
    for path_pdb in paths_pdbs:
        try:
            result_dict = calculate_charges(path_pdb, outpath, result_dict)
            path_out = os.path.join(outpath, 'atom_features.json')
            dump_dict_to_json(result_dict, path_out)
        except Exception as e:
            print(f"Error processing {path_pdb}: {e}")
            discard_pdb(path_pdb, path_discarded, "atom features calculation", e)
    return result_dict
   
### RESIDUE FEATURES ########################################################## 
def update_residue_feature_result_dict(fname_pdb, result_list, result_dict, dict_key):
    if fname_pdb in result_dict:
        result_dict[fname_pdb].update({dict_key: result_list})
    else:
        result_dict[fname_pdb] = {dict_key: result_list}
    return result_dict

def process_metric(df, result_dict, fname_pdb, regex_pattern, dict_key):
    cols_hbonds = df.filter(regex=regex_pattern)
    sorted_hbond_cols = sorted(cols_hbonds.columns, key=lambda x: int(x.split('_')[-1]))
    df = df[sorted_hbond_cols]
    for _, row in df.iterrows():
        value_list = row[sorted_hbond_cols].tolist()
        update_residue_feature_result_dict(fname_pdb, value_list, result_dict, dict_key)
    return result_dict

def process_string_metric(df, result_dict, metric, fname_pdb, dict_key):
    metric_string = df[metric].iloc[0]
    characters = [char for char in metric_string]
    character_list = list(characters)
    update_residue_feature_result_dict(fname_pdb, character_list, result_dict, dict_key)
    return result_dict

def calculate_atom_features(paths_pdbs, outpath, path_discarded):
    result_dict = {}
    for path_pdb in paths_pdbs:
        try:
            result_dict = calculate_charges(path_pdb, outpath, result_dict)
            path_out = os.path.join(outpath, 'atom_features.json')
            dump_dict_to_json(result_dict, path_out)
        except Exception as e:
            print(f"Error processing {path_pdb}: {e}")
            discard_pdb(path_pdb, path_discarded, "atom features calculation", e)
            continue
    return result_dict


from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects

def calculate_residue_features_pyrosetta_rs(paths_pdb, outpath, path_discarded):
    """
    Runs RosettaScripts SimpleMetrics XML via PyRosetta using XmlObjects,
    extracting residue features from the pose, and writes residue_features.json.
    """
    # Initialization
    init("-mute all")
    path_xml = pkg_resources.resource_filename('elen.data_preparation', 'metrics_for_residue_features.xml')
    result_dict = {}

    # Read XML and prepare the protocol
    with open(path_xml, 'r') as f:
        xml_string = f.read()
    xml_obj = XmlObjects.create_from_string(xml_string)
    mover = xml_obj.get_mover("simple_metric")

    # Define which per-residue metrics to extract
    per_res_metrics = [
        ("hbonds_", "hbonds"),
        ("res_energy_", "energies"),
        ("res_sap_score_", "sap-score"),
        ("res_sasa_", "sasa")
    ]
    # And which string metrics
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

            # Collect per-residue metrics
            for metric_key, dict_key in per_res_metrics:
                values = []
                for i in range(1, n_res+1):
                    val = pose.scores.get(f"{metric_key}{i}", 0.0)
                    values.append(val)
                feature_dict[dict_key] = values

            # Collect string metrics
            for metric_key, dict_key in string_metrics:
                # Extract, split into list of characters
                val = pose.scores.get(metric_key, "")
                feature_dict[dict_key] = list(val) if isinstance(val, str) else val

            # Add residue ids
            feature_dict["res_id"] = res_ids

            # Store in result_dict
            result_dict[fname_pdb] = feature_dict

        except Exception as e:
            print(f"Error processing {fname_pdb}: {e}")
            if path_discarded:
                discard_pdb(path_pdb, path_discarded, "residue features calculation", e)
            continue  # Skip to the next pdb file if there's an error

    # Write the JSON as in your pipeline
    path_out = os.path.join(outpath, 'residue_features.json')
    dump_dict_to_json(result_dict, path_out)


def calculate_residue_features(paths_pdb, outpath, path_discarded):
    PATH_ROSETTA_BIN = "/home/florian_wieser/Rosetta_10-2022/main/source/bin/"
    path_xml = pkg_resources.resource_filename('elen.data_preparation', 'metrics_for_residue_features.xml')

    result_dict = {}
    for path_pdb in paths_pdb:
        fname_pdb = os.path.basename(path_pdb)
        try:
            subprocess.run([f"{PATH_ROSETTA_BIN}/rosetta_scripts.linuxgccrelease",
                            "-in:file:s", path_pdb,
                            "-parser:protocol", path_xml,
                            "-nstruct", "1",
                            "-out:path:all", outpath,
                            "--overwrite"],
                           stdout=subprocess.DEVNULL, check=True)

            df = pd.read_csv(os.path.join(outpath, "score.sc"), sep='\s+', skiprows=1)
            # Process per residue float metrics
            result_dict = process_metric(df, result_dict, fname_pdb, "^hbonds_", "hbonds")
            result_dict = process_metric(df, result_dict, fname_pdb, "^res_energy_", "energies")
            result_dict = process_metric(df, result_dict, fname_pdb, "^res_sap_score", "sap-score")
            result_dict = process_metric(df, result_dict, fname_pdb, "^res_sasa_", "sasa")
            # Process per residue string metrics (DSSP, residue identity)
            process_string_metric(df, result_dict, "secondary_structure", fname_pdb, "secondary_structure")
            process_string_metric(df, result_dict, "sequence", fname_pdb, "sequence")
            # Add residue ids 
            res_ids = get_residue_ids(path_pdb)
            update_residue_feature_result_dict(fname_pdb, res_ids, result_dict, "res_id")

            os.remove(os.path.join(outpath, "score.sc"))
        except Exception as e:
            print(f"Error processing {fname_pdb}: {e}")
            discard_pdb(path_pdb, path_discarded, "residue features calculation", e)
            continue  # Skip to the next pdb file if there's an error

    path_out = os.path.join(outpath, 'residue_features.json')
    dump_dict_to_json(result_dict, path_out)
    
    
def get_residue_ids(path_pdb: str) -> list:
    """
    Parse a PDB file to extract a list of residue identifiers.
    
    Parameters:
        path_pdb (str): Path to the PDB file.
    
    Returns:
        list: A list of strings representing the chain and residue numbers in the PDB structure 
              in the format "Chain_ResidueNumber", e.g., "A_1".
    """
    parser = PDBParser(QUIET=True)  # Quiet mode to suppress warnings
    resnum_list = []
    
    try:
        structure = parser.get_structure("structure", path_pdb)
        for model in structure:
            for chain in model:
                chain_id = chain.id  # Get the chain identifier
                for residue in chain:
                    # Process only standard residues (skip heteroatoms)
                    if residue.id[0] == ' ':
                        res_num = residue.id[1]
                        resnum_list.append(f"{chain_id}_{res_num}")
    except FileNotFoundError:
        print(f"Error: The file {path_pdb} does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred while parsing the PDB file: {e}")
    return resnum_list

def add_features_to_df(dict_features):
    feature_rows = []
    for fname, features in dict_features.items():
        n_res = len(features["res_id"])
        for i in range(n_res):
            row = {
                "fname_pdb": fname,
                "res_id": features["res_id"][i],  # e.g., 'A_1'
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