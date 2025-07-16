import os
import re
import sys
import h5py
import torch
import numpy as np
import pandas as pd
from elen.shared_utils.utils_pdb import get_residue_ids

### HELPERS ###################################################################

def get_fname_original_pdb(path_pdb, prediction_type):
    """
    Removes specific patterns from a pdb file name based on the prediction type.

    Args:
        path_pdb (str): The file path or name of the pdb file.
        prediction_type (str): The type of prediction ('LP' or 'RP').

    Returns:
        str: The modified file name after removing the specified pattern.
    """
    if prediction_type == "LP":
        pattern = re.compile(r'(_\d+_(HH|EE|HE|EH))\.pdb$')
    elif prediction_type == "RP":
        pattern = re.compile(r'(_\d+)\.pdb$')
    else:
        raise ValueError("Invalid prediction type. Expected 'LP' or 'RP'.")
    fname = re.sub(pattern, '.pdb', path_pdb)
    return fname


def transform_and_scale_features(hparams, feature_list_stacked, scale_factor):
    """
    Transforms features by scaling them according to a specified method in hparams.
    """
    if hparams.reslevel_feature_scale_type == "standardization":
        mean = torch.mean(feature_list_stacked)
        std = torch.std(feature_list_stacked)
        if std.item() == 0:
            raise ValueError("Standard deviation is zero, division by zero encountered in standardization.")
        feature_list_stacked = (feature_list_stacked - mean) / std

    elif hparams.reslevel_feature_scale_type == "normalization":
        min_val = torch.min(feature_list_stacked)
        max_val = torch.max(feature_list_stacked)
        if max_val == min_val:
            # Instead of erroring, subtract min to produce zeros.
            feature_list_stacked = feature_list_stacked - min_val
        else:
            feature_list_stacked = (feature_list_stacked - min_val) / (max_val - min_val)

    elif hparams.reslevel_feature_scale_type == "centralization":
        feature_list_stacked = feature_list_stacked - torch.mean(feature_list_stacked)

    return feature_list_stacked * scale_factor


def one_hot_encode_feature_list(feature, feature_list):
    """
    One-hot encodes a list of categorical features based on a provided mapping.
    """
    feature_maps = {
        'secondary_structure': {'H': 0, 'E': 1, 'L': 2},  # Helix, Sheet, Loop
        'sequence': {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 
                     'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16,
                     'W': 17, 'Y': 18, 'V': 19}
    }
    try:
        feature_mapping = feature_maps[feature]
        features_int = np.array([feature_mapping[f] for f in feature_list])
    except KeyError as e:
        raise ValueError(f"Invalid feature in feature_list: {e}")
    # Create a zero matrix with shape (number of features, number of unique feature categories)
    one_hot = np.zeros((len(feature_list), len(feature_mapping)))
    one_hot[np.arange(len(feature_list)), features_int] = 1

    return one_hot


def get_filtered_feature_list(self, item, feature, path_saprot_embeddings):
    """
    Filters and returns features based on the pdb path and feature type.
    """
    path_pdb = item['file_path']
    fname_pdb = os.path.basename(path_pdb)
    fname_pdb_json = get_fname_original_pdb(fname_pdb, self.prediction_type)
    # Determine the path based on feature type
    #path_LLM_data = os.path.join(self.path_features, f"{feature}.h5")

    feature_list = []
    # Load feature based on its type
    if "saprot_" in feature:
        with h5py.File(path_saprot_embeddings, 'r') as file:
            #print(f"fname_pdb_json: {fname_pdb_json}")
            if fname_pdb_json in file:
                feature_list = file[fname_pdb_json][:].tolist()
            else:
                raise KeyError(f"Key {fname_pdb_json} not found in the file.")
    else:
        feature_list = self.features_json[fname_pdb_json].get(feature, [])

    # Retrieve list of residue numbers
    if self.use_labels:
        #resnum_list = self.labels[fname_pdb]['res_id']
        resnum_list = item['labels'].get('res_id')
    else:
        resnum_list = get_residue_ids(path_pdb)

    # Filter features based on residue numbers
    feature_list_filtered = [feature_list[i - 1] for i in resnum_list if i <= len(feature_list)]
    return feature_list_filtered


def add_features_tensor_to_out0(self, feature, feature_array, out0):
    """
    Adds a feature tensor to an existing tensor (out0) after converting the feature array
    to the appropriate tensor format and applying any necessary transformations.
    """
    # Convert feature array to tensor and ensure it is on the same device as out0
    feature_tensor = torch.tensor(feature_array, device=out0.device, dtype=torch.float32)

    # Handle different feature cases
    if feature in ["secondary_structure", "sequence"]:
        feature_tensor = feature_tensor.unsqueeze(1) if feature_tensor.dim() == 1 else feature_tensor
    else:
        if feature in ["sasa", "energies", "sap-score", "hbonds"]:
            feature_tensor = feature_tensor.view(-1, 1) if feature_tensor.dim() == 1 else feature_tensor.view(feature_tensor.size(0), 1)
        
        # Ensure feature tensor is correctly shaped (squeezing unnecessary dimensions)
        if feature_tensor.dim() >= 3:
            feature_tensor = feature_tensor.squeeze(1)

        # Apply transformation specified in hparams (e.g., standardization)
        feature_tensor = transform_and_scale_features(self.hparams, feature_tensor, self.hparams.atomlevel_feature_scale_factor)

    # Concatenate the new feature tensor to the existing tensor (out0)
    out0 = torch.cat((out0, feature_tensor), dim=1)
    return out0


def filter_none_features(features):
    return [feature for feature in features if feature != "none"]


def add_atom_features_to_atom_features(self, features: torch.Tensor, item: dict[str, any],
                                          atom_features: list[str]) -> torch.Tensor:
    """
    Enhance the (learned) atom features tensor by appending further atomic features (like charges).
    """
    for feature in filter_none_features(atom_features):
        fname_pdb_json = get_fname_original_pdb(item['id'], self.prediction_type)
        if self.hparams.use_labels and self.prediction_type == "RP":
            fname_pdb_json = re.sub(r'_\d+(?=\.pdb$)', '', fname_pdb_json)
        atom_features_json = self.atomlevel_features_json[fname_pdb_json]
        feature_list = atom_features_json[feature]
        atom_name_list = atom_features_json["atom_names"]
        res_ids_list = atom_features_json["res_ids"]
        
        atoms_identities = set(zip(item['atoms']['name'], item['atoms']['residue'])) 
    
        # Filter the feature_list based on the atom names and residue numbers
        filtered_feature_list = [
            f for f, name, rid in zip(feature_list, atom_name_list, res_ids_list)
            if (name, rid) in atoms_identities
        ]
        # Ensure the number of filtered features matches the number of atoms
        assert len(filtered_feature_list) == len(item['atoms']), "Error: Atom features missing or misaligned."

        features_final = np.array(filtered_feature_list)
        feature_tensor = torch.tensor(features_final, dtype=torch.float32).to(features.device)

        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(1)
        
        # Apply standardization, normalization, or centralization based on hyperparameters
        feature_tensor = transform_and_scale_features(self.hparams, feature_tensor, self.hparams.reslevel_feature_scale_factor)
        features = torch.cat([features, feature_tensor], dim=1)
    return features


def add_residue_features_to_atom_features(self, out0, atom3d_item):
    """
    Enhances the atom-level features by adding residue-level features for each atom in the dataset.
    """
    for feature in filter_none_features(self.atomlevel_features):
        try:
            feature_list_filtered = get_filtered_feature_list(self, atom3d_item, feature, self.path_saprot_embeddings)
        except KeyError as e:
            print(f"Warning: {e}. Skipping feature '{feature}' for file {atom3d_item['file_path']}")
            continue

        # One-hot encode specific features
        if feature in ["secondary_structure", "sequence"]:
            feature_list_filtered = one_hot_encode_feature_list(feature, feature_list_filtered)

        # Convert the filtered feature list to a DataFrame and reset index
        df_feature_list = pd.DataFrame(np.array(feature_list_filtered)).reset_index()

        # Retrieve or generate residue numbers
        if self.use_labels: 
            #resnum_list = self.labels[atom3d_item['id']]['res_id']
            resnum_list = atom3d_item['labels'].get('res_id')
        else:
            resnum_list = get_residue_ids(atom3d_item['file_path'])
        
        # Combine residue numbers with the feature list
        df_res_ids = pd.DataFrame({'res_id': resnum_list})
        df_feature_list = pd.concat([df_res_ids, df_feature_list], axis=1)

        # Merge with the residue list from atoms and select relevant columns
        df_res_list = pd.DataFrame(atom3d_item['atoms']['residue'].values, columns=['res_id'])
        merged = df_feature_list.merge(df_res_list, on='res_id')
        features_final = merged.iloc[:, 2:].to_numpy()
       
        # Add the newly created feature matrix to the initial tensor
        out0 = add_features_tensor_to_out0(self, feature, features_final, out0)

    return out0


def get_residue_feature_tensor(self, atom3d_item: dict) -> np.array:
    """
    Stack residue-level features for each residue in a 3D atom structure into a single tensor.
    """
    residues = atom3d_item['atoms']['residue'].to_list()
    nr_residues = len(set(residues))
    residue_features = []

    for feature in filter_none_features(self.hparams.reslevel_features):
        try:
            feature_data = get_filtered_feature_list(self, atom3d_item, feature, self.path_saprot_embeddings)
            if feature in ["secondary_structure", "sequence"]:
                feature_data = one_hot_encode_feature_list(feature, feature_data)
            
            feature_array = np.array(feature_data, dtype=float).reshape(nr_residues, -1)
            residue_features.append(feature_array)
        
        except Exception as e:
            print(f"Error processing feature '{feature}': {e}")
            continue
    
    if residue_features:
        return np.concatenate(residue_features, axis=1)
    else:
        return np.array([])

"""
import torch_geometric as tg
from elen.training.utils_data import add_atom_features_to_atom_features
from elen.training.utils_data import add_residue_features_to_atom_features
from elen.training.utils_data import get_residue_feature_tensor
from elen.shared_utils.utils_pdb import get_residue_ids
from elen.shared_utils.utils_io import load_from_json


class EDN_Transform:
    def __init__(self, scaled_labels, hparams, path_features, use_labels, prediction_type, **kwargs):
        self.scaled_labels = scaled_labels + [scaled_labels[0]] * (3 - len(scaled_labels)) if len(scaled_labels) < 3 else scaled_labels
        self.hparams = hparams
        self.path_features = path_features
        self.num_nearest_neighbors = hparams.num_nearest_neighbors
        self.hr_atomlevel_features = hparams.hr_atomlevel_features
        self.atomlevel_features = hparams.atomlevel_features
        self.use_labels = use_labels
        self.prediction_type = prediction_type
        self.atomlevel_features_json = load_from_json(f"{path_features}/atom_features.json")
        self.features_json = load_from_json(f"{path_features}/residue_features.json")
        if self.hparams.test_dir and use_labels:
            self.labels = load_from_json(f"{path_features}/labels.json")

    def __repr__(self):
        return f"EDN_Transform(num_nearest_neighbors={self.num_nearest_neighbors})"
    
    def __call__(self, item):
        try:
            # Get labels based on whether we have test_dir and use_labels
            labels = (
                [torch.zeros(len(get_residue_ids(item['file_path']))) for _ in range(3)]
                if not self.hparams.test_dir or not self.use_labels
                else [self.scaled_labels[i][item['id']].clone() for i in range(3)]
            )
            
            elements = item['atoms']['element'].to_numpy()
            element_mapping = {'C': 0, 'O': 1, 'N': 2, 'S': 3, 'SE': 3}
            sel = np.isin(elements, list(element_mapping.keys()))
            elements_int = np.array([element_mapping[e] for e in elements[sel]])
            one_hot = np.eye(len(set(element_mapping.values())))[elements_int]
            
            coords = item['atoms'][['x', 'y', 'z']].to_numpy()[sel]
            c_alpha_flags = (item['atoms']['name'] == 'CA').to_numpy()[sel]
            geometry = torch.tensor(coords, dtype=torch.float32)
            atom_features = torch.tensor(one_hot, dtype=torch.float32)
            
            atom_features = add_atom_features_to_atom_features(self, atom_features, item, self.hr_atomlevel_features)
            atom_features = add_residue_features_to_atom_features(self, atom_features, item)
            residue_features = get_residue_feature_tensor(self, item)
            
            pdist = torch.cdist(geometry, geometry)
            tmp = torch.topk(-pdist, min(self.num_nearest_neighbors, pdist.shape[0]), dim=1)

            nei_list, geo_list = [], []
            for source, x in enumerate(tmp.indices):
                cart = geometry[x]
                nei_list.append(torch.tensor([[source, dest] for dest in x], dtype=torch.long))
                geo_list.append(cart - geometry[source])
            nei_list = torch.cat(nei_list, dim=0).transpose(1, 0)
            geo_list = torch.cat(geo_list, dim=0)

            data = tg.data.Data(
                x=atom_features,
                edge_index=nei_list,
                edge_attr=geo_list,
                pos=geometry,
                Rs_in=[(len(element_mapping), 0)],
                label_1=labels[0],
                label_2=labels[1],
                label_3=labels[2],
                id=item['id'],
                file_path=item['file_path'],
                select_ca=torch.tensor(c_alpha_flags),
                reslevel_features=residue_features
            )
            return data
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}. Skipping this data point.")
            return None
"""