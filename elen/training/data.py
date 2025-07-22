import sys
import torch
import numpy as np
from typing import Optional, Any
from torch_geometric.data import Data
from elen.training.utils_data import (
    add_atom_features_to_atom_features,
    add_residue_features_to_atom_features,
    get_residue_feature_tensor,
)
from elen.shared_utils.utils_pdb import get_residue_ids
from elen.shared_utils.utils_io import load_from_json
import logging

logger = logging.getLogger(__name__)

class EDN_Transform:
    def __init__(
        self, 
        hparams: Any, 
        path_features: str, 
        path_saprot_embeddings: str, 
        use_labels: bool, 
        prediction_type: str, 
        feature_mode: str,
        **kwargs
    ):
        """
        Args:
            hparams: Hyperparameters object/dict with at least:
                - num_nearest_neighbors, hr_atomlevel_features, atomlevel_features
            path_features: Directory for features (JSONs, etc.)
            path_saprot_embeddings: Directory for SAPROT embeddings
            use_labels: Whether to use/return labels in output
            prediction_type: 'residue' or 'atom', etc.
            feature_mode: Which feature set to load
        """
        self.hparams = hparams
        self.path_features = path_features
        self.path_saprot_embeddings = path_saprot_embeddings
        self.num_nearest_neighbors = hparams.num_nearest_neighbors
        self.hr_atomlevel_features = hparams.hr_atomlevel_features
        self.atomlevel_features = hparams.atomlevel_features
        self.use_labels = use_labels
        self.prediction_type = prediction_type

        # Load precomputed features
        self.features_json = None
        if feature_mode in ("full", "no_saprot"):
            self.features_json = load_from_json(f"{path_features}/residue_features.json")

    def __repr__(self):
        return f"EDN_Transform(num_nearest_neighbors={self.num_nearest_neighbors})"
    
    def __call__(self, item: dict) -> Optional[Data]:
        """
        Args:
            item: dict, typically a sample from LMDB or preprocessing pipeline.
        Returns:
            torch_geometric.data.Data or None if sample is skipped.
        """
        # Handle missing labels
        if self.use_labels and 'labels' in item and item['labels']:
            label_rmsd = torch.tensor(item['labels'].get('rmsd', []), dtype=torch.float32)
            label_lddt = torch.tensor(item['labels'].get('lddt', []), dtype=torch.float32)
            label_CAD  = torch.tensor(item['labels'].get('CAD', []),  dtype=torch.float32)
        else:
            n_res = len(get_residue_ids(item['file_path']))
            label_rmsd = torch.zeros(n_res, dtype=torch.float32)
            label_lddt = torch.zeros(n_res, dtype=torch.float32)
            label_CAD  = torch.zeros(n_res, dtype=torch.float32)
        labels = [label_rmsd, label_lddt, label_CAD]

        # Check DataFrame structure
        atoms_df = item['atoms']
        required_atom_cols = {'element', 'name', 'x', 'y', 'z'}
        if not required_atom_cols.issubset(atoms_df.columns):
            logger.error(f"Missing columns in atoms DataFrame for item {item.get('id', 'unknown')}. Skipping.")
            return None

        # Only keep atoms for which we have mapping
        elements = atoms_df['element'].to_numpy()
        element_mapping = {'C': 0, 'O': 1, 'N': 2, 'S': 3, 'SE': 3}
        sel = np.isin(elements, list(element_mapping.keys()))
        if not np.any(sel):
            logger.warning(f"No recognizable atoms in {item.get('id', 'unknown')}. Skipping.")
            return None
        try:
            elements_int = np.array([element_mapping[e] for e in elements[sel]])
        except KeyError as e:
            logger.warning(f"Unknown atom element {e} in {item.get('id', 'unknown')}. Skipping.")
            return None
        one_hot = np.eye(len(set(element_mapping.values())))[elements_int]

        coords = atoms_df[['x', 'y', 'z']].to_numpy()[sel]
        c_alpha_flags = (atoms_df['name'] == 'CA').to_numpy()[sel]

        geometry = torch.tensor(coords, dtype=torch.float32)
        atom_features = torch.tensor(one_hot, dtype=torch.float32)
        atom_features = add_atom_features_to_atom_features(self, atom_features, item, self.hr_atomlevel_features)

        #try:
        atom_features = add_residue_features_to_atom_features(self, atom_features, item)
        #except Exception as e:
        #    logger.warning(f"Skipping {item.get('id', 'unknown')} due to error in residue feature addition: {e}")
        #    return None
        
        residue_features = get_residue_feature_tensor(self, item)
        
        # Compute edges (nearest neighbors; exclude self if desired)
        pdist = torch.cdist(geometry, geometry)
        num_neighbors = min(self.num_nearest_neighbors, pdist.shape[0])
        # Optional: mask diagonal if you don't want self as neighbor
        values, indices = torch.topk(-pdist, num_neighbors, dim=1)
        nei_list = []
        geo_list = []
        for source, neigh_idx in enumerate(indices):
            for dest in neigh_idx.tolist():
                nei_list.append([source, dest])
                geo_list.append((geometry[dest] - geometry[source]).tolist())
        edge_index = torch.tensor(nei_list, dtype=torch.long).transpose(1, 0)
        edge_attr = torch.tensor(geo_list, dtype=torch.float32)

        # Build Data object
        data = Data(
            x=atom_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=geometry,
            Rs_in=[(len(element_mapping), 0)],
            label_1=labels[0],
            label_2=labels[1],
            label_3=labels[2],
            id=item.get('id', None),
            file_path=item.get('file_path', None),
            select_ca=torch.tensor(c_alpha_flags),
            reslevel_features=residue_features
        )
        return data