import sys
import torch
import numpy as np
import torch_geometric as tg
from elen.training.utils_data import add_atom_features_to_atom_features
from elen.training.utils_data import add_residue_features_to_atom_features
from elen.training.utils_data import get_residue_feature_tensor
from elen.shared_utils.utils_pdb import get_residue_ids
from elen.shared_utils.utils_io import load_from_json


class EDN_Transform:
    def __init__(self, hparams, path_features, path_saprot_embeddings, use_labels, prediction_type, **kwargs):
        """
        :param hparams: Hyperparameters object (or dict) with fields like:
                        - num_nearest_neighbors
                        - hr_atomlevel_features
                        - atomlevel_features
                        - test_dir (optional, if used for ignoring labels)
        :param path_features: Directory containing any additional features needed
        :param use_labels: Whether to attach label tensors in the output
        :param prediction_type: Typically 'residue' or 'atom' or similar, if needed
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
        #self.atomlevel_features_json = load_from_json(f"{path_features}/atom_features.json")
        self.features_json = load_from_json(f"{path_features}/residue_features.json")

    def __repr__(self):
        return f"EDN_Transform(num_nearest_neighbors={self.num_nearest_neighbors})"
    
    def __call__(self, item):
        """
        item is typically a dictionary from an LMDB sample:
          {
            'id': <unique ID>,
            'file_path': <original file path or structure name>,
            'atoms': pandas.DataFrame-like structure with at least
                     columns ['element', 'name', 'x', 'y', 'z', ...],
            'labels': {
                'rmsd': [ ... ],
                'lddt': [ ... ],
                'CAD':  [ ... ],
                ...
            }
          }

        Returns a torch_geometric.data.Data object. 
        """

        # Number of residues in this structure (determined by file_path or other methods)
                
        #print(f"item['labels']: {item['labels'].get('res_id')}") 
        # If using labels and they exist in the LMDB item, pull them out
        # Otherwise, default to zero tensors for each label
        if self.use_labels and 'labels' in item and item['labels']:
            # We expect each label array to match the number of residues, but we'll
            # handle cases where the key might be missing by defaulting to zeros.
            label_rmsd = torch.tensor(item['labels'].get('rmsd'), dtype=torch.float32)
            label_lddt = torch.tensor(item['labels'].get('lddt'), dtype=torch.float32)
            label_CAD  = torch.tensor(item['labels'].get('CAD'), dtype=torch.float32)
        else:
            n_res = len(get_residue_ids(item['file_path']))
            label_rmsd = torch.zeros(n_res, dtype=torch.float32)
            label_lddt = torch.zeros(n_res, dtype=torch.float32)
            label_CAD  = torch.zeros(n_res, dtype=torch.float32)
        
        # Convert them into a list if needed for consistent indexing
        labels = [label_rmsd, label_lddt, label_CAD]

        # 1) Filter atoms to the ones we know how to handle (C, O, N, S/SE)
        elements = item['atoms']['element'].to_numpy()
        element_mapping = {'C': 0, 'O': 1, 'N': 2, 'S': 3, 'SE': 3}
        sel = np.isin(elements, list(element_mapping.keys()))
        elements_int = np.array([element_mapping[e] for e in elements[sel]])
        one_hot = np.eye(len(set(element_mapping.values())))[elements_int]

        # 2) Gather coordinates and identify alpha carbons
        coords = item['atoms'][['x', 'y', 'z']].to_numpy()[sel]
        c_alpha_flags = (item['atoms']['name'] == 'CA').to_numpy()[sel]

        geometry = torch.tensor(coords, dtype=torch.float32)
        atom_features = torch.tensor(one_hot, dtype=torch.float32)

        # Add optional atom-level features 
        atom_features = add_atom_features_to_atom_features(self, atom_features, item, self.hr_atomlevel_features)

        # Add optional residue-level features (broadcast to each atom in that residue)
        # If there's a mismatch or an error in the residue mapping, we skip this sample.
        try:
            atom_features = add_residue_features_to_atom_features(self, atom_features, item)
        except AssertionError as e:
            print(f"Skipping {item['id']} due to error in residue feature addition: {e}")
            return None

        # Optionally store a direct residue-level feature tensor
        residue_features = get_residue_feature_tensor(self, item)

        # 3) Compute edges based on nearest neighbors
        pdist = torch.cdist(geometry, geometry)
        # We take topK of the negative distance => "closest" neighbors
        tmp = torch.topk(-pdist, min(self.num_nearest_neighbors, pdist.shape[0]), dim=1)

        nei_list, geo_list = [], []
        for source, x in enumerate(tmp.indices):
            cart = geometry[x]
            nei_list.append(torch.tensor([[source, dest] for dest in x], dtype=torch.long))
            geo_list.append(cart - geometry[source])
        nei_list = torch.cat(nei_list, dim=0).transpose(1, 0)
        geo_list = torch.cat(geo_list, dim=0)

        # 4) Build torch_geometric data object
        data = tg.data.Data(
            x=atom_features,            # Node features
            edge_index=nei_list,        # Edge indices
            edge_attr=geo_list,         # Edge attributes (vector from source to target)
            pos=geometry,               # 3D coordinates
            Rs_in=[(len(element_mapping), 0)],  # So(3) equiv. representation if using e3nn or similar
            label_1=labels[0],         # e.g. RMSD
            label_2=labels[1],         # e.g. lDDT
            label_3=labels[2],         # e.g. CAD
            id=item['id'], 
            file_path=item['file_path'],
            select_ca=torch.tensor(c_alpha_flags),
            reslevel_features=residue_features
        )
        return data
