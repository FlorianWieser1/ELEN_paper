import re
import torch
from Bio import PDB
from Bio.PDB import PDBParser
import atom3d.protein.sequence as seq
import numpy as np 
import pandas as pd 
import sys
import esm
import time
import wandb
import yaml
import os
from sklearn.preprocessing import MinMaxScaler
sys.path.append("/home/florian_wieser/software/ARES/geometricDL/edn/shared_utils/")
sys.path.append("/home/florian_wieser/software/ARES/geometricDL/edn/edn_multi_labels_pr/")
from elen.shared_utils.utils_trafo_labels import scale_local_labels_back
from elen.shared_utils.utils_plot import calculate_regression_metrics
from elen.shared_utils.utils_plot import plot_target_corr
#474

### TIMER HELPERS ##############################################################
def runtime(prefix="", start_time=None):
    if start_time is not None:
        runtime.start_time = start_time
    elif not hasattr(runtime, 'start_time'):
        runtime.start_time = time.time()
    exe_time = time.time() - runtime.start_time
    days, remainder = divmod(exe_time, 24 * 3600)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"== Runtime '{prefix}': {int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s")

def func_timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        runtime(f"{func.__name__} function", start_time)
        return result
    return wrapper


### LM HELPER FUNCTIONS ###################################################################################
def scale_features(input_ten, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    ten_scaled = torch.tensor(scaler.fit_transform(input_ten), dtype=torch.float32)
    return ten_scaled

# Helper function to extract a number after 't' from a string
def extract_number_after_t(input_string):
    pattern = r't(\d+)'  # Define a regular expression pattern to match 't' followed by one or more digits
    match = re.search(pattern, input_string)
    if match:
        number_after_t = int(match.group(1))
        return number_after_t
    else:
        return None  # Return None if no match is found

# Helper function to extract the amino acid sequence from a PDB file


three_letter_to_one_letter = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                              'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                              'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                              'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
    
def get_resid_and_sequence(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    res_ids = []
    sequences = []
    # Iterate through the structure and extract residue information
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() not in {"HOH", "WAT"}:  # Exclude water molecules
                    res_ids.append(residue.get_id()[1])
                    three_letter_code = residue.get_resname()
                    one_letter_code = three_letter_to_one_letter.get(three_letter_code, 'X')  # Default to 'X' if not found
                    sequences.append(one_letter_code)
    df = pd.DataFrame({"res_id": res_ids, "sequence": sequences})
    return df

def int_to_esm2_string(LM_MB):
     int_to_esm2_string_dict = {0:      "None",
                                8:      "esm2_t6_8M_UR50D", 
                                35:     "esm2_t12_35M_UR50D",
                                150:    "esm2_t30_150M_UR50D",
                                650:    "esm2_t33_650M_UR50D",
                                3000:   "esm2_t36_3B_UR50D",
                                15000:  "esm2_t48_15B_UR50D"}
     return int_to_esm2_string_dict[LM_MB]

model_dict = {
    "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
    "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
    "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
    "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D
}


def get_ESM_sequence_embedding_atom_level(pt_item, LM):
    nr_repr_layers = extract_number_after_t(LM)
    
    # get residue sequence
    sequence_info = seq.get_chain_sequences(pt_item['atoms'])
    sequence = sequence_info[0][1]
    
    # get NLP residue representations from ESM-2 
    model, alphabet = model_dict[LM]()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    _, _, batch_tokens = batch_converter([(pt_item['id'], sequence)]) # start/stop tokens always 0/2!?
    with torch.no_grad():
        results = model(batch_tokens, repr_layers = [nr_repr_layers], return_contacts=True) 

    tokens_representations = results["representations"][nr_repr_layers] # logits, represent., attent., contacts
    tokens_representations = np.squeeze(tokens_representations, axis=0)
    tokens_df = pd.DataFrame(tokens_representations.numpy())
    tokens_df = tokens_df.tail(-1)
    tokens_df = tokens_df.head(-1)
    tokens_df = tokens_df.reset_index()
    df_res_ids = get_resid_and_sequence(pt_item['file_path'])
    tokens_df_final = pd.concat([df_res_ids, tokens_df], axis=1)
    res_list = pt_item['atoms']['residue'].values
    res_list_df = pd.DataFrame(res_list, columns = ['res_id'])
    merged = tokens_df_final.merge(res_list_df, on='res_id')
    merged_final = merged.iloc[:, 3:]
    sequence_rep_tensor = merged_final.to_numpy()
    return sequence_rep_tensor


# Helper function to get ESM (Evolutionary Scale Modeling) sequence embedding
def get_ESM_sequence_embedding_residue_level(sequence, LM):
    nr_rep_layers = extract_number_after_t(LM)  # Extract the number after 't' from the ESM model name
    model, alphabet = model_dict[LM]()
    batch_converter = alphabet.get_batch_converter()  # Get the batch converter for ESM
    data = [("dummy_name", sequence)]  # Create a list of data containing the sequence
    _, _, batch_tokens = batch_converter(data)  # Convert the sequence to tokens
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[nr_rep_layers], return_contacts=True)  # ESM representations
    token_representations = results["representations"][nr_rep_layers]  # Extract the token representations
    return token_representations  # Return the token representations


def get_saprot_sequence_embedding_residue_level(path_pdb, LM):
    # Put here, otherwise model.esm.base will cause circular import error (as I also have a file called model)
    sys.path.append("/home/florian_wieser/software/SaProt")
    from utils.foldseek_util import get_struc_seq
    from model.esm.base import EsmBaseModel
    from transformers import EsmTokenizer

    # Extract the "A" chain from the pdb file and encode it into a struc_seq
    # pLDDT is used to mask low-confidence regions if "plddt_mask" is True
    parsed_seqs = get_struc_seq("/home/florian_wieser/software/SaProt/bin/foldseek", path_pdb)
    if parsed_seqs:  # Check if parsed_seqs is not empty
        combined_seq = next(iter(parsed_seqs.values()))[2]
    else:
        # Handle the case where parsed_seqs is empty
        print("No sequences found in parsed_seqs.")
        return None
    config = {"task": "base",
              "config_path": f"/home/florian_wieser/software/SaProt/weights/PLMs/{LM}/",
              "load_pretrained": True}
    model = EsmBaseModel(**config)
    tokenizer = EsmTokenizer.from_pretrained(config["config_path"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = tokenizer(combined_seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embeddings = model.get_hidden_states(inputs)
    return embeddings[0]


def get_sequence_embeddings_batch(hparams, data):
    LM = int_to_esm2_string(hparams.LM)
    sequence_rep_list = []
    for pdb_path in data.file_path:
        sequence = extract_aa_sequence(pdb_path)
        sequence_rep = get_ESM_sequence_embedding_residue_level(sequence, LM)
        sequence_rep = sequence_rep[:, 1:-1, :]
        sequence_rep = sequence_rep.squeeze(0)
        sequence_rep_list.append(sequence_rep)

    stacked_tensor = sequence_rep_list[0]
    for tensor in sequence_rep_list[1:]:
        stacked_tensor = torch.cat((stacked_tensor, tensor), dim=0)
    return stacked_tensor

secondary_structure_mapping = {
    'H': 0,
    'E': 1,
    'L': 2}

amino_acid_mapping = {
    'A': 0, 
    'R': 1, 
    'N': 2, 
    'D': 3, 
    'C': 4, 
    'E': 5, 
    'Q': 6, 
    'G': 7, 
    'H': 8, 
    'I': 9, 
    'L': 10,
    'K': 11,
    'M': 12,
    'F': 13,
    'P': 14,
    'S': 15,
    'T': 16,
    'W': 17,
    'Y': 18,
    'V': 19}

def one_hot_encode_feature_list(feature, feature_list):
    if feature == "secondary_structure":
        feature_mapping = secondary_structure_mapping
    elif feature == "sequence":
        feature_mapping = amino_acid_mapping
    features_int = np.array([feature_mapping[f] for f in feature_list])
    one_hot = np.zeros((len(feature_list), len(feature_mapping)))
    one_hot[np.arange(len(feature_list)), features_int] = 1
    return one_hot    

### PDB FUNCTIONS ###################################################################################
def extract_aa_sequence(pdb_file_path):
    parser = PDB.PDBParser(QUIET=True)  # Create a PDBParser object
    structure = parser.get_structure("protein", pdb_file_path)  # Parse the PDB file
    sequence = ""  # Initialize an empty sequence
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):  # Check if the residue is an amino acid
                    aa = three_letter_to_one_letter.get(residue.resname, 'X')  # Default to 'X' if not found
                    sequence += aa  # Append the amino acid to the sequence
    return sequence  # Return the amino acid sequence


def get_residue_numbers(path_pdb):
    parser = PDBParser()
    structure = parser.get_structure("structure", path_pdb)
    resnum_list = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resnum_list.append(residue.get_id()[1])
    return resnum_list


def write_pred_to_pdb(pdb_filename, output_filename, res_id_list, pred_list):
    with open(pdb_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            if line.startswith('ATOM'):
                residue_number = int(line[22:26].strip())
                for res_id, pred in zip(res_id_list, pred_list):
                    if residue_number == res_id:
                        start_pos = 60
                        width = 6
                        new_value = "  " + str(pred)[:6]
                        new_value = new_value.ljust(width)
                        outfile.write(line[:start_pos] + new_value + line[start_pos + width:])
    return output_filename


### EXPERIMENTS ##############################################################################################

def run_r1z1_experiment(out_tag, trainer, edn_pl, dataloader, param): 
    edn_pl.predictions.clear()
    out = trainer.test(edn_pl, dataloader)
    predictions = pd.DataFrame(edn_pl.predictions)
    # extract r1/z1 values from pdb filename
    predictions[param] = predictions['id'].str.replace(r'_0001.pdb', '', regex=True)
    predictions[param] = predictions[param].str.replace(r'dummy_[rz]1-', '', regex=True)
    predictions[param] = predictions[param].astype(float)
    if param == "r1":
        plot_pred_vs_r1z1(predictions, "r1", "2.2943", "helix radius (CA atom) [$\mathrm{\AA}$]", 
                      "protARES learned helix features", "geo-r1_" + out_tag, "result_images")
    elif param == "z1":
        plot_pred_vs_r1z1(predictions, "z1", "1.5310", "helix rise per residue [$\mathrm{\AA}$]", 
                      "protARES learned helix features", "geo-z1_" + out_tag, "result_images")

        
### TRAINER HELPERS ###############################################################################

def process_label(label, predictions, min_scale, max_scale, type_label_scale, outpath):
    pred, targ = scale_local_labels_back(predictions, label, min_scale, max_scale, type_label_scale)
    df = pd.DataFrame({f"pred_{label}": pred, f"targ_{label}": targ})
    plot_target_corr(df, f"targ_{label}", f"pred_{label}", label, "black", os.path.join(outpath, f"{label}_corr.png"))
    return df

def log_metrics(label, predictions_df, outpath):
    R, r2, mae, var_out = calculate_regression_metrics(predictions_df[f"pred_{label}"], predictions_df[f"targ_{label}"])
    print(f"{label}\t{R:.3f}\t{r2:.3f}\t{mae:.3f}\t{var_out:.3f}")
    wandb.log({f'R_{label}': R, f'r2_{label}': r2, f'mae_{label}': mae, f'var_out_{label}': var_out})

def set_hparams_from_yaml(path_yaml, hparams):
    with open(hparams.yaml, 'r') as file:
        wandb_config = yaml.safe_load(file)
    for key, value in wandb_config['parameters'].items():
        setattr(hparams, key, value['values'][0])
    return wandb_config, hparams

def set_hparams_from_wandb(wandb_config, hparams):
    for key, value in wandb_config.items():
        setattr(hparams, key, value)
    return hparams


### MODEL HELPERS #################################################################################

def transform_reslevel_features(hparams, feature_list_stacked, scale_factor):
        # standardization: mean 0, standard deviation 1
        if hparams.reslevel_feature_scale_type == "standardization":
            feature_list_stacked = (feature_list_stacked - torch.mean(feature_list_stacked)) / torch.std(feature_list_stacked)
        # normalization: min, max scaling
        elif hparams.reslevel_feature_scale_type == "normalization":
            min_val = torch.min(feature_list_stacked)
            max_val = torch.max(feature_list_stacked)
            feature_list_stacked = (feature_list_stacked - min_val) / (max_val - min_val)
        # centralization: shift mean to zero
        elif hparams.reslevel_feature_scale_type == "centralization":
            feature_list_stacked = feature_list_stacked - torch.mean(feature_list_stacked)
        return feature_list_stacked * scale_factor

###################################################################################################
def add_hr_atomlevel_features_tensor_to_atomlevel_features(self, features, item, list_atomlevel_features):
    for feature in list_atomlevel_features:
        if feature == "none":
            continue
        pattern = re.compile(r'(m[12345]).*?(\.pdb)')
        fname_pdb_json = re.sub(pattern, r'\1\2', item['id'])
        
        feature_list = self.atomlevel_features_json[fname_pdb_json][feature]
        atom_name_list = self.atomlevel_features_json[fname_pdb_json]["atom_names"]
        res_ids_list = self.atomlevel_features_json[fname_pdb_json]["res_ids"]
         
        atoms_identities = set(zip(item['atoms']['name'], item['atoms']['residue'])) 
        
        # Filter the feature_list based on the atom names and residue numbers in pdb_data_set
        filtered_feature_list = [
            feature for feature, atom_name, res_id in zip(feature_list, atom_name_list, res_ids_list)
            if (atom_name, res_id) in atoms_identities
        ]
        assert len(filtered_feature_list) == len(item['atoms']) , "Error atom features missing"
        features_final = np.array(filtered_feature_list)
        # add feature batch to out
        feature_tensor = torch.tensor(features_final).to(features.device)
        
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(1)
            
        # apply standardization, normalization or centralization based on hparams     
        feature_tensor = transform_reslevel_features(self.hparams, feature_tensor, self.hparams.reslevel_feature_scale_factor)
        features = torch.cat((features, feature_tensor), dim=1)
    return features
 

def add_reslevel_features_tensor_to_atomlevel_features(self, out0, item, list_atomlevel_features):
    for feature in list_atomlevel_features:
        if feature == "none":
            continue
            
        fname_pdb = item['id']
        pattern = re.compile(r'(m[12345]).*?(\.pdb)')
        fname_pdb_json = re.sub(pattern, r'\1\2', fname_pdb)

        if "esm2_" not in feature and "saprot_" not in feature:
            feature_list = self.features_json[fname_pdb_json][feature]
        if "esm2_" in feature:
            feature_list = self.esm_h5[fname_pdb_json]
            feature_list = feature_list.tolist()[0] # convert np.ndarray to list of lists
        if "saprot_" in feature:
            feature_list = self.saprot_h5[fname_pdb_json]
            feature_list = feature_list.tolist() # convert np.ndarray to list of lists
        
        # get list of residue numbers
        resnum_list  = self.labels[fname_pdb]['res_id']
        feature_list_filtered = [feature_list[i - 1] for i in resnum_list]
        
        if feature == "secondary_structure" or feature == "sequence":
            feature_list_filtered = one_hot_encode_feature_list(feature, feature_list_filtered)
            
        df_feature_list = pd.DataFrame(np.array(feature_list_filtered))
        df_feature_list = df_feature_list.reset_index()
        df_res_ids = get_resid_and_sequence(item['file_path'])
        df_feature_list = pd.concat([df_res_ids, df_feature_list], axis=1)
        res_list = item['atoms']['residue'].values
        df_res_list = pd.DataFrame(res_list, columns = ['res_id'])
        merged = df_feature_list.merge(df_res_list, on='res_id')
        merged_final = merged.iloc[:, 3:]
        features_final = merged_final.to_numpy()

        # add feature batch to out0
        if feature == "secondary_structure" or feature == "sequence":
            feature_tensor = torch.tensor(features_final)
            feature_tensor = feature_tensor.float()
            out0 = torch.cat((out0, feature_tensor.to(out0.device)), dim=1)
        else:
            feature_tensor = torch.tensor(features_final).to(out0.device)
            if feature == "sasa" or feature == "energies" or feature == "sap-score" or feature == "hbonds":
                feature_tensor = feature_tensor.to(out0.device).view(feature_tensor.shape[0], 1)
            if feature_tensor.dim() >= 3:
                feature_tensor = feature_tensor.squeeze(1)
           
            # apply standardization, normalization or centralization based on hparams     
            feature_tensor = transform_reslevel_features(self.hparams, feature_tensor, self.hparams.atomlevel_feature_scale_factor)
            out0 = torch.cat((out0, feature_tensor), dim=1)
    return out0
   
def add_reslevel_features_tensor_to_out0(self, out0, data, hparams):
    for feature in hparams.reslevel_features:
        feature_list_stacked = []
        one_hot_dim = {"secondary_structure": 3, "sequence": 20}
        
        # break if hparams.reslevel_feature is None 
        if feature == "none":
            continue
    
        if feature == "sequence" or feature == "secondary_structure":
            feature_array_stacked = np.empty((0, one_hot_dim[feature]), dtype=float)
            
        # put batch of features together
        for path_pdb in data.file_path:
            fname_pdb = os.path.basename(path_pdb)
            pattern = re.compile(r'(m[12345]).*?(\.pdb)')
            fname_pdb_json = re.sub(pattern, r'\1\2', fname_pdb)
            #fname_pdb_json = fname_pdb[:6] + ".pdb" # if relax models
            
            if "esm2_" not in feature and "saprot_" not in feature:
                feature_list = self.features_json[fname_pdb_json][feature]
            if "esm2_" in feature:
                feature_list = self.esm_h5[fname_pdb_json]
                feature_list = feature_list.tolist()[0] # convert np.ndarray to list of lists
            if "saprot_" in feature:
                feature_list = self.saprot_h5[fname_pdb_json]
                feature_list = feature_list.tolist() # convert np.ndarray to list of lists
                
            # get list of residue numbers
            resnum_list  = self.labels[fname_pdb]['res_id']
            feature_list_filtered = [feature_list[i - 1] for i in resnum_list]
            
            if feature == "secondary_structure" or feature == "sequence":
                feature_list_filtered = one_hot_encode_feature_list(feature, feature_list_filtered)
                feature_array_stacked = np.concatenate((feature_array_stacked, feature_list_filtered), axis=0)
            else:
                feature_list_stacked.extend(feature_list_filtered)

        # add feature batch to out0
        if feature == "secondary_structure" or feature == "sequence":
            feature_array_stacked = torch.tensor(feature_array_stacked)
            feature_array_stacked = feature_array_stacked.float()
            out0 = torch.cat((out0, feature_array_stacked.to(out0.device)), dim=1)
        else:
            feature_list_stacked = torch.tensor(feature_list_stacked).to(out0.device)
            if feature == "sasa" or feature == "energies" or feature == "sap-score" or feature == "hbonds":
                feature_list_stacked = feature_list_stacked.to(out0.device).view(feature_list_stacked.shape[0], 1)
            if feature_list_stacked.dim() >= 3:
                feature_list_stacked = feature_list_stacked.squeeze(1)
            # apply standardization, normalization or centralization based on hparams     
            feature_list_stacked = transform_reslevel_features(hparams, feature_list_stacked, hparams.atomlevel_feature_scale_factor)
            out0 = torch.cat((out0, feature_list_stacked), dim=1)
    return out0

def calculate_multilabel_loss(self, batch, y_hat):
    loss = 0
    for idx, key in enumerate(["label_1", "label_2", "label_3"]):
        if getattr(self.hparams, key):
            loss += torch.nn.functional.huber_loss(y_hat[idx], getattr(batch, key).float())
    return loss
