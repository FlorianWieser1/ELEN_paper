import torch
import torch.nn.functional as F
import numpy as np

def get_dimension(hparams, feature_dict: dict) -> int:
    """
    Calculates the dimension based on hyperparameters and feature contributions.

    Parameters:
        hparams: A configuration object with attributes for dimension calculation.
        feature_dict (dict): A dictionary where keys are feature names and values are their contributions to the dimension.

    Returns:
        int: The calculated dimension.
    """
    dimension = 4  # Base dimension
    if hasattr(hparams, 'one_hot_encoding') and hparams.one_hot_encoding == "Derevyanko":
        dimension = 11  # Adjust base dimension for specific encoding

    for feature_list in [hparams.hr_atomlevel_features, hparams.atomlevel_features]:
        for feature in feature_list:
            if feature != "none":
                dimension += feature_dict.get(feature, 0)  # Safely add feature contribution

    return dimension


def split_activations_batch_into_dicts(activations: np.ndarray, ids: list) -> dict:
    """
    Splits a batch of activations into a dictionary keyed by IDs.

    Parameters:
        activations (np.ndarray): The activations array from a model layer.
        ids (list): A list of identifiers corresponding to each split in the activations.

    Returns:
        dict: A dictionary where each key is an identifier and the value is the corresponding activation slice.
    """
    num_files = len(ids)
    split_size = activations.shape[0] // num_files  # Calculate size of each split
    activations_dict = {}

    if activations.shape[0] % num_files != 0:
        raise ValueError("Number of activations does not evenly divide by the number of IDs")

    for i, file_id in enumerate(ids):
        start_idx = i * split_size
        end_idx = start_idx + split_size
        activations_dict[file_id] = activations[start_idx:end_idx]

    return activations_dict


def calculate_multilabel_loss(self, batch, y_hat: torch.Tensor) -> torch.Tensor:
    """
    Calculate the sum of Huber losses for multiple labels in a batch against predictions.
    
    Parameters:
        batch (object): An object containing the batch data with labels as attributes.
        y_hat (torch.Tensor): The predictions for the batch, expected to have a shape that matches the number of labels.
    
    Returns:
        torch.Tensor: The cumulative Huber loss for all specified labels.
    """
    loss = 0
    label_keys = ["label_1", "label_2", "label_3"]

    dict_losses = {
        "huber": lambda pred, target: torch.nn.functional.huber_loss(pred, target, delta=self.hparams.huber_delta),
        "mse": torch.nn.functional.mse_loss,
        "mae": torch.nn.functional.l1_loss,
        "smooth_l1": torch.nn.functional.smooth_l1_loss,
        "cosine_similarity": lambda pred, target: 1 - torch.nn.functional.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).mean(),
    }
    loss_function = dict_losses.get(self.hparams.loss_type, torch.nn.functional.huber_loss)

    for idx, key in enumerate(label_keys):
        if hasattr(self.hparams, key) and getattr(self.hparams, key):
            try:
                label_data = getattr(batch, key).float()
                #loss += torch.nn.functional.huber_loss(y_hat[idx], label_data)
                
                loss += loss_function(y_hat[idx], label_data)
            except AttributeError:
                print(f"Warning: {key} not found in batch.")
            except Exception as e:
                print(f"Error processing {key}: {e}")
                
    return loss


### activation functions ######################################################

class Mish(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)