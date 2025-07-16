
import os
import sys
import json
import numpy as np
import torch

def min_max_scaler(data: np.ndarray, scale_type: str = "normalization") -> tuple:
    """Scales data using min-max normalization.

    Args:
        data (np.ndarray): The input data array.
        scale_type (str): Type of scaling, defaults to "normalization".

    Returns:
        tuple: Scaled data, minimum value, and maximum value.
    """
    min_val = data.min()
    max_val = data.max()
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data, min_val, max_val


def centralization_scaler(data: np.ndarray) -> tuple:
    """Centers data by subtracting the mean.

    Args:
        data (np.ndarray): The input data array.

    Returns:
        tuple: Centralized data and the mean of the original data.
    """
    mean = data.mean()
    centralized_data = data - mean
    return centralized_data, mean


def standardization_scaler(data: np.ndarray) -> tuple:
    """Standardizes data by subtracting the mean and dividing by the standard deviation.

    Args:
        data (np.ndarray): The input data array.

    Returns:
        tuple: Standardized data, mean, and standard deviation.
    """
    mean = data.mean()
    std = data.std()
    standardized_data = (data - mean) / std
    return standardized_data, mean, std


def inverse_min_max_scaler(scaled_data: np.ndarray, min_val: float, max_val: float) -> list:
    """Reverts data scaled by min-max normalization back to its original scale.

    Args:
        scaled_data (np.ndarray): Scaled data.
        min_val (float): Minimum value of the original data.
        max_val (float): Maximum value of the original data.

    Returns:
        list: Data reverted to its original scale.
    """
    original_data = (scaled_data * (max_val - min_val)) + min_val
    return list(original_data)


def scale_local_labels_back(predictions: dict, label: str, min_val: float, max_val: float, label_scale_type: str) -> tuple:
    """Scales local labels back to their original scale based on the label type.

    Args:
        predictions (dict): Dictionary containing prediction and target data.
        label (str): Key for the specific label in predictions.
        min_val (float): Minimum value used during scaling.
        max_val (float): Maximum value used during scaling.
        label_scale_type (str): Type of scaling applied ('normalization', 'centralization', etc.).

    Returns:
        tuple: Lists of predictions and targets scaled back to their original values.
    """
    prediction_list = []
    target_list = []
    results_dict = {}
    for id, values in predictions[label].items():
        pred = values['pred']
        targ = values['target']

        if label_scale_type == "normalization":
            pred_back = inverse_min_max_scaler(np.array(pred), min_val, max_val)
            targ_back = inverse_min_max_scaler(np.array(targ), min_val, max_val)
        elif label_scale_type == "centralization":
            pred_back = [p + min_val for p in pred]
            targ_back = [t + min_val for t in targ]
        elif label_scale_type == "standardization":
            pred_back = [(p * max_val) + min_val for p in pred]
            targ_back = [(t * max_val) + min_val for t in targ]
        elif label_scale_type == "none":
            pred_back = pred
            targ_back = targ

        prediction_list.extend(pred_back)
        target_list.extend(targ_back)
        results_dict[id] = (pred_back, targ_back)
    
    return prediction_list, target_list, results_dict


def scale_local_labels_from_json(label_tag: str, data_labels: dict, label_scale_type: str) -> tuple:
    """Scales local labels from JSON based on the provided scaling type.

    Args:
        label_tag (str): Key for the specific label to be scaled.
        data_labels (dict): Dictionary containing all data labels.
        label_scale_type (str): Type of scaling to apply ('normalization', 'centralization', 'standardization', 'none').

    Returns:
        tuple: Dictionary of scaled labels, minimum value, and maximum value used for scaling.
    """
    label_list = [data_labels[id][label_tag] for id in data_labels]
    id_list = list(data_labels.keys())
    label_array = np.array(label_list, dtype=float)
    if label_scale_type == "normalization":
        scaled_labels, min_val, max_val = min_max_scaler(label_array)
    elif label_scale_type == "centralization":
        scaled_labels, min_val = centralization_scaler(label_array)
        max_val = min_val  # Not used but provided for consistency in the return values
    elif label_scale_type == "standardization":
        scaled_labels, min_val, max_val = standardization_scaler(label_array)
    elif label_scale_type == "none":
        scaled_labels = label_array
        min_val, max_val = None, None  # No scaling applied, hence no min/max

    scaled_labels_dict = {id: torch.tensor(scaled_labels[idx]) for idx, id in enumerate(id_list)}

    return scaled_labels_dict, min_val, max_val


def get_scaled_labels_and_min_max_scale(hparams) -> tuple:
    """Loads and scales labels from a JSON file based on hyperparameters.

    Args:
        hparams: Hyperparameters containing label configuration and paths.

    Returns:
        tuple: Scaled labels, minimum scales, and maximum scales.
    """
    path_labels = os.path.join(hparams.test_dir, "../../labels.json")
    try:
        with open(path_labels) as file:
            data_labels = json.load(file)
    except FileNotFoundError:
        print(f"Label file not found {path_labels}")
        return [], [], []
    scaled_labels, min_scale, max_scale = [], [], []
    for label_attr in ['label_1', 'label_2', 'label_3']:
        if getattr(hparams, label_attr):
            label_tag = getattr(hparams, label_attr)
            scaled_label, min_val, max_val = scale_local_labels_from_json(label_tag, data_labels, hparams.label_scale_type)
            scaled_labels.append(scaled_label)
            min_scale.append(min_val)
            max_scale.append(max_val)
    
    return scaled_labels, min_scale, max_scale