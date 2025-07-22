import os
import sys
import yaml
import json
import torch
import wandb
import random
import numpy as np
import pandas as pd
from typing import Dict, Any
from typing import Any, Tuple, Dict
import elen.training.model as m
import pytorch_lightning as pl
#from elen.training.utils_training import scale_local_labels_back
from elen.shared_utils.utils_plot import plot_target_corr
from elen.shared_utils.utils_plot import calculate_regression_metrics

def seed_everything(seed: int) -> None:
    """
    Seeds all necessary random number generators to ensure reproducibility across multiple libraries.

    Parameters:
        seed (int): The seed value to use for all random number generators.
    """
    random.seed(seed)  # Seed Python's built-in random module
    np.random.seed(seed)  # Seed NumPy's random module
    torch.manual_seed(seed)  # Seed PyTorch's random number generator

    # Seed for CUDA (GPU) operations if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Seed all GPUs if multiple GPUs are available

    # Ensure that PyTorch's operations are deterministic on GPU (useful for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Seed all workers if using PyTorch Lightning
    pl.seed_everything(seed, workers=True)
    
    
def set_hparams_from_yaml(hparams: Any) -> Tuple[Dict[str, Any], Any]:
    """
    Load hyperparameters from a YAML file and update the hparams object.

    Parameters:
    - hparams: A namespace or object where hyperparameters are stored.

    Returns:
    - A tuple containing the configuration from the YAML file and the updated hparams object.
    
    Raises:
    - FileNotFoundError: If the YAML file does not exist.
    - yaml.YAMLError: If the YAML file is not correctly formatted.
    """
    try:
        with open(hparams.yaml, 'r') as file:
            wandb_config = yaml.safe_load(file)
        for key, value in wandb_config['parameters'].items():
            setattr(hparams, key, value['values'][0])
        return wandb_config, hparams
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The specified file {hparams.yaml} was not found.") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError("There was an error parsing the YAML file.") from e


def set_hparams_from_wandb(wandb_config: Dict[str, Any], hparams: Any) -> Any:
    """
    Update the hparams object with configurations from wandb.

    Parameters:
    - wandb_config: A dictionary containing the wandb configurations.
    - hparams: A namespace or object where hyperparameters are stored.

    Returns:
    - The updated hparams object.
    """
    for key, value in wandb_config.items():
        setattr(hparams, key, value)
    return hparams


def initialize_training_script(hparams: Any, logger: Any) -> Tuple[Any, Dict[str, Any], Any]:
    """
    Initialize or resume an experiment run based on hyperparameters.

    Parameters:
    - hparams: A namespace or object containing hyperparameters and run configuration.
    - logger: Logger for logging messages.
    - m: Module containing the EDN_PL class definition.

    Returns:
    - A tuple containing the possibly updated hparams object, a dictionary of keyword arguments, and the initialized EDN_PL model.
    """
    PROC_ID = os.environ.get('SLURM_PROCID')  # Check if the script is run via SLURM
    keyword_args = {}
    edn_pl = None
    # Handle checkpoint and wandb initialization
    if (hparams.wandb in ['online', 'offline']) and hparams.checkpoint:
        logger.info(f"Loading model from checkpoint file {hparams.checkpoint}.")
        run_id = os.path.basename(hparams.checkpoint)[:8]
        if PROC_ID == "0" or PROC_ID is None:
            wandb.init(id=run_id, resume='must', project='edn-experiment', mode=hparams.wandb)
        checkpoint = torch.load(hparams.checkpoint)
        hparams.__dict__.update(checkpoint['hyper_parameters'])
        hparams.checkpoint = hparams.checkpoint  # Ensure 'checkpoint' keeps its original value.
        edn_pl = m.EDN_PL.load_from_checkpoint(hparams.checkpoint)
        keyword_args['ckpt_path'] = hparams.checkpoint

    # Initialize new runs
    elif not hparams.checkpoint:
        if hparams.wandb in ['online', 'offline']:
            if hparams.yaml:  # Initialize from YAML file
                logger.info(f"Initializing from {hparams.yaml} yaml file.")
                wandb_config, hparams = set_hparams_from_yaml(hparams)
                if PROC_ID == "0" or PROC_ID is None:
                    wandb.init(mode=hparams.wandb, project='edn-experiment', config=wandb_config)
            if not hparams.yaml:  # Initialize from command line arguments
                logger.info("Initializing new model from args.")
                if PROC_ID == "0" or PROC_ID is None:
                    wandb.init(mode=hparams.wandb, project='edn-experiment')
                hparams = set_hparams_from_wandb(wandb.config, hparams)

            # Convert residue level features to list if provided as a string
            if isinstance(hparams.reslevel_features, str):
                hparams.reslevel_features = [hparams.reslevel_features]
        
        elif hparams.wandb == 'disabled':
            if PROC_ID == "0" or PROC_ID is None:
                wandb.init(mode="disabled", project='edn-experiment')
                hparams.max_epochs = hparams.epochs

        dict_args = vars(hparams)
        edn_pl = m.EDN_PL(**dict_args)  # Instantiate the EDN_PL class

    return hparams, keyword_args, edn_pl


def transform_and_plot_label(label, predictions, min_scale, max_scale, type_label_scale, outpath):
    """
    Transforms and scales label predictions, then plots the correlation between targets and predictions.

    Args:
        label (str): The name of the label to be transformed and plotted.
        predictions (torch.Tensor or np.ndarray): The predicted values for the label.
        min_scale (float): Minimum value used for scaling.
        max_scale (float): Maximum value used for scaling.
        type_label_scale (str): Type of scaling to be applied.
        outpath (str): Directory path to save the output plot.

    Returns:
        pd.DataFrame: A DataFrame containing the transformed target and prediction values.
    """
    # Scale predictions and target values back to their original range
    pred, targ, _ = scale_local_labels_back(predictions, label, min_scale, max_scale, type_label_scale)
    
    # Create a DataFrame to hold target and prediction values
    df = pd.DataFrame({f"pred_{label}": pred, f"targ_{label}": targ})
    
    # Plot the correlation between target and prediction values
    plot_path = os.path.join(outpath, f"{label}_corr.png")
    plot_target_corr(df, f"targ_{label}", f"pred_{label}", label, color="black", save_path=plot_path)
    
    return df


def log_metrics(label, predictions_df):
    """
    Calculates regression metrics for the given predictions and logs them using Weights & Biases (wandb).

    Args:
        label (str): The name of the label for which metrics are being logged.
        predictions_df (pd.DataFrame): A DataFrame containing columns for the target and predicted values.
    """
    # Calculate regression metrics
    R, spearman, r2, mae, var_out = calculate_regression_metrics(predictions_df[f"pred_{label}"],
                                                                 predictions_df[f"targ_{label}"])
    
    # Print the metrics in a formatted string
    print(f"{label}\tR: {R:.3f}\tSpearman: {spearman:.3f}\tr^2: {r2:.3f}\tMAE: {mae:.3f}\tVariance: {var_out:.3f}")
    
    # Log the metrics to wandb
    wandb.log({
        f'R_{label}': R,
        f'spear_{label}': spearman,
        f'mae_{label}': mae,
        f'var_out_{label}': var_out
    })
    
    
### utils for label transformation and scaling

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