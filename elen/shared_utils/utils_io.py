import csv
import json
import h5py
from typing import Dict, Any

def load_from_hdf5(path: str) -> Dict[str, Any]:
    """
    Load data from an HDF5 file and return it as a dictionary.

    Parameters:
        path (str): The path to the HDF5 file.

    Returns:
        Dict[str, Any]: A dictionary containing the data from the HDF5 file.
    """
    dict_data = {}
    try:
        with h5py.File(path, 'r') as f:
            for protein_id in f.keys():
                dict_data[protein_id] = f[protein_id][()]  # Load the dataset into memory
    except (OSError, KeyError) as e:
        print(f"Error loading HDF5 file: {e}")
    return dict_data

def load_from_json(path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file and return it as a dictionary.

    Parameters:
        path (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: A dictionary containing the data from the JSON file.
    """
    try:
        with open(path, 'r') as f:
            dict_data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")
        dict_data = {}
    return dict_data

def load_from_csv(path: str) -> Dict[str, Any]:
    """
    Load data from a CSV file and return it as a dictionary.

    Each row in the CSV is stored under a numerical index as its key, and the row data
    is a dictionary mapping column headers to their respective values in that row.

    Parameters:
        path (str): The path to the CSV file.

    Returns:
        Dict[str, Any]: A dictionary containing the data from the CSV file.
                        The outer dictionary is keyed by row index (0-based),
                        and each value is another dictionary of column_name -> column_value.
    """
    data = {}
    try:
        with open(path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                data[idx] = row
    except (OSError, csv.Error) as e:
        print(f"Error loading CSV file: {e}")
    return data

def dump_dict_to_json(dict_serialized: Dict[str, Any], path_json: str) -> None:
    """
    Write a dictionary to a JSON file.

    Parameters:
        dict_serialized (Dict[str, Any]): The dictionary to write.
        path_json (str): The path to the JSON file where the dictionary should be saved.

    Returns:
        None
    """
    try:
        with open(path_json, "w") as f:
            json.dump(dict_serialized, f, indent=4)
    except OSError as e:
        print(f"Error writing to JSON file: {e}")