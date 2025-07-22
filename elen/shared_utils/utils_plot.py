import os
import wandb
import warnings
import subprocess
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
from elen.shared_utils.utils_others import calculate_regression_metrics

def plot_target_corr(data, xaxis, yaxis, label, color, save_path):
    """
    Plots the correlation between target and predicted values, including regression metrics.

    Args:
        data (pd.DataFrame): DataFrame containing the target and predicted values.
        xaxis (str): Column name for the x-axis (target values).
        yaxis (str): Column name for the y-axis (predicted values).
        label (str): Label name for the plot title.
        color (str): Color of the scatter plot points and regression line.
        save_path (str): Path to save the output plot.
    """
    # Create the scatter plot with regression line
    sns.regplot(data=data, x=xaxis, y=yaxis, color=color, scatter_kws={'s': 12, 'alpha': 0.3}, line_kws={'lw': 1.5})
    
    # Add metrics as legend to the plot
    pearson, spearman, _, mae, rmse = calculate_regression_metrics(data[xaxis], data[yaxis])
    plt.plot([], [], ' ', label=f"n: {len(data[yaxis])}")
    plt.plot([], [], ' ', label=f"R: {pearson:4f}")
    plt.plot([], [], ' ', label=f"spear: {spearman:4f}")
    plt.plot([], [], ' ', label=f"MAE: {mae:4f}")
    plt.plot([], [], ' ', label=f"RMSE: {rmse:4f}")
    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left')
    
    # Set plot labels and title
    plt.title(f"{label} Predicted vs. Target")
    plt.xlabel(f"Target {label}")
    plt.ylabel(f"Predicted {label}")
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
    
    
def log_plots(outpath, globtag, wandbtag):
    """
    Logs all plot files matching a specific tag from the output directory to Weights & Biases (wandb).

    Args:
        outpath (str): Directory path where the plot files are stored.
        globtag (str): File extension or tag to filter plot files.
        wandbtag (str): Tag to use when logging the images to wandb.
    """
    all_files = os.listdir(outpath)
    plot_files = [f for f in all_files if f.endswith(globtag)]

    for plot_file in plot_files:
        file_path = os.path.join(outpath, plot_file)
        wandb.log({wandbtag: [wandb.Image(file_path, caption=plot_file)]})
        
        
        
def concatenate_plots(plot_list: List[str], mode: str, outpath: str) -> str:
    """
    Concatenates multiple image files into a single image using the 'convert' command from ImageMagick.

    This function uses the 'convert' command to combine multiple plot images into one, depending on the specified mode
    which determines how the images are combined (e.g., horizontally or vertically).

    Parameters:
        plot_list (List[str]): A list of file paths to the plot images to be concatenated.
        mode (str): The mode of concatenation ('+append' for horizontal, '-append' for vertical).
        outpath (str): The file path where the concatenated image will be saved.

    Returns:
        str: The path to the output file where the concatenated image is saved.

    Raises:
        subprocess.CalledProcessError: If the 'convert' command fails.
        ValueError: If the mode is not one of the expected options.
    """
    if mode not in ['+append', '-append']:
        raise ValueError("Invalid mode. Use '+append' for horizontal or '-append' for vertical concatenation.")
    
    try:
        # Ensure the command is executed correctly
        subprocess.run(['convert'] + plot_list + [mode, outpath], check=True)
    except subprocess.CalledProcessError as e:
        warnings.warn(f"[WARN] Failed to concatenate plots ({len(plot_list)} images). Skipping final concat.\nError: {e}")
    return outpath

### for data_preparation
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Plotting functions ########################################################

def plot_labels_bar_rmsd(labels_json_path, id_to_plot, outpath):
    """
    Plot a grouped bar plot of per-residue RMSD labels for a specified protein.
    Only RMSD metrics (rmsd_nat, rmsd_md, rmsd_avg_md) are plotted.
    """
    try:
        with open(labels_json_path, 'r') as f:
            labels_dict = json.load(f)
    except Exception as e:
        logging.error(f"Could not load labels from {labels_json_path}: {e}")
        return

    matched_key = None
    for key in labels_dict.keys():
        if key.startswith(id_to_plot):
            matched_key = key
            break
    if matched_key is None:
        logging.error(f"No label entry found for identifier {id_to_plot}.")
        return

    data = labels_dict[matched_key]
    required_keys = ["res_id", "rmsd_nat", "rmsd_md", "rmsd_avg_md"]
    for rk in required_keys:
        if rk not in data:
            logging.error(f"Missing {rk} in data for {matched_key}.")
            return

    # Create a DataFrame for convenience
    df = pd.DataFrame({
        "rmsd_nat": data["rmsd_nat"],
        "rmsd_md": data["rmsd_md"],
        "rmsd_avg_md": data["rmsd_avg_md"]
    }, index=data["res_id"])

    n = len(df)
    x = np.arange(n)
    bar_width = 0.2

    # Offsets for RMSD bars
    offset_rmsd_nat = -bar_width
    offset_rmsd_md = 0
    offset_rmsd_avg = bar_width

    fig, ax = plt.subplots(figsize=(18, 6))
    bars1 = ax.bar(x + offset_rmsd_nat, df["rmsd_nat"], width=bar_width, label="rmsd_nat", color="C0")
    bars2 = ax.bar(x + offset_rmsd_md, df["rmsd_md"], width=bar_width, label="rmsd_md", color="C1")
    bars3 = ax.bar(x + offset_rmsd_avg, df["rmsd_avg_md"], width=bar_width, label="rmsd_avg_md", color="C2")

    ax.set_xlabel("Residue ID")
    ax.set_ylabel("RMSD")
    ax.set_title(f"Per-residue RMSD labels for {matched_key}")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plot_file = os.path.join(outpath, f"{id_to_plot}_rmsd_barplot.png")
    plt.savefig(plot_file)
    logging.info(f"Saved RMSD barplot for {id_to_plot} to {plot_file}")
    plt.close()


def plot_labels_bar_lddt(labels_json_path, id_to_plot, outpath):
    """
    Plot a grouped bar plot of per-residue LDDT labels for a specified protein.
    Only LDDT metrics (lddt_nat, lddt_md) are plotted.
    """
    try:
        with open(labels_json_path, 'r') as f:
            labels_dict = json.load(f)
    except Exception as e:
        logging.error(f"Could not load labels from {labels_json_path}: {e}")
        return

    matched_key = None
    for key in labels_dict.keys():
        if key.startswith(id_to_plot):
            matched_key = key
            break
    if matched_key is None:
        logging.error(f"No label entry found for identifier {id_to_plot}.")
        return

    data = labels_dict[matched_key]
    required_keys = ["res_id", "lddt_nat", "lddt_md"]
    for rk in required_keys:
        if rk not in data:
            logging.error(f"Missing {rk} in data for {matched_key}.")
            return

    # Create a DataFrame for convenience
    df = pd.DataFrame({
        "lddt_nat": data["lddt_nat"],
        "lddt_md": data["lddt_md"]
    }, index=data["res_id"])

    n = len(df)
    x = np.arange(n)
    bar_width = 0.25

    # Offsets for LDDT bars
    offset_lddt_nat = -bar_width / 2
    offset_lddt_md = bar_width / 2
    
    fig, ax = plt.subplots(figsize=(18, 6))
    bars1 = ax.bar(x + offset_lddt_nat, df["lddt_nat"], width=bar_width, label="lddt_nat", color="C3")
    bars2 = ax.bar(x + offset_lddt_md, df["lddt_md"], width=bar_width, label="lddt_md", color="C4")
    plt.margins(x=0.02)

    ax.set_xlabel("Residue ID")
    ax.set_ylabel("LDDT (0 to 1)")
    ax.set_title(f"Per-residue LDDT labels for {matched_key}")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plot_file = os.path.join(outpath, f"{id_to_plot}_lddt_barplot.png")
    plt.savefig(plot_file)
    logging.info(f"Saved LDDT barplot for {id_to_plot} to {plot_file}")
    plt.close()


def plot_labels_violin(labels_json_path, outpath):
    """
    Plot violin plots for label distributions across the dataset.
    RMSD metrics are plotted on the primary y-axis and LDDT metrics on a secondary y-axis.
    The plotting is done manually using matplotlib's violinplot.
    """
    try:
        with open(labels_json_path, 'r') as f:
            labels_dict = json.load(f)
    except Exception as e:
        logging.error(f"Could not load labels from {labels_json_path}: {e}")
        return

    # Collect values for each metric across all proteins
    rmsd_data = {"rmsd_nat": [], "rmsd_md": [], "rmsd_avg_md": []}
    lddt_data = {"lddt_nat": [], "lddt_md": []}
    for key, data in labels_dict.items():
        for metric in rmsd_data.keys():
            if metric in data:
                rmsd_data[metric].extend(data[metric])
        for metric in lddt_data.keys():
            if metric in data:
                lddt_data[metric].extend(data[metric])

    # Define positions for each category on the common x-axis
    pos_rmsd = [0, 1, 2]
    pos_lddt = [3, 4]
    all_positions = pos_rmsd + pos_lddt
    all_labels = ["rmsd_nat", "rmsd_md", "rmsd_avg_md", "lddt_nat", "lddt_md"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()

    # Plot violins for RMSD metrics on primary y-axis
    data_rmsd = [rmsd_data["rmsd_nat"], rmsd_data["rmsd_md"], rmsd_data["rmsd_avg_md"]]
    parts1 = ax.violinplot(data_rmsd, positions=pos_rmsd, widths=0.8, showmeans=True)
    for pc in parts1['bodies']:
        pc.set_facecolor("C0")
        pc.set_alpha(0.6)

    # Plot violins for LDDT metrics on secondary y-axis
    data_lddt = [lddt_data["lddt_nat"], lddt_data["lddt_md"]]
    parts2 = ax2.violinplot(data_lddt, positions=pos_lddt, widths=0.8, showmeans=True)
    for pc in parts2['bodies']:
        pc.set_facecolor("C3")
        pc.set_alpha(0.6)

    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels)
    ax.set_ylabel("RMSD")
    ax2.set_ylabel("LDDT (0 to 1)")
    ax.set_title("Distribution of labels across all proteins")

    plt.tight_layout()
    plot_file = os.path.join(outpath, "labels_violin.png")
    plt.savefig(plot_file)
    logging.info(f"Saved violin plot to {plot_file}")
    plt.close()

# for train_val_test.py
def plot_histogram(data, label, color, max_label, path_output):
    print(f"Plotting histogram for {label}.")
    num_bins = 20
    bins = np.linspace(0, max_label, num_bins + 1)
    plt.hist(data, bins=bins, color=color, log=True)
    plt.xlabel(f"{label}", fontsize=18)
    plt.ylabel('log count [1]', fontsize=18)
    plt.xlim(0,  max_label)
    plt.xticks(np.arange(0, max_label, step=(max_label / num_bins)), rotation='vertical', fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot([], [], ' ', label = "n: " + str(len(data)))
    plt.grid(color="#DDDDDD", which="both")
    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left', fontsize=16)
    plt.title(f"Distribution of {label} label", fontsize=20)
    plt.savefig(path_output, bbox_inches='tight')
    plt.clf()
    return path_output

def plot_violinplot(data, label, color, path_output):
    print(f"Plotting violinplot for {label}.")
    violin_parts = plt.violinplot(data, vert=False, showmeans=True, showextrema=True)
    for pc in violin_parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
    violin_parts['cmeans'].set_edgecolor(color)
    violin_parts['cmaxes'].set_edgecolor(color)
    violin_parts['cmins'].set_edgecolor(color)
    violin_parts['cbars'].set_edgecolor(color)
    plt.xlabel(f"{label}", fontsize=18)
    plt.ylabel('count [1]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot([], [], ' ', label = "n: " + str(len(data)))
    plt.grid(color="#DDDDDD", which="both")
    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left', fontsize=16)
    plt.title(f"Distribution of {label} label", fontsize=20)
    plt.savefig(path_output, bbox_inches='tight')
    plt.clf()
    return path_output