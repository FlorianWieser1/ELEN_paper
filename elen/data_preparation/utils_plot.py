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
