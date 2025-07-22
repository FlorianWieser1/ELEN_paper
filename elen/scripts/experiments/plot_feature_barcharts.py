#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

def preprocess_labels(df, layer):
    # Remove brackets and quotes from feature lists for nicer labels
    df[layer] = df[layer].str.replace(r'[\[\]\"]', '', regex=True)
    df[layer] = df[layer].str.replace(r"\s*,\s*", ",", regex=True)  # no space here, easier to check comma presence
    df[layer] = df[layer].replace("none", "none")  # keep none as is
    return df

def get_ordered_labels(labels):
    # Sort labels putting 'none' first if present,
    # then 'all' (combined features) second if present,
    # then the rest sorted alphabetically.
    new_labels = []
    labels_set = set(labels)

    if 'none' in labels_set:
        new_labels.append('none')

    # Detect combined feature sets (labels containing commas)
    all_label = None
    combined_labels = [lbl for lbl in labels_set if ',' in lbl]
    if combined_labels:
        # There could be multiple combined labels, but here we treat all as one "All"
        # Pick one combined label to represent the "All" bar
        # (You can only have one "All" bar, so pick the first sorted)
        all_label = sorted(combined_labels)[0]
        new_labels.append(all_label)

    # Add all other single feature labels (without comma and not none)
    for lbl in sorted(lbl for lbl in labels_set if lbl not in ['none', all_label]):
        new_labels.append(lbl)

    return new_labels

def label_fmt(l, all_label):
    if l == "none":
        return "None"
    elif l == all_label:
        return "All"
    else:
        # Replace keys with nicer labels inside the feature string
        dict_xlabels = {
            "sap-score": "SAP-score",
            "sasa": "SASA",
            "energies": "Energies",
            "secondary_structure": "Secondary Structure",
            "sequence": "Sequence"
        }
        for key, val in dict_xlabels.items():
            l = l.replace(key, val)
        return l

def plot_average_bar(df, metrics, args, figsize=(5, 3.2)):
    save_path = args.out
    layer = args.layer
    panel_letter = args.panel_letter

    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.4,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.weight": "semibold"
    })

    df = preprocess_labels(df, layer)
    feature_col = [c for c in df.columns if c.endswith("_features")][0]

    group = df.groupby(layer)

    means = group[metrics].mean()
    stds = group[metrics].std()

    avg_means = means.mean(axis=1)
    avg_stds = means.std(axis=1)

    labels = get_ordered_labels(list(means.index))
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)

    parts = str(args.csv).split("_")
    type_data = parts[0]
    dict_colors = {
        'floats': 'Greens',
        'onehot': 'Reds',
        'hbonds': 'Blues',
        'saprot': 'Oranges'
    }

    # Separate 'none' and combined features label from others
    labels_set = set(labels)
    combined_labels = [lbl for lbl in labels if ',' in lbl]
    all_label = combined_labels[0] if combined_labels else None

    # Colors for all bars except 'none'
    other_labels = [lbl for lbl in labels if lbl != 'none']
    cmap = cm.get_cmap(dict_colors.get(type_data, 'Greens'))
    norm = mcolors.Normalize(vmin=0.0, vmax=max(len(other_labels) - 1, 1))
    other_colors = [cmap(norm(i)) for i in range(len(other_labels))]

    color_map = {'none': '#4c4c4c'}  # dark grey for none
    for lbl, col in zip(other_labels, other_colors):
        color_map[lbl] = col

    bars = []
    for i, lbl in enumerate(labels):
        val = avg_means.get(lbl, np.nan)
        err = avg_stds.get(lbl, np.nan)
        if np.isnan(val):
            continue
        color = color_map.get(lbl, "#4daf4a")  # fallback green
        bar = ax.bar(
            i, val, yerr=err,
            capsize=3, alpha=0.92, color=color, width=0.7, linewidth=1.0, edgecolor='black',
            error_kw={'elinewidth': 0.8, 'alpha': 0.7},
            label=None
        )
        bars.append(bar[0])
        ax.annotate(f'{val:.4f}',
                    xy=(i, val),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='regular')

    ax.set_xticks(x)
    ax.set_xticklabels([label_fmt(l, all_label) for l in labels], rotation=0, fontsize=12)

    ax.set_ylabel("Pearson R" if args.metric == "R" else "MAE", fontsize=10, fontweight="semibold", labelpad=7)
    xlabel = 'Atom-level Features' if layer == "atomlevel_features" else "Residue-level Features"
    ax.set_xlabel(xlabel, fontsize=10, labelpad=4)

    ax.tick_params(axis='x', labelrotation=0, labelsize=8, length=4)
    ax.tick_params(axis='y', labelsize=10, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.22 * np.nanmax(avg_means.values))

    if panel_letter is not None:
        ax.text(-0.25, 1.11, panel_letter, fontsize=19, fontweight='bold',
                transform=ax.transAxes, va='top', ha='left')

    plt.tight_layout(pad=0.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)

    # --- Save metrics summary csv ---
    group_counts = group[metrics].count()
    print(f"group_counts:\n{group_counts}")
    summary_df = means.copy()
    summary_df.columns = [f"{c}_mean" for c in summary_df.columns]

    std_df = stds.copy()
    std_df.columns = [f"{c}_std" for c in std_df.columns]

    count_df = group_counts.copy()
    count_df.columns = [f"{c}_count" for c in count_df.columns]

    summary_df['Avg_mean'] = summary_df[[f"{m}_mean" for m in metrics]].mean(axis=1)
    summary_df = summary_df.join(std_df)
    summary_df = summary_df.join(count_df)

    summary_df.reset_index(inplace=True)

    csv_path = os.path.splitext(save_path)[0] + '_data.csv' if save_path else 'metrics_summary_data.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path}")

###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Publication-ready bar plot of average metric from CSV, grouped by atom-level features."
    )
    parser.add_argument('csv', help="CSV file to plot")
    parser.add_argument('--out', help="Output image file (e.g. plot.png). If omitted, no image is saved.")
    parser.add_argument('--panel_letter', default=None, help="Panel letter (A, B, etc.) for figure")
    parser.add_argument('--layer', type=str, default="atomlevel_features", choices=['atomlevel_features', 'reslevel_features', 'hr_atomlevel_features'])
    parser.add_argument("--metric", type=str, default="R", choices=["R", "mae"])

    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    metrics = [f'{args.metric}_CAD', f'{args.metric}_lddt', f'{args.metric}_rmsd']
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')

    plot_average_bar(df, metrics, args)

if __name__ == "__main__":
    main()
