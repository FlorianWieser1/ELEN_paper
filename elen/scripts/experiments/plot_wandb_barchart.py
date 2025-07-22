#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def plot_bar(
    df, metric, save_path, layer, 
    figsize=(3, 3),  # Small, publication-friendly size
    panel_letter=None
):
    # Matplotlib publication settings
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

    # Group and order
    grouped = df.groupby(layer)[metric].mean().sort_values(ascending=False)
    stds = df.groupby(layer)[metric].std().reindex(grouped.index)
    labels = grouped.index.str.replace(r'[\[\]"]', '', regex=True)
    labels = ["None" if lbl.lower() == "none" else lbl for lbl in labels]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        labels, grouped, yerr=stds, 
        capsize=4, alpha=0.92, color="#5a5a5a", linewidth=0, width=0.63
    )

    ax.set_ylabel(metric, fontsize=12, fontweight="semibold", labelpad=6)
    xlabel = 'Atom-level Feature' if layer == "atomlevel_features" else "Residue-level Feature"
    ax.set_xlabel(xlabel, fontsize=10, labelpad=3)
    ax.tick_params(axis='x', labelrotation=0, labelsize=10, length=4)
    ax.tick_params(axis='y', labelsize=10, length=4)
    # ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='regular')

    ax.set_ylim(0, grouped.max() * 1.18)

    # Optional: Add panel letter for multi-panel figures
    if panel_letter is not None:
        ax.text(-0.23, 1.07, panel_letter, fontsize=18, fontweight='bold', 
                transform=ax.transAxes, va='top', ha='left')

    plt.tight_layout(pad=0.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Condensed, publication-ready bar plot from CSV, grouped by atom-level features."
    )
    parser.add_argument('csv', help="CSV file to plot")
    parser.add_argument('--metric', default='R_CAD', choices=['R_CAD', 'R_lddt', 'R_rmsd'],
                        help="Metric to plot (default: R_CAD)")
    parser.add_argument('--out', help="Output image file (e.g. plot.png). If omitted, no image is saved.")
    parser.add_argument('--panel_letter', default=None, help="Panel letter (A, B, etc.) for figure")
    parser.add_argument('--layer', type=str, default="atomlevel_features", choices=['atomlevel_features', 'reslevel_features'])
    args = parser.parse_args()

    # Load and parse CSV
    df = pd.read_csv(args.csv)
    for metric in ['R_CAD', 'R_lddt', 'R_rmsd']:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')

    plot_bar(
        df, args.metric, args.out, args.layer,
        figsize=(2.2, 2.6), 
        panel_letter=args.panel_letter
    )

if __name__ == "__main__":
    main()
