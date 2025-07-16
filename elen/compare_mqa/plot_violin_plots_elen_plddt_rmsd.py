#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(args):
    # Publication-quality font/appearance
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
        "figure.titlesize": 18,
    })
    sns.set_style("whitegrid")
    plt.rcParams["axes.linewidth"] = 2.0  # Thicker axis frame

    # Read data
    df = pd.read_csv(args.csv_file)

    # Melt to long-form for seaborn violinplot
    data_long = pd.DataFrame({
        "ELEN": pd.concat([df["orig_elen"], df["rede_elen"]], ignore_index=True),
        "pLDDT": pd.concat([df["orig_plddt"], df["rede_plddt"]], ignore_index=True),
        "RMSD": pd.concat([df["orig_rmsd"], df["rede_rmsd"]], ignore_index=True),
        "Group": ["Original design"] * len(df) + ["Redesign"] * len(df)
    })

    # Set up plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200, sharey=False)
    metrics = [
        ("ELEN", (0.7, 0.95), "ELEN Score"),
        ("pLDDT", (50, 95), "pLDDT"),
        ("RMSD", (0, 8), "RMSD [Å]")
    ]
    colors = ["#5078D1", "#EA6A47"]  # blue, orange

    for i, (metric, ylim, ylabel) in enumerate(metrics):
        ax = axes[i]
        sns.violinplot(
            x="Group", y=metric, data=data_long, ax=ax,
            palette=colors, cut=0, inner=None, width=0.8,
            linewidth=2.0, zorder=1
        )
        sns.boxplot(
            x="Group", y=metric, data=data_long, ax=ax,
            palette=colors, width=0.25, fliersize=2, showcaps=True,
            boxprops=dict(alpha=0.7, linewidth=2.0, edgecolor="black"),
            whiskerprops=dict(linewidth=2.0, color="black"),
            medianprops=dict(linewidth=2.2, color="black"),
            capprops=dict(linewidth=2.0, color="black"),
            zorder=2
        )
        # Make all spines (axes) visible and thick
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_color("black")

        ax.set_title(metric, fontsize=17, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_ylim(ylim)
        ax.grid(True, axis='y', zorder=0)

    # Panel letter at top left OUTSIDE the axes
    #fig.text(0.06, 0.97, args.panel_letter, fontsize=22, fontweight='bold', ha='left', va='top')

    plt.subplots_adjust(top=0.88, wspace=0.28)
    plt.savefig(f"{args.output_prefix}_violin.png", bbox_inches='tight', dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create side-by-side violin+box plots for ELEN, pLDDT, RMSD (original vs redesign) for loop region."
    )
    parser.add_argument(
        "--csv-file",
        required=True,
        help="Input CSV with columns: orig_elen, orig_plddt, orig_rmsd, rede_elen, rede_plddt, rede_rmsd"
    )
    parser.add_argument(
        "--output-prefix",
        default="output",
        help="Prefix for output plot filename."
    )
    parser.add_argument(
        "--panel-letter",
        default="A",
        help="Panel letter for figure (e.g. 'A', 'B')."
    )
    args = parser.parse_args()
    main(args)
