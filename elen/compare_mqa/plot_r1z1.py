#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3

import argparse
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

def parse_filename_for_value(filename: str) -> float:
    match = re.search(r'(?:r1|z1)-([0-9.]+)', filename)
    if match:
        return float(match.group(1))
    else:
        return float('nan')

def main(args):
    # Font settings for publication quality
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
        "figure.titlesize": 18,
    })

    # Read CSV into DataFrame
    df = pd.read_csv(args.csv_file)
    df["extracted_value"] = df["fname_pdb"].apply(parse_filename_for_value)
    
    # Separate rows for r1 and z1
    df_r1 = df[df["fname_pdb"].str.contains("r1-")]
    df_z1 = df[df["fname_pdb"].str.contains("z1-")]
    
    # Group by extracted_value and take the mean of avg_per_chain (ELEN score).
    r1_group = df_r1.groupby("extracted_value", as_index=False)["avg_per_chain"].mean()
    z1_group = df_z1.groupby("extracted_value", as_index=False)["avg_per_chain"].mean()

    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    
    # --- Plot z1 ---
    ax = axes[0]
    ax.grid(True, zorder=0)
    ax.scatter(z1_group["extracted_value"], z1_group["avg_per_chain"], s=15, color="black", zorder=2)
    ax.axvline(x=1.1, color="red", linestyle="--", label="π-helix (z1 = 1.1 Å)")
    ax.axvline(x=1.5, color="red", linestyle="--", label="α-helix (z1 = 1.5 Å)")
    ax.axvline(x=2.0, color="red", linestyle="--", label="3₁₀ helix (z1 = 2.0 Å)")
    ax.set_xlabel("Helical rise per residue z₁")
    ax.set_ylabel("ELEN score")
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc="lower left")
    ax.set_title("Helical rise per residue", fontsize=16, pad=12)
    
    # --- Plot r1 ---
    ax = axes[1]
    ax.grid(True, zorder=0)
    ax.scatter(r1_group["extracted_value"], r1_group["avg_per_chain"], s=15, color="black", zorder=2)
    ax.axvline(x=1.9, color="red", linestyle="--", label="3₁₀ helix (r1 = 1.9 Å)")
    ax.axvline(x=2.3, color="red", linestyle="--", label="α-helix (r1 = 2.3 Å)")
    ax.axvline(x=2.8, color="red", linestyle="--", label="π-helix (r1 = 2.8 Å)")
    ax.set_xlabel("Helical radius r₁")
    ax.set_ylabel("ELEN score")
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc="lower left")
    ax.set_title("Helical radius", fontsize=16, pad=12)

    # Panel letter at top left OUTSIDE the axes (adjust x/y as needed)
    fig.text(0.04, 0.97, args.panel_letter, fontsize=22, fontweight='bold', ha='left', va='top')

    # Model name centered just above the plots, close to the axes
    fig.suptitle(args.model_name, fontsize=18, fontweight='semibold', y=1.04)  # y=0.98 puts it closer

    # Reduce space between title and plots
    plt.subplots_adjust(top=0.90, wspace=0.28)  # top controls space below suptitle

    # Save merged plot
    plt.savefig(f"{args.output_prefix}_merged.png", bbox_inches='tight', dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract values from filenames in CSV and create side-by-side z1/r1 plot per model."
    )
    parser.add_argument(
        "--csv-file",
        required=True,
        help="Path to the input CSV file (must have columns 'fname_pdb' and 'avg_per_chain')."
    )
    parser.add_argument(
        "--output-prefix",
        default="output_plot",
        help="Prefix for the output plot filename."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model variant name for plot title (e.g. 'ELEN-NoSeq')"
    )
    parser.add_argument(
        "--panel-letter",
        default="A",
        help="Panel letter for suptitle (e.g. 'A', 'B', 'C')."
    )
    args = parser.parse_args()
    main(args)
