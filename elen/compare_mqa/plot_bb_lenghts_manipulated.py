#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3

import re
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

"""
Script: filter_and_plot_elen_scores_shift.py

Description:
    This script reads a CSV file containing ELEN scores (and other metrics) for various residues
    across different PDB files. It filters the data to include only the specified residue IDs,
    extracts a bond length shift (positive or negative) from each PDB filename (e.g., "1ubq_plu10A_A_RP_elen_scored.pdb"
    -> +10, "1ubq_min50A_A_RP_elen_scored.pdb" -> -50), computes the average ELEN score across
    the specified residues for each PDB file, and finally plots the averaged ELEN score vs. the
    extracted bond length shift.

Usage example:
    python filter_and_plot_elen_scores_shift.py \
        --csv_path elen_scores_RP.csv \
        --res_id 61 62 63 \
        --output_plot my_plot.png
"""

def extract_bond_length_shift_from_fname(fname):
    """
    Given a filename like:
      '1ubq_plu50A_A_RP_elen_scored.pdb' -> +50
      '1ubq_min40A_A_RP_elen_scored.pdb' -> -40
      '1ubq_plu00A_A_RP_elen_scored.pdb' -> +0  (or just 0.0)
    
    Returns the numeric shift as a float. If the pattern is not found,
    returns None or 0.0 (based on preference).
    """
    # Look for "..._plu##A_..." which should be a positive shift
    match_plu = re.search(r'_plu(\d+)A_', fname)
    if match_plu:
        val_str = match_plu.group(1)  # e.g. '50', '10', '00'
        return float(val_str)

    # Look for "..._min##A_..." which should be a negative shift
    match_min = re.search(r'_min(\d+)A_', fname)
    if match_min:
        val_str = match_min.group(1)  # e.g. '50', '10', '00'
        return -float(val_str)

    # If we don't match either pattern, return None or 0.0
    return None

def main(args):
    # 1. Read the CSV file into a pandas DataFrame
    df = pd.read_csv(args.csv_path)

    # 2. Filter for only the specified residues
    df_filtered = df[df['res_id'].isin(args.res_id)].copy()

    # 3. Keep only the relevant columns
    df_filtered = df_filtered[['fname_pdb', 'res_id', 'ELEN_score']]

    # 4. Extract the bond length shift from the PDB filename into a new column
    df_filtered['bond_shift'] = df_filtered['fname_pdb'].apply(extract_bond_length_shift_from_fname)

    # 5. Compute the average ELEN score for each PDB (over the filtered residues)
    df_summary = df_filtered.groupby('fname_pdb', as_index=False).agg({
        'bond_shift': 'first',
        'ELEN_score': 'mean'
    })
    df_summary.rename(columns={'ELEN_score': 'avg_ELEN_score'}, inplace=True)

    # 6. Print intermediate DataFrames for inspection
    print("Filtered Data (One row per specified residue):")
    print(df_filtered)
    print("\nSummary Data (One row per PDB):")
    print(df_summary[['fname_pdb', 'bond_shift', 'avg_ELEN_score']])

    # 7. Plot avg_ELEN_score vs. bond length shift
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)  # Quadratic figure, high DPI
    
    # Place grid behind scatter points
    ax.set_axisbelow(True)
    ax.grid(True, zorder=-1)

    # Plot scatter in black
    ax.scatter(df_summary['bond_shift'], df_summary['avg_ELEN_score'], color='black', zorder=2)

    ax.set_xlabel('Bond Length Shift (Å)', fontsize=14)
    ax.set_ylabel('Average ELEN Score', fontsize=14)
    #ax.set_title('Average ELEN Score vs. Bond Length Shift')
    plt.tight_layout()
    plt.savefig(args.output_plot)
    plt.close()
    print(f'Plot saved to {args.output_plot}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filter residues, compute averages, and plot results with bond length shifts.'
    )
    parser.add_argument('--csv_path', type=str, required=True, default="elen_scores_RP.csv",
                        help='Path to the CSV file containing ELEN scores.')
    parser.add_argument('--res_id', type=int, nargs='+', required=True,
                        help='Residue IDs to filter (e.g., --res_id 61 62 63).')
    parser.add_argument('--output_plot', type=str, default='avg_elen_score_vs_shift.png',
                        help='Name of the output plot image file.')
    args = parser.parse_args()
    main(args)
