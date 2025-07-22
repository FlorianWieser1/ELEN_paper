#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3

import re
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

"""
Script: filter_and_plot_elen_scores.py

Description:
    This script reads a CSV file containing ELEN scores (and other metrics) for various residues
    across different PDB files. It filters the data to include only the specified residue IDs,
    extracts the rotation degree from each PDB filename, computes the average ELEN score across
    the specified residues for each PDB file, and finally plots the averaged ELEN score versus
    the extracted (wrapped) dihedral angle.

Usage example:
    python filter_and_plot_elen_scores.py \
        --csv_path elen_scores_RP.csv \
        --res_id 61 62 63 \
        --output_plot my_plot.png
"""

def extract_degree_from_fname(fname):
    """
    Given a filename like '1ubq_m1619_A.pdb' or '1ubq_p1619_A.pdb',
    extract the rotation degree as a float.

    Examples:
      1ubq_m1619_A.pdb -> -161.9
      1ubq_p1619_A.pdb -> +161.9
    """
    match = re.search(r'_([mp])(\d+)_', fname)
    if not match:
        # If we can't parse it, return None or 0.0 depending on your preference
        return None

    sign_char = match.group(1)  # 'm' or 'p'
    digits = match.group(2)     # e.g. '1619'

    # Insert a decimal before the last digit: '1619' -> '161.9'
    if len(digits) > 1:
        degree_str = digits[:-1] + '.' + digits[-1]
    else:
        # Fallback if digits is somehow only 1 digit long
        degree_str = digits

    degree_val = float(degree_str)
    if sign_char == 'm':
        degree_val = -degree_val
    return degree_val

def main(args):
    # 1. Read the CSV file into a pandas DataFrame
    df = pd.read_csv(args.csv_path)

    # 2. Filter for only the specified residues
    df_filtered = df[df['res_id'].isin(args.res_id)].copy()

    # 3. Keep only the relevant columns
    df_filtered = df_filtered[['fname_pdb', 'res_id', 'ELEN_score']]

    # 4. Extract the degree value from the PDB filename into a new column
    df_filtered['degree'] = df_filtered['fname_pdb'].apply(extract_degree_from_fname)

    # 5. Compute the average ELEN score for each PDB (over the filtered residues)
    df_summary = df_filtered.groupby('fname_pdb', as_index=False).agg({
        'degree': 'first',
        'ELEN_score': 'mean'
    })
    df_summary.rename(columns={'ELEN_score': 'avg_ELEN_score'}, inplace=True)

    # 6. "Wrap" angles so values near -180 become close to +180
    df_summary['degree_wrapped'] = df_summary['degree'].apply(
        lambda x: x + 360 if x is not None and x < 0 else x
    )

    # 7. Print intermediate DataFrames for inspection
    print("Filtered Data (One row per specified residue):")
    print(df_filtered)
    print("\nSummary Data (One row per PDB) - including wrapped angles:")
    print(df_summary[['fname_pdb', 'degree', 'degree_wrapped', 'avg_ELEN_score']])

    # 8. Plot avg_ELEN_score vs. wrapped degree
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)  # Quadratic figure, high DPI
    # Place grid behind scatter points
    ax.set_axisbelow(True)
    ax.grid(True, zorder=-1)
    
    # Plot scatter in black
    ax.scatter(df_summary['degree_wrapped'], df_summary['avg_ELEN_score'], color='black', zorder=2)
    
    ax.set_xlabel('Dihedral Angle (°)', fontsize=14)
    ax.set_ylabel('Average ELEN Score', fontsize=14)
    #ax.set_title('Average ELEN Score vs. ed Dihedral Angle')
    plt.tight_layout()
    plt.savefig(args.output_plot)
    plt.close()
    print(f'Plot saved to {args.output_plot}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filter residues, compute averages, and plot results.'
    )
    parser.add_argument('--csv_path', type=str, required=True, default="elen_scores_RP.csv",
                        help='Path to the CSV file containing ELEN scores.')
    parser.add_argument('--res_id', type=int, nargs='+', required=True,
                        help='Residue IDs to filter (e.g., --res_id 61 62 63).')
    parser.add_argument('--output_plot', type=str, default='avg_elen_score_vs_dihedral.png',
                        help='Name of the output plot image file.')
    args = parser.parse_args()
    main(args)
