#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3

import re
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def extract_bond_length_shift_from_fname(fname):
    match_plu = re.search(r'_plu(\d+)A_', fname)
    if match_plu:
        return float(match_plu.group(1))
    match_min = re.search(r'_min(\d+)A_', fname)
    if match_min:
        return -float(match_min.group(1))
    return None

def extract_degree_from_fname(fname):
    """
    Extracts degree values from two supported patterns in filename:
    1. '1ubq_m1619_A.pdb' or '1ubq_p1619_A.pdb' --> -161.9 / +161.9
    2. '1ubq_min10d_A.pdb' or '1ubq_plu5d_A.pdb' --> -10 / +5
    """
    # Style 1: _m1619_ or _p1619_
    match1 = re.search(r'_([mp])(\d+)_', fname)
    if match1:
        sign_char = match1.group(1)
        digits = match1.group(2)
        degree_str = digits[:-1] + '.' + digits[-1] if len(digits) > 1 else digits
        degree_val = float(degree_str)
        if sign_char == 'm':
            degree_val = -degree_val
        return degree_val

    # Style 2: _min10d_ or _plu5d_
    match2 = re.search(r'_(plu|min)(\d+)d_', fname)
    if match2:
        sign, digits = match2.groups()
        val = float(digits)
        return val if sign == "plu" else -val

    # If nothing matched:
    print(f"Could not parse degree from filename: {fname}")
    return None

def fxtract_degree_from_fname(fname):
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
        print(f"here:")
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
    print(f"degree_val: {degree_val}")
    return degree_val


def process_bond_length_csv(csv_path, res_ids):
    df = pd.read_csv(csv_path)
    df = df[df['res_id'].isin(res_ids)].copy()
    df = df[['fname_pdb', 'res_id', 'ELEN_score']]
    df['bond_shift'] = df['fname_pdb'].apply(extract_bond_length_shift_from_fname)
    df = df.dropna(subset=['bond_shift'])
    df_summary = df.groupby('fname_pdb', as_index=False).agg({
        'bond_shift': 'first',
        'ELEN_score': 'mean'
    }).rename(columns={'ELEN_score': 'avg_ELEN_score'})
    return df_summary

def process_dihedral_csv(csv_path, res_ids):
    df = pd.read_csv(csv_path)
    print(f"df: {df}")
    df = df[df['res_id'].isin(res_ids)].copy()
    df = df[['fname_pdb', 'res_id', 'ELEN_score']]
    df['degree'] = df['fname_pdb'].apply(extract_degree_from_fname)
    df = df.dropna(subset=['degree'])
    df_summary = df.groupby('fname_pdb', as_index=False).agg({
        'degree': 'first',
        'ELEN_score': 'mean'
    }).rename(columns={'ELEN_score': 'avg_ELEN_score'})
    df_summary['degree_wrapped'] = df_summary['degree'].apply(
        lambda x: x + 360 if x is not None and x < 0 else x
    )
    return df_summary

def main(args):
    # Publication font settings
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
        "figure.titlesize": 18,
        "axes.linewidth": 1.1,
    })

    bond_summary = process_bond_length_csv(args.csv_bond, args.res_id)
    degree_summary = process_dihedral_csv(args.csv_dihedral, args.res_id)
    
    # ---- Plotting ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # --- Bond length shift subplot (left) ---
    ax = axes[0]
    ax.set_axisbelow(True)
    ax.grid(True, zorder=0)
    ax.scatter(bond_summary['bond_shift'], bond_summary['avg_ELEN_score'], color='black', s=16, zorder=2)
    ax.set_xlabel('Bond Length Shift (Å)')
    ax.set_ylabel('Average ELEN Score')
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Bond length shift", fontsize=16, pad=12)

    # --- Dihedral degree subplot (right) ---
    ax = axes[1]
    ax.set_axisbelow(True)
    ax.grid(True, zorder=0)
    ax.scatter(degree_summary['degree_wrapped'], degree_summary['avg_ELEN_score'], color='black', s=16, zorder=2)
    ax.set_xlabel('Dihedral Angle (°)')
    ax.set_ylabel('Average ELEN Score')
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Dihedral angle", fontsize=16, pad=12)

    # Panel letter (left outside)
    fig.text(0.035, 0.97, args.panel_letter, fontsize=22, fontweight='bold', ha='left', va='top')
    # Model name (centered above)
    fig.suptitle(args.model_name, fontsize=18, fontweight='semibold', y=1.04)

    plt.subplots_adjust(top=0.88, wspace=0.25)
    plt.savefig(f"{args.output_prefix}_merged.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Merged plot saved to {args.output_prefix}_merged.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create merged ELEN plots for bond length and dihedral angle from two CSV files."
    )
    parser.add_argument('--csv_bond', type=str, required=True,
                        help='CSV file with manipulated bond lengths.')
    parser.add_argument('--csv_dihedral', type=str, required=True,
                        help='CSV file with manipulated dihedral angles.')
    parser.add_argument('--res_id', type=int, nargs='+', required=True,
                        help='Residue IDs to filter (e.g., --res_id 61 62 63).')
    parser.add_argument('--output_prefix', type=str, default='avg_elen_panel',
                        help='Prefix for output plot file (will add _merged.png).')
    parser.add_argument('--model_name', type=str, default='ELEN',
                        help='Model name for plot suptitle.')
    parser.add_argument('--panel_letter', type=str, default='A',
                        help='Panel letter for the merged panel (A, B, ...).')
    args = parser.parse_args()
    main(args)
