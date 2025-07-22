#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
import sys
import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

RESNAME_MAP = {
    'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'GLY': 'G', 'LEU': 'L', 'ILE': 'I',
    'MET': 'M', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'CYS': 'C', 'TYR': 'Y',
    'TRP': 'W', 'ASP': 'D', 'GLU': 'E', 'ASN': 'N', 'GLN': 'Q', 'HIS': 'H',
    'LYS': 'K', 'ARG': 'R'
}

def prettify_name(fname):
    if "wt_" in os.path.basename(fname):
        return f"{fname[:4]}_wt"
    elif "model_" in fname:
        m = re.search(r'model_(\d+)', fname)
        if m:
            return f"{fname[:4]}_model_{m.group(1)}"

def parse_pdb_sequence(pdb_path, chain_id='A'):
    sequence = {}
    if not os.path.isfile(pdb_path):
        return sequence

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21:22].strip()
                if chain != chain_id:
                    continue
                try:
                    res_id = int(line[22:26].strip())
                except ValueError:
                    continue
                if res_id not in sequence:
                    res_3letter = line[17:20].strip()
                    sequence[res_id] = RESNAME_MAP.get(res_3letter, '?')
    return sequence

def main(args):
    df = pd.read_csv(args.input_csv)
    native_mask = df['fname_pdb'].str.contains(args.reference_tag)
    native_df = df[native_mask].drop_duplicates(['res_id'])

    if native_df.empty:
        print(f"No native model found using --reference_tag {args.reference_tag}")
        sys.exit(1)

    native_scores = native_df.set_index('res_id')['ELEN_score']
    mutant_df = df[~native_mask][['fname_pdb', 'res_id', 'ELEN_score']]

    pivot = mutant_df.pivot(index='fname_pdb', columns='res_id', values='ELEN_score')
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    native_vec = native_scores.reindex(pivot.columns)
    delta_df = pivot.subtract(native_vec, axis=1)

    model_files = pivot.index.tolist()
    all_files = [os.path.basename(args.reference_pdb)] + model_files
    
    sequences = {}
    for fname in all_files:
        pdb_pattern = os.path.join(args.scored_elen_models, os.path.splitext(fname)[0] + "*.pdb")
        candidates = glob.glob(pdb_pattern)
        pdb_path = candidates[0] if candidates else None
        sequences[fname] = parse_pdb_sequence(pdb_path, args.chain_id) if pdb_path else {}
    aa_df = pd.DataFrame(index=all_files, columns=pivot.columns)
    
    for fname in all_files:
        for res_id in pivot.columns:
            aa_df.at[fname, res_id] = sequences[fname].get(res_id, '?')
    delta_df.loc[args.reference_pdb] = 0
    delta_df = delta_df.reindex([os.path.basename(args.reference_pdb)] + model_files)
    
    # Build DataFrames for abs and delta, with pretty row names if desired
    abs_df = pivot.copy()
    abs_df.loc[os.path.basename(args.reference_pdb)] = native_scores.reindex(pivot.columns)
    abs_df = abs_df.reindex([os.path.basename(args.reference_pdb)] + model_files)
    
    # Apply pretty names for row index if wanted
    pretty_names = {idx: prettify_name(idx) for idx in abs_df.index}
    abs_df.index = [pretty_names[idx] for idx in abs_df.index]
    delta_df.index = [pretty_names.get(idx, idx) for idx in delta_df.index]
    
    # Concatenate: all abs columns, then all delta columns
    combined = pd.concat(
        [abs_df.add_suffix("_abs"), delta_df.add_suffix("_delta")],
        axis=1
    )
    
    # Optional: move pretty_name column to front (if you want pretty/short labels)
    combined.index.name = "model"
    
    combined_csv = f"native_favoring_combined_{args.plot_title}.csv"
    combined.to_csv(combined_csv)
    print(f"Saved combined CSV (all abs then all delta): {combined_csv}")
    
    #delta_csv = f"native_favoring_{args.plot_title}.csv"
    #delta_df.to_csv(delta_csv)
    pretty_names = {fname: prettify_name(fname) for fname in all_files}
    
    # Rename indices for aa_df and delta_df to the short names
    aa_df.index = [pretty_names[f] for f in aa_df.index]
    delta_df.index = [pretty_names.get(f, f) for f in delta_df.index]
    plt.figure(figsize=(18, max(6, len(delta_df) * 0.35)))
    ax = sns.heatmap(
        delta_df,
        cmap='bwr',
        center=0,
        annot=aa_df,
        fmt='',
        linewidths=0.4,
        linecolor='grey',
        xticklabels=True,
        yticklabels=True,
        cbar=True,
        annot_kws={"size": 9},
        cbar_kws={'label': r'$\Delta$ ELEN Score'}
    )

    cbar = ax.collections[0].colorbar
    cbar.set_label(r'$\Delta$ ELEN Score', fontsize=14)
    cbar.ax.tick_params(labelsize=10)

    plt.xlabel('Residue Number', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8)

    if args.panel_label:
        plt.gcf().text(0.01, 0.92, args.panel_label, fontsize=22, fontweight='bold', ha='left', va='top')

    if args.plot_title:
        plt.title(args.plot_title, fontsize=16, fontweight='semibold', pad=18)

    plt.tight_layout()
    figfile = f"native_favoring_heat_{args.plot_title}.png"
    plt.savefig(figfile, bbox_inches='tight')
    print(f"Saved heatmap: {figfile}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot per-residue ELEN score deltas relative to native model with sequences; saves CSV and heatmap."
    )
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--reference_tag', type=str, required=True)
    parser.add_argument('--reference_pdb', type=str, required=True)
    parser.add_argument('--chain_id', type=str, default='A')
    parser.add_argument('--scored_elen_models', type=str, required=True)
    parser.add_argument('--plot_title', type=str, default='')
    parser.add_argument('--panel_label', type=str, default='')
    args = parser.parse_args()
    main(args)
