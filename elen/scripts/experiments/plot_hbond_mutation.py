#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
import sys
import os
import glob
import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Map common 3-letter amino acid codes to 1-letter codes.
RESNAME_MAP = {
    'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'GLY': 'G', 'LEU': 'L', 'ILE': 'I',
    'MET': 'M', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'CYS': 'C', 'TYR': 'Y',
    'TRP': 'W', 'ASP': 'D', 'GLU': 'E', 'ASN': 'N', 'GLN': 'Q', 'HIS': 'H',
    'LYS': 'K', 'ARG': 'R'
}

def parse_pdb_for_residues(pdb_path, residues_of_interest, chain_id='A'):
    residue_map = {}
    if not os.path.isfile(pdb_path):
        return residue_map

    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21:22].strip()
            if chain != chain_id:
                continue
            try:
                res_id = int(line[22:26].strip())
            except ValueError:
                continue
            if res_id not in residues_of_interest:
                continue
            res_3letter = line[17:20].strip()
            one_letter = RESNAME_MAP.get(res_3letter, '?')
            if res_id not in residue_map:
                residue_map[res_id] = one_letter
    return residue_map

def parse_pdb_for_all_residues(pdb_path, chain_id='A'):
    all_resids = set()
    if not os.path.isfile(pdb_path):
        return all_resids
    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21:22].strip()
            if chain != chain_id:
                continue
            try:
                res_id = int(line[22:26].strip())
                all_resids.add(res_id)
            except ValueError:
                continue
    return sorted(all_resids)

def main(args):
    df = pd.read_csv(args.input_csv)
    all_resids = parse_pdb_for_all_residues(
        os.path.join(args.scored_elen_models, args.reference_pdb),
        chain_id=args.chain_id
    )
    all_resids_set = set(all_resids)
    fixed_positions = {}
    found_structure = False
    with open(args.fixed_positions, 'r') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if args.structure_id in data:
                chain_dict = data[args.structure_id]
                fixed_positions_list = chain_dict.get(args.chain_id, [])
                fixed_positions_set = set(fixed_positions_list)
                found_structure = True
                break
    if not found_structure:
        print(f"ERROR: No entry for {args.structure_id} found in {args.fixed_positions}!")
        sys.exit(1)
    mutated_residues = all_resids_set - fixed_positions_set
    if not mutated_residues:
        print("WARNING: No mutated residues inferred. Check your fixed_positions.jsonl or reference PDB!")
    else:
        print(f"Inferred mutated residues: {sorted(mutated_residues)}")
    df = df[df['res_id'].isin(mutated_residues)]
    reference = df[df['fname_pdb'].str.contains(args.reference_tag)]
    mutants = df[~df['fname_pdb'].str.contains(args.reference_tag)]
    pivot_ref = (
        reference[['res_id', 'ELEN_score']]
        .set_index('res_id')
        .rename(columns={'ELEN_score': 'original'})
    )
    mutant_pivot = mutants.pivot(index='res_id', columns='fname_pdb', values='ELEN_score')
    combined = pivot_ref.join(mutant_pivot)
    diff_df = combined.copy()
    for col in mutant_pivot.columns:
        diff_df[f'delta_{col}'] = diff_df[col] - diff_df['original']
    diff_df = diff_df.round(3)
    diff_df.to_csv(f"hbond_mutation_{args.suffix}.csv")
    ########################################################################
    # Scatter Plot
    ########################################################################
    plt.figure(figsize=(12, 6))
    plt.grid(True, zorder=0)
    plt.scatter(
        combined.index,
        combined['original'],
        label='Original',
        marker='o',
        color='black',
        alpha=1.0,
        zorder=3
    )
    for col in mutant_pivot.columns:
        plt.scatter(
            combined.index,
            combined[col],
            label=col,
            alpha=0.7,
            marker='x',
            zorder=3
        )
    plt.xlabel('Residue Number', fontsize=14)
    plt.ylabel('ELEN Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=8)
    if args.plot_title:
        plt.title(args.plot_title, fontsize=16, pad=10)
    if args.panel_label:
        plt.text(-0.10, 1.02, args.panel_label, fontsize=22, fontweight='bold',
                 transform=plt.gca().transAxes, va='top', ha='right')
    plt.tight_layout()
    plt.savefig(f"hbond_mutation_scatter_{args.suffix}", bbox_inches='tight')
    plt.close()
    ########################################################################
    # Figure 2: Heatmap of numeric deltas, annotated with one-letter amino acids
    ########################################################################
    mutant_files = list(mutant_pivot.columns)
    delta_cols = [f'delta_{mf}' for mf in mutant_files]
    delta_df = diff_df[delta_cols].copy()
    delta_df.columns = mutant_files
    delta_df = delta_df.T
    sorted_residues = sorted(mutated_residues)
    delta_df = delta_df[sorted_residues]
    aa_df = pd.DataFrame(index=mutant_files, columns=sorted_residues, dtype=object)
    for model_file in mutant_files:
        base_name = os.path.splitext(model_file)[0]
        pattern = os.path.join(args.scored_elen_models, base_name + "*")
        candidates = glob.glob(pattern)
        if not candidates:
            for res_id in sorted_residues:
                aa_df.loc[model_file, res_id] = '?'
            continue
        pdb_path = candidates[0]
        res_map = parse_pdb_for_residues(pdb_path, mutated_residues, chain_id=args.chain_id)
        for res_id in sorted_residues:
            aa_df.loc[model_file, res_id] = res_map.get(res_id, '?')
    # Include the reference model
    ref_pdb_names = reference['fname_pdb'].unique()
    if len(ref_pdb_names) != 1:
        print(
            "WARNING: Expected exactly one reference PDB filename matching "
            f"tag '{args.reference_tag}', but found: {ref_pdb_names}"
        )
    ref_pdb_name = ref_pdb_names[0]
    ref_base_name = os.path.splitext(ref_pdb_name)[0]
    ref_pattern = os.path.join(args.scored_elen_models, ref_base_name + "*")
    ref_candidates = glob.glob(ref_pattern)
    if ref_candidates:
        ref_pdb_path = ref_candidates[0]
        ref_res_map = parse_pdb_for_residues(ref_pdb_path, mutated_residues, chain_id=args.chain_id)
    else:
        print(f"WARNING: No PDB found for reference pattern: {ref_pattern}")
        ref_res_map = {r: '?' for r in sorted_residues}
    delta_df.loc[ref_pdb_name] = 0.0
    aa_df.loc[ref_pdb_name] = [ref_res_map.get(r, '?') for r in sorted_residues]
    new_index = [ref_pdb_name] + mutant_files
    delta_df = delta_df.reindex(new_index)
    aa_df = aa_df.reindex(new_index)
    # Plot the heatmap, with optional panel label and title
    fig = plt.figure(figsize=(14, 4))
    ax = sns.heatmap(
        delta_df,
        cmap='bwr',
        center=0,
        annot=aa_df,
        fmt='',
        cbar=True,
        annot_kws={"size": 12},
        cbar_kws={'label': r'$\Delta$ ELEN Score'}
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label(r'$\Delta$ ELEN Score', fontsize=14)
    cbar.ax.tick_params(labelsize=10)
    plt.xlabel('Residue Number', fontsize=14)
    plt.ylabel('Models', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # Add panel label to the left (like "A", "B", etc.)
    if args.panel_label:
        # Add a big bold letter just outside the left of the figure
        fig.text(0.01, 0.92, args.panel_label, fontsize=22, fontweight='bold', ha='left', va='top')
    # Add title above the heatmap
    if args.plot_title:
        plt.title(args.plot_title, fontsize=16, fontweight='semibold', pad=18)
    plt.tight_layout()
    plt.savefig(f"hbond_mutation_heat_{args.suffix}", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot ELEN scores for mutated residues, generate CSV of deltas, and "
            "show heatmap of deltas + annotated amino acids, using a fixed_positions.jsonl file "
            "to determine which residues are fixed (and thus not mutated)."
        )
    )
    parser.add_argument('--input_csv', type=str, required=True,
                        help="Input CSV file containing ELEN scores.")
    parser.add_argument('--reference_tag', type=str, required=True,
                        help="Identifier (regex) for the reference (original) structure in 'fname_pdb'.")
    parser.add_argument('--reference_pdb', type=str, required=True,
                        help="Name of the reference PDB file (within the scored_elen_models directory).")
    parser.add_argument('--fixed_positions', type=str, required=True,
                        help="JSONL file specifying which residues were fixed (not mutated).")
    parser.add_argument('--structure_id', type=str, default='4uos',
                        help="Key in the JSONL file for the target structure (e.g. '4uos').")
    parser.add_argument('--chain_id', type=str, default='A',
                        help="Chain ID to consider for fixed/mutated residues.")
    parser.add_argument('--scored_elen_models', type=str, required=True,
                        help="Directory containing the PDB files (including mutants).")
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--plot_title', type=str, default='', help="Optional title to display above plots.")
    parser.add_argument('--panel_label', type=str, default='', help="Optional panel label (e.g., 'A', 'B') to display at left.")
    args = parser.parse_args()
    main(args)
