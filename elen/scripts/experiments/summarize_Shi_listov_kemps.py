#!/usr/bin/env python3
"""
plot_elen_heatmap_v2_split_vertical_tight.py

Splits sequence into 4 vertical chunks.
For each chunk plots xtal B-factors (top row) and model ELEN scores (below),
with separate colormaps sharing x-axis, minimal vertical spacing,
and both colorbars on the right stacked vertically (xtal on top, model below).

Usage:
    python plot_elen_heatmap_v2_split_vertical_tight.py --xtal xtal.pdb --models-dir ./models/ --chain A --out heatmap.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

_three_to_one = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'SEC':'U','PYL':'O','ASX':'B','GLX':'Z','XLE':'J','UNK':'X'
}

def parse_pdb(pdb_path, chain_id='A'):
    res_b = {}
    res_aa = {}
    with open(pdb_path) as f:
        for L in f:
            if not L.startswith(('ATOM','HETATM')):
                continue
            ch = L[21].strip()
            if ch != chain_id:
                continue
            resnum = int(L[22:26])
            if resnum in res_b:
                continue
            resname = L[17:20].strip()
            bfac = float(L[60:66])
            res_b[resnum] = bfac
            res_aa[resnum] = _three_to_one.get(resname, 'X')
    return res_aa, res_b

def apply_scientific_style():
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.2,
        "figure.dpi": 220,
        "savefig.dpi": 330,
        "font.family": "sans-serif",
        "font.weight": "bold",
        "axes.grid": False,
    })

def main(args):
    apply_scientific_style()

    xtal_aa, xtal_b = parse_pdb(args.xtal, chain_id=args.chain)
    all_resnums = sorted(set(xtal_aa.keys()))
    min_res = min(all_resnums)
    max_res = max(all_resnums)
    full_resnums = list(range(min_res, max_res + 1))

    xtal_seq = [xtal_aa.get(r, 'X') for r in full_resnums]
    xtal_b_full = [xtal_b.get(r, 0.0) for r in full_resnums]

    model_files = sorted([
        fn for fn in os.listdir(args.models_dir)
        if fn.lower().endswith('.pdb')
    ])
    model_names = []
    elen_scores = []
    model_seqs = []

    for fn in model_files:
        model_aa, model_b = parse_pdb(os.path.join(args.models_dir, fn), chain_id=args.chain)
        seq_aligned = [model_aa.get(r, 'X') for r in full_resnums]
        b_aligned = [model_b.get(r, 0.0) for r in full_resnums]
        model_names.append(fn)
        model_seqs.append(seq_aligned)
        elen_scores.append(b_aligned)

    n_res = len(full_resnums)
    n_models = len(model_names)

    all_data = np.vstack([np.array(xtal_b_full)[np.newaxis, :], np.array(elen_scores)])
    all_seqs = [xtal_seq] + model_seqs
    row_labels = ['Xtal (B-factor)'] + [f'{m} (ELEN)' for m in model_names]

    n_chunks = args.n_chunks
    chunk_size = (n_res + n_chunks - 1) // n_chunks  # ceil division

    fig_width = max(8, chunk_size * 0.30)
    fig_height = (n_models + 1) * n_chunks * 0.40

    # Reserve extra space on the right for two vertical colorbars
    fig = plt.figure(figsize=(fig_width + 2.8, fig_height))
    main_grid = plt.GridSpec(n_chunks, 1, left=0.08, right=0.82, hspace=0.18, wspace=0.06)

    # Color limits
    xtal_nonzero = all_data[0, all_data[0, :] > 0]
    xtal_vmin, xtal_vmax = (0, 1)
    if len(xtal_nonzero) > 0:
        xtal_vmin = np.percentile(xtal_nonzero, 2)
        xtal_vmax = np.percentile(xtal_nonzero, 98)

    models_nonzero = all_data[1:, :][all_data[1:, :] > 0]
    model_vmin, model_vmax = (0, 1)
    if len(models_nonzero) > 0:
        model_vmin = np.percentile(models_nonzero, 2)
        model_vmax = np.percentile(models_nonzero, 98)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_res)

        chunk_data = all_data[:, start:end]
        chunk_seqs = [row[start:end] for row in all_seqs]
        chunk_xticks = full_resnums[start:end]

        # Per-row colormaps: top is Blues (xtal), rest YlOrRd
        facecolors = np.empty(chunk_data.shape + (4,), dtype=float)
        norm_xtal = plt.Normalize(xtal_vmin, xtal_vmax)
        norm_model = plt.Normalize(model_vmin, model_vmax)
        facecolors[0] = plt.cm.Blues(norm_xtal(chunk_data[0]))
        for j in range(1, chunk_data.shape[0]):
            facecolors[j] = plt.cm.YlOrRd(norm_model(chunk_data[j]))

        ax = fig.add_subplot(main_grid[i, 0])
        ax.imshow(facecolors, aspect='auto', interpolation='nearest')

        # Text annotation: sequence
        for ridx, row_seq in enumerate(chunk_seqs):
            for cidx, aa in enumerate(row_seq):
                ax.text(cidx, ridx, aa, fontsize=8, weight='bold', ha='center', va='center', color='black' if facecolors[ridx, cidx, :3].mean() > 0.6 else 'white')

        # Row labels
        ax.set_yticks(np.arange(chunk_data.shape[0]))
        ax.set_yticklabels(row_labels if i == 0 else ['']*len(row_labels), fontsize=12, rotation=0)
        # X tick labels only at bottom
        if i == n_chunks - 1:
            ax.set_xticks(np.arange(end-start))
            ax.set_xticklabels(chunk_xticks, rotation=90, fontsize=12)
        else:
            ax.set_xticks([])

        ax.set_ylabel('' if i > 0 else 'Model', fontsize=20, fontweight='semibold')
        ax.set_xlabel('Residue Number' if i == n_chunks - 1 else '', fontsize=20, fontweight='semibold')

        # Remove ticks, spines for clean look
        ax.tick_params(axis='x', bottom=False, top=False)
        ax.tick_params(axis='y', left=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- Colorbars, side-by-side axes, stacked ---
    # Position in figure coordinates (0,0)-(1,1)
    cbar_width = 0.02
    cbar_height = 0.36
    cbar_pad = 0.025

    # Top colorbar for Xtal (blue)
    cbar_ax_xtal = fig.add_axes([0.86, 0.54, cbar_width, cbar_height])
    sm_xtal = plt.cm.ScalarMappable(cmap='Blues', norm=norm_xtal)
    sm_xtal.set_array([])
    cbar_xtal = fig.colorbar(sm_xtal, cax=cbar_ax_xtal, orientation='vertical')
    cbar_xtal.set_label('Xtal B-factor', fontsize=16, fontweight='semibold', labelpad=10)

    # Bottom colorbar for Model (YlOrRd)
    cbar_ax_model = fig.add_axes([0.86, 0.10, cbar_width, cbar_height])
    sm_model = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm_model)
    sm_model.set_array([])
    cbar_model = fig.colorbar(sm_model, cax=cbar_ax_model, orientation='vertical')
    cbar_model.set_label('Model ELEN score', fontsize=16, fontweight='semibold', labelpad=10)

    plt.suptitle("Xtal B-factors and Model ELEN Scores per Residue", fontsize=20, fontweight='bold', y=0.92)
    plt.tight_layout(rect=[0, 0, 0.84, 0.97])

    plt.savefig(args.out, bbox_inches='tight', transparent=True)
    print(f"Wrote heatmap to {args.out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="plot_elen_heatmap_v2_split_vertical_tight.py", description="ELEN/B-factor heatmap vertical split with tight layout and stacked colorbars")
    parser.add_argument('--xtal', required=True, help="Reference xtal structure PDB file (with B-factors)")
    parser.add_argument('--models-dir', required=True, help="Directory of model PDB files (ELEN scores in B-factor column)")
    parser.add_argument('--chain', default='A', help="Chain identifier to parse (default: A)")
    parser.add_argument('--out', default='elen_heatmap_shi_xtal.png', help="Output image file")
    parser.add_argument('--n_chunks', type=int, default=4, help="Number of image splits")
    args = parser.parse_args()
    main(args)
