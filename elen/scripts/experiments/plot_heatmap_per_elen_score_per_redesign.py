#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
#TODO rework to look like other heatmap plot
#TODO rework to have panel naems
#TODO fix too high values
"""
plot_elen_heatmap.py

Read a native PDB and a folder of model PDBs, extract per-residue ELEN scores
from the B-factor column, compute ΔELEN = model - native, and plot a heatmap
with one-letter sequence codes overlaid.

Dependencies:
    pandas, matplotlib, seaborn
"""
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# standard three→one AA code map
_three_to_one = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    # others:
    'SEC':'U','PYL':'O','ASX':'B','GLX':'Z','XLE':'J','UNK':'X'
}

def parse_pdb(pdb_path, chain_id='A'):
    """
    Parse a PDB, extract for the given chain:
      - ordered list of residue numbers
      - one-letter sequence
      - per-residue B-factor (first atom seen)
    """
    res_b   = {}
    res_aa  = {}
    seen    = []

    with open(pdb_path) as f:
        for L in f:
            if not L.startswith(('ATOM','HETATM')): 
                continue
            ch = L[21].strip()
            if ch != chain_id:
                continue

            resnum = int(L[22:26])
            if resnum in res_b:
                continue  # already saw this residue
            resname = L[17:20].strip()
            bfac    = float(L[60:66])

            res_b[resnum]  = bfac
            res_aa[resnum] = _three_to_one.get(resname, 'X')
            seen.append(resnum)

    # sort by residue number
    resnums = sorted(seen)
    seq     = [res_aa[r] for r in resnums]
    bfacs   = [res_b[r]  for r in resnums]

    return resnums, seq, bfacs

def main(args):
    # parse native
    print(f"Parsing native: {args.native}")
    ref_resnums, seq, ref_b = parse_pdb(args.native, chain_id=args.chain)
    print(len(ref_resnums))
    print(len(seq))
    print(len(ref_b))
    
    # collect ΔELEN for each model
    deltas = []
    model_names = []

    files = sorted(os.listdir(args.models_dir))
    for fn in files:
        if not fn.lower().endswith('.pdb'):
            continue
        path = os.path.join(args.models_dir, fn)
        print(f"  → {fn}")
        resnums, seq2, b = parse_pdb(path, chain_id=args.chain)

        if resnums != ref_resnums:
            raise ValueError(
                f"Residue numbering mismatch in {fn}: "
                f"{resnums[:3]}… vs {ref_resnums[:3]}…"
            )
        # compute Δ
        delta = np.array(b) - np.array(ref_b)
        deltas.append(delta)
        model_names.append(fn)

    # build DataFrame
    df = pd.DataFrame(
        data = deltas,
        index = model_names,
        columns = ref_resnums
    )
    print(df)
    
    # build annotation grid of sequence letters
    annot = pd.DataFrame(
        [seq]*len(model_names),
        index = model_names,
        columns = ref_resnums
    )

    # plot
    plt.figure(
        figsize = (max(8, len(ref_resnums)*0.2), 
                   max(4, len(model_names)*0.5))
    )
    sns.heatmap(
        df,
        cmap        = 'RdBu_r',
        center      = 0,
        annot       = annot,
        fmt         = '',
        cbar_kws    = {'label': 'ΔELEN Score'}
    )
    plt.xlabel('Residue Number')
    plt.ylabel('Model PDB')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Wrote heatmap to {args.out}")

###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="plot_elen_heatmap.py", description="ΔELEN heatmap from PDB B-factors")
    parser.add_argument('--native','-n', required=True, help="Native/reference PDB file")
    parser.add_argument('--models-dir','-i', required=True, help="Directory containing model PDB files")
    parser.add_argument('--chain','-c', default='A', help="Chain identifier to parse (default: A)")
    parser.add_argument('--out','-o', default='heatmap.png', help="Output image file (default: heatmap.png)")
    args = parser.parse_args()
    main(args)
