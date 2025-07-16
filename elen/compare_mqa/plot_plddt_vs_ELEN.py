#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import linregress, pearsonr, spearmanr

def extract_bfactors(pdb_file):
    residue_bfactors = defaultdict(list)
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("ATOM"):
                res_num = int(line[22:26])
                try:
                    b = float(line[60:66])
                except ValueError:
                    continue
                residue_bfactors[res_num].append(b)
    residue_numbers = []
    bfactor_per_residue = []
    for res_num in sorted(residue_bfactors):
        residue_numbers.append(res_num)
        bfactor_per_residue.append(np.mean(residue_bfactors[res_num]))
    return residue_numbers, bfactor_per_residue

def match_files(af2_dir, elen_dir):
    af2_files = {f.split('_relaxed')[0]: f for f in os.listdir(af2_dir) if f.endswith('.pdb')}
    elen_files = {f.split('_relaxed')[0]: f for f in os.listdir(elen_dir) if f.endswith('.pdb')}
    matches = []
    for key in af2_files.keys() & elen_files.keys():
        matches.append((
            os.path.join(af2_dir, af2_files[key]),
            os.path.join(elen_dir, elen_files[key]),
            key
        ))
    return matches

def main(args):
    # Publication-style
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
        "figure.titlesize": 18,
        "axes.linewidth": 1.3,
        "axes.edgecolor": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "figure.dpi": 200,
    })

    matches = match_files(args.af2_dir, args.elen_dir)
    if not matches:
        print("No matching files found.")
        return

    all_plddt = []
    all_elen = []
    for af2_path, elen_path, _ in matches:
        _, plddt = extract_bfactors(af2_path)
        _, elen = extract_bfactors(elen_path)
        n = min(len(plddt), len(elen))
        all_plddt.extend(plddt[:n])
        all_elen.extend(elen[:n])
    
    all_plddt = np.array(all_plddt)
    all_elen = np.array(all_elen)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(all_plddt, all_elen)
    reg_x = np.linspace(all_plddt.min(), all_plddt.max(), 100)
    reg_y = slope * reg_x + intercept

    # Correlations
    pearson_corr, pearson_p = pearsonr(all_plddt, all_elen)
    spearman_corr, spearman_p = spearmanr(all_plddt, all_elen)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(True, zorder=0, alpha=0.28)
    ax.scatter(
        all_plddt, all_elen,
        s=15, color="black", zorder=2, edgecolors='none', alpha=0.75, label="Data"
    )
    ax.plot(reg_x, reg_y, color="red", linewidth=2, zorder=3, label="Linear fit")

    ax.set_xlabel("AF2 pLDDT", fontsize=16)
    ax.set_ylabel("ELEN Score", fontsize=16)

    # Correlation text
    textstr = (
        f"Pearson $r$ = {pearson_corr:.3f}\n"
        f"Spearman $\\rho$ = {spearman_corr:.3f}"
    )
    # Text box in upper left inside axes
    ax.text(
        0.04, 0.97, textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.35', alpha=0.93)
    )

    # Panel letter and title
    if args.panel_letter:
        fig.text(0.03, 0.97, args.panel_letter, fontsize=22, fontweight='bold', ha='left', va='top')
    if args.model_name:
        fig.suptitle(args.model_name, fontsize=18, fontweight='semibold', y=1.01)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(args.output_file, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved plot as {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a publication-style PLDDT vs ELEN scatter plot from two PDB directories."
    )
    parser.add_argument("af2_dir", help="Folder with AF2-relaxed PDBs (PLDDT in b-factor).")
    parser.add_argument("elen_dir", help="Folder with ELEN-scored PDBs (ELEN in b-factor).")
    parser.add_argument(
        "--output-file", "-o",
        default="plddt_vs_elen_pub.png",
        help="Filename for the output plot."
    )
    parser.add_argument(
        "--panel-letter",
        default="",
        help="Panel letter to annotate plot (e.g. 'A', 'B')."
    )
    parser.add_argument(
        "--model-name",
        default="",
        help="Model variant name for plot title."
    )
    args = parser.parse_args()
    main(args)
