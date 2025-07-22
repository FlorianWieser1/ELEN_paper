#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J ELEN_inference
#SBATCH -o ELEN_inference_%j.log
#SBATCH -e ELEN_inference_%j.err

import os
import re
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

"""
Example script: parse_hbond_and_score.py

Now updated to:
  1) Produce both a scatter plot and a violin plot.
  2) Filter out any rows where the 'bfactor_score' is zero (0.0).

Steps:
  1) Recursively read all PDB files from a given folder (specified by --inpath).
  2) Parse each file to extract:
      - The "hbonds_XA Y" lines (where X is residue number, A is chain, Y is hbond count).
      - The B-factor (which we assume is the 'ELEN-Score'), 
        taken only from 'CA' atoms, so that we have a single score per residue.
  3) Combine that data into a single pandas DataFrame with columns:
       ['pdb_file', 'residue_number', 'chain', 'hbonds', 'bfactor_score'].
  4) Remove any rows where 'bfactor_score' is 0.0 (score cannot be zero).
  5) Plot a scatter plot of number of hbonds (x) vs ELEN-Score (y).
  6) Plot a violin plot grouping data by the number of hbonds.
"""

def parse_hbonds_from_pdb(lines):
    """
    Parse lines of a PDB file and collect the residue -> hbond_count from lines like:
        hbonds_9A 2
      This means residue 9 on chain A has 2 hbonds.

    Return a dict:
       { (chain, residue_number) : hbond_count, ... }
    """
    hbond_pattern = re.compile(r"^hbonds_(\d+)([A-Za-z])\s+(\d+)$")
    residue_hbonds = {}
    for line in lines:
        line = line.strip()
        match = hbond_pattern.match(line)
        if match:
            resid_str, chain, hbond_str = match.groups()
            resid = int(resid_str)
            hbond = int(hbond_str)
            residue_hbonds[(chain, resid)] = hbond
    return residue_hbonds


def parse_score_from_pdb(lines):
    """
    Parse the 'score' (which we assume is stored in the B-factor column)
    from only CA atoms in ATOM lines. (So we pick the B-factor of CA.)
    Return a dict of { (chain, residue_number) : bfactor_score, ... }
    """
    residue_scores = {}

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue  # skip non-CA
            chain = line[21].strip()
            resid_str = line[22:26].strip()
            bfactor_str = line[60:66].strip()

            if not resid_str.isdigit():
                continue
            resid = int(resid_str)
            try:
                bfactor = float(bfactor_str)
            except ValueError:
                continue

            key = (chain, resid)
            residue_scores[key] = bfactor  # store CA's bfactor

    return residue_scores


def collect_data_from_pdb(infile):
    """
    Parse a single PDB file to retrieve (chain,resid)->hbond_count
    and (chain,resid)->bfactor_score.
    Combine them row by row.
    Return a list of rows:
        [pdb_file, residue_number, chain, hbonds, bfactor_score]
    """
    with open(infile, "r") as f:
        lines = f.readlines()

    # parse
    hbonds_dict = parse_hbonds_from_pdb(lines)
    bfactor_dict = parse_score_from_pdb(lines)

    # combine
    rows = []
    all_keys = set(hbonds_dict.keys()) | set(bfactor_dict.keys())
    for key in all_keys:
        chain, resid = key
        hbond_count = hbonds_dict.get(key, 0)
        bfactor = bfactor_dict.get(key, None)
        rows.append([os.path.basename(infile), resid, chain, hbond_count, bfactor])
    return rows


def main(args):
    # Gather .pdb files (non-recursive)
    pdb_files = []
    for fname in os.listdir(args.inpath):
        if fname.lower().endswith(".pdb"):
            fullpath = os.path.join(args.inpath, fname)
            pdb_files.append(fullpath)

    # Collect data
    all_data = []
    for pdbf in pdb_files:
        data_rows = collect_data_from_pdb(pdbf)
        all_data.extend(data_rows)

    df = pd.DataFrame(all_data, columns=["pdb_file", "residue_number", "chain", "hbonds", "bfactor_score"])
    # Drop rows that have no bfactor_score or are 0.0
    df = df.dropna(subset=["bfactor_score"])         # drop missing scores
    df = df[df["bfactor_score"] != 0.0]              # also drop 0.0

    # 1) SCATTER PLOT
    plt.figure(figsize=(6, 4))
    plt.scatter(
        df["hbonds"], 
        df["bfactor_score"], 
        alpha=0.5,     # half transparent
        c="black",     # black color
        s=10           # smaller marker size
    )
    plt.xlabel("Number of hbonds per residue")
    plt.ylabel("ELEN Score")
    plt.tight_layout()
    plt.savefig("score_vs_hbonds_scatter.png", dpi=150)
    plt.close()

    # 2) VIOLIN PLOT
    # Group by "hbonds" and gather bfactor_scores
    grouped_scores = df.groupby("hbonds")["bfactor_score"].apply(list).to_dict()
    sorted_hbond_counts = sorted(grouped_scores.keys())
    data_for_violin = [grouped_scores[k] for k in sorted_hbond_counts]
    positions = range(1, len(sorted_hbond_counts) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(
        data_for_violin,
        positions=positions,
        showmeans=False,
        showextrema=True,
        showmedians=True
    )

    # styling for the violin
    for pc in parts["bodies"]:
        pc.set_facecolor("black")
        pc.set_alpha(0.5)
    # median lines
    if "cmedians" in parts:
        parts["cmedians"].set_edgecolor("black")
        parts["cmedians"].set_linewidth(1)
    # other lines
    for part_name in ("cbars", "cmins", "cmaxes"):
        if part_name in parts:
            parts[part_name].set_edgecolor("black")
            parts[part_name].set_linewidth(1)

    ax.set_xticks(positions)
    ax.set_xticklabels(sorted_hbond_counts)
    ax.set_xlabel("Number of hbonds per residue")
    ax.set_ylabel("ELEN Score")

    plt.tight_layout()
    plt.savefig("score_vs_hbonds_violin.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect residue-level hbond counts and CA B-factor scores from PDBs, then produce scatter+violin plots.")
    parser.add_argument("--inpath", required=True, help="Folder containing .pdb files to parse.")
    args = parser.parse_args()
    main(args)
