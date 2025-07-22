#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J get_features
#SBATCH -o get_features.log
#SBATCH -e get_features.err
#TODO add sap-score, SS,
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

from pyrosetta import init, pose_from_pdb, create_score_function
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector, ResidueSpanSelector
from pyrosetta.rosetta.core.simple_metrics.per_residue_metrics import (
    HbondMetric,
    SidechainNeighborCountMetric,
    PerResidueSasaMetric,
    PerResidueEnergyMetric,
)
from pyrosetta.rosetta.protocols.moves import DsspMover
from pyrosetta.rosetta.core.sequence import ABEGOManager
from elen.shared_utils.constants import AA_THREE_TO_ONE


###############################################################################
# Utility and feature functions
###############################################################################
def read_loop_positions_from_pdb(pdb_file):
    """
    Reads the 'loop_position' line at the end of the PDB to identify loop start/end.
    Example line:  loop_position 4 5
    Returns (start, end) as integers or (None, None) if not found.
    """
    loop_start, loop_end = None, None
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("loop_position "):
                tokens = line.strip().split()
                try:
                    loop_start, loop_end = int(tokens[1]), int(tokens[2])
                except (IndexError, ValueError):
                    pass
                break
    return loop_start, loop_end


def getChi1(pose, res):
    if pose.residue(res).nchi() >= 1:
        return pose.chi(1, res)
    return np.nan


def getChi2(pose, res):
    if pose.residue(res).nchi() >= 2:
        return pose.chi(2, res)
    return np.nan


def to_list(pyro_map):
    return list(dict(pyro_map.items()).values())


def getSingleMetric(pose, metric, res):
    selector = ResidueIndexSelector(res)
    metric.set_residue_selector(selector)
    return to_list(metric.calculate(pose))[0]


def getSequenceRange(pose, start, stop):
    sequence = []
    for r in range(start, stop + 1):
        sequence.append(pose.residue(r).name1())
    return "".join(sequence)


def stripRosettaName(full_resname):
    return full_resname.split(':')[0]


def getABEGOScheme(pose, res):
    abego_manager = ABEGOManager()
    idx = abego_manager.torsion2index_level1(
        pose.phi(res), pose.psi(res), pose.omega(res)
    )
    return abego_manager.index2symbol(idx)


def getABEGORange(pose, start, stop):
    return "".join(getABEGOScheme(pose, r) for r in range(start, stop + 1))


def getSMFeatures(pose, metric, tag, feat, cols, ccap, ncap):
    loop_selector = ResidueSpanSelector(ccap, ncap)
    metric.set_residue_selector(loop_selector)
    loop_vals = to_list(metric.calculate(pose))
    loop_mean = np.mean(loop_vals)
    val_ccap = getSingleMetric(pose, metric, ccap)
    val_ncap = getSingleMetric(pose, metric, ncap)

    cols += [f"{tag}_loop", f"{tag}_ccap", f"{tag}_ncap"]
    feat += [loop_mean, val_ccap, val_ncap]
    return feat, cols


def generate_features(pose, pdb_name, nres, ccap, ncap):
    feat = []
    cols = []

    # 1) General info
    feat += [pdb_name, nres, ccap, ncap]
    cols += ["id", "nres", "ccap", "ncap"]

    # 2) Sequence / side-chain identity
    id_loop = getSequenceRange(pose, ccap + 1, ncap + 1)
    ccap_base = stripRosettaName(pose.residue(ccap + 1).name())
    ncap_base = stripRosettaName(pose.residue(ncap + 1).name())
    id_ccap = AA_THREE_TO_ONE[ccap_base]
    id_ncap = AA_THREE_TO_ONE[ncap_base]
    feat += [id_loop, id_ccap, id_ncap]
    cols += ["id_loop", "id_ccap", "id_ncap"]

    # 3) ABEGO
    ab_loop = getABEGORange(pose, ccap, ncap)
    ab_ccap = getABEGOScheme(pose, ccap)
    ab_ncap = getABEGOScheme(pose, ncap)
    feat += [ab_loop, ab_ccap, ab_ncap]
    cols += ["ab_loop", "ab_ccap", "ab_ncap"]

    # 4) Dihedrals
    feat += [pose.phi(ccap), pose.phi(ncap)]
    feat += [pose.psi(ccap), pose.psi(ncap)]
    feat += [getChi1(pose, ccap), getChi1(pose, ncap)]
    feat += [getChi2(pose, ccap), getChi2(pose, ncap)]
    cols += [
        "phi_ccap", "phi_ncap",
        "psi_ccap", "psi_ncap",
        "chi1_ccap", "chi1_ncap",
        "chi2_ccap", "chi2_ncap",
    ]

    # 5) Per-residue metrics aggregated over the loop
    metrics = [
        (HbondMetric(), "hb"),
        (SidechainNeighborCountMetric(), "scnc"),
        (PerResidueSasaMetric(), "prsm"),
        (PerResidueEnergyMetric(), "E")
    ]
    for metric_obj, tag in metrics:
        feat, cols = getSMFeatures(pose, metric_obj, tag, feat, cols, ccap, ncap)

    return feat, cols


def generate_per_residue_features(pose, pdb_name):
    """
    For each residue in the pose, compute a set of features and return rows.
    """
    rows = []
    cols = [
        "id", "residue_index", "res_name", "abego",
        "phi", "psi", "chi1", "chi2",
        "hb", "scnc", "prsm", "E"
    ]
    # metrics to compute per residue
    metric_classes = [
        (HbondMetric, "hb"),
        (SidechainNeighborCountMetric, "scnc"),
        (PerResidueSasaMetric, "prsm"),
        (PerResidueEnergyMetric, "E"),
    ]

    total = pose.total_residue()
    for res in range(1, total + 1):
        res_name = pose.residue(res).name1()
        abego = getABEGOScheme(pose, res)
        phi = pose.phi(res)
        psi = pose.psi(res)
        chi1 = getChi1(pose, res)
        chi2 = getChi2(pose, res)

        feats = [pdb_name, res, res_name, abego, phi, psi, chi1, chi2]
        # compute each metric at this residue
        for MetricClass, tag in metric_classes:
            metric = MetricClass()
            val = getSingleMetric(pose, metric, res)
            feats.append(val)

        rows.append(feats)

    return rows, cols


def main(args):
    # Initialize PyRosetta
    init("-mute all")
    scorefxn = create_score_function("ref2015")

    # Prepare to store results
    loop_rows = []
    loop_cols = None
    res_rows = []
    res_cols = None

    pattern = os.path.join(args.inpath_extracted_loops, "*.pdb")
    pdb_files = glob.glob(pattern)

    for pdb_path in pdb_files:
        print(f"Processing PDB: {pdb_path}")
        try:
            pose = pose_from_pdb(pdb_path)
            pdb_name = os.path.basename(pdb_path)

            loop_start, loop_end = read_loop_positions_from_pdb(pdb_path)
            if loop_start is None or loop_end is None:
                print("No 'loop_position' found in PDB, skipping.")
                continue

            nres = loop_end - loop_start + 1
            ccap = loop_start
            ncap = loop_end
            if ncap - ccap < 1:
                print(f"Invalid loop region: ccap={ccap}, ncap={ncap}, skipping.")
                continue

            # per-residue features
            this_res_rows, this_res_cols = generate_per_residue_features(pose, pdb_name)
            if res_cols is None:
                res_cols = this_res_cols
            res_rows.extend(this_res_rows)

            # per-loop aggregated features
            this_loop_row, this_loop_cols = generate_features(pose, pdb_name, nres, ccap, ncap)
            if loop_cols is None:
                loop_cols = this_loop_cols
            loop_rows.append(this_loop_row)

        except Exception as e:
            print(f"Error processing {pdb_path}: {e}. Skipping.", file=sys.stderr)
            continue

    # Write per-residue features CSV
    if res_rows and res_cols:
        df_res = pd.DataFrame(res_rows, columns=res_cols)
        df_res.to_csv(args.output_per_res_csv, index=False)
        print(f"Per-residue features saved to {args.output_per_res_csv}")
    else:
        print("No per-residue data rows were collected. Check input folder and filtering conditions.")

    # Write per-loop aggregated features CSV
    if loop_rows and loop_cols:
        df_loop = pd.DataFrame(loop_rows, columns=loop_cols)
        df_loop.to_csv(args.output_loop_csv, index=False)
        print(f"Loop-aggregated features saved to {args.output_loop_csv}")
    else:
        print("No loop-aggregated data rows were collected. Check input folder and filtering conditions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute both per-residue and loop-aggregated features from loop pocket PDB files using PyRosetta."
    )
    parser.add_argument(
        "--inpath_extracted_loops", type=str, required=True,
        help="Folder containing input PDB files."
    )
    parser.add_argument(
        "--output_per_res_csv", type=str, default="per_res_features.csv",
        help="Output CSV file for per-residue features."
    )
    parser.add_argument(
        "--output_loop_csv", type=str, default="loop_features.csv",
        help="Output CSV file for loop-aggregated features."
    )
    args = parser.parse_args()
    main(args)
