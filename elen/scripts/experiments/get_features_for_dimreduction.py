#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
# Compute both per-residue and loop-aggregated features from loop pocket PDB files using PyRosetta.
# Updated to include aggregation of per-residue metrics over loop spans.
#SBATCH -J generate_features
#SBATCH -o generate_features_%j.log
#SBATCH -e generate_features_%j.err
import os
import sys
import glob
import shutil
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
from elen.shared_utils.utils_extraction import extract_loops, clean_pdb
from elen.config import PATH_ROSETTA_TOOLS

def read_loop_type_from_pdb(pdb_file):
    loop_type = ""
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("loop_type"):
                tokens = line.strip().split()
                try:
                    loop_type = str(tokens[1])
                except (IndexError, ValueError):
                    pass
                break
    return loop_type

def read_loop_positions_from_pdb(pdb_file, pattern):
    """
    Reads the 'loop_position_target' line at the end of the PDB to identify loop start/end.
    Example line:  loop_position_target 4 5
    Returns (start, end) as integers or (None, None) if not found.
    """
    loop_start, loop_end = None, None
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(pattern):
                tokens = line.strip().split()
                try:
                    loop_start, loop_end = int(tokens[1]), int(tokens[2])
                except (IndexError, ValueError):
                    pass
                break
    return loop_start, loop_end


# Helper functions for dihedrals

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
        (PerResidueEnergyMetric(), "E"),
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
        for MetricClass, tag in metric_classes:
            metric = MetricClass()
            val = getSingleMetric(pose, metric, res)
            feats.append(val)

        rows.append(feats)

    return rows, cols


def main(args):
    # Prepare output folder
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)

    # Initialize PyRosetta
    init()
    scorefxn = create_score_function("ref2015")

    # Clean .pdbs
    paths_pdbs = glob.glob(os.path.join(args.inpath, f"*.pdb"))
    for path_pdb in paths_pdbs:
        clean_pdb(path_pdb, args.outpath, PATH_ROSETTA_TOOLS)
    
    # Extract loops
    paths_pdbs = glob.glob(os.path.join(args.outpath, f"*.pdb"))
    for path_pdb in paths_pdbs:
        extract_loops(path_pdb, args.outpath, "None", 4, 10, 40)
        
    # 1) Compute or load per-residue features
    path_per_res_csv = os.path.join(args.outpath, "features_per_residue.csv")
    if os.path.exists(path_per_res_csv) and not args.overwrite:
        df_per = pd.read_csv(path_per_res_csv)
    else:
        data_per_residue = []
        for path_pdb in paths_pdbs:
            pose = pose_from_pdb(path_pdb)
            pdb_name = os.path.basename(path_pdb)
            rows, cols_per = generate_per_residue_features(pose, pdb_name)
            data_per_residue.extend(rows)
        df_per = pd.DataFrame(data_per_residue, columns=cols_per)
        df_per.to_csv(path_per_res_csv, index=False)
    
    # 2) Read loop positions
    loop_positions = {}
    for path_loop in glob.glob(os.path.join(args.outpath, "extracted_loops", "*.pdb")):
        loop_start, loop_stop = read_loop_positions_from_pdb(path_loop, "loop_position_target")
        loop_positions[os.path.basename(path_loop)] = (loop_start, loop_stop)
    print(f"loop_positions: {loop_positions}")
    # 3) Aggregate per-residue metrics over loops
    loop_rows = []
    # Define which numeric columns to average
    numeric_cols = ["phi", "psi", "chi1", "chi2", "hb", "scnc", "prsm", "E"]
    for loop_name, (start, stop) in loop_positions.items():
        print(f"here")
        if start is None or stop is None:
            continue
        # select residues in the loop span
        mask = (df_per['residue_index'] >= start) & (df_per['residue_index'] <= stop)
        print(f"mask: {mask}")
        df_span = df_per.loc[mask]
        if df_span.empty:
            continue
        # compute mean of each metric
        means = df_span[numeric_cols].mean()
        row = {'id': loop_name}
        row.update(means.to_dict())
        loop_rows.append(row)
    df_loop = pd.DataFrame(loop_rows)
    
    # add other features
    dict_other_features = {}
    other_features = [] 
    for loop_name, (start, stop) in loop_positions.items():
        nres = stop - start + 1
        path_extracted_loop = os.path.join(args.outpath, "extracted_loops", loop_name)
        loop_type = read_loop_type_from_pdb(path_extracted_loop) 
        dict_other_features[loop_name] = {'nres': nres, 'loop_type': loop_type}

        other_features = [nres, loop_type]
        dict_other_features[loop_name] = other_features
    df_other_features = pd.DataFrame.from_dict(dict_other_features, orient='index').reset_index()
    df_other_features.columns = ['id', 'nres', 'loop_type']
    
    # merge and to .csv 
    df_merged = df_loop.merge(df_other_features, on='id', how='inner')
    path_loop_csv = os.path.join(args.outpath, args.output_loop_csv)
    df_merged.to_csv(path_loop_csv, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute both per-residue and loop-aggregated features from loop pocket PDB files using PyRosetta.")
    parser.add_argument(
        "--inpath", type=str, required=True,
        help="Folder containing input PDB files."
    )
    parser.add_argument(
        "--outpath", type=str, default="generated_features",
        help="Folder containing output files."
    )
    parser.add_argument(
        "--output_per_res_csv", type=str, default="features_per_residue.csv",
        help="Output CSV file for per-residue features."
    )
    parser.add_argument(
        "--output_loop_csv", type=str, default="loop_features.csv",
        help="Output CSV file for loop-aggregated features."
    )
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
