#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3

import os
import sys
import glob
import math
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

# PyRosetta imports
try:
    import pyrosetta
    from pyrosetta import pose_from_pdb
    from rosetta.core.scoring import calc_per_res_sasa
except ImportError:
    print("PyRosetta not found. Please install it: pip install pyrosetta")
    #sys.exit(1)

def parse_pdb_file(filename):
    """
    Parse ATOM lines from a PDB file, returning a list of tuples:
        (chain_id, residue_number, residue_name, ELEN_score)
    The 'ELEN_score' is read from the column where B-factor normally resides (columns [60:66]).
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain_id = line[21].strip()                    # chain is column 21
                res_number_str = line[22:26].strip()           # residue number is columns [22:26]
                res_name = line[17:20].strip()                 # residue name is columns [17:20]
                elen_score_str = line[60:66].strip()           # ELEN score is columns [60:66]

                # Convert residue number to integer
                try:
                    res_number = int(res_number_str)
                except ValueError:
                    continue

                # Convert ELEN score to float
                try:
                    elen_score = float(elen_score_str)
                except ValueError:
                    continue

                if math.isnan(elen_score):
                    continue

                data.append((chain_id, res_number, res_name, elen_score))

    return data

def main(args):
    # Initialize PyRosetta (do this once)
    pyrosetta.init(options="-ignore_unrecognized_res true -ignore_zero_occupancy false")

    # Collect PDB files containing "RP" in the name
    pdb_files = glob.glob(os.path.join(args.pdb_dir, "*RP*.pdb"))
    if not pdb_files:
        print(f"No PDB files found in directory: {args.pdb_dir}")
        return

    # Dictionary to store ELEN scores by (res_name, 'surface') or (res_name, 'buried')
    # e.g. elen_score_dict[(res_name, "surface")] = [list_of_elen_scores]
    elen_score_dict = defaultdict(list)

    # We'll define a simple threshold for surface classification
    surface_threshold = 30.0  # Adjust as needed

    for pdb_file in pdb_files:
        # Parse data from the PDB file (chain, residue_number, residue_name, elen_score)
        parsed_data = parse_pdb_file(pdb_file)

        # Load the same PDB with PyRosetta and calculate per-residue SASA
        try:
            pose = pose_from_pdb(pdb_file)
        except Exception as e:
            print(f"Could not load {pdb_file} into PyRosetta. Skipping. Error: {e}")
            continue

        # Calculate per-residue SASA
        # 'True' here indicates that the function will do a fast (approximate) calculation
        per_res_sasa = calc_per_res_sasa(pose, True)

        # Build a helper dict to map (chain, pdb_resnum) -> pose_res_index
        # Naive approach: we rely on `pose.pdb_info().chain(i)` and `pose.pdb_info().number(i)`
        # to match the chain and residue numbers from the original PDB
        pose_map = {}
        for i in range(1, pose.size() + 1):
            c = pose.pdb_info().chain(i)
            n = pose.pdb_info().number(i)
            pose_map[(c, n)] = i

        # Now assign each atom line's residue to either surface or buried
        for (chain_id, res_num, res_name, elen_score) in parsed_data:
            if (chain_id, res_num) not in pose_map:
                # Could not map this residue to the pose (common if mismatch in chain or numbering)
                continue

            pose_res_i = pose_map[(chain_id, res_num)]
            sasa_value = per_res_sasa[pose_res_i]

            # Classify residue as surface or buried
            if sasa_value >= surface_threshold:
                location_class = "surface"
            else:
                location_class = "buried"

            elen_score_dict[(res_name, location_class)].append(elen_score)

    # Now, for each (res_name, location_class), compute average ELEN score
    # We'll keep track of distinct residue names so we can produce a grouped bar chart
    residue_names = set()
    for (res_name, loc_class) in elen_score_dict.keys():
        residue_names.add(res_name)
    residue_names = sorted(list(residue_names))

    # We'll gather average surface and average buried for each residue in separate lists
    avg_surface = []
    avg_buried = []

    for res_name in residue_names:
        surface_scores = elen_score_dict.get((res_name, "surface"), [])
        buried_scores  = elen_score_dict.get((res_name, "buried"), [])

        # Mean of the scores if we have them, else 0 or None (up to you)
        if surface_scores:
            avg_surf = sum(surface_scores) / len(surface_scores)
        else:
            avg_surf = 0.0  # or float('nan')

        if buried_scores:
            avg_bur = sum(buried_scores) / len(buried_scores)
        else:
            avg_bur = 0.0  # or float('nan')

        avg_surface.append(avg_surf)
        avg_buried.append(avg_bur)

    # -----------
    # First plot: Grouped bar chart comparing average ELEN scores for surface vs. buried
    # -----------
    plt.figure()
    indices = range(len(residue_names))
    bar_width = 0.4

    plt.bar(indices, avg_surface, width=bar_width, label="Surface")
    plt.bar([i + bar_width for i in indices], avg_buried, width=bar_width, label="Buried")

    plt.xlabel("Residue Name")
    plt.ylabel("Average ELEN Score")
    plt.title("Average ELEN Score by Residue (Surface vs. Buried)")
    plt.xticks([i + bar_width/2 for i in indices], residue_names, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    print(f"Bar plot saved to {args.output_plot}")

    # -----------
    # Second plot: Scatter plot of average ELEN score vs. approximate side-chain H-bond capacity
    # We'll separate surface vs. buried points.
    # -----------
    # Approximate maximum hydrogen bond capacity (side chain). Adjust if needed.
    hbond_capacity_dict = {
        "ALA": 0,
        "ARG": 6,
        "ASN": 4,
        "ASP": 4,
        "CYS": 2,
        "GLN": 4,
        "GLU": 4,
        "GLY": 0,
        "HIS": 5,
        "ILE": 0,
        "LEU": 0,
        "LYS": 3,
        "MET": 1,
        "PHE": 0,
        "PRO": 0,
        "SER": 3,
        "THR": 3,
        "TRP": 1,
        "TYR": 3,
        "VAL": 0
    }

    # We'll make two sets of points: surface and buried.
    # Use the same residue_names and the average surface/buried arrays we already computed.
    # If a residue isn't in hbond dict, skip it.
    x_surface, y_surface, label_surface = [], [], []
    x_buried, y_buried, label_buried = [], [], []

    for i, r in enumerate(residue_names):
        if r not in hbond_capacity_dict:
            continue
        # surface
        if avg_surface[i] != 0.0:  # or check for NaN if you used that
            x_surface.append(hbond_capacity_dict[r])
            y_surface.append(avg_surface[i])
            label_surface.append(r)
        # buried
        if avg_buried[i] != 0.0:
            x_buried.append(hbond_capacity_dict[r])
            y_buried.append(avg_buried[i])
            label_buried.append(r)

    plt.figure()
    plt.scatter(x_surface, y_surface, label="Surface")
    plt.scatter(x_buried, y_buried, label="Buried")
    plt.xlabel("Approx. Max # of H-bonds (side chain)")
    plt.ylabel("Average ELEN Score")
    plt.title("Avg ELEN Score vs. H-Bond Capacity (Surface vs. Buried)")
    plt.legend()

    # Optionally, label each point with the residue name
    for xi, yi, lab in zip(x_surface, y_surface, label_surface):
        plt.text(xi, yi, lab)
    for xi, yi, lab in zip(x_buried, y_buried, label_buried):
        plt.text(xi, yi, lab)

    plt.tight_layout()
    scatter_output = args.output_plot.replace(".png", "_hbond.png")
    plt.savefig(scatter_output, dpi=150)
    print(f"Scatter plot saved to {scatter_output}")

###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute average ELEN scores per residue identity across multiple PDB files, "
                    "classify them as surface or buried via PyRosetta SASA, and plot the results."
    )
    parser.add_argument(
        "--pdb_dir",
        required=True,
        help="Path to directory containing one or more .pdb files."
    )
    parser.add_argument(
        "--output_plot",
        default="avg_elen_score_per_residue.png",
        help="Filename for the primary output bar plot (PNG). A second scatter plot is also generated."
    )
    args = parser.parse_args()

    main(args)