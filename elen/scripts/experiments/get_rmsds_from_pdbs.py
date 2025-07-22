#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
import argparse
import os
import pandas as pd
import re


# Function to parse RMSD values from PDB file
def parse_rmsd_values(pdb_file):
    rmsd_dict = {}
    with open(pdb_file, 'r') as file:
        for line in file:
            match = re.match(r'res_rmsd_(\d+)\s+(\d+\.\d+)', line)
            if match:
                res_id = int(match.group(1))
                rmsd_value = float(match.group(2))
                rmsd_dict[res_id] = rmsd_value
    return rmsd_dict


# Collect RMSD values from all PDB files in a directory
def collect_rmsd(directory):
    pdb_files = [f for f in os.listdir(directory) if f.endswith('.pdb')]
    all_data = []

    for pdb_file in pdb_files:
        full_path = os.path.join(directory, pdb_file)
        rmsd_values = parse_rmsd_values(full_path)

        avg_all = pd.Series(rmsd_values).mean()
        avg_region = pd.Series({k: v for k, v in rmsd_values.items() if 180 <= k <= 190}).mean()
        avg_except_region = pd.Series({k: v for k, v in rmsd_values.items() if not (180 <= k <= 190)}).mean()

        all_data.append({
            'model': pdb_file,
            'avg_rmsd_all': avg_all,
            'avg_rmsd_180_190': avg_region,
            'avg_rmsd_except_180_190': avg_except_region
        })

    return pd.DataFrame(all_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and average RMSD values from PDB files.")
    parser.add_argument('-d', '--directory', required=True, help="Directory containing PDB files")
    parser.add_argument('-o', '--output_csv', required=False, default='rmsd_summary.csv', help="Output CSV file")

    args = parser.parse_args()

    rmsd_df = collect_rmsd(args.directory)
    rmsd_df.to_csv(args.output_csv, index=False)

    print(f"RMSD summary written to {args.output_csv}")
