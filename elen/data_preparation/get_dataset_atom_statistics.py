#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J count_elements
#SBATCH -o count_elements.log
#SBATCH -e count_elements.err
import sys
import os
import atom3d.datasets as da
from collections import defaultdict
import pandas as pd
import argparse as ap

def main(args):
    """
    Loop through PDB files in 'folder' using atom3d,
    count the occurrences of each element, and write results to CSV.
    """
    # Dictionary to keep track of element -> count
    element_counts = defaultdict(int)

    # Load the entire folder as a PDB dataset using atom3d
    dataset = da.load_dataset(args.inpath, filetype='pdb')

    # Loop over each structure in the dataset
    for item in dataset:
        # item['atoms'] is a pandas DataFrame with columns like "element", "name", etc.
        elements = item['atoms']['element'].to_numpy()
        for el in elements:
            element_counts[el] += 1

    # Convert counts into a DataFrame
    data = [{'element': e, 'count': c} for e, c in element_counts.items()]
    df = pd.DataFrame(data)
    # Sort by element name for clarity
    df = df.sort_values(by='element')

    # Write to CSV in the same folder (or change to a custom path if needed)
    tag = str(args.inpath).rstrip("/")
    print(f"tag {tag}")
    
    out_file = f"{tag}_element_counts.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved element counts to {out_file}")

###########################################################################
if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--inpath', type=str, default="AF3_models")
    args = parser.parse_args()
    main(args)
