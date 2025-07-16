#!/usr/bin/env python3
import os
import glob
import shutil
import matplotlib.pyplot as plt
import argparse

def count_residues_in_pdb(file_path):
    """
    Count the number of unique residues in a PDB file.

    The function reads each line of the file and processes lines that start with "ATOM".
    It extracts the chain identifier (column 22), the residue sequence number (columns 23-26),
    and the insertion code (column 27) to form a unique identifier for each residue.
    """
    residues = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    # Ensure the line is long enough to include the necessary columns
                    if len(line) < 27:
                        continue
                    chain = line[21].strip()          # Column 22 (index 21)
                    res_seq = line[22:26].strip()       # Columns 23-26 (indices 22-26)
                    insertion_code = line[26].strip()   # Column 27 (index 26)
                    residue_id = (chain, res_seq, insertion_code)
                    residues.add(residue_id)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return len(residues)

###############################################################################
def main(args):
    input_folder = args.input_folder
    pdb_files = glob.glob(os.path.join(input_folder, "*.pdb"))
    
    if not pdb_files:
        print(f"No .pdb files found in folder: {input_folder}")
        return

    # Create the filtered folder inside the input folder if it doesn't exist.
    filtered_folder = os.path.join(input_folder, args.filtered_dir)
    if not os.path.exists(filtered_folder):
        os.makedirs(filtered_folder)

    residue_counts = []
    print("Processing files:")
    for pdb_file in pdb_files:
        count = count_residues_in_pdb(pdb_file)
        residue_counts.append(count)
        print(f"  {os.path.basename(pdb_file)}: {count} residues")
        
        # If the residue count is greater than the threshold, move the file.
        if count > args.threshold:
            dest = os.path.join(filtered_folder, os.path.basename(pdb_file))
            print(f"    Moving {os.path.basename(pdb_file)} to '{filtered_folder}'")
            try:
                shutil.move(pdb_file, dest)
            except Exception as e:
                print(f"Error moving {pdb_file}: {e}")

    # Plot histogram if not disabled.
    if not args.no_plot:
        plt.figure(figsize=(8, 6))
        plt.hist(residue_counts, bins=args.bins, edgecolor='black')
        plt.xlabel("Number of Residues per File")
        plt.ylabel("Frequency (Number of Files)")
        plt.title("Histogram of Residue Counts in PDB Files")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Plotting disabled (--no-plot flag set).")
        
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count residues in PDB files, filter those with residue counts above a threshold, and plot a histogram.")
    parser.add_argument("--input_folder",
                        help="Path to the input folder containing .pdb files")
    parser.add_argument("--threshold", type=int, default=500,
                        help="Residue count threshold for filtering PDB files (default: 500)")
    parser.add_argument("--bins", type=int, default=20,
                        help="Number of bins for the histogram (default: 20)")
    parser.add_argument("--filtered_dir", default="filtered",
                        help="Name of the folder where filtered files will be moved (default: filtered)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Do not display the histogram plot")
    
    args = parser.parse_args()
    main(args)
