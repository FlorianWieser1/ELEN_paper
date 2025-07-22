#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
# Updated: extract labels directly from LMDB (no labels.json usage)
import os
import sys
import glob
import json
import shutil
import argparse
import atom3d.datasets as da
import csv

### HELPERS ###################################################################

def load_json_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def write_to_file(path_fnames, base_fnames):
    with open(path_fnames, 'w') as f:
        for fname in base_fnames:
            f.write(fname + '\n')

def load_from_file(path_fnames):
    with open(path_fnames, 'r') as f:
        loaded_fnames = set(line.strip() for line in f)
    return loaded_fnames

### MAIN ######################################################################

def main(args):
    if os.path.exists(args.outpath) and args.overwrite:
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)

    # load dataset
    dataset_test = da.load_dataset(args.dataset_dir, 'lmdb')
    base_fnames = set()
    base_fnames_loops = set()
    path_fnames = os.path.join(args.outpath, "fnames.txt")
    path_fnames_loops = os.path.join(args.outpath, "fnames_loops.txt")
    path_labels_csv = os.path.join(args.outpath, "labels_all.csv")
    
    if not os.path.exists(path_fnames):
        print(f"Generating fnames.txt")
        # gather file names and labels
        with open(path_labels_csv, "w", newline='') as csvfile:
            fieldnames = ["id", "res_id", "rmsd", "lddt", "CAD"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for idx in range(len(dataset_test)):
                entry = dataset_test[idx]
                fname_loop = entry['id']
                labels = entry['labels']
                # For base_fnames, use the first 17 chars of the id as key
                base_fnames.add(fname_loop[:17])
                base_fnames_loops.add(fname_loop)
                # Write all labels per residue as individual rows
                # Use consecutive index instead of original res_id
                for i in range(len(labels["res_id"])):
                    writer.writerow({
                        "id": fname_loop,
                        "res_id": i,  # consecutive index per id starting at 0
                        "rmsd": labels["rmsd"][i],
                        "lddt": labels["lddt"][i],
                        "CAD": labels["CAD"][i]
                    })
        write_to_file(path_fnames, base_fnames)
        write_to_file(path_fnames_loops, base_fnames_loops)
    else:
        print(f"Loading base fnames from fnames.txt")
        base_fnames = load_from_file(path_fnames)

    print(f"Number of unique ids {len(base_fnames)}")
    
    outpath_pdbs = os.path.join(args.outpath, "pdbs") 
    os.makedirs(outpath_pdbs, exist_ok=True)
    
    all_pdb_files = glob.glob(os.path.join(args.base_pdbs_dir, "*.pdb"))
    pdb_dict = {}
    for pdb_file in all_pdb_files:
        base = os.path.basename(pdb_file)
        # Extract key matching your base_fname[:17] logic (adjust as needed)
        key = base[:17]
        if key not in pdb_dict:
            pdb_dict[key] = pdb_file

    for idx, base_fname in enumerate(base_fnames):
        if base_fname in pdb_dict:
            print(f"Copying {base_fname} ({idx+1}/{len(base_fnames)})")
            shutil.copy(pdb_dict[base_fname], outpath_pdbs)
        else:
            print(f"No match found for {base_fname}") 
    print("Done.")

###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze curated protein loop pocket dataset with residue stats and label histograms. Optionally analyze loop lengths from original PDB folder or a precomputed file.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to LMDB dataset folder")
    parser.add_argument("--base_pdbs_dir", type=str, required=True, help="Path to dataset folder containing base .pdbs (those from whom loops are extracted)")
    parser.add_argument("--outpath", type=str, default="out_base_pdbs", help="Path to dataset folder containing base .pdbs (those from whom loops are extracted)")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing run.")
    args = parser.parse_args()
    main(args)
