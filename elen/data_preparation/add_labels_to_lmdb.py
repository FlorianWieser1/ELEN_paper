#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
import os
import json
import argparse
import atom3d.datasets.datasets as LMDBDataset

def add_labels_to_lmdb_atom3d(in_path, out_path, labels_json):
    """Load an LMDB dataset, attach labels, then write a new LMDB."""
    print(f"Loading LMDB dataset from: {in_path}")
    dataset = LMDBDataset.LMDBDataset(in_path)  # read-only dataset

    print(f"Reading labels from: {labels_json}")
    with open(labels_json, 'r') as f:
        labels_dict = json.load(f)

    new_data = []
    for sample in dataset:
        # sample is a dictionary with keys like: sample['id'], sample['atoms'], etc.
        record_id = sample['id']
        if record_id in labels_dict:
            # Attach the entire label sub-dict as sample['labels']
            sample['labels'] = labels_dict[record_id]
        else:
            sample['labels'] = {}
        new_data.append(sample)

    # Now write out the new LMDB
    print(f"Writing new LMDB to: {out_path}")
    LMDBDataset.make_lmdb_dataset(new_data, out_path)
    print("Done.")

###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Attach labels from a JSON file to an ATOM3D LMDB dataset."
    )
    parser.add_argument(
        "in_lmdb", 
        help="Path to the input LMDB dataset."
    )
    parser.add_argument(
        "out_lmdb", 
        help="Path to the output LMDB dataset (with labels)."
    )
    parser.add_argument(
        "labels_json", 
        help="Path to the JSON file containing labels."
    )
    args = parser.parse_args()
    add_labels_to_lmdb_atom3d(args.in_lmdb, args.out_lmdb, args.labels_json)

if __name__ == "__main__":
    main()
