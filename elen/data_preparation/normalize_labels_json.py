#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
import json
import argparse

def main(labels_in, labels_out, scales_out):
    """
    1) Load labels_in (JSON).
    2) Compute global min/max for each label.
    3) Normalize each label array.
       - If label == 'rmsd': do inverted min–max, so smaller RMSD => higher scaled value.
       - Else: do standard min–max scaling.
    4) Write normalized labels to labels_out.
    5) Write min/max (and whether inverted) to scales_out.
    """

    print(f"Loading {labels_in} ...")
    with open(labels_in, 'r') as f:
        data = json.load(f)
    # Example structure of data:
    # {
    #   "5fji_A_ur-r004-m2_6_HE.pdb": {
    #       "rmsd": [0.566, 0.587, ...],
    #       "lddt": [...],
    #       "CAD":  [...],
    #       ...
    #   },
    #   ...
    # }

    # Collect all possible label names from across all entries
    label_names = set()
    for struct_id, label_dict in data.items():
        label_names.update(label_dict.keys())
    label_names = sorted(label_names)

    # -----------------------------------------------------------------
    # 1) Compute global min and max for each label (across all entries)
    # -----------------------------------------------------------------
    scales = {}
    for lbl in label_names:
        if lbl in ('res_id'):
            continue
        # Gather all numeric values
        all_values = []
        for struct_id, label_dict in data.items():
            if lbl in label_dict:
                # label_dict[lbl] is typically a list of floats
                all_values.extend(label_dict[lbl])
        if not all_values:
            print(f"Warning: no values found for label '{lbl}'—skipping.")
            continue

        mn = float(min(all_values))
        mx = float(max(all_values))
        scales[lbl] = {
            'min': mn,
            'max': mx,
            'inverted': (lbl == 'rmsd')  # We'll invert 'rmsd'
        }

    # ---------------------------------------------
    # 2) Transform each value according to min/max
    #    - If 'rmsd', do inverted: 1 - (x - mn)/(mx - mn)
    #    - Else, do normal:       (x - mn)/(mx - mn)
    # ---------------------------------------------
    for struct_id, label_dict in data.items():
        for lbl, arr in label_dict.items():
            if lbl not in scales:
                continue  # e.g. label has no data or was skipped
            mn = scales[lbl]['min']
            mx = scales[lbl]['max']
            invert = scales[lbl]['inverted']

            if mx == mn:
                # Avoid divide-by-zero if all values are constant
                print(f"Warning: label '{lbl}' for '{struct_id}' has constant value {mn}.")
                if invert:
                    # If we invert but min==max, everything is the same => call it 1.0
                    label_dict[lbl] = [1.0] * len(arr)
                else:
                    # Normal scaling for constant => 0.0
                    label_dict[lbl] = [0.0] * len(arr)
            else:
                new_vals = []
                for val in arr:
                    norm_val = (val - mn) / (mx - mn)
                    if invert:
                        norm_val = 1.0 - norm_val
                    new_vals.append(norm_val)
                label_dict[lbl] = new_vals

    # --------------------------------------
    # 3) Write out the normalized JSON data
    # --------------------------------------
    print(f"Writing normalized labels to {labels_out} ...")
    with open(labels_out, 'w') as f:
        json.dump(data, f, indent=2)

    # ---------------------------------------
    # 4) Write out the scale parameters used
    # ---------------------------------------
    # We store min, max, and whether we inverted each label
    print(f"Writing scale parameters to {scales_out} ...")
    with open(scales_out, 'w') as f:
        json.dump(scales, f, indent=2)

    print("Done.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Normalize label values in a JSON file. Inverts RMSD values by default.")
    parser.add_argument(
        "labels_in", 
        help="Path to the input JSON file containing raw label data."
    )
    parser.add_argument(
        "labels_out", 
        help="Path to output the JSON file with normalized label data."
    )
    parser.add_argument(
        "scales_out", 
        help="Path to output the JSON file with scale parameters (min/max/inverted)."
    )
    return parser.parse_args()

###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize label values in a JSON file. Inverts RMSD values by default.")
    parser.add_argument(
        "labels_in", 
        help="Path to the input JSON file containing raw label data."
    )
    parser.add_argument(
        "labels_out", 
        help="Path to output the JSON file with normalized label data."
    )
    parser.add_argument(
        "scales_out", 
        help="Path to output the JSON file with scale parameters (min/max/inverted)."
    )
    args = parser.parse_args()
    main(args.labels_in, args.labels_out, args.scales_out)