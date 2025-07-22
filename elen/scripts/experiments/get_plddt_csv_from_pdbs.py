#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
import os
import argparse
import pandas as pd
import re

def extract_pLDDT_from_pdb(pdb_path):
    # residue_id -> (resname, chain, [all pLDDTs])
    res_dict = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            # Parse ATOM line
            # Columns: https://www.wwpdb.org/documentation/file-format-content/format23/sect9.html
            # Example: ATOM      1  N   TYR A   1      26.683   0.851  -2.044  1.00 89.09           N  
            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = int(line[22:26].strip())
            plddt = float(line[60:66].strip())
            key = (chain, resseq)
            if key not in res_dict:
                res_dict[key] = {"resname": resname, "chain": chain, "resseq": resseq, "plddts": []}
            res_dict[key]["plddts"].append(plddt)
    # Take average pLDDT per residue
    rows = []
    for key, v in res_dict.items():
        avg_plddt = sum(v["plddts"]) / len(v["plddts"])
        rows.append({
            "metric": "plddt",
            "fname_pdb": os.path.basename(pdb_path),
            "res_id": v["resseq"],
            "ELEN_score": avg_plddt,
            "avg_per_chain": "",
        })
    return rows

def main():
    parser = argparse.ArgumentParser(description="Extract pLDDT per residue from all PDBs in a folder to CSV.")
    parser.add_argument("pdb_folder", help="Folder containing .pdb files")
    parser.add_argument("-o", "--output_csv", required=True, help="Output CSV filename")
    args = parser.parse_args()

    all_rows = []
    for filename in sorted(os.listdir(args.pdb_folder)):
        if filename.endswith(".pdb"):
            pdb_path = os.path.join(args.pdb_folder, filename)
            rows = extract_pLDDT_from_pdb(pdb_path)
            all_rows.extend(rows)
    if not all_rows:
        print("No PDB files found or no ATOM entries found!")
        return
    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_csv, index=False)
    print(f"Written {len(df)} rows for {df['fname_pdb'].nunique()} models to {args.output_csv}")

if __name__ == "__main__":
    main()
