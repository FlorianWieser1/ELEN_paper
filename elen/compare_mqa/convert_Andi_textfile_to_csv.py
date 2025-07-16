#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
import argparse
import pandas as pd
import re

def parse_plddt_out_file(input_file):
    results = []
    pdb_file = None

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Processing PDB file:'):
                # Extract pdb file name (use only basename)
                pdb_file = line.split(':',1)[1].strip().split('/')[-1]
            elif line.startswith('Residue:'):
                # Parse residue line
                m = re.match(r'Residue: (\w+) (\d+) \| pLDDT: ([\d\.]+)', line)
                if m and pdb_file is not None:
                    resname, resid, plddt = m.groups()
                    results.append({
                        'metric': 'plddt',
                        'fname_pdb': pdb_file,
                        'res_id': int(resid),
                        'ELEN_score': float(plddt),
                        'avg_per_chain': '', # keep column for compatibility
                    })
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Convert AlphaFold pLDDT .out to compatible .csv")
    parser.add_argument("input_out", help="Input .out file")
    parser.add_argument("-o", "--output_csv", required=True, help="Output CSV file name")
    args = parser.parse_args()

    df = parse_plddt_out_file(args.input_out)
    df.to_csv(args.output_csv, index=False)
    print(f"Written {len(df)} rows to {args.output_csv}")

if __name__ == "__main__":
    main()
