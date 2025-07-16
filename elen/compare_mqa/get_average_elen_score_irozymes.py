#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
import argparse
import pandas as pd

def main(input_csv, output_csv=None):
    # Load CSV
    df = pd.read_csv(input_csv)
    
    # Get all unique models
    models = df['fname_pdb'].unique()
    
    summary = []
    for model in models:
        sub = df[df['fname_pdb'] == model]
        avg_total = sub['ELEN_score'].mean()
        
        # Residues 180-190
        res_180_190 = sub[(sub['res_id'] >= 180) & (sub['res_id'] <= 190)]
        avg_180_190 = res_180_190['ELEN_score'].mean() if not res_180_190.empty else float('nan')
        
        # All except 180-190
        not_180_190 = sub[(sub['res_id'] < 180) | (sub['res_id'] > 190)]
        avg_not_180_190 = not_180_190['ELEN_score'].mean() if not not_180_190.empty else float('nan')
        
        summary.append({
            'model': model,
            'avg_total': avg_total,
            'avg_180_190': avg_180_190,
            'avg_not_180_190': avg_not_180_190,
        })
    
    result_df = pd.DataFrame(summary)
    
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"Results written to {output_csv}")
    else:
        print(result_df.to_csv(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ELEN scores per model and region.")
    parser.add_argument("--input_csv", help="Input CSV file with ELEN scores")
    parser.add_argument("-o", "--output_csv", help="Optional output CSV file for results")
    args = parser.parse_args()
    
    main(args.input_csv, args.output_csv)
