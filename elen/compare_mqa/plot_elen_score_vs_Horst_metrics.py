#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def main(args):
    # Load the two CSVs
    try:
        df_metrics = pd.read_csv(args.metrics_csv)
    except Exception as e:
        print(f"Error reading metrics CSV: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_elen = pd.read_csv(args.elen_csv)
    except Exception as e:
        print(f"Error reading ELEN CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Normalise filenames so they match
    df_metrics['filename'] = df_metrics['filename'].str.replace(r'\.fa$', '', regex=True)
    df_elen['filename'] = df_elen['fname_pdb'].str.replace(r'_\d+_unrelaxed.*$', '', regex=True)

    # Merge on filename
    df_merged = pd.merge(
        df_metrics,
        df_elen[['filename', 'ELEN_score']],
        how='inner',
        left_on='filename',
        right_on='filename'
    )
    print(df_merged)

    if df_merged.empty:
        print("Warning: No matching filenames found between the two files.", file=sys.stderr)
        sys.exit(0)

    # Define which metrics to plot against ELEN_score
    metrics_to_plot = [
        'overall_confidence',
        'ligand_confidence',
        'SaProt_mut_score',
        'Average negative pseudo-log-likelihood'
    ]

    for col in metrics_to_plot:
        if col not in df_merged.columns:
            print(f"Column '{col}' not found in merged DataFrame; skipping.", file=sys.stderr)
            continue

        # Extract clean (non‑NaN) data for correlation / regression
        x = df_merged[col].values
        y = df_merged['ELEN_score'].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() == 0:
            print(f"Column '{col}' contains only NaNs after merge; skipping.", file=sys.stderr)
            continue
        x = x[mask]
        y = y[mask]

        # Linear regression (least‑squares fit)
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept

        # Correlations
        pearson_r, _ = pearsonr(x, y)
        spearman_r, _ = spearmanr(x, y)

        # Plot
        plt.figure()
        plt.scatter(x, y, label='data points')
        plt.plot(np.sort(x), slope * np.sort(x) + intercept, linewidth=2, label='linear fit')
        plt.xlabel(col)
        plt.ylabel('ELEN_score')
        plt.title(f'{col} vs ELEN_score')
        plt.legend(title=f"Pearson r={pearson_r:.3f}\nSpearman ρ={spearman_r:.3f}")

        out_file = f"{col.replace(' ', '_')}_vs_ELEN_score.png"
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()
        print(f"Saved scatter plot with regression and correlations: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge two CSVs and generate scatter plots of metrics vs ELEN_score"
    )
    parser.add_argument(
        "--metrics_csv",
        required=True,
        help="Path to the first CSV containing your various scores (must have a 'filename' column)"
    )
    parser.add_argument(
        "--elen_csv",
        required=True,
        help="Path to the second CSV containing ELEN_score (must have 'fname_pdb' and 'ELEN_score' columns)"
    )
    args = parser.parse_args()
    main(args)
