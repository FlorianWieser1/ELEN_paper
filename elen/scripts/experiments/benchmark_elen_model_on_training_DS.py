#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#TODO wait for bigger data - test, val, train
import os
import re
import sys
import json
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

### HELPERS ###################################################################
def load_from_file(path_fnames):
    with open(path_fnames, 'r') as f:
        loaded_fnames = set(line.strip() for line in f)
    return loaded_fnames

def append_images_horizontally(output_path, input_paths):
    """
    Horizontally append images using ImageMagick's convert +append.
    """
    cmd = ['convert', '+append'] +  input_paths + [output_path]
    try:
        subprocess.run(cmd, check=True)
        print(f"Created appended image: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error while appending images: {e}")
        
def set_scientific_plot_style():
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.5,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.weight": "bold",
        "legend.frameon": False,
    })

def plot_perres_correlation(df, pred_col, label_col, title, color, path_plot, panel_letter=None):
    """
    Plot scatter + regression line with correlations in legend.
    """
    set_scientific_plot_style()
    fig, ax = plt.subplots(figsize=(5,5))

    sns.regplot(data=df, x=label_col, y=pred_col, color=color,
                scatter_kws={'s':20, 'alpha':0.4}, line_kws={'lw':2}, ax=ax)

    pearson_corr = df[label_col].corr(df[pred_col], method='pearson')
    spearman_corr = df[label_col].corr(df[pred_col], method='spearman')
    kendall_corr = df[label_col].corr(df[pred_col], method='kendall')
    mae = mean_absolute_error(df[label_col], df[pred_col])

    ax.plot([], [], ' ', label=f'n = {len(df):,}')
    ax.plot([], [], ' ', label=rf'$R$ = {pearson_corr:.3f}')
    ax.plot([], [], ' ', label=rf'$\rho$ = {spearman_corr:.3f}')
    ax.plot([], [], ' ', label=rf'$\tau$ = {kendall_corr:.3f}')
    ax.plot([], [], ' ', label=f'MAE = {mae:.3f}')

    ax.legend(loc='upper left', handletextpad=0.3, fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=18)
    ax.set_xlabel(f"Label {title}", fontweight='semibold', fontsize=14)
    ax.set_ylabel(f"Predicted {title}", fontweight='semibold', fontsize=14)

    if panel_letter is not None:
        ax.text(-0.15, 1.05, panel_letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left')

    plt.tight_layout(pad=1.0)
    plt.savefig(path_plot, bbox_inches='tight', transparent=True)
    plt.close(fig)
    return path_plot

### MAIN ######################################################################

def main(args):
    if os.path.exists(args.outpath) and args.overwrite:
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    path_tmp_data = os.path.join(args.outpath, "df_elen_with_labels.csv")
    path_metrics_csv = os.path.join(args.outpath, "metrics_summary.csv")

    if not os.path.exists(path_tmp_data): 
        print(f"Generating tmp data .csv")
        # Load dataset filenames
        dataset_fnames = load_from_file(args.target_ds_fnames)
        ds_fnames = {re.sub(r'_[EH]{2}\.pdb$', '.pdb', filename) for filename in dataset_fnames}
        
        # Load batch predictions from elen_scores.json  
        with open(args.elen_scores_json, 'r') as f:
            data_elen_scores = json.load(f)
        df = pd.DataFrame(data_elen_scores)
        df.index = df.index.astype(int)
        df = df.sort_index()
        
        # Filter out unwanted metrics like 'all'
        df_filtered = df[~df['metric'].isin(['all'])].copy()
        
        # Clean 'filename'
        df_filtered.loc[:, 'filename'] = df_filtered['filename'].str.replace(r'(m[1-5]_)[A-Z]_', r'\1', regex=True)
        df_filtered.loc[:, 'filename'] = df_filtered['filename'].str.replace(r'_[EH]{2}\.pdb$', '.pdb', regex=True)
        
        # Filter to dataset filenames
        df_filtered = df_filtered[df_filtered['filename'].isin(ds_fnames)].copy()
        print(f"df_filtered sample:\n{df_filtered.head()}")
        print(f"Unique metrics: {df_filtered['metric'].unique()}")

        # Load labels CSV
        labels_df = pd.read_csv(args.labels_csv)
        # The columns: id (filename), res_id (index), lddt, CAD, rmsd
        # Prepare for merge: ensure types are correct and names align
        labels_df = labels_df.rename(columns={'id': 'filename', 'res_id': 'index'})
        # Convert index to int to match with df_filtered
        labels_df['index'] = labels_df['index'].astype(int)
        labels_df['filename'] = labels_df['filename'].str.replace(r'_(HH|HE|EE|EH)(?=\.pdb$)', '', regex=True)
        print(f"labels_df: {labels_df}")

        # Separate DataFrames for each metric with renamed prediction columns
        dfs = []
        for metric in ['lddt', 'CAD', 'rmsd']:
            dfm = df_filtered[df_filtered['metric'] == metric].copy()
            dfm = dfm.rename(columns={'pred': f'pred_{metric}'})
            dfs.append(dfm[['filename', 'index', f'pred_{metric}']])

        # Merge prediction columns on filename, index
        df_preds = dfs[0]
        for dfm in dfs[1:]:
            df_preds = pd.merge(df_preds, dfm, on=['filename', 'index'], how='outer')
        # Merge predictions with labels
        df_elen = pd.merge(df_preds, labels_df, on=['filename', 'index'], how='left')
        
        # Rename label columns for consistency
        df_elen = df_elen.rename(columns={
            'lddt': 'label_lddt',
            'CAD': 'label_CAD',
            'rmsd': 'label_rmsd'
        })

        df_elen = df_elen.sort_values(['filename', 'index']).reset_index(drop=True)

        print(f"df_elen sample:\n{df_elen.head()}")

        # Save labeled dataframe
        df_elen.to_csv(path_tmp_data, index=False)
        print(f"Saved labeled df to {path_tmp_data}")
    else:
        print("Loading tmp data .csv")
        df_elen = pd.read_csv(path_tmp_data)

    # Calculate metrics and save to CSV
    metrics = []
    for metric in ['lddt', 'CAD', 'rmsd']:
        pred_col = f'pred_{metric}'
        label_col = f'label_{metric}'
        df_sub = df_elen.dropna(subset=[pred_col, label_col])
        if len(df_sub) == 0:
            print(f"No data for metric {metric}, skipping.")
            continue
        pearson_corr = df_sub[label_col].corr(df_sub[pred_col], method='pearson')
        spearman_corr = df_sub[label_col].corr(df_sub[pred_col], method='spearman')
        mae = mean_absolute_error(df_sub[label_col], df_sub[pred_col])
        metrics.append({
            'metric': metric,
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr,
            'mean_absolute_error': mae,
            'n_points': len(df_sub)
        })
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(path_metrics_csv, index=False)
    print(f"Saved metrics summary to {path_metrics_csv}")

    # Plotting
    plot_colors = {'lddt': 'black', 'CAD': 'black', 'rmsd': 'black'}
    panel_letters = {'lddt': 'A', 'CAD': 'B', 'rmsd': 'C'}
    titles = {'lddt': 'lDDT', 'CAD': 'CAD-Score', 'rmsd': 'RMSD'}
    input_paths = [] 
    for metric in ['lddt', 'CAD', 'rmsd']:
        pred_col = f'pred_{metric}'
        label_col = f'label_{metric}'
        df_sub = df_elen.dropna(subset=[pred_col, label_col])
        if len(df_sub) == 0:
            continue
        outpath_scatter = os.path.join(args.outpath, f"correlation_scatter_{metric}.png")
        plot_perres_correlation(df_sub, pred_col, label_col, titles[metric], plot_colors[metric], outpath_scatter, panel_letters[metric])
        input_paths.append(outpath_scatter)
    outpath_scatter_final = os.path.join(args.outpath, "correlation_scatter.png")
    append_images_horizontally(outpath_scatter_final, input_paths)
    print("Done.")

###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze curated protein loop pocket dataset with residue stats and label histograms.")
    parser.add_argument("--elen_scores_json", type=str, required=True)
    parser.add_argument("--target_ds_fnames", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="CSV with columns id,res_id,rmsd,lddt,CAD (from LMDB, not JSON!)")
    parser.add_argument("--outpath", type=str, default="out_elen_train_val_test_bench",
                        help="Path to output folder.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing run.")
    args = parser.parse_args()
    main(args)
