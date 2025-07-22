#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J plot_dataset
#SBATCH -o plot_dataset.log
#SBATCH -e plot_dataset.err
"""
Dataset characteristics analysis:
- Number of samples (train/val/test)
- Residue identity frequencies across all splits (with bar plot)
- Label histograms + stats (lddt, CAD, rmsd)
"""

import sys
import argparse
import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import skew, kurtosis
import matplotlib.ticker as mticker
import atom3d.datasets as da
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
barcolor="#7463A5"

### HELPERS
def load_json_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

# Then in analyze_and_plot_loop_lengths (just update this block):
def analyze_and_plot_loop_lengths(loop_lengths, outdir, panel_letter=None):
    if not loop_lengths:
        print("Warning: No loop_length records found.")
        return
    print(f"Found {len(loop_lengths)} loop lengths.")
    # Save CSV
    df_loop = pd.DataFrame({'loop_length': loop_lengths})
    csv_path = os.path.join(outdir, "loop_length_distribution.csv")
    df_loop.to_csv(csv_path, index=False)
    print(f"Saved loop length CSV to {csv_path}")
    # Plot improved histogram
    hist_path = os.path.join(outdir, "loop_length_histogram.png")
    plot_integer_histogram(
        loop_lengths, "Loop length", hist_path, legend_position='upper right', panel_letter="B")
    print(f"Saved loop length histogram to {hist_path}")
    
def compute_statistics(values):
    arr = np.array(values)
    stats = {
        "count": len(arr),
        "mean": np.mean(arr),
        "std": np.std(arr),
        "median": np.median(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "percentile_25": np.percentile(arr, 25),
        "percentile_75": np.percentile(arr, 75),
        "skewness": skew(arr),
        "kurtosis": kurtosis(arr),
    }
    return stats

def analyze_residue_identities(datasets):
    residue_counter = Counter()
    total_residues = 0
    for ds in datasets:
        for i in range(len(ds)):
            resnames = ds[i]['atoms']['resname']  # pandas Series
            counts = resnames.value_counts()
            residue_counter.update(counts.to_dict())
            total_residues += len(resnames)
    return residue_counter, total_residues


def load_loop_lengths_file(loop_lengths_file):
    with open(loop_lengths_file, 'r') as f:
        lines = f.readlines()
    loop_lengths = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                loop_lengths.append(int(line))
            except ValueError:
                continue
    return loop_lengths

### PLOTTING FUNCTIONS
def plot_histogram(
    values, xlabel, xtick_nlocator, save_path, legend_position, panel_letter=None,
    figsize=(5.1,3),
    bins=40,
    x_min=0.0,
    x_max=1.0,
    decimals_xticks=1,
    ):
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.4,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "figure.dpi": 250,
        "savefig.dpi": 350,
        "font.family": "sans-serif",
        "font.weight": "semibold",
    })
    fig, ax = plt.subplots(figsize=figsize)
    clipped_values = [v for v in values if x_min <= v <= x_max]
    n, bins_edges, patches = ax.hist(
        clipped_values, bins=bins, range=(x_min, x_max), 
        edgecolor='black', alpha=0.95, color="#556B2F"
    )
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(plt.MaxNLocator(xtick_nlocator))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f'%.{decimals_xticks}f'))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel(xlabel, fontsize=13, fontweight='semibold', labelpad=4)
    ax.set_ylabel("Frequency", fontsize=13, fontweight='semibold', labelpad=4)
    ax.tick_params(axis='x', labelrotation=0, labelsize=10, length=1)
    ax.tick_params(axis='y', labelsize=11, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    mean = np.mean(clipped_values)
    median = np.median(clipped_values)
    ax.axvline(mean, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='Mean')
    ax.axvline(median, color='blue', linestyle='--', linewidth=1.4, alpha=0.8, label='Median')
    ax.legend(fontsize=10, frameon=False, loc=legend_position, handlelength=1.0, borderpad=0.7)
    if panel_letter:
        ax.text(-0.18, 1.08, panel_letter, fontsize=17, fontweight='bold', transform=ax.transAxes, va='top', ha='left')
    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, bbox_inches='tight', dpi=350, transparent=True)
    plt.close(fig)
    
def plot_integer_histogram(
    values, xlabel, save_path, legend_position='upper right', panel_letter=None,
    figsize=(3.0, 3.22)):
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 1.5,
        "figure.dpi": 220,
        "savefig.dpi": 330,
        "font.family": "sans-serif",
        "font.weight": "bold",
    })
    values = np.array(values)
    x_min, x_max = int(values.min()), int(values.max())
    bins = np.arange(x_min, x_max + 2)
    fig, ax = plt.subplots(figsize=figsize)
    n, bins_edges, patches = ax.hist(
        values, bins=bins, edgecolor='black', alpha=0.93, color=barcolor, align='left', rwidth=0.75
    )
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_xticks(np.arange(x_min, x_max + 1, 1))
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold', labelpad=7)
    ax.set_ylabel("Frequency", fontsize=14, fontweight='bold', labelpad=7)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.tick_params(axis='x', labelrotation=0, labelsize=14, length=5)
    ax.tick_params(axis='y', labelsize=14, length=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    mean = np.mean(values)
    median = np.median(values)
    ax.axvline(mean, color='red', linestyle='-', linewidth=2.1, alpha=0.9, label=f'Mean')
    ax.axvline(median, color='blue', linestyle='--', linewidth=2.1, alpha=0.9, label=f'Median')
    ax.legend(fontsize=13, frameon=False, loc=legend_position, handlelength=1.3, borderpad=1.0)
    ax.text(-0.25, 1.05, panel_letter, fontsize=16, fontweight='bold', transform=ax.transAxes, va='top', ha='left')
    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.close(fig)
    
    
def plot_violin(data, x_max, legend_position, metric, save_path, panel_letter=None, figsize=(3.2, 4)):
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.4,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "figure.dpi": 250,
        "savefig.dpi": 350,
        "font.family": "sans-serif",
        "font.weight": "semibold",
    })

    fig, ax = plt.subplots(figsize=figsize)

    # Use seaborn violinplot for easier styling and jittered scatter overlay
    sns.violinplot(data=data, ax=ax, color=barcolor, inner=None)  # No inner parts, custom below
    
    # Make violin contours black and thicker
    for collection in ax.collections:
        collection.set_edgecolor('black')
        collection.set_linewidth(1.2)

   # Overlay boxplot
    sns.boxplot(
        data=data, 
        ax=ax, 
        width=0.1, 
        showcaps=True,
        boxprops={'facecolor':'white', 'edgecolor':'black', 'linewidth': 1.2},
        whiskerprops={'color':'black', 'linewidth':1.2},
        capprops={'color':'black', 'linewidth':1.2},
        medianprops={'color':'red', 'linewidth':2},
        flierprops={'markerfacecolor':'grey', 'markeredgecolor':'black', 'markersize': 3}  # if you want black fliers
    )
    # Overlay jittered points for raw data
    #sns.stripplot(data=data, ax=ax, color='black', size=4, jitter=0.15, alpha=0.5)

    # Axis labels and limits
    ax.set_ylabel(metric.upper(), fontsize=16, fontweight='semibold', labelpad=6)
    ax.set_xticks([0])
    ax.set_xticklabels([metric.upper()], fontsize=14)
    ax.set_ylim(0.0, x_max)  # Assuming metric like LDDT bounded between 0 and 1
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add summary statistics annotation (mean and median)
    mean_val = np.mean(data)
    median_val = np.median(data)
    ax.text(0.05, legend_position, f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}", transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='left')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if panel_letter:
        ax.text(-0.45, 1.05, panel_letter, fontsize=16, fontweight='bold',
                transform=ax.transAxes, va='top', ha='left')

    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.close(fig)
    

def plot_aa_composition(freq_df, save_path, panel_letter=None, figsize=(4.5,3)):
    """
    freq_df: DataFrame with index=residue, column='frequency'
    Plot a bar plot of AA frequencies.
    """
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.4,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "figure.dpi": 250,
        "savefig.dpi": 350,
        "font.family": "sans-serif",
        "font.weight": "semibold",
    })
    fig, ax = plt.subplots(figsize=figsize)
    residues = freq_df.index
    frequencies = freq_df['frequency']
    cmap = cm.get_cmap('tab20')
    colors = [cmap(i / max(len(residues)-1, 1)) for i in range(len(residues))]
    bars = ax.bar(residues, frequencies, color=colors, edgecolor='black', alpha=0.95) 
    ax.set_xticklabels(residues, rotation=50, ha='right', fontsize=9)
    ax.set_xlabel("Residue", fontsize=14, fontweight='semibold', labelpad=4)
    ax.set_ylabel("Frequency", fontsize=14, fontweight='semibold', labelpad=4)
    #ax.set_title("Amino Acid Composition", fontsize=15, fontweight='semibold', pad=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
   
    ax.text(-0.3, 1.00, panel_letter, fontsize=16, fontweight='bold',
                transform=ax.transAxes, va='top', ha='left')
    
    plt.tight_layout(pad=0.8)
    plt.savefig(save_path, bbox_inches='tight', transparent=True)

###############################################################################
def main(dataset_dir, outdir, panel_letter=None, loop_lengths_file=None):
    os.makedirs(outdir, exist_ok=True)

    #######################################################################
    # LOOP LENGHTS
    # Handle the two fast loop length input options first:
    if loop_lengths_file is not None:
        print(f"Loading loop lengths from file: {loop_lengths_file}")
        loop_lengths = load_loop_lengths_file(loop_lengths_file)
        analyze_and_plot_loop_lengths(loop_lengths, outdir, panel_letter=panel_letter)

    #######################################################################
    # NR OF SAMPLES
    # Load LMDB datasets    
    path_train_dir = os.path.join(dataset_dir, "lmdbs", "train")
    path_val_dir = os.path.join(dataset_dir, "lmdbs", "val")
    path_test_dir = os.path.join(dataset_dir, "lmdbs", "test")
    
    dataset_train = da.load_dataset(path_train_dir, 'lmdb')
    dataset_val = da.load_dataset(path_val_dir, 'lmdb')
    dataset_test = da.load_dataset(path_test_dir, 'lmdb')
    
    nr_train = len(dataset_train)
    nr_val = len(dataset_val)
    nr_test = len(dataset_test)
    print(f"Number of samples train, val, test: {(nr_train, nr_val, nr_test)}")
    
    #######################################################################
    # AA IDENTITY HISTOGRAM
    # Analyze residue identities
    residue_counter, total_residues = analyze_residue_identities([dataset_train, dataset_val, dataset_test])

    # DataFrame for residues
    df_res = pd.DataFrame.from_dict(residue_counter, orient='index', columns=['count'])
    df_res.index.name = 'residue'
    df_res = df_res.sort_index()
    df_res['frequency'] = df_res['count'] / total_residues

    # Add sample counts as rows with NaN frequency for clarity
    sample_counts = pd.DataFrame({
        'residue': ['num_samples_train', 'num_samples_val', 'num_samples_test'],
        'count': [nr_train, nr_val, nr_test],
        'frequency': [np.nan, np.nan, np.nan]
    }).set_index('residue')

    df_summary = pd.concat([sample_counts, df_res])

    csv_res_path = os.path.join(outdir, "residue_identity_statistics.csv")
    df_summary.to_csv(csv_res_path)
    print(f"Saved residue identity statistics to {csv_res_path}")

    # Plot amino acid composition bar plot
    aa_comp_plot_path = os.path.join(outdir, "amino_acid_composition.png")
    plot_aa_composition(df_res[['frequency']], aa_comp_plot_path, panel_letter="C")
    print(f"Saved amino acid composition plot to {aa_comp_plot_path}")

    #######################################################################
    # LABEL HISTOGRAM
    labels_path = os.path.join(dataset_dir, "labels.json")

    data_labels = load_json_file(labels_path)
    metrics = defaultdict(list)
    for fname_loop, lab in data_labels.items():
        for key in ['lddt', 'CAD', 'rmsd']:
            if key in lab:
                if isinstance(lab[key], (float, int)):
                    metrics[key].append(lab[key])
                elif isinstance(lab[key], list):
                    metrics[key].extend(lab[key])

    for key, values in metrics.items():
        stats = compute_statistics(values)
        df = pd.DataFrame({key: values})
        stats_row = {key: [f"stat_{k}:{v:.4f}" for k,v in stats.items()]}
        df_stats = pd.DataFrame(stats_row)
        df_full = pd.concat([df, df_stats], ignore_index=True)
        csv_path = os.path.join(outdir, f"{key}_distribution.csv")
        df_full.to_csv(csv_path, index=False)
        print(f"Saved CSV for {key} with summary stats to {csv_path}")

        plot_path = os.path.join(outdir, f"{key}_distribution.png")
        x_max = 10.0 if key == "rmsd" else 1.0
        legend_position = 'upper right' if key == 'rmsd' else 'upper left'
        plot_histogram(values, key.upper(), 5, plot_path, legend_position, panel_letter=panel_letter, x_min=0.0, x_max=x_max, decimals_xticks=1)
        legend_position = 0.90 if key == 'rmsd' else 0.01
        panel_letter = 'A' if key == "lddt" else None
        plot_violin(values, x_max, legend_position, key.upper(), plot_path.replace("distribution", "violin"), panel_letter=panel_letter)

        print(f"Saved histogram plot to {plot_path}")

    print("Analysis complete. CSVs and plots saved to", outdir)

###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze curated protein loop pocket dataset with residue stats and label histograms. Optionally analyze loop lengths from original PDB folder or a precomputed file.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to LMDB dataset folder")
    parser.add_argument("--outdir", type=str, default="dataset_analysis_out", help="Output directory for CSVs and plots")
    parser.add_argument("--panel_letter", type=str, default=None, help="Panel letter for figures (optional)")
    parser.add_argument("--loop_lengths_file", type=str, default=None, help="Path to a text file with one loop length per line (alternative to parsing PDBs)")
    args = parser.parse_args()
    main(args.dataset_dir, args.outdir, args.panel_letter, args.loop_lengths_file)