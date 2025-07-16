#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
"""
Dataset characteristics analysis:
- Number of samples (train/val/test)
- Residue identity frequencies across all splits (with bar plot)
- Label histograms + stats (lddt, CAD, rmsd)
- Optional: loop length distribution from original pdb dataset folder (--data_set_for_loop_length)
  with caching to speed up repeated runs.
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

def load_json_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def plot_histogram(
    values, metric, save_path, panel_letter=None,
    figsize=(3,2.8),
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
        edgecolor='black', alpha=0.95, color="#6e6e6e"
    )
    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f'%.{decimals_xticks}f'))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel(metric.upper(), fontsize=13, fontweight='semibold', labelpad=4)
    ax.set_ylabel("Frequency", fontsize=13, labelpad=2)
    ax.tick_params(axis='x', labelrotation=0, labelsize=10, length=1)
    ax.tick_params(axis='y', labelsize=11, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    mean = np.mean(clipped_values)
    median = np.median(clipped_values)
    ax.axvline(mean, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='Mean')
    ax.axvline(median, color='blue', linestyle='--', linewidth=1.4, alpha=0.8, label='Median')
    ax.legend(fontsize=10, frameon=False, loc='upper left', handlelength=1.0, borderpad=0.7)
    if panel_letter:
        ax.text(-0.18, 1.08, panel_letter, fontsize=17, fontweight='bold', transform=ax.transAxes, va='top', ha='left')
    plt.tight_layout(pad=0.6)
    plt.savefig(save_path, bbox_inches='tight', dpi=350, transparent=True)
    plt.close(fig)

def plot_violin(data, metric, save_path, panel_letter=None, figsize=(4,3)):
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
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#6e6e6e')
        pc.set_edgecolor('black')
        pc.set_alpha(0.85)
    ax.set_ylabel(metric.upper(), fontsize=14, fontweight='semibold', labelpad=6)
    ax.set_xticks([1])
    ax.set_xticklabels([metric.upper()])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if panel_letter:
        ax.text(-0.15, 1.05, panel_letter, fontsize=17, fontweight='bold',
                transform=ax.transAxes, va='top', ha='left')
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, bbox_inches='tight', dpi=350, transparent=True)
    plt.close(fig)

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

def parse_loop_lengths_from_pdb_folder(pdb_folder, cache_csv="loop_length_cache.csv"):
    cache_path = os.path.join(pdb_folder, cache_csv)
    if os.path.exists(cache_path):
        print(f"Loading cached loop lengths from {cache_path}")
        df_cache = pd.read_csv(cache_path)
        return df_cache['loop_length'].tolist()
    else:
        print(f"No cache found. Parsing PDB files in {pdb_folder} ... this may take a while.")
        loop_lengths = []
        for filename in os.listdir(pdb_folder):
            if filename.endswith(".pdb"):
                filepath = os.path.join(pdb_folder, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        if line.lower().startswith('loop_length'):
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    length = int(parts[1])
                                    loop_lengths.append(length)
                                except ValueError:
                                    pass
                            break  # stop after first loop_length line
        # Save cache CSV
        df_cache = pd.DataFrame({'loop_length': loop_lengths})
        df_cache.to_csv(cache_path, index=False)
        print(f"Cached {len(loop_lengths)} loop lengths to {cache_path}")
        return loop_lengths

def main(dataset_dir, outdir, panel_letter=None, pdb_loop_dir=None):
    os.makedirs(outdir, exist_ok=True)

    if pdb_loop_dir is not None:
        print(f"Parsing loop lengths from PDB folder: {pdb_loop_dir}")
        loop_lengths = parse_loop_lengths_from_pdb_folder(pdb_loop_dir)
        if not loop_lengths:
            print("Warning: No loop_length records found in PDB folder.")
        else:
            print(f"Found {len(loop_lengths)} loop lengths.")
            # Save CSV
            df_loop = pd.DataFrame({'loop_length': loop_lengths})
            csv_path = os.path.join(outdir, "loop_length_distribution.csv")
            df_loop.to_csv(csv_path, index=False)
            print(f"Saved loop length CSV to {csv_path}")
            # Plot violin
            violin_path = os.path.join(outdir, "loop_length_violinplot.png")
            plot_violin(loop_lengths, "loop_length", violin_path, panel_letter=panel_letter)
            print(f"Saved loop length violin plot to {violin_path}")
            # Plot histogram
            hist_path = os.path.join(outdir, "loop_length_histogram.png")
            plot_histogram(loop_lengths, "loop_length", hist_path, panel_letter=panel_letter,
                           x_min=min(loop_lengths) - 1, x_max=max(loop_lengths) + 1, decimals_xticks=0, bins=20)
            print(f"Saved loop length histogram to {hist_path}")
        return

    # Continue with LMDB dataset residue and label analysis if no pdb_loop_dir

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
    
    residue_counter, total_residues = analyze_residue_identities([dataset_train, dataset_val, dataset_test])
    print(f"Total residues counted across all splits: {total_residues}")
    print("Residue counts:", residue_counter)

    df_res = pd.DataFrame.from_dict(residue_counter, orient='index', columns=['count'])
    df_res.index.name = 'residue'
    df_res = df_res.sort_index()
    df_res['frequency'] = df_res['count'] / total_residues

    sample_counts = pd.DataFrame({
        'residue': ['num_samples_train', 'num_samples_val', 'num_samples_test'],
        'count': [nr_train, nr_val, nr_test],
        'frequency': [np.nan, np.nan, np.nan]
    }).set_index('residue')

    df_summary = pd.concat([sample_counts, df_res])

    csv_res_path = os.path.join(outdir, "residue_identity_statistics.csv")
    df_summary.to_csv(csv_res_path)
    print(f"Saved residue identity statistics to {csv_res_path}")

    aa_comp_plot_path = os.path.join(outdir, "amino_acid_composition.png")
    plot_aa_composition(df_res[['frequency']], aa_comp_plot_path, panel_letter=panel_letter)
    print(f"Saved amino acid composition plot to {aa_comp_plot_path}")

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

        print(f"Statistics for {key}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")

        plot_path = os.path.join(outdir, f"{key}_distribution.png")
        x_max = 10.0 if key == "rmsd" else 1.0
        plot_histogram(values, key, plot_path, panel_letter=panel_letter, x_min=0.0, x_max=x_max, decimals_xticks=1)
        print(f"Saved histogram plot to {plot_path}")

    print("Analysis complete. CSVs and plots saved to", outdir)


def plot_aa_composition(freq_df, save_path, panel_letter=None, figsize=(6,4)):
    import matplotlib.cm as cm
    import matplotlib.ticker as mticker
    
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 10,
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
    ax.set_xticklabels(residues, rotation=45, ha='right')
    ax.set_xlabel("Residue", fontsize=14, fontweight='semibold', labelpad=4)
    ax.set_ylabel("Frequency", fontsize=14, fontweight='semibold', labelpad=4)
    ax.set_title("Amino Acid Composition", fontsize=15, fontweight='semibold', pad=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    if panel_letter:
        ax.text(-0.08, 1.05, panel_letter, fontsize=17, fontweight='bold',
                transform=ax.transAxes, va='top', ha='left')
    
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, bbox_inches='tight', dpi=350, transparent=True)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze curated protein loop pocket dataset with residue stats and label histograms. Optionally analyze loop lengths from original PDB folder.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to LMDB dataset folder")
    parser.add_argument("--outdir", type=str, default="dataset_analysis_out", help="Output directory for CSVs and plots")
    parser.add_argument("--panel_letter", type=str, default=None, help="Panel letter for figures (optional)")
    parser.add_argument("--data_set_for_loop_length", type=str, default=None, help="Path to original PDB dataset folder with loop_length records")
    args = parser.parse_args()
    main(args.dataset_dir, args.outdir, args.panel_letter, args.data_set_for_loop_length)
