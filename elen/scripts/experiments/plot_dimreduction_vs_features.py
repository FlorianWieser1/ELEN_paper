#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
#sbatch -j plot_dimred
#sbatch -o plot_dimred.log
#sbatch -e plot_dimred.err

import os
import glob
import shutil
import logging
import numpy as np
import pandas as pd
import argparse as ap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn.manifold import TSNE
import umap
from Bio.PDB import PDBParser

from elen.shared_utils.utils_io import load_from_json, load_from_csv
from get_features_for_dimreduction import read_loop_positions_from_pdb

logging.basicConfig(level=logging.INFO, format='elen-dimred-%(levelname)s(%(asctime)s): %(message)s', datefmt='%y-%m-%d %H:%M:%S')

# Dimensionality reduction methods
def calculate_pca(X):
    return pd.DataFrame(decomposition.PCA(n_components=3, random_state=43).fit_transform(X),
                        columns=['pca_1', 'pca_2', 'pca_3'])

def calculate_umap(X):
    return pd.DataFrame(umap.UMAP(n_components=3, n_neighbors=100, min_dist=0.5, random_state=15).fit_transform(X),
                        columns=['umap_1', 'umap_2', 'umap_3'])

def calculate_tsne(X):
    return pd.DataFrame(TSNE(init='pca', n_components=3, perplexity=10, n_iter=5000, random_state=15).fit_transform(X),
                        columns=['tsne_1', 'tsne_2', 'tsne_3'])

DIMRED_METHODS = {'pca': calculate_pca, 'umap': calculate_umap, 'tsne': calculate_tsne}

# Plotting 2D scatter
def plot_2d(df, reduction, feature, outpath):
    plt.figure(figsize=(8, 6))
    x, y, feat_data = df[f'{reduction}_1'], df[f'{reduction}_2'], df[feature]
    
    if pd.api.types.is_numeric_dtype(feat_data):
        scatter = plt.scatter(x, y, c=feat_data, cmap='viridis', s=20, alpha=0.8)
        plt.colorbar(scatter, label=feature)
    else:
        categories = pd.Categorical(feat_data)
        palette = sns.color_palette('tab10', len(categories.categories))
        sns.scatterplot(x=x, y=y, hue=categories, palette=palette, s=30, alpha=0.8)
        plt.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel(f'{reduction.upper()} Component 1')
    plt.ylabel(f'{reduction.upper()} Component 2')
    plt.title(f'{reduction.upper()} colored by {feature}')
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()

# Data loading and preparation
def load_prepare_data(args):
    activations_raw = load_from_json(args.inpath_activations)
    activations = {k: np.array(v) for k, v in activations_raw.items() if np.array(v).shape == (40, 300)}

    loop_positions = {os.path.basename(p): read_loop_positions_from_pdb(p, "loop_position_target")
                      for p in glob.glob(os.path.join(args.inpath_extracted_loops, "*.pdb"))}

    features_res = pd.DataFrame.from_dict(load_from_csv(args.inpath_features_per_residue), orient="index").reset_index(drop=True)
    features_res["residue_index"] = features_res["residue_index"].astype(int)

    rows, emb_rows = [], []
    for loop_fn, (start, stop) in loop_positions.items():
        base_id = loop_fn.rsplit('_', 2)[0] + ".pdb"
        df_loop = features_res[(features_res["id"] == base_id) & (features_res["residue_index"].between(start, stop))].copy()
        df_loop["loop_filename"] = loop_fn
        rows.append(df_loop)

        key = loop_fn.replace("_m1_", "_m1_A_")
        acts = activations.get(key, [])[start-1:stop]
        for offset, vec in enumerate(acts):
            emb_rows.append({"loop_filename": loop_fn, "residue_index": start + offset, **{f"act_{j}": val for j, val in enumerate(vec)}})

    df_features = pd.concat(rows, ignore_index=True)
    df_emb = pd.DataFrame(emb_rows)
    return pd.merge(df_features, df_emb, on=["loop_filename", "residue_index"], how="inner")

# Main execution
def main(args):
    os.makedirs(args.outpath, exist_ok=True)
    merged_path = os.path.join(args.outpath, "df_merged.csv")

    if args.overwrite or not os.path.exists(merged_path):
        df_merged = load_prepare_data(args)
        df_merged.to_csv(merged_path, index=False)
    else:
        df_merged = pd.read_csv(merged_path)

    X = df_merged.filter(like="act_").values
    for method in args.dimred_methods:
        reducer = DIMRED_METHODS[method]
        df_components = reducer(X)
        for feature in args.feature_list:
            df_components[feature] = df_merged[feature].values
            plot_2d(df_components, method, feature, os.path.join(args.outpath, f"{method}_colored_by_{feature}.png"))

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('--inpath_features_per_residue', default="features.csv")
    parser.add_argument('--inpath_extracted_loops', default="extracted_loops")
    parser.add_argument('--inpath_activations', default="activations.json")
    parser.add_argument('--outpath', default="dimreduction_out")
    parser.add_argument('--feature_list', nargs='+', default=['residue_index', 'res_name', 'abego', 'phi', 'psi', 'chi1', 'chi2', 'hb', 'scnc', 'prsm', 'E'])
    parser.add_argument('--dimred_methods', nargs='+', choices=['pca', 'umap', 'tsne'], default=['pca', 'umap'])
    parser.add_argument('--overwrite', action='store_true')
    main(parser.parse_args())
