#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
#SBATCH -J plot_dimred
#SBATCH -o plot_dimred.log
#SBATCH -e plot_dimred.err

import os
import sys
import shutil
import logging
import numpy as np
import pandas as pd
import argparse
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn.manifold import TSNE
import umap

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

# Custom utilities
from elen.shared_utils.utils_io import load_from_json, load_from_csv

dict_panel_name = {
    'residue_index' : 'H',
    'res_name' : 'A',
    'abego' : 'B',
    'phi' : 'E',
    'psi' : 'F',
    'hb' : 'D',
    'prsm' : 'C',
    'E': 'G',
}

dict_titles = {
    'residue_index' : 'Residue Index',
    'res_name' : 'Side-chain Identity',
    'abego' : 'ABEGO',
    'phi' : 'Phi Angle',
    'psi' : 'Psi Angle',
    'hb' : 'Number of H-bonds',
    'scnc' : 'Number of Side-chain contacts',
    'prsm' : 'Per-residue SASA',
    'E': 'Rosetta Energy',
    'nres': 'Number of Residues',
    'ccap': 'Residue Number Ccap',
    'ncap': 'Residue Number Ncap',
    'id_loop': 'Side-chain Identities Loop',
    'id_ccap': 'Side-chain Identity Ccap',
    'id_ncap': 'Side-chain Identity Ncap',
    'ab_loop': 'ABEGO Loop',
    'ab_ccap': 'ABEGO Ccap',
    'ab_ncap': 'ABEGO Ncap',
    'phi_ccap': 'Phi Angle Ccap',
    'phi_ncap': 'Phi Angle Ncap',
    'psi_ccap': 'Psi Angle Ccap',
    'psi_ncap': 'Psi Angle Ncap',
    'chi1_ccap': 'Chi1 Angle Ccap',
    'chi1_ncap': 'Chi1 Angle Ncap',
    'chi2_ccap': 'Chi2 Angle Ccap',
    'chi2_ncap': 'Chi2 Angle Ncap',
    'hb_loop': 'Number of H-bonds Loop',
    'hb_ccap': 'Number of H-bonds Ccap',
    'hb_ncap': 'Number of H-bonds Ncap',
    'scnc_loop': 'Number of Side-chain Loop',
    'scnc_ccap': 'Number of Side-chain Ccap',
    'scnc_ncap': 'Number of Side-chain Ncap',
    'prsm_loop': 'Per-residue SASA Loop',
    'prsm_ccap': 'Per-residue SASA Ccap',
    'prsm_ncap': 'Per-residue SASA Ncap',
    'E_loop': 'Rosetta Energy Loop',
    'E_ccap': 'Rosetta Energy Ccap',
    'E_ncap': 'Rosetta Energy Ncap',
}
###############################################################################
# Dimensionality reduction methods
def calculate_pca(activations):
    pca = decomposition.PCA(n_components=2, random_state=123)
    components = pca.fit_transform(activations)
    return pd.DataFrame(components, columns=['pca_1', 'pca_2'])

def calculate_umap(activations):
    reducer = umap.UMAP(n_components=2, n_neighbors=100, min_dist=0.5, random_state=123)
    components = reducer.fit_transform(activations)
    return pd.DataFrame(components, columns=['umap_1', 'umap_2'])

def calculate_tsne(activations):
    tsne_model = TSNE(init='pca', n_components=2, verbose=0, random_state=123, perplexity=10, n_iter=5000)
    components = tsne_model.fit_transform(activations)
    return pd.DataFrame(components, columns=['tsne_1', 'tsne_2'])

###############################################################################

def plot_2d_dimreduction(df, reduction, feature, outpath):
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,  # Smaller legend font
        "figure.titlesize": 16,
    })

    plt.figure(figsize=(8, 6))

    categorical = {'res_name', 'abego', 'aa', 
                   'loop_filename', 'id', 'ab_loop', 'ab_ccap', 'ab_ncap',
                   'id_loop', 'id_ccap', 'id_ncap'}
    feature_data = df[feature]
    
    if feature not in categorical:
        df = df.copy()
        feature_data = df[feature].replace('', np.nan).astype(float)
        z_scores = zscore(feature_data)
        mask = abs(z_scores) < 5
        df = df[mask]    # <<---- FILTER THE WHOLE DATAFRAME!
        feature_data = df[feature].replace('', np.nan).astype(float)
    else:
        feature_data = df[feature]
    
    x, y = df[f'{reduction}_1'], df[f'{reduction}_2']

    if pd.api.types.is_numeric_dtype(feature_data):
        scatter = plt.scatter(x, y, c=feature_data, cmap='turbo', s=20, alpha=0.8)
        cbar = plt.colorbar(scatter)
    else:
        categories = pd.Categorical(feature_data)
        palette = sns.color_palette('Set1', len(categories.categories))
        palette_map = dict(zip(categories.categories, palette))
        scatter = sns.scatterplot(
            x=x, y=y, hue=feature_data, palette=palette_map,
            s=22, alpha=0.85, edgecolor='none', legend=True
        )

        # Custom compact legend outside plot
        handles, labels = scatter.get_legend_handles_labels()
        plt.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=(1.01, 1),
            loc='upper left',
            borderaxespad=0.,
            frameon=False,
            ncol=1,
            markerscale=0.8,     # Shrink legend marker size
            handlelength=1.5,    # Shorten legend handles
            handletextpad=0.6,
            columnspacing=0.7,
            title_fontsize=12,
            fontsize=10
        )
    
    #plt.text(-0.10, 1.02, dict_panel_name[feature], fontsize=22, fontweight='bold',
    plt.text(-0.10, 1.02, "A", fontsize=22, fontweight='bold',
                 transform=plt.gca().transAxes, va='top', ha='right')
    
    plt.xlabel(f'{reduction.upper()} 1', fontsize=14)
    plt.ylabel(f'{reduction.upper()} 2', fontsize=14)
    plt.title(f'{dict_titles[feature]}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.82, 1])  # Leave room for legend
    plt.savefig(outpath, dpi=100, bbox_inches='tight')
    plt.close()
    

###############################################################################
def main(args):
    # Overwrite output if requested
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)

    # --- Load and filter activations ---
    activations_raw = load_from_json(args.inpath_activations)
    desired_shape = (40, 300)
    filtered = {name: acts for name, acts in activations_raw.items()
                if np.array(acts).shape == desired_shape}
    num_dropped = len(activations_raw) - len(filtered)
    if num_dropped:
        logging.warning(f"dropped {num_dropped} activations not of shape {desired_shape}")

    # --- Load features per residue ---
    dict_features_per_residue = load_from_csv(args.inpath_features_per_residue)
    df_features = pd.DataFrame.from_dict(dict_features_per_residue, orient="index").reset_index(drop=True)
    cols_to_convert = df_features.columns.difference(
        ['id', 'res_name', 'abego', 'aa']
    )
    df_features[cols_to_convert] = df_features[cols_to_convert].apply(pd.to_numeric, errors='raise')
    df_features.set_index('id', inplace=True)

    # --- Merge activations with features ---
    merged_data = []
    for key, acts in filtered.items():
        df_act = pd.DataFrame(acts, columns=[f'act_{i}' for i in range(len(acts[0]))])
        try:
            df_feat = df_features.loc[[key]].reset_index(drop=True)
        except KeyError:
            logging.warning(f"Skipping {key} because id not found in features")
            continue
        if len(df_feat) != len(df_act):
            logging.warning(f"Skipping {key} due to mismatch in activation/features length")
            continue
        df_merged = pd.concat([df_act, df_feat], axis=1)
        df_merged["id"] = key
        merged_data.append(df_merged)

    if not merged_data:
        raise ValueError("No objects to concatenate after filtering.")

    df_all = pd.concat(merged_data, ignore_index=True)

    # --- Extract activation vectors (per-residue) ---
    X = df_all[[col for col in df_all.columns if col.startswith('act_')]].to_numpy()
    reduction_map = {'umap': calculate_umap, 'pca': calculate_pca, 'tsne': calculate_tsne}
    valid_per_residue_features = [f for f in args.pr_feature_list if f in df_all.columns]

    # --- Per-residue plots ---
    for dimred_method in args.dimred_methods:
        df_components = reduction_map[dimred_method](X)
        df_components = pd.concat([df_components, df_all[valid_per_residue_features + ['id']].reset_index(drop=True)], axis=1)
        for feature in valid_per_residue_features:
            outpath = os.path.join(args.outpath, f"residue_{dimred_method}_colored_by_{feature}.png")
            print(f"Plotting per-residue dimreduction plot for features {feature}")
            plot_2d_dimreduction(df_components, dimred_method, feature, outpath)
            
    # --- Load and prepare loop features ---
    dict_features_per_loop = load_from_csv(args.inpath_features_per_loop)
    df_loops = pd.DataFrame.from_dict(dict_features_per_loop, orient="index").reset_index()
    df_loops.rename(columns={'index': 'loop_filename'}, inplace=True)
    df_loops = df_loops.apply(pd.to_numeric, errors='ignore')

    loop_feature_cols = df_loops.columns.difference(['loop_filename'])
    df_numeric = df_loops[loop_feature_cols].select_dtypes(include=[np.number]).dropna()
    X_loop = df_numeric.to_numpy()
    df_loops_clean = df_loops.loc[df_numeric.index].reset_index(drop=True)
    
    # --- Plotting per-loop features ---
    for dimred_method in args.dimred_methods:
        for feature in args.pl_feature_list:
            y = df_loops_clean[feature]
            df_components = reduction_map[dimred_method](X_loop)
            df_components["loop_filename"] = df_loops_clean["loop_filename"]
            df_components[feature] = y.values
            outpath = os.path.join(args.outpath, f"loop_{dimred_method}_colored_by_{feature}.png")
            print(f"Plotting per-loop dimreduction plot for features {feature}")
            plot_2d_dimreduction(df_components, dimred_method, feature, outpath)

###############################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='elen-dimred-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %h:%m:%S'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath_features_per_residue', default="features_per_residues.csv")
    parser.add_argument('--inpath_features_per_loop', default="loop_features.csv")
    parser.add_argument('--inpath_activations', default="activations.json")
    parser.add_argument('--outpath', default="dimreduction_out")
    parser.add_argument('--pr_feature_list', nargs='+', type=str,
                        default=['residue_index', 'res_name', 'abego', 'phi', 'psi', 'hb', 'prsm', 'E'])

    parser.add_argument('--pl_feature_list', nargs='+', type=str,
                        default=['nres', 'ccap', 'ncap', 'id_loop', 'id_ccap',
                                 'id_ncap', 'ab_loop', 'ab_ccap', 'ab_ncap', 'phi_ccap', 'phi_ncap',
                                 'psi_ccap', 'psi_ncap', 'chi1_ccap', 'chi1_ncap', 'chi2_ccap',
                                 'chi2_ncap', 'hb_loop', 'hb_ccap', 'hb_ncap', 'scnc_loop', 'scnc_ccap',
                                 'scnc_ncap', 'prsm_loop', 'prsm_ccap', 'prsm_ncap', 'E_loop', 'E_ccap', 'E_ncap'])
    parser.add_argument('--dimred_methods', nargs='+', type=str, default=['pca'],
                        choices=['pca', 'umap', 'tsne'])
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
