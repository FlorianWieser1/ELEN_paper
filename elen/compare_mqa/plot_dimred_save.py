#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
#SBATCH -J plot_dimred
#SBATCH -o plot_dimred.log
#SBATCH -e plot_dimred.err
# TODO naive way doesn't make things better, go on with this script
#TODO  run on bigger dataset, add number of residues features, add loop type, exposure
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import os
import sys
import glob
import shutil
import logging
import numpy as np
import pandas as pd
import argparse as ap
from sklearn import decomposition
from sklearn.manifold import TSNE
from scipy.stats import zscore

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.DSSP import DSSP

# custom utilities
from elen.shared_utils.utils_io import load_from_json, load_from_csv
from elen.compare_mqa.get_features_for_dimreduction import read_loop_positions_from_pdb
 
###############################################################################
def calculate_pca(activations):
    pca = decomposition.PCA(n_components=2, random_state=123)
    components = pca.fit_transform(activations)
    return pd.DataFrame(components, columns=['pca_1', 'pca_2'])


def calculate_umap(activations):
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=123)
    components = reducer.fit_transform(activations)
    return pd.DataFrame(components, columns=['umap_1', 'umap_2'])


def calculate_tsne(activations):
    tsne_model = TSNE(init='pca', n_components=2, verbose=0, random_state=123, perplexity=10, n_iter=5000)
    components = tsne_model.fit_transform(activations)
    return pd.DataFrame(components, columns=['tsne_1', 'tsne_2'])

###############################################################################

def plot_2d_dimreduction(df, reduction, feature, outpath):
    plt.figure(figsize=(8, 6))

    categorical_features = ['res_name', 'abego', 'loop_type']
    feature_data = df[feature]

    if feature not in categorical_features:
        # Filter out outliers using z-score
        z_scores = zscore(feature_data)
        mask = abs(z_scores) < 3

        # Apply mask to full DataFrame
        df = df[mask]
        feature_data = df[feature]

        # Clean up feature_data
        feature_data = feature_data.replace('', np.nan).astype(float)

    x = df[f'{reduction}_1']
    y = df[f'{reduction}_2']

    if pd.api.types.is_numeric_dtype(feature_data):
        scatter = plt.scatter(x, y, c=feature_data, cmap='turbo', s=20, alpha=0.8)
        cbar = plt.colorbar(scatter)
        cbar.set_label(feature)
    else:
        categories = pd.Categorical(feature_data)
        palette = sns.color_palette('Set2', len(categories.categories))
        sns.scatterplot(x=x, y=y, hue=categories, palette=palette, s=30, alpha=0.8)
        plt.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.plot([], [], ' ', label="n: " + str(len(df))[:5])
    plt.legend(title=feature, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel(f'{reduction.upper()} Component 1', fontsize=14)
    plt.ylabel(f'{reduction.upper()} Component 2', fontsize=14)
    plt.title(f'{reduction.upper()} colored by {feature}', fontsize=16)
    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()


def load_and_filter_activations(path_activations, loop_positions):
    activations_raw = load_from_json(path_activations)
    # only keep the ones whose shape is (40, 300)
    desired_shape = (40, 300)
    filtered = {
        name: acts
        for name, acts in activations_raw.items()
        if np.array(acts).shape == desired_shape
    }
    num_dropped = len(activations_raw) - len(filtered)
    if num_dropped:
        logging.warning(f"dropped {num_dropped} activations not of shape {desired_shape}")
        
    filtered_activations = {}
    for name, acts in activations_raw.items():
        key = name.replace("_m1_A_", "_m1_")
        if key not in loop_positions:
            print(f"{key} not found in loop.")
            continue
    
        # one‐off conversion of this entry to ndarray
        acts_arr = np.asarray(acts)       # now shape (40, feat_dim)
        print(f"loop_positions: {loop_positions.keys()}")
        start, stop = loop_positions[key]
        loop_acts = acts_arr[start - 1 : stop, :]
        filtered_activations[key] = loop_acts
        
    print(f"len(filtered_activations): {len(filtered_activations)}")
    return filtered_activations


def get_loop_positions_dict_from_pdbs(inpath_loops, pattern):
    loop_positions = {}
    for path_loop in glob.glob(os.path.join(inpath_loops, "*.pdb")):
        loop_start, loop_stop = read_loop_positions_from_pdb(path_loop, pattern)
        loop_positions[os.path.basename(path_loop)] = (loop_start, loop_stop)
    return loop_positions

###############################################################################
def main(args):
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    
    path_tmp_data = os.path.join(args.outpath, "df_merged.csv")
    if os.path.exists(path_tmp_data):
        print(f"Loading existing merged dataframe from {path_tmp_data}")
        df_merged = pd.read_csv(path_tmp_data)
    elif not os.path.exists(path_tmp_data):
        loop_positions = get_loop_positions_dict_from_pdbs(args.inpath_extracted_loops, "loop_position ")
        filtered_activations = load_and_filter_activations(args.inpath_activations, loop_positions)
        
        # TODO move on to features
        # 1) load your features and make sure residue_index is int
        loop_positions_target = get_loop_positions_dict_from_pdbs(args.inpath_extracted_loops, "loop_position_target")
        
        dict_features_per_residue = load_from_csv(args.inpath_features_per_residue)
        features_per_residue = (pd.DataFrame.from_dict(dict_features_per_residue, orient="index").reset_index(drop=True))
        cols_to_convert = features_per_residue.columns.difference(['id', 'res_name', 'abego'])
        features_per_residue[cols_to_convert] = features_per_residue[cols_to_convert].apply(pd.to_numeric, errors='raise')
        print(f"filtered_activations: {filtered_activations}") 
        print(f"loop_positions_target: {loop_positions_target}")
        print(f"features_per_residue: {features_per_residue}")
        sys.exit(0)
        
        # 3) build mapping from base‐id → list of loop fns
        loops_by_base = {}
        for loop_fn, (start, stop) in loop_positions_target.items():
            no_ext = os.path.splitext(loop_fn)[0]
            parts = no_ext.rsplit("_", 2)
            if len(parts) != 3:
                continue
            base_id = parts[0] + ".pdb"
            loops_by_base.setdefault(base_id, []).append(loop_fn)
        
        # 4) for each base_id, slice out only the loop residues
        filtered_rows = []
        for base_id, fn_list in loops_by_base.items():
            df_base = features_per_residue[features_per_residue["id"] == base_id]
            if df_base.empty:
                continue
            for loop_fn in fn_list:
                start, stop = loop_positions_target[loop_fn]
                mask = (df_base["residue_index"] >= start) & \
                       (df_base["residue_index"] <= stop)
                df_loop = df_base.loc[mask].copy()
                df_loop["id"] = loop_fn
                filtered_rows.append(df_loop)
        
        if filtered_rows:
            filtered_df = pd.concat(filtered_rows, ignore_index=True)
        else:
            filtered_df = pd.DataFrame(
                columns=list(features_per_residue.columns) + ["id"]
            )
        
        emb_rows = []
        for loop_fn, arr in filtered_activations.items():
            start, stop = loop_positions_target[loop_fn.replace("_m1_A_", "_m1_")]
            for offset, vec in enumerate(arr):
                resi = start + offset
                row = {
                    "id": loop_fn,
                    "residue_index": resi,
                }
                row.update({ f"act_{j}": float(vec[j]) for j in range(vec.shape[0]) })
                emb_rows.append(row)
        df_emb = pd.DataFrame(emb_rows)
        df_merged = pd.merge(
            filtered_df,
            df_emb,
            on=["id", "residue_index"],
            how="inner"
        )
        print(f"Merged has {len(df_merged)} residues for PCA")
        df_merged.to_csv(path_tmp_data)

    # Plotting per-residue activations
    act_cols = [c for c in df_merged.columns if c.startswith("act_")]
    X = df_merged[act_cols].to_numpy()
    filenames = df_merged["id"].tolist()
    resi_idx  = df_merged["residue_index"].tolist()

    for dimred_method in args.dimred_methods:
        for feature in args.feature_list:
            print(f"Plotting {dimred_method} for feature {feature}.") 
            y = df_merged[feature]
            
            reduction_map = {
                'umap': calculate_umap,
                'pca': calculate_pca,
                'tsne': calculate_tsne}
            
            df_components = reduction_map[dimred_method](X)
            df_components["id"] = filenames
            df_components["residue_index"] = resi_idx
            df_components[feature]         = y.values

            outpath = os.path.join(args.outpath, f"{dimred_method}_colored_by_{feature}.png")
            plot_2d_dimreduction(df_components, dimred_method, feature, outpath)
        print("Done.")

    # Plotting per-loop features
    dict_features_per_loop = load_from_csv(args.inpath_features_per_loop)
    df_loops = pd.DataFrame.from_dict(dict_features_per_loop, orient="index")
    df_loops.reset_index(inplace=True)
    df_loops.rename(columns={'index': 'i'}, inplace=True)
    df_loops = df_loops.apply(pd.to_numeric, errors='ignore')
    print(f"df_loops: {df_loops}")
    loop_feature_cols = df_loops.columns.difference(['id'])
    
    X_loop = df_loops[loop_feature_cols].select_dtypes(include=[np.number]).to_numpy()

    for dimred_method in args.dimred_methods:
        numeric_loop_features = df_loops[loop_feature_cols].select_dtypes(include=[np.number]).columns
        for feature in list(numeric_loop_features) + ['nres', 'loop_type']:

            print(f"Plotting loop {dimred_method} for feature {feature}.")
            y = df_loops[feature]

            df_components = reduction_map[dimred_method](X_loop)
            df_components["id"] = df_loops["id"]
            df_components[feature] = y.values

            outpath = os.path.join(args.outpath, f"loop_{dimred_method}_colored_by_{feature}.png")
            plot_2d_dimreduction(df_components, dimred_method, feature, outpath)

    print(f"dict_features_per_loop: {dict_features_per_loop}")

###############################################################################
if __name__ == "__main__":
    logger = logging.basicConfig(level=logging.INFO, format='elen-dimred-%(levelname)s(%(asctime)s): %(message)s',
                        datefmt='%y-%m-%d %h:%m:%s')
    parser = ap.ArgumentParser()
    parser.add_argument('--inpath_features_per_residue', default="features_per_residues.csv")
    parser.add_argument('--inpath_features_per_loop', default="loop_features.csv")
    parser.add_argument('--inpath_extracted_loops', default="extracted_loops")
    parser.add_argument('--inpath_activations', default="activations.json")
    parser.add_argument('--outpath', default="dimreduction_out")
    parser.add_argument('--feature_list', nargs='+', type=str, default=['residue_index', 'res_name', 'abego','phi', 'psi', 'chi1', 'chi2', 'hb', 'scnc', 'prsm', 'E'])
    parser.add_argument('--dimred_methods', nargs='+', type=str, default=['pca', 'umap'], choices=['pca', 'umap', 'tsne'])
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
