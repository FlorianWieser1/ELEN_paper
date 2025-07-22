#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
#SBATCH -J plot_dimred
#SBATCH -o plot_dimred.log
#SBATCH -e plot_dimred.err

import umap
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit
import matplotlib.cm as cm
import json
import os
import sys
import shutil
import numpy as np
import pandas as pd
import argparse as ap
from sklearn import decomposition
from sklearn.manifold import TSNE
DATA_PATH = os.environ.get('DATA_PATH')
PATH_PROJECT = os.environ.get("PROJECT_PATH")
sys.path.append(f"{PATH_PROJECT}/geometricDL/ARES_PL_sweep/scripts")
import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.DSSP import DSSP
from elen.shared_utils.utils_extraction import get_BioPython_DSSP
import re
PATH_DSSP="/home/florian_wieser/miniconda3/envs/elen_test/bin/mkdssp"
# TODO 
# Usage: ./dimension_reduction_feature_plots.py --inpath_dataset PATH --feature_list loop_length loop_type --outpath PATH
# Input: needs activations.json by predict.py and ground truth features e.g. entries loop_length in .pdb 
# TODO
# - extract features from .pdbs residue-wise (e.g. SS) - propably to be done one extract_RP script scale
# - put together dataframes residue-wise - each label has a feature, e.g. sasa, identity, SS

def plot_dimreduction_vs_feature(data, method, feat, outpath):
    light = np.array([1, 1, 1.5])
    norm = np.sqrt(data[f"{method}_1"]**2 + data[f"{method}_2"]**2 + data[f"{method}_3"]**2)
    light_effect = (data[f"{method}_1"] + data[f"{method}_2"] + 1.5*data[f"{method}_3"]) / norm
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    nr_elements = str(len(data[f"{method}_1"]))[:6]

    dict_categories = {
            'loop_length': ["2", "3", "4", "5", "6", "7", "8", "9", "10"],
            'surf_stat': ["exposed", "buried"],
            'loop_type': ["HH", "HE", "EH", "EE"],
            'aa': ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']}
    dict_cmaps = {
            'loop_length': "tab10",
            'surf_stat': "rainbow",
            'loop_type': "rainbow",
            'sasa': "rainbow",
            'aa': "tab20"}
    dict_titles = {
            'loop_length': "Loop Length",
            'surf_stat': "Surface Exposure",
            'loop_type': "Loop Type (SS)",
            'sasa': "Solvent Accessible Surface Area",
            'aa': "Amio Acid Identity"}
   
    if feat in ['surf_stat', 'loop_type', 'loop_length', 'aa']:
        categories = dict_categories[feat]
        cmap = plt.get_cmap(dict_cmaps[feat])
        col_list = cmap(np.linspace(0, 1, len(categories)))
        col_dict = dict(zip(categories, col_list))
        col = [col_dict.get(col).tolist() for col in data[feat]]

        _ = ax.scatter(data[f"{method}_1"], data[f"{method}_2"], data[f"{method}_3"], c=col, s=100, alpha=0.5, edgecolors='k', linewidth=0.5)
        markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') for color in col_dict.values()]
        plt.plot([], [], ' ', label = f"n: {nr_elements}")
        plt.legend(markers, col_dict.keys(), numpoints=1, loc='upper right', fontsize=12, framealpha=1)
    else: # float features
        cmap = plt.get_cmap(dict_cmaps['sasa'])
        feat_data = np.nan_to_num(data[feat], nan=0).astype(float)
        norm = plt.Normalize(np.nanmin(feat_data), np.nanmax(feat_data))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        _ = ax.scatter(data[f"{method}_1"], data[f"{method}_2"], data[f"{method}_3"], c=feat_data, cmap=cmap, s=100, alpha=0.5, edgecolors='k', linewidth=0.5)
        ax.tick_params(labelsize=10)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.tick_params(labelsize=14)
        plt.plot([], [], ' ', label = f"n: {nr_elements}")
        plt.legend(frameon=False, handletextpad=-2.0, loc='upper right', fontsize=12, framealpha=1)
        
    ax.set_xlabel(f"{method}_1", fontsize=14)
    ax.set_ylabel(f"{method}_2", fontsize=14)
    ax.set_zlabel(f"{method}_3", fontsize=14)
    plt.title(str(dict_titles[feat]), fontsize=22)
    wbtag = re.findall(r'_[a-z\d]{8}_', args.inpath_activations)[0][1:9]
    plt.savefig(outpath)#, bbox_inches='tight')
    plt.clf()


# 3 Analysis types: PCA, U-MAP, t-SNE
def calculate_PCA(activations):
    pca = decomposition.PCA(n_components=3, random_state=43)
    components = pca.fit_transform(activations)
    df = pd.DataFrame(components, columns=['pca_1', 'pca_2', 'pca_3'])
    return df

def calculate_UMAP(activations):
    reducer = umap.UMAP(n_components=3, n_neighbors=100, min_dist=0.5, random_state=15)
    components = reducer.fit_transform(activations)
    df = pd.DataFrame(components, columns=['umap_1', 'umap_2', 'umap_3'])
    return df

def calculate_TSNE(activations):
    tsne = TSNE(init='pca', n_components=3, verbose=0, random_state=15, perplexity=10, n_iter=5000)
    components = tsne.fit_transform(activations)
    df = pd.DataFrame(components, columns=['tsne_1', 'tsne_2', 'tsne_3'])
    return df

### HELPERS
def load_from_json(path):
    with open(path, "r") as file:
        data = json.load(file) 
    return data

def get_dict_of_avg_or_perres_act(data, feature):
    dict_act = {}
    for id, activations_list in data.items(): 
        activations = np.array(activations_list)
        if feature != "aa":
            dict_act[id] = np.mean(activations, axis=0)
        else:
            for idx, tensor in enumerate(activations):
                dict_act[f"{id}_{idx + 1}"] = tensor
    return dict_act


# gather features from .pdbs
def get_features_from_pdbs(feature_list, ids, path_dataset):
    df = pd.DataFrame()
    df['id'] = ids
    for feature in feature_list:
        list_feat_extr = []
        for id in ids:
            path_pdb = os.path.join(path_dataset, id)
            list_feat_extr.append(extract_feature_of_pdb(path_pdb, feature))
        df[feature] = list_feat_extr
    return df

# helper for ^
def extract_feature_of_pdb(path_test_pdb, feature):
    with open(path_test_pdb, 'r') as f:
        for line in f:
            if line[:len(feature) + 1] == f"{feature} ":  # not rms_stem
                feature_extracted = line[len(feature) + 1:].strip()
                return feature_extracted

# gather features from .pdbs
def get_aa_from_pdbs(ids, path_dataset):
    data = {} 
    for id in ids:
        fname, resnum = id.split(".pdb_", 1)
        path_pdb = os.path.join(path_dataset, fname + ".pdb")
        print(f"path_pdb: {path_pdb}")
        path_pdb = path_pdb.replace("_m1_A_", "_A_")
        print(f"path_pdb: {path_pdb}")
        _, sequence = get_BioPython_DSSP(path_pdb, PATH_DSSP)
        data[id] = sequence[int(resnum) - 1]
    df = pd.DataFrame(list(data.values()), index=data.keys(), columns=['aa'])
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id'}, inplace=True)
    return df


#######################################
def process_dimension_reduction(reduction_type, feature, df_features, activations, ids, outpath):
    """
    Handles the computation and plotting of dimension reduction techniques like UMAP or PCA.

    :param reduction_type: str, either 'umap' or 'pca'
    :param df_features: DataFrame, contains features data
    :param activations: numpy array, activation data for dimension reduction
    :param ids: list, identifiers for each data point
    :param outdir: str, path to save plots
    """
    # Map of reduction functions
    reduction_functions = {
        'umap': calculate_UMAP,
        'pca': calculate_PCA,
        'tsne': calculate_TSNE
    }

    # Calculate dimension reduction embeddings
    df_components = reduction_functions[reduction_type](activations)
    df_components['id'] = ids  # add identifiers to dimension reduction results
    df_final = pd.merge(df_features, df_components, on='id')
    # Print and plot results
    print(f"Plotting {reduction_type} for feature {feature}.")
    plot_dimreduction_vs_feature(df_final, reduction_type, feature, outpath)
    
    
       
###############################################################################
def main(args):
    wbtag = re.findall(r'_[a-z\d]{8}_', args.inpath_activations)[0][1:9]
    outdir = f"out_dimred_{wbtag}"
    
    if args.overwrite and os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    data = load_from_json(args.inpath_activations)
   
    for feature in args.feature_list:
        dict_act = get_dict_of_avg_or_perres_act(data, feature)
        activations = np.array(list(dict_act.values()))
        ids = list(dict_act.keys()) # add identifiers
        
        if feature != "aa":
            df_features = get_features_from_pdbs(args.feature_list, ids, args.inpath_dataset)
        else:
            df_features = get_aa_from_pdbs(ids, args.inpath_dataset)
        
        # run dimensionality reduction methods and plot analysis
        if args.umap:  
            outpath = os.path.join(outdir, f"umap_{feature}_{wbtag}.png")
            process_dimension_reduction('umap', feature, df_features, activations, ids, outpath)
        if args.pca: 
            outpath = os.path.join(outdir, f"pca_{feature}_{wbtag}.png")
            process_dimension_reduction('pca', feature, df_features, activations, ids, outpath)
        if args.tsne: 
            outpath = os.path.join(outdir, f"tsne_{feature}_{wbtag}.png")
            process_dimension_reduction('tsne', feature, df_features, activations, ids, outpath)
    print("Done.")

###############################################################################    
if __name__ == "__main__":
    parser = ap.ArgumentParser()
    #parser.add_argument('--inpath_dataset', default=f"{DATA_PATH}/LP_20/pdbs/test")
    parser.add_argument('--inpath_dataset', default=f"{DATA_PATH}/LP_AF_500k/pdbs/test")
    parser.add_argument('--inpath_activations', default=f"/home/florian_wieser/software/ARES/geometricDL/edn/edn_multi_labels_pr/out/activations.json")
    parser.add_argument('--feature_list', nargs='+', type=str, default=['loop_type', 'loop_length', 'sasa', 'surf_stat'])
    #parser.add_argument('--feature_list', nargs='+', type=str, default=['aa'])
    parser.add_argument('--umap', action='store_true', default=False, help='Calculate and plot U-MAP of activations.')
    parser.add_argument('--pca', action='store_true', default=True, help='Calculate and plot PCA of activations.')
    parser.add_argument('--tsne', action='store_true', default=False, help='Calculate and plot tSNE of activations.')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing output.')
    args = parser.parse_args()
    main(args)
