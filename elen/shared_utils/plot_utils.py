import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os, sys, glob
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn import decomposition
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import wandb 
#import pymol
#from pymol import cmd
import matplotlib.cm as cm
import subprocess

### PLOTTING HELPERS ##########################################################################################################
def log_plots(outpath, globtag, wandbtag):
    all_files = os.listdir(outpath)
    plot_files = [f for f in all_files if f.endswith(globtag)]

    for plot_file in plot_files:
            file_path = os.path.join(outpath, plot_file)
            wandb.log({wandbtag: [wandb.Image(file_path, caption=plot_file)]})


### PLOTTING FUNCTIONS ##########################################################################################################
def plot_target_corr(data, xaxis, yaxis, label, color, outpath):
    sns.regplot(data=data, x=xaxis, y=yaxis, color=color, scatter_kws={'s':12, 'alpha':0.3}, line_kws={'lw':1.5})
    pearson = round(data[yaxis].corr(data[xaxis]), 4)
    r2 = r2_score(data[xaxis], data[yaxis])
    mae = mean_absolute_error(data[xaxis], data[yaxis])
    rmse = np.sqrt(mean_absolute_error(data[xaxis], data[yaxis]))
    plt.plot([], [], ' ', label = "n: " + str(len(data[yaxis]))[:4])
    plt.plot([], [], ' ', label = '$\mathrm{R^{2}}:$ ' + str(pearson))
    plt.plot([], [], ' ', label = "r2: " + str(r2)[:4])
    plt.plot([], [], ' ', label = "MAE: " + str(mae)[:4])
    plt.plot([], [], ' ', label = "RMSE: " + str(rmse)[:4])
    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left')
    plt.title(f"{label} predicted vs. {label} target")
    plt.xlabel(f"{label}")
    plt.ylabel(f"{label}")
    plt.savefig(outpath, bbox_inches='tight')
    plt.clf()

def plot_target_correlation(data, label, color, outpath):
    print(f"Plotting prediction vs. target correlation scatter plot.")
    # calculate correlation
    pearson = round(data["pred"].corr(data["target"]), 4)
    print(f"Pearson correlation with {label}: {pearson}") 

    # make xy-scatterplot with regression line
    sns.regplot(data=data, x="target", y="pred", color=color, scatter_kws={'s':12, 'alpha':0.3}, line_kws={'lw':1.5})
    r2 = r2_score(data['target'], data['pred'])
    mae = mean_absolute_error(data['target'], data['pred'])
    rmse = np.sqrt(mean_absolute_error(data['target'], data['pred']))
    plt.plot([], [], ' ', label = "n: " + str(len(data['pred']))[:4])
    plt.plot([], [], ' ', label = '$\mathrm{R^{2}}:$ ' + str(pearson))
    plt.plot([], [], ' ', label = "r2: " + str(r2)[:4])
    plt.plot([], [], ' ', label = "MAE: " + str(mae)[:4])
    plt.plot([], [], ' ', label = "RMSE: " + str(rmse)[:4])
    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left')
    plt.title(f"{label} predicted vs. {label} target")
    plt.xlabel(f"{label}")
    plt.ylabel(f"{label}")
    plt.savefig(f"{outpath}", bbox_inches='tight')
    plt.clf()


def plot_per_residue_target_barcharts(df, label, outpath):
    for fname_pdb, row in df.iterrows():
        # parse data out of DataFrame
        target = row['target']
        pred = row['pred']
        delta = row['delta']
        res_id = row['res_id']
        print(f"Plotting per-residue prediction barplot for {fname_pdb[:-4]}.pdb." )

        width = 0.3  # Width of each bar
        positions = list(range(len(res_id)))
        plt.figure(figsize=(12,2))
        # Getting current axes
        ax = plt.gca()
        ax.grid(True, which='both', axis='y', linestyle='-', linewidth=0.7, zorder=0)

        # Plotting each category
        plt.bar(positions, target, width=width, color='forestgreen', label='target', zorder=3)
        plt.bar([p + width for p in positions], pred, width=width, color='royalblue', label='pred', zorder=3)
        plt.bar([p + width*2 for p in positions], delta, width=width, color='tomato', label='delta', zorder=3)
        
        # Labeling
        plt.xlabel('Residue id [1]', fontsize=12)
        plt.ylabel(f"{label}", fontsize=12)
        plt.title('Per residue predictions vs. targets', fontsize=12)
        plt.xlim(min(positions) - 0.5, max(positions) + width*3 + 0.5)
        plt.xticks([p + width for p in positions], res_id)
        plt.plot([], [], ' ', label = "avg delta: " + str(np.mean(delta))[:6])
        plt.legend(frameon=False, handletextpad=-2.0, loc='upper left', framealpha=0.5)
        outfile = os.path.join(outpath, fname_pdb[:-4] + "_per_res_prediction") 
        plt.savefig(outfile, bbox_inches='tight')
        plt.clf() 
    print("Plotting barplots done.")

"""
def plot_pdb(fname_pdb):
    os.environ['OMP_NUM_THREADS'] = '1'
    pymol.finish_launching(['pymol', '-c']) # Initialize PyMOL in command-line (headless) mode
    cmd.set("max_threads", 1)
    cmd.load(fname_pdb, "my_protein") # Load the PDB file
    cmd.show("cartoon", "my_protein")
    cmd.show("sticks", "my_protein")
    cmd.spectrum("b", "rainbow", "my_protein") # Color the protein according to B-factor values using the rainbow color scheme
    cmd.bg_color("white") # Set the background color to white
    cmd.orient("my_protein")
    cmd.zoom("my_protein")
    # Adjust the image resolution (optional)
    cmd.set("ray_trace_mode", 1)  # Ray tracing
    cmd.set("ray_shadows", 0)     # Disable shadows
    cmd.ray(600, 400)              # Set resolution (800x600)
    cmd.png(f"{fname_pdb[:-4]}_target_at_bfactor", dpi=200) # Save the image as a PNG file
    #pymol.cmd.quit() # Quit pymol
"""
def plot_pred_vs_r1z1(data, feature, target_value, xlabel, title, outname, outdir):
    g = sns.scatterplot(data=data, x=data[feature], y=data['pred'], color='black', s=14)
    #sns.lineplot(data=data, x=data[feature], y=data['pred'], color='black', lw=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('protARES score')
    optimum = plt.axvline(x=float(target_value), color='red', lw=1)
    plt.plot([], [], ' ', color='red', label = 'ideally: ' + str(target_value))
    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left')
    plt.savefig(f"{outdir}/{outname}", bbox_inches='tight')
    plt.clf()


### STATISTICS PLOTS
def plot_histogram(data, label, color, max_label, path_output):
    print(f"Plotting histogram for {label}.")
    num_bins = 20
    bins = np.linspace(0, max_label, num_bins + 1)
    plt.hist(data, bins=bins, color=color, log=True)
    plt.xlabel(f"{label}", fontsize=18)
    plt.ylabel('log count [1]', fontsize=18)
    plt.xlim(0,  max_label)
    plt.xticks(np.arange(0, max_label, step=(max_label / num_bins)), rotation='vertical', fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot([], [], ' ', label = "n: " + str(len(data)))
    plt.grid(color="#DDDDDD", which="both")
    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left', fontsize=16)
    plt.title(f"Distribution of {label} label", fontsize=20)
    plt.savefig(path_output, bbox_inches='tight')
    plt.clf()
    return path_output

def plot_violinplot(data, label, color, path_output):
    print(f"Plotting violinplot for {label}.")
    violin_parts = plt.violinplot(data, vert=False, showmeans=True, showextrema=True)
    for pc in violin_parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
    violin_parts['cmeans'].set_edgecolor(color)
    violin_parts['cmaxes'].set_edgecolor(color)
    violin_parts['cmins'].set_edgecolor(color)
    violin_parts['cbars'].set_edgecolor(color)
    plt.xlabel(f"{label}", fontsize=18)
    plt.ylabel('count [1]', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.plot([], [], ' ', label = "n: " + str(len(data)))
    plt.grid(color="#DDDDDD", which="both")
    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left', fontsize=16)
    plt.title(f"Distribution of {label} label", fontsize=20)
    plt.savefig(path_output, bbox_inches='tight')
    plt.clf()
    return path_output

def concatenate_plots(plot_list, mode, outpath):
    subprocess.run(['convert'] + plot_list + [mode, outpath])
    return outpath 
   
### PCA
to_use = ['nres', # general features - references
          'cpr',
          'id_ccap', 'id_cpr',  # sequence identity
          'ab_ccap', 'ab_cpr',  # abego
          'phi_cpr', 'psi_ccap',
          'chi1_ccap', 'chi2_cpr', #sc
          'hb_tot', 'hb_cpr',  # nr of hbonds
          'scnc_tot','scnc_cpr', # side-chain neighbors
          'prsm_tot', 'prsm_cpr', # SASA 
          'E_tot', 'E_cpr']

def runPCA(data):
    pca = decomposition.PCA(n_components=10, random_state=321)
    df = data[['pca_{}'.format(x) for x in range(1, 151)]]
    Y_pca = pca.fit_transform(df)
    exp_var = pca.explained_variance_ratio_
    #plt.bar(range(0,len(exp_var)), exp_var, alpha=0.5, align='center', label='Individual explained variance')
    #plt.savefig("expl_var")

    data['pca0'] = Y_pca[:,0]
    data['pca1'] = Y_pca[:,1]
    data['pca2'] = Y_pca[:,2]
    data['pca3'] = Y_pca[:,3]
    return data


def cleanData(data):
    # remove redundancy in amino-acids
    for hue_feature in ['id_npr', 'id_ncap', 'id_ccap', 'id_cpr']:  # sequence identity
        data[hue_feature] = np.where(data[hue_feature] == "HIS_D", "HIS", data[hue_feature])
        data[hue_feature] = np.where(data[hue_feature] == "CYS:disulfide", "CYS", data[hue_feature])
        data[hue_feature] = np.where(data[hue_feature] == "LEU:MP-C-connect", "LEU", data[hue_feature])
        data[hue_feature] = np.where(data[hue_feature] == "ASP:MP-N-connect", "ASP", data[hue_feature])
   
    # remove outliers from Rosetta score 
    for energy in ['E_tot', 'E_loop', 'E_npr', 'E_ncap', 'E_ccap', 'E_cpr']:
        data[energy] = data[energy].astype(float)
        data['z'] = np.abs(stats.zscore(data[energy]))
        data.drop(data[data['z'] >= 3].index, inplace = True)
    return data 


# plot PCA: either with string or float valued HHL features
def plot_PCA_vs_feature(data, comp1, comp2, feat, outdir):
    # different way of coloring if there are string features
    if "id" in feat or "ab" in feat:
        #print("Sequence feature!")
        if "id" in feat:
            cmap = plt.get_cmap('tab20')
            aa_list = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
        else:
            cmap = plt.get_cmap('hsv')
            aa_list = ["A", "B", "E", "G", "O"]
        col_list = cmap(np.linspace(0, 1, len(aa_list)))
        col_dict = dict(zip(aa_list, col_list))
        col = [col_dict.get(col).tolist() for col in data[feat]]
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.scatter(x=data[comp1], y=data[comp2], s=35, c=col, ec='grey', lw=0.2, alpha=0.3)
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in col_dict.values()]
        plt.plot([], [], ' ', label = f"n: {str(len(data[comp1]))[:6]}")
        plt.legend(markers, col_dict.keys(), numpoints=1, loc='upper right')
    else: # float features
        #print("Float feature!")
        cmap = cm.get_cmap('hsv')
        feat_data = np.nan_to_num(data[feat], nan=0).astype(float)
        norm = plt.Normalize(np.nanmin(feat_data), np.nanmax(feat_data))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.scatter(x=data[comp1], y=data[comp2], s=35, c=feat_data, cmap=cmap, norm=norm, ec='grey', lw=0.2, alpha=0.3)
        ax.tick_params(labelsize=10)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(str(feat), rotation=270, fontsize=16)
        plt.plot([], [], ' ', label = f"n: {str(len(data[comp1]))[:6]}")
        plt.legend(frameon=False, handletextpad=-2.0, loc='upper left')
    
    ax.set_xlabel(str(comp1), fontsize=20)
    ax.set_ylabel(str(comp2), fontsize=20)
    plt.title(str(feat), fontsize=24)
    outpath = os.path.join(outdir, f"{feat}_{comp1}-{comp2}_PCA.png")
    plt.savefig(outpath, bbox_inches='tight')
    plt.tight_layout()
    plt.clf()


def plot_PCAs(data, outdir):
    data = cleanData(data)
    data = runPCA(data) 
    for feat in to_use:
        plot_PCA_vs_feature(data, 'pca0', 'pca1', feat, outdir)

def calculate_regression_metrics(target, pred):
    R, _ = pearsonr(target, pred)
    spearman, _ = spearmanr(target, pred)
    r2 = r2_score(target, pred)
    mae = mean_absolute_error(target, pred)
    var_out = np.var(pred)
    return R, spearman, r2, mae, var_out
