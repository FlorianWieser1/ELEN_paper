#!/home/florian_wieser/anaconda3/envs/ares/bin/python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.cm 
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.manifold import TSNE
from sklearn import decomposition
import matplotlib.cm as cm
from scipy import stats
import sys

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


def plotCorrelation(data, xaxis, xlabel, color, outname, outdir):
    # calculate correlation
    pearson = round(data["pred"].corr(data[xaxis]), 4)
    print(f"Pearson correlation with {xaxis}: {pearson}") 

    # make xy-scatterplot with regression line
    sns.regplot(data=data, x=xaxis, y="pred", color=color, scatter_kws={'s':12, 'alpha':0.3}, line_kws={'lw':1.5})
    plt.plot([], [], ' ', label = '$\mathrm{R^{2}}:$ ' + str(pearson))
    if xaxis == "target" :
        r2 = r2_score(data['target'], data['pred'])
        mae = mean_absolute_error(data['target'], data['pred'])
        rmse = np.sqrt(mean_absolute_error(data['target'], data['pred']))

        plt.plot([], [], ' ', label = "n: " + str(len(data['pred']))[:4])
        plt.plot([], [], ' ', label = "r2: " + str(r2)[:4])
        plt.plot([], [], ' ', label = "MAE: " + str(mae)[:4])
        plt.plot([], [], ' ', label = "RMSE: " + str(rmse)[:4])
    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left')
    plt.title('rms predicted vs. ' + xaxis)
    plt.xlabel(xlabel)
    plt.ylabel('$\mathrm{rms _{pred}}$ [$\mathrm{\AA}$]')
    plt.savefig(f"{outdir}/{outname}", bbox_inches='tight')
    plt.clf()


def plotFeatureCorrelation(data, hue_feature, xaxis, xlabel, color, outname, outdir):
    pearson = round(data["pred"].corr(data[xaxis]), 4)
    print(f"Pearson correlation with {xaxis}: {pearson}") 

    colors = ["yellow", "orange", "red", "violet", "blue", "cyan", "green", "brown"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)
    # if its a string feature, treat plotting differently
    if "id" in hue_feature or "ab" in hue_feature:
        if "id" in hue_feature:
            aa_list = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
            cmap = plt.get_cmap('tab20')
        else:
            aa_list = ["A", "B", "E", "G", "O"]

        col_list = cmap(np.linspace(0, 1, len(aa_list)))
        col_dict = dict(zip(aa_list, col_list))
        ax = sns.scatterplot(data=data, x=xaxis, y="pred", hue=hue_feature, palette=col_dict, s=14, alpha=0.3, linewidth=0, legend=True)
        sns.regplot(data=data, x=xaxis, y="pred", color='black', scatter=False, line_kws={'lw':0.5})
        ax.legend(bbox_to_anchor=(1.0, 1.01), fontsize=8)
    else:
        #colors = ["yellow", "orange", "red", "violet", "blue", "cyan", "green", "brown"]
        #cmap = LinearSegmentedColormap.from_list("mycmap", colors)
        cmap = plt.get_cmap('hsv')

        ax = sns.scatterplot(data=data, x=xaxis, y="pred", hue=hue_feature, cmap=cmap, s=14, alpha=0.3, linewidth=0, legend=False)
        sns.regplot(data=data, x=xaxis, y="pred", color='black', scatter=False, line_kws={'lw':0.5})
        r2 = r2_score(data['target'], data['pred'])
        mae = mean_absolute_error(data['target'], data['pred'])
        rmse = np.sqrt(mean_absolute_error(data['target'], data['pred']))
        plt.plot([], [], ' ', label = '$\mathrm{R^{2}}:$ ' + str(pearson))
        plt.plot([], [], ' ', label = "n: " + str(len(data['pred']))[:4])
        plt.plot([], [], ' ', label = "r2: " + str(r2)[:4])
        plt.plot([], [], ' ', label = "MAE: " + str(mae)[:4])
        plt.plot([], [], ' ', label = "RMSE: " + str(rmse)[:4])
        feat_data = np.nan_to_num(data[hue_feature], nan=0).astype(float)
        norm = plt.Normalize(np.nanmin(feat_data), np.nanmax(feat_data))
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(str(hue_feature), rotation=270, fontsize=14)

        plt.legend(frameon=False, handletextpad=-2.0, loc='upper left')
    plt.title(f"rms predicted and {hue_feature} vs. {xaxis}")
    plt.xlabel(xlabel)
    plt.ylabel('$\mathrm{rms _{pred}}$ [$\mathrm{\AA}$]')
    plt.savefig(f"{outdir}/{outname}-{hue_feature}", bbox_inches='tight', dpi=150)
    plt.clf()

def plotScoreR1Z1(data, feature, target_value, xlabel, title, outname, outdir):
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

# ideal z1 for alpha_helix "1.5310"
# ideal ncc r1 for alpha_helix "1.5367 2.2943 1.7332"
#plotScore(z1_data, 'z1', 1.5310, "helix rise per residue [$\mathrm{\AA}$]", "protARES learned helix features", "score-vs-z1")
#plotScore(r1_data, 'r1', 2.2943, "helix radius (CA atom) [$\mathrm{\AA}$]", "protARES learned helix features", "score-vs-r1")
#to_use = ['nres_tot', 'id_ncap','cpr_HB']

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

"""
to_use = ['nres_tot', 'nres_loop', # general features - references
          'npr_HB', 'ncap_HB', 'ccap_HB', 'cpr_HB',
          'id_npr', 'id_ncap', 'id_ccap', 'id_cpr',  # sequence identity
          'HH_ang', 'HH_dihe', 'HL_ang', 'LH_ang', 'T4_ncap', 'T4_ccap', # global angles 
          'ab_npr','ab_ncap', 'ab_ccap', 'ab_cpr',  # abego
          'phi_npr', 'phi_ncap', 'phi_ccap','phi_cpr', 'psi_npr', 'psi_ncap', 'psi_ccap', 'psi_cpr',
          'chi1_npr','chi1_ncap', 'chi1_ccap', 'chi1_cpr', 'chi2_npr', 'chi2_ncap', 'chi2_ccap', 'chi2_cpr', #sc
          'hb_HHL_tot', 'hb_HHL_loop', 'hb_HHL_npr', 'hb_HHL_ncap', 'hb_HHL_ccap', 'hb_HHL_cpr',  # nr of hbonds
          'scnc_HHL_tot', 'scnc_HHL_loop', 'scnc_HHL_npr', 'scnc_HHL_ncap', 'scnc_HHL_ccap', 'scnc_HHL_cpr', # side-chain neighbors
          'prsm_HHL_tot', 'prsm_HHL_loop', 'prsm_HHL_npr', 'prsm_HHL_ncap', 'prsm_HHL_ccap', 'prsm_HHL_cpr', # SASA 
          'E_HHL_tot', 'E_HHL_loop', 'E_HHL_npr', 'E_HHL_ncap', 'E_HHL_ccap', 'E_HHL_cpr']

          # 'hb_HB_tot', 'hb_HB_loop', 'hb_HB_npr', 'hb_HB_ncap', 'hb_HB_ccap', 'hb_HB_cpr',  # number of hbonds
          # 'scnc_HB_tot', 'scnc_HB_loop', 'scnc_HB_npr', 'scnc_HB_ncap', 'scnc_HB_ccap', 'scnc_HB_cpr', # side-chain neighbors
          # 'prsm_HB_tot', 'prsm_HB_loop', 'prsm_HB_npr', 'prsm_HB_ncap', 'prsm_HB_ccap', 'prsm_HB_cpr', # SASA 
          # 'E_HB_tot', 'E_HB_loop', 'E_HB_npr', 'E_HB_ncap', 'E_HB_ccap', 'E_HB_cpr'] # energy
"""

def runTSNE(data):
    tsne = TSNE(init='pca', n_components=2, verbose=1, random_state=1, perplexity=30, n_iter=5000)
    df = data[['pca_{}'.format(x) for x in range(1, 151)]]
    Y_tsne = tsne.fit_transform(df)
    data['tsne0'] = Y_tsne[:,0]
    data['tsne1'] = Y_tsne[:,1]
    return data

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

# plot PCA: either with string or float valued HHL features
def PlotPCA(data, comp1, comp2, feat, outtag, outdir):
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
    plt.savefig(f"{outdir}/{outtag}_{comp1}-{comp2}-{feat}.png", bbox_inches='tight')
    plt.tight_layout()
    plt.clf()


def plotPCAs(data, outtag, outdir):
    data = cleanData(data)
    data = runPCA(data) 
    for feat in to_use:
        PlotPCA(data, 'pca0', 'pca1', feat, outtag, outdir)

def plotTSNEs(data, outtag, outdir):
    data = cleanData(data)
    data = runTSNE(data) 
    for feat in to_use:
        PlotPCA(data, 'tsne0', 'tsne1', feat, outtag, outdir)

def plotFeatureCorrelations(data, outtag, outdir):
    data = cleanData(data)
    for feat in to_use:
        plotFeatureCorrelation(data, feat, "target", "rms [A]", "blue", outtag, outdir)
