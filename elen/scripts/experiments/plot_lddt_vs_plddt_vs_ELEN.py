#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, pearsonr, spearmanr

# ---- Publication-style settings ----
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
    "figure.titlesize": 18,
    "axes.linewidth": 1.3,
    "axes.edgecolor": "black",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "figure.dpi": 200,
})

# --- Load data ---
df = pd.read_csv(sys.argv[1])

# Rename columns for plot labels
df = df.rename(columns={
    "lddt": "lDDT",
    "elen": "ELEN Score",
    "plddt": "pLDDT"
})

cols = ["lDDT", "ELEN Score", "pLDDT"]

# ---- Plotting ----
sns.set_style("white")

def corr_regplot(x, y, **kwargs):
    # Scatter points
    ax = plt.gca()
    ax.scatter(x, y, s=16, color="black", alpha=0.75, edgecolor='none')
    # Regression line
    if len(x) > 1:
        slope, intercept, _, _, _ = linregress(x, y)
        xs = np.linspace(np.min(x), np.max(x), 100)
        ys = slope * xs + intercept
        ax.plot(xs, ys, color="red", linewidth=2, zorder=3)
        # Correlations
        pear, _ = pearsonr(x, y)
        spear, _ = spearmanr(x, y)
        txt = f"Pearson $r$ = {pear:.3f}\nSpearman $\\rho$ = {spear:.3f}"
        ax.text(
            0.05, 0.95, txt, fontsize=12, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.93)
        )
    ax.grid(True, zorder=0, alpha=0.28)
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def hist_diag(x, **kwargs):
    ax = plt.gca()
    ax.hist(x, bins=28, color="black", alpha=0.75)
    ax.grid(True, zorder=0, alpha=0.28)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

g = sns.PairGrid(df[cols], diag_sharey=False, despine=False)
g.map_upper(corr_regplot)
g.map_lower(corr_regplot)
g.map_diag(hist_diag)

# Axis labels
for i, label in enumerate(cols):
    g.axes[-1, i].set_xlabel(label)
    g.axes[i, 0].set_ylabel(label)

# Title (set closer to plot)
g.fig.suptitle("Scatter Matrix of lDDT, ELEN Score, and pLDDT", fontsize=18, y=0.98, weight="semibold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{sys.argv[2]}_scatter_matrix_publication_style.png", bbox_inches='tight', dpi=200)
plt.close()
print("Saved as scatter_matrix_publication_style.png")
