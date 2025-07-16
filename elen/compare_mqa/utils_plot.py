import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np 
import os
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib as mpl


### PLOTTING FUNCTIONS ##########################################################################################################

### correlation experiment
def plot_perres_correlation(df, method, color, path_plot):
    #df = df.astype(float)
    plt.figure(figsize=(6, 6))
    
    sns.regplot(data=df, x="GT", y=method, color=color, 
                scatter_kws={'s':12, 'alpha':0.3}, line_kws={'lw':1.5})

    pearson_corr = df['GT'].corr(df[method], method='pearson')
    spearman_corr = df['GT'].corr(df[method], method='spearman')
    kendall_corr = df['GT'].corr(df[method], method='kendall')
    #r2 = r2_score(df['ground_truth'], df[method])
    #mae = mean_absolute_error(df['ground_truth'], df[method])
    #rmse = np.sqrt(mean_absolute_error(df['ground_truth'], df[method]))

    plt.plot([], [], ' ', label = "n: " + str(len(df[method]))[:5])
    plt.plot([], [], ' ', label = r'$r$: ' + f"{str(pearson_corr)[:5]}")
    plt.plot([], [], ' ', label = r'$\rho$: ' + f"{str(spearman_corr)[:5]}")
    plt.plot([], [], ' ', label = r'$\tau$: ' + f"{str(kendall_corr)[:5]}")
    #plt.plot([], [], ' ', label = "r2: " + str(r2)[:5])
    #plt.plot([], [], ' ', label = "MAE: " + str(mae)[:5])
    #plt.plot([], [], ' ', label = "RMSE: " + str(rmse)[:5])
    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left')
    plt.title(f"Correlation of {method} predictions with ground truth lddt")
    plt.xlabel(f"target lddt")
    plt.ylabel(f"{method} lddt")
    plt.savefig(path_plot, bbox_inches='tight')
    plt.clf()
    return path_plot

def plot_perres_correlation_density(df, method, color, path_plot):
    """
    Plots the correlation between the 'GT' column and the given 'method' column.
    Instead of plotting thousands of individual scatter points, it uses a 2D density
    (histogram) to show where data is concentrated, along with a regression line.
    """
    plt.figure(figsize=(7.5, 6))
    
    # 2D histogram to visualize data density
    sns.histplot(data=df, x="GT", y=method, bins=50, cbar=True) 
    
    # Overlay a regression line (no scatter points here)
    sns.regplot(
        data=df, x="GT", y=method,
        scatter=False, 
        color=color,
        line_kws={'lw':1.5}
    )
    
    # Calculate correlations
    pearson_corr = df['GT'].corr(df[method], method='pearson')
    spearman_corr = df['GT'].corr(df[method], method='spearman')
    kendall_corr = df['GT'].corr(df[method], method='kendall')

    # You can also compute these if needed:
    # r2 = r2_score(df['GT'], df[method])
    # mae = mean_absolute_error(df['GT'], df[method])
    # rmse = np.sqrt(mean_absolute_error(df['GT'], df[method]))

    # Populate the legend with stats
    plt.plot([], [], ' ', label=f"n: {len(df):,}")
    plt.plot([], [], ' ', label=rf'$r$ (Pearson): {pearson_corr:.3f}')
    plt.plot([], [], ' ', label=rf'$\rho$ (Spearman): {spearman_corr:.3f}')
    plt.plot([], [], ' ', label=rf'$\tau$ (Kendall): {kendall_corr:.3f}')
    # plt.plot([], [], ' ', label=f"R^2: {r2:.3f}")
    # plt.plot([], [], ' ', label=f"MAE: {mae:.3f}")
    # plt.plot([], [], ' ', label=f"RMSE: {rmse:.3f}")

    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left')
    plt.title(f"Correlation of {method} predictions with ground truth LDDT")
    plt.xlabel("Target LDDT (GT)")
    plt.ylabel(f"{method} LDDT")
    
    plt.savefig(path_plot, bbox_inches='tight')
    plt.clf()
    
    return path_plot

def plot_perres_correlation_kde(df, method, path_plot):
    # Create KDE plot
    sns.kdeplot(x=df['ground_truth'], y=df[method], cmap="Blues", fill=True, thresh=0, levels=100)
    # Add titles and labels
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('KDE Density Plot')
    plt.savefig(path_plot, bbox_inches='tight')
    plt.clf()
    return path_plot


### Top1 loss experiment
def plot_top1_loss(df, method_1, method_2, path_plot):
    df[f"{method_1}_better"] = np.where(df[method_1] < df[method_2], 1, 0)
    df[f"{method_1}_same"] = np.where(df[method_1] == df[method_2], 1, 0)
    percentage = np.sum(df[f"{method_1}_better"] == 1) / (len(df) - np.sum(df[f"{method_1}_same"] == 1)) * 100.0
    min_val = 0
    max_val = max(df[method_1].max(), df[method_2].max())
    _, ax = plt.subplots()                                                       
    ax.scatter(df[method_2], df[method_1], s=30, alpha=0.3, marker='o', zorder=10, color='black', lw=1)
    line = mlines.Line2D([0, 1], [0, 1], color='black', lw=0.5)
    transform = ax.transAxes
    line.set_transform(transform)                                         
    ax.add_line(line)                                                   
    plt.xlabel(f"Top-1 loss {method_2} lddt [1]")
    plt.ylabel(f"Top-1 loss {method_1} lddt [1]")
    ax.set_xlim(min_val, max_val)                                                  
    ax.set_ylim(min_val, max_val)                                                  
    ax.set_aspect('equal', 'box')                                            
    plt.xticks()                                           
    plt.yticks()
    plt.minorticks_on()
    plt.grid(True, which='major', color='#a6a6a6', lw=0.25)
    plt.grid(True, which='minor', color='#DDDDDD', lw=0.25)
    plt.plot([], [], ' ', label = str(percentage)[:5] + "%")
    plt.plot([], [], ' ', label = "n: " + str(len(df[method_1]))[:5])
    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left')
    plt.title(f"Top-1 loss {method_1} vs. {method_2}")        
    plt.savefig(path_plot, bbox_inches='tight')
    plt.clf()
    return path_plot


import sys
def plot_heatmap(df, path_plot, range_min_max, cmap, sort_pattern, width=6):
    df = df.abs()
    if 'ground_truth' in df.index:
        df = df.drop('ground_truth')
    df = df.sort_values(by=sort_pattern, ascending=False)
    plt.figure(figsize=(width, 8))  # You can adjust the size as needed
    ax = sns.heatmap(df, annot=True, cmap=cmap, fmt=".3f", annot_kws={'size':12}, vmin=range_min_max[0], vmax=range_min_max[1])
    plt.title(f"Heatmap of {os.path.basename(path_plot)}")
    ax.xaxis.tick_top()  # Moves x-axis labels to the top
    plt.savefig(path_plot, bbox_inches='tight')
    plt.clf()
    return path_plot
