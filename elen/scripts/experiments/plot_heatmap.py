#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse as ap
import os
import sys


def plot_heatmap(df, tag, range_min_max, cmap, width=6):
    # Creating the heatmap
    plt.figure(figsize=(width, 8))  # You can adjust the size as needed
    ax = sns.heatmap(df, annot=True, cmap=cmap, fmt=".3f", annot_kws={'size':12},
                     vmin=range_min_max[0], vmax=range_min_max[1])  # 'fmt' is optional and can be adjusted
    plt.title(f"Heatmap of {tag}")
    ax.xaxis.tick_top()  # Moves x-axis labels to the top
    plt.savefig(f"{tag}_heatmap.png", bbox_inches='tight')
    
def load_df(inpath):
    df = pd.read_csv(inpath, index_col=0)
    df = df.abs()
    if 'ground_truth' in df.index:
        df = df.drop('ground_truth')
        
    n_train = df.loc['n_train']
    df_n_train = pd.DataFrame(n_train).transpose()
    
    print(df)
    plt.figure(figsize=(5, 5))
    plt.scatter(df.loc['n_train'], df.loc['spear_lddt'], color='blue')
    plt.xlabel("Size train data")
    plt.ylabel("spear lddt")
    #sns.regplot(data=df, x='n_train', y='R_lddt')
    plt.savefig('scatter.png')
    plt.clf()
    
    if 'n_train' in df.index:
        df = df.drop('n_train')

    print(df)
    return df, df_n_train


###############################################################################
def main(args):
    # Load your CSV data into a DataFrame
    #print(df)
    
    #df['mean'] = df.drop('control', axis=1).mean(axis=1)
    #df['delta'] = df['mean'] - df['control']
    tag = os.path.basename(args.inpath).replace(".csv", "")
    df, df_n_train = load_df(args.inpath)
    print(df_n_train)
    
    if args.mode == "correlation":
        if args.inpath_reference:
            tag_ref = os.path.basename(args.inpath_reference).replace(".csv", "")
            plot_heatmap(df, tag, [0, 1], "viridis")      
            df = df.drop('elen')
            
            df_ref = load_df(args.inpath_reference)
            df_ref = df_ref.rename(columns={
                                            'R_all_loc': 'R_loo_loc',
                                            'τ_all_loc': 'τ_loo_loc',
                                            'y_all_loc': 'ρ_loo_loc'})
            plot_heatmap(df_ref, tag_ref, [0, 1], "viridis")      
            diff = df_ref.subtract(df)
            plot_heatmap(diff, f"{tag}_diff_{tag_ref}", [-0.15, 0.15], "coolwarm")
        else:             

            # Select columns from 'HH_2' to 'HH_10'
            selected_columns = df.loc[:, 'HH_2':'HH_10']
            rowwise_average = selected_columns.mean(axis=1)
            df['HH_2_to_HH_10_avg'] = rowwise_average
            print(df)
            #df_sorted = df.sort_values(by='ρ_loo_loc', ascending=False)
            #plot_heatmap(df_sorted, f"{tag}_sorted", [None, None], "viridis")      
            plot_heatmap(df, f"{tag}", [None, None], "inferno", 16)      
            print(df_n_train)
            df_n_train = df_n_train.T
            plt.clf()
            plt.figure(figsize=(10, 2))  # You can adjust the size as needed
            plt.bar(df_n_train.index, df_n_train['n_train'], color='blue')
            plt.savefig("barplot.png", bbox_inches='tight') 
            
    if args.mode == "top1":
        if args.inpath_reference:
            tag_ref = os.path.basename(args.inpath_reference).replace(".csv", "")

            df_sorted = df.sort_values(by='avg_lddt_loo_loc', ascending=False)
            plot_heatmap(df_sorted, f"{tag}_sorted", [None, None], "Greys", 2)      
            plot_heatmap(df, tag, [None, None], "Greys", 2)      
           
            df = df.drop('elen')

            df_ref = load_df(args.inpath_reference)
            print(df_ref)
            df_ref = df_ref.rename(columns={'avg_lddt_all_loc': 'avg_lddt_loo_loc'})
            plot_heatmap(df_ref, tag_ref, [None, None], "Greys", 2)      
            diff = df.subtract(df_ref)
            plot_heatmap(diff, f"{tag}_diff_{tag_ref}", [-0.05, 0.05], "coolwarm", 2)
        else:             
            df_sorted = df.sort_values(by='avg_lddt_loo_loc', ascending=False)
            plot_heatmap(df_sorted, tag, [None, None], "Greys", 2)      
   
        
    
    

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    # paths are subpaths from dir_??, for slurm parallelization
    parser.add_argument('--inpath', type=str, default=f"/home/florian_wieser/software/ARES/geometricDL/edn/ELEN_testing/cameo_3month/out_9fcwo4ii/correlations_loc_loo-res.csv")
    parser.add_argument('--mode', type=str, default="correlation", choices=["correlation", "top1"])
    parser.add_argument('--inpath_reference', default=None)
    #parser.add_argument('--inpath_reference', default=f"/home/florian_wieser/software/ARES/geometricDL/edn/ELEN_testing/cameo_3month/out_9fcwo4ii/correlations_loc_all-res.csv")

    args = parser.parse_args()
    main(args)
