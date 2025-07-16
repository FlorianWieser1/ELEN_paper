import os
import sys
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from elen.compare_mqa.utils_plot import plot_perres_correlation, plot_perres_correlation_density
from elen.compare_mqa.utils_plot import plot_top1_loss, plot_heatmap
from elen.shared_utils.utils_plot import concatenate_plots

### EXPERIMENTS ###############################################################
### CORRELATION
import os
import pandas as pd
from scipy.stats import spearmanr, pearsonr

def experiment_correlation(args, df):
    """
    Main correlation experiment, adapted for multiple ELEN columns.
    """
    # 1) Keep only valid GT rows (where GT <= 1.0, and not NaN).
    df = df[df['GT'].notna() & (df['GT'] <= 1.0)]
    
    # 2) Optionally, average or group data by loop/identifier for "global" scope.
    if args.scope == "global":
        # Average over numeric columns.
        numeric_cols = df.select_dtypes(include='number').columns
        # Decide on grouping column:
        group_col = 'loop_id' if args.pocket_type == 'LP' else 'identifier'
        # Group + mean
        df = df.groupby(group_col, as_index=False)[numeric_cols].mean()
        
        # Identify columns to drop:
        #   - If we're working with loop-based structures (LP), drop resid & loop_id 
        #   - else drop resid only
        columns_to_drop = ['resid', 'loop_id'] if args.pocket_type == 'LP' else ['resid']
    else:
        # If we're not in "global" scope, we remove these columns for correlation calculations.
        columns_to_drop = ['identifier', 'resid', 'loop_id'] if args.pocket_type == 'LP' \
                          else ['identifier', 'resid']
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # 3) Compute correlation metrics & generate a heatmap across *all* remaining columns (except GT).
    #    We'll store the CSV and heatmap in `args.outpath`.
    calculate_correlation_metrics(args.pocket_type, args.scope, args.outtag, df, args.outpath)

    # 4) Now generate correlation density plots for each method-like column:
    #    "methods" + any "ELEN_*" columns + "GT".
    prefix = f"{args.outtag}_corr_{args.scope}_{args.pocket_type}"
    
    # methods from the command line
    methods_base = args.methods if hasattr(args, 'methods') else []
    
    # find columns that start with 'ELEN_'
    elen_cols = [c for c in df.columns if c.startswith("ELEN_")]

    # additional columns to consider
    additional_methods = ["GT"]  # always include GT in the density plots

    # combine into a single list
    plot_methods = methods_base + elen_cols + additional_methods
    
    # 5) Create a separate correlation-density plot for each column in `plot_methods`,
    #    then vertically concatenate them into one final PNG.
    plots_methods = []
    for method in plot_methods:
        # output path for the single-plot
        outpath_plot_corr = os.path.join(args.outpath, f"{prefix}_{method}.png")
        plot_corr = plot_perres_correlation_density(df, method, "black", outpath_plot_corr)
        plots_methods.append(plot_corr)
    
    # The final “appended” plot that stacks them all:
    outpath_plots_methods = os.path.join(args.outpath, f"{prefix}_ALL_METHODS.png")
    concatenate_plots(plots_methods, "-append", outpath_plots_methods)
    
    return outpath_plots_methods


def calculate_correlation_metrics(pocket_type, scope, outtag, df, outpath):
    """
    Computes Spearman & Pearson correlations of each column vs. GT, saves a CSV, and
    plots a heatmap. This no longer references a single `elen_model`, so the
    output filenames are based only on pocket_type and scope.
    """
    # 1) Drop any rows with NaN
    df = df.dropna()

    # 2) We only want to compute correlation for columns that differ from 'GT'.
    #    But for the raw correlation, we'll keep 'GT' in the DataFrame to
    #    correlate *against* it.
    
    # Prepare a results DataFrame for storing correlations
    short_scope = scope[:2]   # e.g. 'gl' for 'global'
    short_pock  = pocket_type[:2]  # e.g. 'LP' -> 'LP'
    col_spearman = f"ρ_{short_scope}_{short_pock}"
    col_pearson  = f"R_{short_scope}_{short_pock}"
    
    df_results = pd.DataFrame(columns=["method", col_spearman, col_pearson])
    
    # 3) For each column (except GT itself), calculate Spearman & Pearson vs. GT.
    if "GT" not in df.columns:
        # If GT is missing entirely, just return an empty result
        return
    print(f"df: {df}")
    print(df.dtypes)
    print(df.head()) 
    
    for col in df.columns:
        if col != "GT" and col != "identifier":
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.to_csv("asdf.csv")
    for col in df.columns:
        if col == "GT" or col == "identifier":
            continue  # skip correlating GT vs GT
        spearman_corr, _ = spearmanr(df[col], df["GT"])
        pearson_corr, _ = pearsonr(df[col], df["GT"])
        df_results = pd.concat([
            df_results,
            pd.DataFrame({
                "method": [col],
                col_spearman: [spearman_corr],
                col_pearson:  [pearson_corr],
            })
        ], ignore_index=True)
    # 4) Write out the CSV with absolute values (common in MQA correlation tables),
    #    and also produce a heatmap of the two correlation columns.
    #    The final CSV is e.g. "corr_global_LP.csv"
    path_corr_csv = os.path.join(outpath, f"{outtag}_corr_{scope}_{pocket_type}.csv")

    # Convert correlation columns to numeric and then use absolute values
    df_results[col_spearman] = pd.to_numeric(df_results[col_spearman], errors='coerce').abs()
    df_results[col_pearson]  = pd.to_numeric(df_results[col_pearson],  errors='coerce').abs()

    # Optionally, sort the results by method name or correlation magnitude
    df_results = df_results.round(4)
    df_results.sort_values("method", ascending=True, inplace=True)

    df_results.to_csv(path_corr_csv, index=False)
    print(df_results)

    # 5) Generate a heatmap from the correlation columns.
    heatmap_out = os.path.join(outpath, f"{outtag}_heat_corr_{scope}_{pocket_type}.png")
    plot_heatmap(df_results.set_index("method"), heatmap_out,
                 [None, None],
                 "viridis",
                 f"ρ_{scope[:2]}_{pocket_type}", 
                 4)

def experiment_top1_loss(args, 
                         df: pd.DataFrame, 
                         methods_where_highest_score_is_best: list):
    """
    Conducts a top1-loss experiment:
    
    1) Identify "method columns" (everything except 'identifier', 'resid', 'GT', 'loop_id').
    2) Apply min–max scaling so that bigger == better for each method:
       - If method is in methods_where_highest_score_is_best: direct min–max
       - Else invert so bigger is better.
    3) For each (identifier, method), pick the row with the highest scaled value.
    4) From the original DF, record the GT at that row => yields top1 GT.
    5) Build df_top1: rows = targets, columns = method top1 GT scores.
    6) Pass df_top1 to a function that computes average top1 loss and logs/plots results.
    7) Make correlation-style plots comparing every 'ELEN_<model>' column to each method,
       then concatenate them.
    """

    # --- (A) Determine which columns are "methods" (to be scaled) ---
    ignore_cols = {'identifier', 'resid', 'GT', 'loop_id'}
    # This leaves "args.methods", plus any "ELEN_..." columns, or any other numeric MQA columns
    method_columns = [c for c in df.columns if c not in ignore_cols]

    # --- (B) Create a scaled DataFrame so we don't lose original GT references ---
    df_scaled = df.copy()

    # For each method-like column, do min–max scaling where bigger=better
    for method in method_columns:
        method_vals = df_scaled[method].dropna()
        if method_vals.empty:
            # Skip entirely if all values are NaN
            continue

        min_val, max_val = method_vals.min(), method_vals.max()
        if min_val == max_val:
            # All values identical in this column → set scaled=1.0 (arbitrary but consistent)
            df_scaled[method] = 1.0
        else:
            if method in methods_where_highest_score_is_best:
                # direct min–max scale
                df_scaled[method] = (df_scaled[method] - min_val) / (max_val - min_val)
            else:
                # invert scale so bigger = better
                df_scaled[method] = 1.0 - (df_scaled[method] - min_val) / (max_val - min_val)

    # --- (C) For each target (identifier) & each method column, pick the row with best (max) scaled score ---
    top1_by_target = {}
    for target_id, group_scaled in df_scaled.groupby('identifier'):
        group_original = df[df['identifier'] == target_id]
        
        best_gt_for_this_target = {}
        for method in method_columns:
            # The index of the row with the highest scaled value in group_scaled
            idx_best = group_scaled[method].idxmax()
            # Retrieve GT from the original DF at that index
            best_gt_for_this_target[method] = group_original.loc[idx_best, 'GT']

        top1_by_target[target_id] = best_gt_for_this_target

    # --- (D) Convert to a DF: rows = target, columns = top1 GT for each method ---
    df_top1 = pd.DataFrame.from_dict(top1_by_target, orient='index')

    # --- (E) Perform further analysis: average top1 loss, heatmaps, etc. ---
    df_top1_avg = calculate_average_top1_loss(args, df_top1)

    # --- (F) Optionally produce comparisons: e.g. compare each 'ELEN_<model>' to each method. ---
    #     We'll do something similar to your original loop, but handle multiple ELEN columns.
    prefix = f"{args.outtag}_top1_{args.scope}_{args.pocket_type}"

    # Identify which columns are "ELEN" columns
    elen_cols = [m for m in df_top1.columns if m.startswith("ELEN_")]
    # You might want to skip if no ELEN columns exist
    if elen_cols:
        plots_methods = []
        for elen_col in elen_cols:
            plots_top1 = []
            for method_2 in args.methods:
                outpath_plot_top1 = os.path.join(args.outpath, f"{prefix}_{elen_col}-{method_2}.png")
                plot_top1 = plot_top1_loss(df_top1, elen_col, method_2, outpath_plot_top1)
                plots_top1.append(plot_top1)

            # Combine the per-method plots vertically
            outpath_plots_top1 = os.path.join(args.outpath, f"{prefix}_{elen_col}.png")
            plot_method = concatenate_plots(plots_top1, "-append", outpath_plots_top1)
            plots_methods.append(plot_method)

        # Combine all the "ELEN_<model>" summary plots horizontally
        outpath_plots_methods = os.path.join(args.outpath, f"{prefix}_ALL_ELEN.png")
        if len(plots_methods) > 1:
            concatenate_plots(plots_methods, "+append", outpath_plots_methods)
        else:
            # If only one or zero ELEN columns, just rename the single plot:
            if plots_methods:
                os.rename(plots_methods[0], outpath_plots_methods)
        return outpath_plots_methods
    else:
        print("[Info] No ELEN_* columns found. Skipping ELEN-based top1 plots.")
        return None
    
def calculate_average_top1_loss(args, df_top1):
    """
    Given df_top1 (rows=targets, columns=methods),
    compute the average GT across each method (mean of each column).
    Write CSV of results and produce a heatmap.
    Does NOT reference a single elen_model, so the CSV is named:
        top1_{scope}_{pocket_type}.csv
    """

    # Take the mean across all targets (rows)
    mean_values = df_top1.mean()

    # Build a small DataFrame to store results
    pocket_type = args.pocket_type
    scope = args.scope
    short_name = f"avg_lddt_{scope}_{pocket_type}"

    df_out = pd.DataFrame({
        'method': mean_values.index,
        short_name: mean_values.values
    }).set_index('method')

    # Convert to numeric (in case of any weird dtypes), then round
    df_out = df_out.apply(pd.to_numeric, errors='coerce').round(4)
    df_out.sort_values(by="method", ascending=True, inplace=True)
    
    # Write CSV
    path_csv = os.path.join(args.outpath, f"{args.outtag}_top1_{scope}_{pocket_type}.csv")
    df_out.to_csv(path_csv)
    print(f"[Top1] CSV saved to: {path_csv}")

    # Optionally produce a heatmap from these means
    heatmap_path = path_csv.replace("top1_", "heat_top1_").replace(".csv", ".png")
    plot_heatmap(df_out, heatmap_path, [None, None], "plasma", f"avg_lddt_{scope}_{pocket_type}", 2)
    return df_out

def experiment_auc(df, args, threshold=0.7):
    """
    Computes and plots ROC curves + AUC scores for one or more methods vs. ground truth.

    df : pd.DataFrame
        Must include:
        - "GT": ground truth numeric scores (0..1).
        - One or more method columns to compare (strings in pred_method).
        - Possibly extra columns like "identifier", "resid", "loop_id".
    pred_method : str or list of str
        Name(s) of columns in df to compare to "GT".
    args : Namespace
        Contains:
        - args.scope: "global" or something else
        - args.pocket_type: "LP" or some other string
        - args.outpath: directory for saving plots/results
    threshold : float
        Binarization threshold for GT: if GT >= threshold => 1, else 0
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    def compute_auc_for_method(method_col):
        """
        Returns (auc_value, normalized_pred_scores).
          - If the raw AUC < 0.5, we invert predictions, re-check AUC, and return that.
        """
        gt_scores = df["GT"].values
        method_scores = df[method_col].values

        # Binarize ground truth based on threshold
        gt_labels = [1 if score >= threshold else 0 for score in gt_scores]

        # Normalize predictions to [0,1] if they fall outside [0,1]
        min_val, max_val = min(method_scores), max(method_scores)
        if max_val > 1 or min_val < 0:
            norm_pred = [(x - min_val) / (max_val - min_val) for x in method_scores]
        else:
            norm_pred = method_scores

        # Compute AUC
        auc_val = roc_auc_score(gt_labels, norm_pred)

        # If AUC < 0.5, invert the predictions
        if auc_val < 0.5:
            norm_pred = [-x for x in norm_pred]
            auc_val = roc_auc_score(gt_labels, norm_pred)

        return auc_val, norm_pred

    # If you have a list of "ELEN" columns to add:
    elen_cols = [f"ELEN_{m}" for m in args.elen_models]
    pred_method = args.methods + elen_cols

    # 1) Optionally average numeric columns by group if scope=="global"
    if args.scope == "global":
        numeric_cols = df.select_dtypes(include='number').columns
        group_col = 'loop_id' if args.pocket_type == 'LP' else 'identifier'
        df = df.groupby(group_col, as_index=False)[numeric_cols].mean()

    # 2) Ensure pred_method is a list
    if isinstance(pred_method, str):
        pred_method = [pred_method]

    # 3) Prepare for ROC curve plotting
    auc_results = {}
    roc_data_list = []  # We'll store (method_col, fpr, tpr, auc_val) here

    # 4) For each method, compute AUC and store ROC data
    for method_col in pred_method:
        auc_val, norm_pred_scores = compute_auc_for_method(method_col)
        auc_results[method_col] = auc_val

        # Build ROC data
        gt_labels = [1 if g >= threshold else 0 for g in df["GT"].values]
        fpr, tpr, _ = roc_curve(gt_labels, norm_pred_scores)

        # Store data for sorting/plotting later
        roc_data_list.append((method_col, fpr, tpr, auc_val))

    # Sort by AUC (descending)
    roc_data_list.sort(key=lambda x: x[3], reverse=True)

    # 5) Plot the curves in sorted order, with a square figure
    plt.figure(figsize=(6, 6))
    for method_col, fpr, tpr, auc_val in roc_data_list:
        plt.plot(fpr, tpr, label=f"{method_col} (AUC={auc_val:.4f})")
        print(f"ROC AUC for {method_col}: {auc_val:.4f}")

    # Plot a diagonal line (random performance): black and dashed
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")

    # Legend half as big
    plt.legend(prop={'size': 6})

    plt.tight_layout()

    # 6) Save the figure
    outpath_plot = os.path.join(args.outpath, f"{args.outtag}_roc_auc_{args.scope}_{args.pocket_type}.png")
    plt.savefig(outpath_plot)
    plt.close()
    print(f"ROC curve plot saved to {outpath_plot}")

    # 7) Write AUC results to CSV
    auc_df = pd.DataFrame(list(auc_results.items()), columns=["method", "auc"])
    auc_df["auc"] = auc_df["auc"].round(4)
    auc_df.sort_values(by="method", ascending=True, inplace=True)
    outpath_csv = os.path.join(args.outpath, f"{args.outtag}_roc_auc_{args.scope}_{args.pocket_type}.csv")
    auc_df.to_csv(outpath_csv, index=False)
    print(f"AUC results CSV saved to {outpath_csv}")

    return auc_results