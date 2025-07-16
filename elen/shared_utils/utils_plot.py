import os
import wandb
import warnings
import subprocess
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
from elen.shared_utils.utils_others import calculate_regression_metrics

def plot_target_corr(data, xaxis, yaxis, label, color, save_path):
    """
    Plots the correlation between target and predicted values, including regression metrics.

    Args:
        data (pd.DataFrame): DataFrame containing the target and predicted values.
        xaxis (str): Column name for the x-axis (target values).
        yaxis (str): Column name for the y-axis (predicted values).
        label (str): Label name for the plot title.
        color (str): Color of the scatter plot points and regression line.
        save_path (str): Path to save the output plot.
    """
    # Create the scatter plot with regression line
    sns.regplot(data=data, x=xaxis, y=yaxis, color=color, scatter_kws={'s': 12, 'alpha': 0.3}, line_kws={'lw': 1.5})
    
    # Add metrics as legend to the plot
    pearson, spearman, _, mae, rmse = calculate_regression_metrics(data[xaxis], data[yaxis])
    plt.plot([], [], ' ', label=f"n: {len(data[yaxis])}")
    plt.plot([], [], ' ', label=f"R: {pearson:4f}")
    plt.plot([], [], ' ', label=f"spear: {spearman:4f}")
    plt.plot([], [], ' ', label=f"MAE: {mae:4f}")
    plt.plot([], [], ' ', label=f"RMSE: {rmse:4f}")
    plt.legend(frameon=False, handletextpad=-2.0, loc='upper left')
    
    # Set plot labels and title
    plt.title(f"{label} Predicted vs. Target")
    plt.xlabel(f"Target {label}")
    plt.ylabel(f"Predicted {label}")
    
    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
    
    
def log_plots(outpath, globtag, wandbtag):
    """
    Logs all plot files matching a specific tag from the output directory to Weights & Biases (wandb).

    Args:
        outpath (str): Directory path where the plot files are stored.
        globtag (str): File extension or tag to filter plot files.
        wandbtag (str): Tag to use when logging the images to wandb.
    """
    all_files = os.listdir(outpath)
    plot_files = [f for f in all_files if f.endswith(globtag)]

    for plot_file in plot_files:
        file_path = os.path.join(outpath, plot_file)
        wandb.log({wandbtag: [wandb.Image(file_path, caption=plot_file)]})
        
        
        
def concatenate_plots(plot_list: List[str], mode: str, outpath: str) -> str:
    """
    Concatenates multiple image files into a single image using the 'convert' command from ImageMagick.

    This function uses the 'convert' command to combine multiple plot images into one, depending on the specified mode
    which determines how the images are combined (e.g., horizontally or vertically).

    Parameters:
        plot_list (List[str]): A list of file paths to the plot images to be concatenated.
        mode (str): The mode of concatenation ('+append' for horizontal, '-append' for vertical).
        outpath (str): The file path where the concatenated image will be saved.

    Returns:
        str: The path to the output file where the concatenated image is saved.

    Raises:
        subprocess.CalledProcessError: If the 'convert' command fails.
        ValueError: If the mode is not one of the expected options.
    """
    if mode not in ['+append', '-append']:
        raise ValueError("Invalid mode. Use '+append' for horizontal or '-append' for vertical concatenation.")
    
    try:
        # Ensure the command is executed correctly
        subprocess.run(['convert'] + plot_list + [mode, outpath], check=True)
    except subprocess.CalledProcessError as e:
        warnings.warn(f"[WARN] Failed to concatenate plots ({len(plot_list)} images). Skipping final concat.\nError: {e}")
    return outpath