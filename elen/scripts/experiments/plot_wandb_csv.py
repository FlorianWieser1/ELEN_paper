#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def average_training_loss_per_epoch(train_csv, val_csv, step_col_train, loss_col_train, step_col_val, loss_col_val):
    df_val = pd.read_csv(val_csv)
    df_train = pd.read_csv(train_csv)
    epoch_steps = df_val[step_col_val].values
    epoch_starts = np.insert(epoch_steps[:-1], 0, 0)
    epoch_ends = epoch_steps

    avg_train_losses = []
    for start, end in zip(epoch_starts, epoch_ends):
        mask = (df_train[step_col_train] >= start) & (df_train[step_col_train] < end)
        avg_loss = df_train.loc[mask, loss_col_train].mean()
        avg_train_losses.append(avg_loss)

    epochs = np.arange(1, len(df_val) + 1)
    return epochs, avg_train_losses, df_val[loss_col_val].values

def plot_train_val_loss(epochs, avg_train_loss, val_loss, output_file=None):
    # --- Publication-style settings ---
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
        "figure.titlesize": 18,
        "axes.linewidth": 1.1,
    })

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_axisbelow(True)
    ax.grid(True, zorder=0, linestyle='--', alpha=0.4)

    ax.plot(
        epochs, avg_train_loss, marker='o', linestyle='-', linewidth=2.1, color="#0064B5",
        label="Training Loss", zorder=3, markersize=6
    )
    ax.plot(
        epochs, val_loss, marker='s', linestyle='--', linewidth=2.1, color="#EA6615",
        label="Validation Loss", zorder=4, markersize=6
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    #ax.set_title("Training and Validation Loss per Epoch")
    ax.legend(loc='best', frameon=False)
    ax.tick_params(axis='both', which='major', length=6, width=1.1, direction='out')
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publication-style training/validation loss curve plot.")
    parser.add_argument("--train-csv", required=True, help="CSV file with training loss (frequent steps).")
    parser.add_argument("--val-csv", required=True, help="CSV file with validation loss (per epoch).")
    parser.add_argument("--output", default=None, help="Output plot filename (e.g., loss_curves.png).")
    parser.add_argument("--step-col-train", default="Step", help="Step column in training loss CSV.")
    parser.add_argument("--loss-col-train", default=None, help="Loss column in training loss CSV.")
    parser.add_argument("--step-col-val", default="Step", help="Step column in validation loss CSV.")
    parser.add_argument("--loss-col-val", default=None, help="Loss column in validation loss CSV.")
    args = parser.parse_args()

    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)
    if args.loss_col_train is None:
        args.loss_col_train = [c for c in df_train.columns if "loss" in c and "__" not in c][0]
    if args.loss_col_val is None:
        args.loss_col_val = [c for c in df_val.columns if "val_loss" in c and "__" not in c][0]

    epochs, avg_train_loss, val_loss = average_training_loss_per_epoch(
        args.train_csv, args.val_csv, args.step_col_train, args.loss_col_train, args.step_col_val, args.loss_col_val
    )
    plot_train_val_loss(epochs, avg_train_loss, val_loss, output_file=args.output)
