#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

def main(args):
    # Publication font settings
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

    # Read both images
    img1 = mpimg.imread(args.input_left)
    img2 = mpimg.imread(args.input_right)

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Show images on axes, remove ticks, axis, etc.
    for ax, img in zip(axes, [img1, img2]):
        ax.imshow(img)
        ax.axis('off')
        # Optionally add a thin black frame
        if args.frame:
            # Draw rectangle: (x, y) = (0,0), width=1, height=1 in axes coords
            rect = patches.Rectangle(
                (0, 0), 1, 1,
                linewidth=1.5,
                edgecolor='black',
                facecolor='none',
                transform=ax.transAxes,
                zorder=10,
                clip_on=False  # So frame goes outside of image if needed
            )
            ax.add_patch(rect)
        # Optionally set empty title
        ax.set_title("")

    # Panel letter (left, outside)
    fig.text(0.08, 0.97, args.panel_letter, fontsize=22, fontweight='bold', ha='left', va='top')

    plt.subplots_adjust(top=0.88, wspace=0.01)
    plt.savefig(args.output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Merged plot saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge two plot images into a publication-style side-by-side panel with model name and panel letter."
    )
    parser.add_argument('--input_left', type=str, required=True, help='Path to left input plot (e.g., bond shift plot).')
    parser.add_argument('--input_right', type=str, required=True, help='Path to right input plot (e.g., dihedral plot).')
    parser.add_argument('--output_file', type=str, default='merged_panel.png', help='Output filename for merged figure.')
    parser.add_argument('--panel_letter', type=str, default='A', help='Panel letter for the merged panel (A, B, ...).')
    parser.add_argument('--frame', action='store_true', help='If set, draws a thin black frame around each image panel.')
    args = parser.parse_args()
    main(args)
