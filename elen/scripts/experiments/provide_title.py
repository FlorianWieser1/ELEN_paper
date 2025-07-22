#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    parser = argparse.ArgumentParser(description="Add a centered, semibold title to an existing image.")
    parser.add_argument('image', type=str, help='Path to the input image file')
    parser.add_argument('--title', type=str, required=True, help='Title to display above the image')
    parser.add_argument('--output', type=str, default=None, help='Path to save the output image (optional)')
    args = parser.parse_args()

    # Load and plot the image
    img = mpimg.imread(args.image)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.axis('off')  # Hide axes

    # Add centered, semibold title
    plt.title(args.title, loc='center', fontsize=8, fontweight='semibold')

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, bbox_inches='tight', dpi=300)
        print(f"Saved figure with title to {args.output}")
    else:
        plt.show()

###############################################################################
if __name__ == "__main__":
    main()
