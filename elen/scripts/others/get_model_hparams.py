#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J get_model_hparams
#SBATCH -o get_model_hparams.log
#SBATCH -e get_model_hparams.err

import argparse
import torch

def main():
    parser = argparse.ArgumentParser(
        description="Load hyperparameters from a trained ELEN model checkpoint without initializing the model."
    )
    parser.add_argument(
        '--path_model',
        type=str,
        required=True,
        help='Path to the trained ELEN model checkpoint (e.g., my_model.ckpt).'
    )
    args = parser.parse_args()

    # Load the checkpoint as a dictionary
    checkpoint = torch.load(args.path_model, map_location='cpu')

    # 'hyper_parameters' should contain your training hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})

    if not hparams:
        print("No hyperparameters found in the checkpoint.")
    else:
        print("Hyperparameters stored in the checkpoint:")
        for param_name, param_value in hparams.items():
            print(f"{param_name}: {param_value}")

if __name__ == "__main__":
    main()

