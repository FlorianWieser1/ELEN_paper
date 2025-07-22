#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J ⭐e2e 
#SBATCH -o e2e.log
#SBATCH -e e2e.err
##SBATCH --gres gpu:1
import os
import sys
import logging
import subprocess
import argparse as ap
from elen.config import PATH_PROJECT

def run_script(command, output_path, description):
    logging.info(f"Running {description}...")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        logging.info(f"✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗")
        logging.info(f"Error: {description} failed.")
        logging.info(f"✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗")
        sys.exit(1)
    if os.path.exists(output_path):
        logging.info(f"✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓")
        logging.info(f"{description} output produced successfully.")
        logging.info(f"✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓")
    else:
        logging.info(f"✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗")
        logging.info(f"Error: {description} output not found at {output_path}.")
        logging.info(f"✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗✗")
        sys.exit(1)


###############################################################################

def main(args):
    if args.training:
        # Run training script
        DEFAULT_PATH="/home/florian_wieser/projects/ELEN/elen_training/datasets/LP_20"
        TRAINING_OUTPUT = "out_training/activations.json"
        command_training = (
                   f"{PATH_PROJECT}/training/train_sweep.py"
                   f" --train_dir {DEFAULT_PATH}/pdbs/train"
                   f" --val_dir {DEFAULT_PATH}/pdbs/val"
                   f" --test_dir {DEFAULT_PATH}/pdbs/test"
                   " --out out_training"
                   " --use_labels True"
                   " --prediction_type LP"
                   f" --yaml {PATH_PROJECT}/training/LP_20_acluster.yaml"
                   " --wandb disabled"
                   " --filetype pdb"
                   " --random_seed 123"
                   " --epochs 2"
                   " --batch_size 4"
        )
        run_script(command_training, TRAINING_OUTPUT, "training script")
        
    if args.inference: 
        # Run inference script
        DEFAULT_PATH="/home/florian_wieser/projects/ELEN/elen_testing/inference"
        INFERENCE_OUTPUT = "out_inference/1ubq_A_elen_scored.pdb"
        command_inference = (
                   f"{PATH_PROJECT}/inference/ELEN_inference.py"
                   f" --inpath {DEFAULT_PATH}/input_1ubq"
                   " --outpath out_inference"
                   f" --path_model {PATH_PROJECT}/../models/jwwrx159.ckpt"
                   " --overwrite"
        )
        run_script(command_inference, INFERENCE_OUTPUT, "inference script")
       
    if args.mqa:
        # Run compare_mqa script
        DEFAULT_PATH="/home/florian_wieser/projects/ELEN/elen_testing/MQA/cameo_1"
        COMPARE_MQA_OUTPUT = "out_compare_mqa/corr_perres_LP_jwwrx159.png"
        command_mqa = (
                   f"{PATH_PROJECT}/compare_mqa/compare_mqa_methods.py"
                   f" --inpath_natives {DEFAULT_PATH}/natives"
                   f" --inpath_methods {DEFAULT_PATH}/methods"
                   f" --inpath_models {DEFAULT_PATH}/models"
                   " --outpath out_compare_mqa"
                   " --elen_model jwwrx159"
                   " --exp_correlation"
                   " --pocket_type LP"
                   " --scope perres"
                   " --overwrite"
        )
        run_script(command_mqa, COMPARE_MQA_OUTPUT, "compare_mqa script")
    logging.info("Done.")


###############################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='ELEN-e2e-%(levelname)s(%(asctime)s): %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S')
    parser = ap.ArgumentParser()
    parser.add_argument("--training", action="store_true", default=False)
    parser.add_argument("--inference", action="store_true", default=False)
    parser.add_argument("--mqa", action="store_true", default=False)
    args = parser.parse_args()
    main(args)