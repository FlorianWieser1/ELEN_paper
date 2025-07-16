#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J compare_mqa_methods
#SBATCH -o compare_mqa_methods_%j.log
#SBATCH -e compare_mqa_methods_%j.err
#SBATCH --gres gpu:1

# TODO make handles for different elen scores 
# Description: Script to compare ELEN to other methods in the CAMEO QE category
# Usage: ./test_mqa_methods.py --inpath af_predictions --outpath results --methods enqa
# Author: Florian Wieser (florian.wieser@tugraz.at)
import os
import sys
import glob 
import shutil
import argparse as ap
import logging
import pandas as pd

#TODO get rid of fucking warnings
import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from elen.compare_mqa.utils_mqa import get_mqa_data_inference_LP, get_mqa_data_inference_RP
from elen.compare_mqa.experiments import experiment_correlation, experiment_top1_loss, experiment_auc
from elen.shared_utils.shared_utils import func_timer

### EXPERIMENTS ###############################################################
### CORRELATION
import os
import pandas as pd

@func_timer
def main(args):
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    
    logging.info(f"{args.scope}/{args.pocket_type} scoring problem.")
    path_json = os.path.join(args.outpath, f"{args.pocket_type}_mqa_data.json")
    
    if not os.path.exists(path_json): 
        logging.info(f"Processing mqa data of {args.pocket_type} scoring problem.")
        if args.pocket_type == "LP": 
            df = get_mqa_data_inference_LP(args)
        elif args.pocket_type == "RP":
            df = get_mqa_data_inference_RP(args)
        df.to_json(path_json, orient='records', lines=True)  
    logging.info(f"Loading data from .json.")
    df = pd.read_json(path_json, orient='records', lines=True)
    
    if args.exp_correlation:
        logging.info("Running correlation experiment.") 
        experiment_correlation(args, df)
    if args.exp_top1_loss: 
        logging.info("Running Top1 loss experiment.") 
        experiment_top1_loss(args, df, args.methods)
    if args.exp_roc: 
        logging.info("Running ROC experiment.") 
        experiment_auc(df, args, 0.8)
        
    # clean up
    path_plots = os.path.join(args.outpath, "plots")
    os.makedirs(path_plots, exist_ok=True)
    for to_move in args.methods + ['GT', 'ELEN']:
        files_to_move = glob.glob(os.path.join(args.outpath, f"*{to_move}.png"))
        for file in files_to_move:
            shutil.move(file, os.path.join(path_plots, os.path.basename(file)))
    logging.info(f"Done.")

    
###############################################################################
if __name__ == "__main__":
    # setup loggers
    logger = logging.basicConfig(level=logging.INFO, format='ELEN-mqa-%(levelname)s(%(asctime)s): %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S')
    #logging.getLogger('matplotlib.font_manager').disabled = True
    parser = ap.ArgumentParser()
    parser.add_argument("--inpath", type=str, default=f"cameo_dev")
    parser.add_argument("--outpath", type=str, default=f"compare_mqa_out")
    parser.add_argument("--outtag", type=str, default=f"mqa")
    parser.add_argument("--scope", type=str, default='perres', choices=['global', 'perres'])
    parser.add_argument("--pocket_type", type=str, default='LP', choices=['LP', 'RP'])
    parser.add_argument("--elen_models", nargs='+', default=["jwwrx159", "j0ribitb"])
    parser.add_argument("--elen_scores", type=str, nargs='+', default=['ELEN'])
    parser.add_argument("--loop_mode", type=str, default="loop_pocket", choices=["loop_pocket", "loop_stretch"])
    parser.add_argument("--ss_frag", type=str, default="None", choices=["None", "helix", "sheet"],
                        help="defaults to extract any kind of loop, e.g. EHEHHELLLHEHEHEH")
    # old extraction parameters: ss_frag_size 6, nr_residues 28, loop_max_size 10
    parser.add_argument("--ss_frag_size", type=int, default=4)
    parser.add_argument("--nr_residues", type=int, default=28)
    parser.add_argument("--loop_max_size", type=int, default=10)
    parser.add_argument("--exp_correlation", action="store_true", default=False, 
                        help="Compute xy-scatter plot between ground truth lddt values and \
                        the models' prediction, either for the whole protein or just the loops, \
                        either local (per residue lddt) or global (average lddt per model)")
    parser.add_argument("--exp_top1_loss", action="store_true", default=False)
    parser.add_argument("--exp_roc", action="store_true", default=False, help="Compute Receiver Operator Characteristics.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing run.")
    args = parser.parse_args()
    args.inpath_natives = os.path.join(args.inpath, "natives")
    args.inpath_methods = os.path.join(args.inpath, "methods")
    args.inpath_models = os.path.join(args.inpath, "models")
    args.methods = []
    for dir in os.listdir(args.inpath_methods):
        path_dir_method = os.path.join(args.inpath_methods, dir)
        if os.path.isdir(path_dir_method):
            args.methods.append(os.path.basename(path_dir_method))
    main(args)
