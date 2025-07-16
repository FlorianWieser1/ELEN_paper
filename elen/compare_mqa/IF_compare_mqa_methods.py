#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J IF_compare_mqa_methods
#SBATCH -o IF_compare_mqa_methods.log
#SBATCH -e IF_compare_mqa_methods.err
##SBATCH --gres gpu:1
import os
import sys
import glob
import subprocess
import argparse as ap
from elen.config import PATH_MQA, PATH_PYTHON

def run_compare_mqa_methods_script(args, elen_model, pocket_type, scope, outpath, overwrite):
    command = [PATH_PYTHON, f"{PATH_MQA}/compare_mqa_methods.py", '--inpath_natives', args.inpath_dataset + "/natives",
                                    '--inpath_models', args.inpath_dataset + "/models",
                                    '--inpath_methods', args.inpath_dataset + "/methods",
                                    '--scope', scope, 
                                    '--pocket_type', pocket_type,
                                    '--outpath', outpath, 
                                    '--elen_model', elen_model,
                                    '--loop_mode', "loop_pocket",
                                    '--ss_frag_size', str(args.ss_frag_size),
                                    '--nr_residues', str(args.nr_residues),
                                    '--loop_max_size', str(args.loop_max_size),
                                    '--exp_correlation']
    if overwrite:
        command.append('--overwrite') 
    try:
        result = subprocess.run(command)
    except subprocess.CalledProcessError:
        print("compare_mqa_methods.py failed. Exiting.")
        sys.exit(1)

def merge_final_plots(args, tag, orientation, outpath, elen_model):
    final_plots = glob.glob(os.path.join(outpath, f"{tag}*.png"))
    final_plots_sorted = sorted(final_plots, key=lambda x: (
        'LP' not in x, 
        'RP' not in x,    
        'perres' not in x,
        'global' not in x 
    ))
    path_final = os.path.join(outpath, f"{tag}_final_{elen_model}.png")
    subprocess.run(['convert'] + final_plots_sorted + [orientation, path_final])

############################################################################## 
def main(args):
    for elen_model in args.elen_models:
        print(f"Running experiments for elen model {elen_model}")
        outpath = f"{args.inpath_dataset}/out_{elen_model}"
        run_compare_mqa_methods_script(args, elen_model, "LP", "perres", outpath, False)
        run_compare_mqa_methods_script(args, elen_model, "RP", "perres", outpath, False)
        #run_compare_mqa_methods_script(args, elen_model, "LP", "global", outpath, False)
        #run_compare_mqa_methods_script(args, elen_model, "RP", "global", outpath, False)
        #merge_final_plots(args, "corr", "+append", outpath, elen_model)
        #merge_final_plots(args, "heat", "+append", outpath, elen_model)
    print("Done.") 

###############################################################################
if __name__ == "__main__":
    parser = ap.ArgumentParser()
    default_path = "/home/florian_wieser/software/ARES/geometricDL/edn/ELEN_testing/"
    parser.add_argument("--inpath_dataset", type=str, default=f"{default_path}/cameo_3month/")
    #parser.add_argument("--elen_models", type=list, nargs='+', default=["dkt62ak2", "7qwiog9g", "9g0ouk9v", "ati0uun2", "bn2z164b", "brf4599v", "esyd8tat", "k4pyvu65", "r3o39uf3"])
    parser.add_argument("--elen_models", type=str, default=["gxf3w3mm", "jwwrx159"])
    parser.add_argument("--ss_frag_size", type=int, default=4)
    parser.add_argument("--loop_max_size", type=int, default=10)
    parser.add_argument("--nr_residues", type=int, default=28)
    args = parser.parse_args()
    main(args)    
