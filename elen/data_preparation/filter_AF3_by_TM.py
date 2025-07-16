#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J filter_TM
#SBATCH -o filter_TM.log
#SBATCH -e filter_TM.err
import os
import re
import sys
import glob
import shutil
import logging
import subprocess
import argparse as ap
from elen.shared_utils.utils_others import func_timer, discard_pdb
from elen.config import PATH_SOFTWARE

### HELPERS ###################################################################

def get_tm_score(path_pdb_model, path_pdb_ref):
    # Using shell=True here to allow shell expansion of "~"
    cmd = f"{PATH_SOFTWARE}/TMalign/TMscore {path_pdb_model} {path_pdb_ref}"
    # Run the command and capture the output
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    # Check if the command ran successfully
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with error: {result.stderr.strip()}")
    output = result.stdout
    # Use regex to search for the TM-score (e.g., "TM-score    = 0.9845  (d0= 5.58)")
    match = re.search(r"TM-score\s*=\s*([0-9]*\.?[0-9]+)", output)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("TM-score not found in the output.")

###############################################################################
@func_timer
def main(args):
    # Define a general label for logging discarded files
    if os.path.exists(args.outpath) and args.overwrite:
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
   
    for path_af3 in glob.glob(os.path.join(args.inpath_AF, "*.pdb")):
        fname_af3 = os.path.basename(path_af3)
        logging.debug(f"Calculating TM-Score of {fname_af3}.")
        identifier = fname_af3[:4]
        try:
            path_native = glob.glob(os.path.join(args.inpath_natives, f"{identifier}*.pdb"))[0]
        except IndexError:
            logging.error(f"No native PDB file found for identifier {identifier}. Skipping.")
            continue

        try:
            TM_score = get_tm_score(path_af3, path_native)
        except Exception as e:
            logging.error(f"Error computing TM-score for {fname_af3}: {e}. Discarding files.")
            try:
                path_md = glob.glob(os.path.join(args.inpath_MD, f"{identifier}*.pdb"))[0]
            except IndexError:
                logging.error(f"No MD PDB file found for identifier {identifier}.")
                path_md = None
            discard_pdb(path_af3, args.path_discarded, "TM-filtering", "TM-filtering")
            discard_pdb(path_native, args.path_discarded, "TM-filtering", "TM-filtering")
            if path_md:
                discard_pdb(path_md, args.path_discarded, "TM-filtering", "TM-filtering")
            continue

        if TM_score > float(args.cutoff):
            logging.info(f"TM-Score of {fname_af3}: {TM_score}")
        else:
            try:
                path_md = glob.glob(os.path.join(args.inpath_MD, f"{identifier}*.pdb"))[0]
            except IndexError:
                logging.error(f"No MD PDB file found for identifier {identifier}.")
                continue
            discard_pdb(path_af3, args.path_discarded, "TM-filtering", "TM-filtering")
            discard_pdb(path_native, args.path_discarded, "TM-filtering", "TM-filtering")
            if path_md:
                discard_pdb(path_md, args.path_discarded, "TM-filtering", "TM-filtering")
    logging.info("Done.")

            
###############################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-Filter_TM-Score-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    parser = ap.ArgumentParser(
        description="Filter AF3 models based on TM-Score by comparing them with native structures."
    )
    default_path = "/home/florian_wieser/projects/ELEN/elen_training/data_preparation/AF3_LiMD/AF_LiMD_200/filter_TM_score_testing"
    parser.add_argument("--inpath_AF", type=str, default=f"{default_path}/AF3_models", help="Input directory for AF3 model PDB files.")
    parser.add_argument("--inpath_natives", type=str, default=f"{default_path}/natives", help="Input directory for native PDB files.")
    parser.add_argument("--inpath_MD", type=str, default=f"{default_path}/MD", help="Input directory for MD frames.")
    parser.add_argument("--outpath", type=str, default=f"{default_path}/filtered_TM", help="Output directory for filtered (discarded) PDB files.")
    parser.add_argument("--cutoff", type=float, default=0.7, help="Cutoff to decide if an AF3 prediction is acceptable.")
    parser.add_argument("--path_discarded", type=str, default=f"discarded", help="Output directory for failed PDB files.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output folders.")
    args = parser.parse_args()
    main(args)
