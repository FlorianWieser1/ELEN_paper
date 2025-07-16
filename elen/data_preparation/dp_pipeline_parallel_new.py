#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J dp_pipeline
#SBATCH -o dp_pipeline.log
#SBATCH -e dp_pipeline.err
# (Add any additional SBATCH directives as needed)

# TODO fix, submitted still not working
# TODO test all modes
# TODO refactor, make output clean
# TODO upscale testing

import os
import sys
import glob
import json
import shutil
import logging
import argparse
import subprocess
import time
import warnings

from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from elen.shared_utils.utils_others import func_timer
from elen.config import PATH_DP_SCRIPTS
from elen.config import PATH_ROSETTA_TOOLS

# Define absolute script path and working directory for Slurm jobs
SCRIPT_PATH = os.path.abspath(sys.argv[0])
WORKDIR = os.getcwd()

################ HELPER FUNCTIONS ################

def rosetta_clean(path_pdb, outpath):
    """
    Cleans a PDB file using Rosetta's clean_pdb.py and places the result in outpath.
    """
    subprocess.run(
        [
            f"{PATH_ROSETTA_TOOLS}/clean_pdb.py",
            path_pdb,
            "--allchains",
            "X"
        ],
        stdout=subprocess.DEVNULL
    )
    fname_pdb = os.path.basename(path_pdb)
    outpath_pdb = os.path.join(outpath, fname_pdb.replace("_X.pdb", ".pdb"))
    shutil.move(fname_pdb.replace(".pdb", "_X.pdb"), outpath_pdb)
    os.remove(fname_pdb.replace(".pdb", "_X.fasta"))

def split_list(lst, n):
    """Split list lst into n nearly equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def wait_for_slurm_jobs(job_ids, poll_interval=10):
    """Wait until all job IDs no longer appear in squeue."""
    while True:
        remaining = []
        for job_id in job_ids:
            result = subprocess.run(f"squeue -j {job_id}", shell=True, stdout=subprocess.PIPE, text=True)
            if job_id in result.stdout:
                remaining.append(job_id)
        if not remaining:
            break
        logging.info(f"Waiting for jobs to finish: {remaining}")
        time.sleep(poll_interval)

def merge_directory_contents(src_dirs, dest_dir):
    """Copy all files from each src_dir into dest_dir (creating subfolders if needed)."""
    os.makedirs(dest_dir, exist_ok=True)
    for sdir in src_dirs:
        if os.path.isdir(sdir):
            for fname in os.listdir(sdir):
                src_file = os.path.join(sdir, fname)
                dest_file = os.path.join(dest_dir, fname)
                if not os.path.exists(dest_file):
                    shutil.copy(src_file, dest_file)

################ PIPELINE STAGE FUNCTIONS ################

def run_harmonization(args, outpath_harmonization, path_discarded):
    subprocess.run([
        f"{PATH_DP_SCRIPTS}/harmonize_pdb_to_af3.py", 
        "--inpath_AF", args.inpath_AF,
        "--inpath_natives", args.inpath_natives,
        "--inpath_MD", args.inpath_MD,
        "--outpath", outpath_harmonization,
        "--path_discarded", path_discarded
    ], check=True)

def run_TM_score_filtering(cutoff, inpath, path_discarded):
    subprocess.run([
        f"{PATH_DP_SCRIPTS}/filter_AF3_by_TM.py", 
        "--inpath_AF", os.path.join(inpath, "AF3_models"),
        "--inpath_natives", os.path.join(inpath, "natives"),
        "--inpath_MD", os.path.join(inpath, "MD_frames"),
        "--cutoff", str(cutoff),
        "--outpath", path_discarded,
        "--path_discarded", path_discarded
    ], check=True)

def run_loop_extraction(inpath, outpath_extraction, path_discarded):
    subprocess.run([
        f"{PATH_DP_SCRIPTS}/extract_LP_AF3_LiMD.py", 
        "--inpath_AF", os.path.join(inpath, "AF3_models"),
        "--inpath_natives", os.path.join(inpath, "natives"),
        "--inpath_MD", os.path.join(inpath, "MD_frames"),
        "--outpath", outpath_extraction,
        "--path_discarded", path_discarded
    ], check=True)

def run_label_calculation(args, inpath_harmonized, inpath_loop, outpath_labels, path_discarded):
    command = [
        f"{PATH_DP_SCRIPTS}/calculate_labels_AF3_LiMD.py", 
        "--inpath_AF", os.path.join(inpath_harmonized, "AF3_models"),
        "--inpath_natives", os.path.join(inpath_harmonized, "natives"),
        "--inpath_MD", os.path.join(inpath_harmonized, "MD_frames"),
        "--inpath_loop", os.path.join(inpath_loop, "EL_AF3_models"),
        "--inpath_MD_simulations", args.inpath_MD_simulations,
        "--outpath", outpath_labels,
        "--path_discarded", path_discarded
    ]
    if args.plot_hist:
        command.append("--plot_hist")
    subprocess.run(command, check=True)
    
def run_calculate_residue_features(inpath, outpath_residue_features, path_discarded):
    subprocess.run([
        f"{PATH_DP_SCRIPTS}/compute_residue_features.py", 
        "--inpath_models", str(inpath),
        "--outpath", str(outpath_residue_features),
        "--residue_features",
        "--path_discarded", path_discarded
    ], check=True)
   
def run_calculate_saprot_embeddings(inpath_models, outpath_saprot_embeddings, path_discarded):
    saprot_model = "saprot_650M"
    subprocess.run([
        f"{PATH_DP_SCRIPTS}/compute_SaProt_embeddings.py",
        "--inpath_models", inpath_models,
        "--outpath", outpath_saprot_embeddings,
        "--saprot_model", saprot_model,
        "--path_discarded", path_discarded
    ], check=True)

def run_create_dataset(args, inpath_extracted_loops, inpath_labels, outpath_dataset, 
                       outpath_residue_features, outpath_saprot_embeddings):
    command = [
        f"{PATH_DP_SCRIPTS}/train_val_test.py", 
        "--inpath", os.path.join(inpath_extracted_loops, "EL_AF3_models"),
        "--outpath", outpath_dataset,
        "--make_dataset",
        "--max_sized_dataset",
        "--path_labels", f"{inpath_labels}/labels.json"
    ]
    if args.plot_hist:
        command.append("--plot_hist")
    subprocess.run(command, check=True)
    
    shutil.copy(f"{outpath_residue_features}/residue_features.json", outpath_dataset)
    shutil.copy(f"{outpath_saprot_embeddings}/saprot_650M.h5", outpath_dataset)
    shutil.copy(f"{inpath_labels}/labels.json", outpath_dataset)

def filter_discarded(path_discarded,
                     path_extracted_loops,
                     path_labels_json,
                     path_residue_features_json,
                     path_saprot_h5):
    discarded_ids = set()
    if os.path.isdir(path_discarded):
        for fname in os.listdir(path_discarded):
            if fname.endswith(".pdb"):
                pdb_id = fname[:4].lower()
                discarded_ids.add(pdb_id)
    logging.info(f"Initial discarded IDs from discarded folder: {discarded_ids}")

    loop_dir = os.path.join(path_extracted_loops, "EL_AF3_models")
    if os.path.isdir(loop_dir):
        labels_data = {}
        if os.path.exists(path_labels_json):
            with open(path_labels_json, "r") as f:
                labels_data = json.load(f)
        residue_features_data = {}
        if os.path.exists(path_residue_features_json):
            with open(path_residue_features_json, "r") as f:
                residue_features_data = json.load(f)
        for pdb_file in os.listdir(loop_dir):
            if not pdb_file.endswith(".pdb"):
                continue
            file_id = pdb_file[:4].lower()
            has_label = any(key.lower().startswith(file_id) for key in labels_data)
            has_feature = any(key.lower().startswith(file_id) for key in residue_features_data)
            if file_id in discarded_ids or not (has_label and has_feature):
                source_path = os.path.join(loop_dir, pdb_file)
                target_path = os.path.join(path_discarded, pdb_file)
                shutil.move(source_path, target_path)
                if file_id in discarded_ids:
                    reason = "identifier already discarded"
                else:
                    missing = []
                    if not has_label:
                        missing.append("label")
                    if not has_feature:
                        missing.append("residue_feature")
                    reason = "missing required entries: " + ", ".join(missing)
                logging.info(f"Moved {pdb_file} to discarded folder because {reason}.")
                discarded_ids.add(file_id)

    if os.path.exists(path_labels_json):
        with open(path_labels_json, "r") as f:
            labels = json.load(f)
        filtered_labels = {}
        for key, value in labels.items():
            key_id = key[:4].lower()
            if key_id not in discarded_ids:
                filtered_labels[key] = value
        with open(path_labels_json, "w") as f:
            json.dump(filtered_labels, f, indent=2)
    
    if os.path.exists(path_residue_features_json):
        with open(path_residue_features_json, "r") as f:
            residue_features = json.load(f)
        filtered_features = {}
        for key, value in residue_features.items():
            key_id = key[:4].lower()
            if key_id not in discarded_ids:
                filtered_features[key] = value
        with open(path_residue_features_json, "w") as f:
            json.dump(filtered_features, f, indent=2)

################ PARALLEL WRAPPERS & WORKER FUNCTIONS ################
# Each parallel stage splits the input, submits sbatch jobs with the worker mode,
# waits for completion, then merges outputs.

# -- Loop Extraction --
def parallel_loop_extraction(args, n_jobs=10):
    import glob
    input_dir = os.path.join(args.outpath, "harmonized")
    final_output_dir = os.path.join(args.outpath, "extracted_loops")
    os.makedirs(final_output_dir, exist_ok=True)
    
    ref_dir = os.path.join(input_dir, "AF3_models")
    file_list = sorted(glob.glob(os.path.join(ref_dir, "*.pdb")))
    subsets = split_list(file_list, n_jobs)
    subset_dirs = []
    base_subset_dir = os.path.join(args.outpath, "extraction_subsets")
    os.makedirs(base_subset_dir, exist_ok=True)
    
    for i, subset in enumerate(subsets):
        subset_dir = os.path.join(base_subset_dir, f"subset_{i}")
        os.makedirs(subset_dir, exist_ok=True)
        for subfolder in ["AF3_models", "natives", "MD_frames"]:
            src_dir = os.path.join(input_dir, subfolder)
            dest_subdir = os.path.join(subset_dir, subfolder)
            os.makedirs(dest_subdir, exist_ok=True)
            for filepath in subset:
                fname = os.path.basename(filepath)
                if subfolder == "AF3_models":
                    src_file = os.path.join(os.path.abspath(src_dir), fname)
                    dst_fname = fname
                else:
                    pdb_id = fname[:4]
                    search_pattern = os.path.join(os.path.abspath(src_dir), f"{pdb_id}*.pdb")
                    matches = sorted(glob.glob(search_pattern))
                    if matches:
                        src_file = matches[0]
                        dst_fname = os.path.basename(src_file)
                    else:
                        print(f"No matching file found in {src_dir} for pdb id {pdb_id}")
                        continue
                dst_file = os.path.join(dest_subdir, dst_fname)
                if not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)
        subset_dirs.append(subset_dir)
    
    processes = []
    for subset_dir in subset_dirs:
        job_name = f"loopext_{os.path.basename(subset_dir)}"
        log_file = os.path.join(subset_dir, "job.log")
        cmd = (
            f"srun --job-name={job_name} --output={log_file} "
            f"python {SCRIPT_PATH} --stage loop_extraction --subset_dir {subset_dir} "
            f"--outpath {final_output_dir} --path_discarded {os.path.join(args.outpath, 'discarded')}"
        )
        logging.info(f"Launching job: {cmd}")
        proc = subprocess.Popen(cmd, shell=True)
        processes.append(proc)
    
    for proc in processes:
        proc.wait()
    
    merge_loop_extraction_outputs(subset_dirs, final_output_dir)

def worker_loop_extraction(args):
    worker_output = os.path.join(args.subset_dir, "extracted_loops")
    os.makedirs(worker_output, exist_ok=True)
    run_loop_extraction(args.subset_dir, worker_output, args.path_discarded)

def merge_loop_extraction_outputs(subset_dirs, final_output_dir):
    for subset_dir in subset_dirs:
        subset_output = os.path.join(subset_dir, "extracted_loops")
        if os.path.isdir(subset_output):
            src_folder = os.path.join(subset_output, "EL_AF3_models")
            dest_folder = os.path.join(final_output_dir, "EL_AF3_models")
            os.makedirs(dest_folder, exist_ok=True)
            for file in os.listdir(src_folder):
                src_file = os.path.join(src_folder, file)
                dest_file = os.path.join(dest_folder, file)
                if not os.path.exists(dest_file):
                    shutil.copy(src_file, dest_file)

# -- Label Calculation --
def parallel_label_calculation(args, n_jobs=10):
    import glob
    inpath_harmonized = os.path.join(args.outpath, "harmonized")
    inpath_loop = os.path.join(args.outpath, "extracted_loops")
    final_output_dir = os.path.join(args.outpath, "labels")
    os.makedirs(final_output_dir, exist_ok=True)
    
    ref_dir = os.path.join(inpath_harmonized, "AF3_models")
    file_list = sorted(glob.glob(os.path.join(ref_dir, "*.pdb")))
    subsets = split_list(file_list, n_jobs)
    subset_dirs = []
    base_subset_dir = os.path.join(args.outpath, "label_subsets")
    os.makedirs(base_subset_dir, exist_ok=True)
    
    for i, subset in enumerate(subsets):
        subset_dir = os.path.join(base_subset_dir, f"subset_{i}")
        os.makedirs(subset_dir, exist_ok=True)
        
        # Create symlinks for harmonized subfolders.
        for subfolder in ["AF3_models", "natives", "MD_frames"]:
            src_dir = os.path.join(inpath_harmonized, subfolder)
            dest_subdir = os.path.join(subset_dir, subfolder)
            os.makedirs(dest_subdir, exist_ok=True)
            for filepath in subset:
                fname = os.path.basename(filepath)
                if subfolder == "AF3_models":
                    src_file = os.path.join(os.path.abspath(src_dir), fname)
                    dst_fname = fname
                else:
                    pdb_id = fname[:4]
                    search_pattern = os.path.join(os.path.abspath(src_dir), f"{pdb_id}*.pdb")
                    matches = sorted(glob.glob(search_pattern))
                    if matches:
                        src_file = matches[0]
                        dst_fname = os.path.basename(src_file)
                    else:
                        print(f"No matching file found in {src_dir} for pdb id {pdb_id}")
                        continue
                dst_file = os.path.join(dest_subdir, dst_fname)
                if not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)
        
        # Create symlinks for the extracted loops folder ("EL_AF3_models").
        src_loop_dir = os.path.join(inpath_loop, "EL_AF3_models")
        dest_loop_dir = os.path.join(subset_dir, "EL_AF3_models")
        os.makedirs(dest_loop_dir, exist_ok=True)
        processed_pdb_ids = set()
        for filepath in subset:
            pdb_id = os.path.basename(filepath)[:4]
            if pdb_id in processed_pdb_ids:
                continue
            processed_pdb_ids.add(pdb_id)
            search_pattern = os.path.join(os.path.abspath(src_loop_dir), f"{pdb_id}*_af3*.pdb")
            matches = sorted(glob.glob(search_pattern))
            if matches:
                for src_file in matches:
                    dst_fname = os.path.basename(src_file)
                    dst_file = os.path.join(dest_loop_dir, dst_fname)
                    if not os.path.exists(dst_file):
                        os.symlink(src_file, dst_file)
            else:
                print(f"No matching extracted loop found in {src_loop_dir} for pdb id {pdb_id}")
        
        subset_dirs.append(subset_dir)
    
    processes = []
    for subset_dir in subset_dirs:
        job_name = f"labelcalc_{os.path.basename(subset_dir)}"
        log_file = os.path.join(subset_dir, "job.log")
        cmd = (
            f"srun --job-name={job_name} --output={log_file} "
            f"python {SCRIPT_PATH} --stage label_calculation --subset_dir {subset_dir} "
            f"--outpath {final_output_dir} --inpath_MD_simulations {args.inpath_MD_simulations} "
            f"--path_discarded {os.path.join(args.outpath, 'discarded')}"
        )
        logging.info(f"Launching job: {cmd}")
        proc = subprocess.Popen(cmd, shell=True)
        processes.append(proc)
    
    for proc in processes:
        proc.wait()
    
    merge_label_calculation_outputs(subset_dirs, final_output_dir)

def farallel_label_calculation(args, n_jobs=10):
    import glob
    inpath_harmonized = os.path.join(args.outpath, "harmonized")
    inpath_loop = os.path.join(args.outpath, "extracted_loops")
    final_output_dir = os.path.join(args.outpath, "labels")
    os.makedirs(final_output_dir, exist_ok=True)
    
    ref_dir = os.path.join(inpath_harmonized, "AF3_models")
    file_list = sorted(glob.glob(os.path.join(ref_dir, "*.pdb")))
    subsets = split_list(file_list, n_jobs)
    subset_dirs = []
    base_subset_dir = os.path.join(args.outpath, "label_subsets")
    os.makedirs(base_subset_dir, exist_ok=True)
    
    for i, subset in enumerate(subsets):
        subset_dir = os.path.join(base_subset_dir, f"subset_{i}")
        os.makedirs(subset_dir, exist_ok=True)
        
        # Create symlinks for harmonized subfolders.
        for subfolder in ["AF3_models", "natives", "MD_frames"]:
            src_dir = os.path.join(inpath_harmonized, subfolder)
            dest_subdir = os.path.join(subset_dir, subfolder)
            os.makedirs(dest_subdir, exist_ok=True)
            for filepath in subset:
                fname = os.path.basename(filepath)
                if subfolder == "AF3_models":
                    src_file = os.path.join(os.path.abspath(src_dir), fname)
                    dst_fname = fname
                else:
                    pdb_id = fname[:4]
                    search_pattern = os.path.join(os.path.abspath(src_dir), f"{pdb_id}*.pdb")
                    matches = sorted(glob.glob(search_pattern))
                    if matches:
                        src_file = matches[0]
                        dst_fname = os.path.basename(src_file)
                    else:
                        print(f"No matching file found in {src_dir} for pdb id {pdb_id}")
                        continue
                dst_file = os.path.join(dest_subdir, dst_fname)
                if not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)
        
        # Create symlinks for the extracted loops folder ("EL_AF3_models").
        src_loop_dir = os.path.join(inpath_loop, "EL_AF3_models")
        dest_loop_dir = os.path.join(subset_dir, "EL_AF3_models")
        os.makedirs(dest_loop_dir, exist_ok=True)
        processed_pdb_ids = set()
        for filepath in subset:
            pdb_id = os.path.basename(filepath)[:4]
            if pdb_id in processed_pdb_ids:
                continue
            processed_pdb_ids.add(pdb_id)
            search_pattern = os.path.join(os.path.abspath(src_loop_dir), f"{pdb_id}*_af3*.pdb")
            matches = sorted(glob.glob(search_pattern))
            if matches:
                for src_file in matches:
                    dst_fname = os.path.basename(src_file)
                    dst_file = os.path.join(dest_loop_dir, dst_fname)
                    if not os.path.exists(dst_file):
                        os.symlink(src_file, dst_file)
            else:
                print(f"No matching extracted loop found in {src_loop_dir} for pdb id {pdb_id}")
        
        subset_dirs.append(subset_dir)
    
    job_ids = []
    for subset_dir in subset_dirs:
        job_name = f"labelcalc_{os.path.basename(subset_dir)}"
        log_file = os.path.join(subset_dir, "job.log")
        cmd = (
            f"sbatch --job-name={job_name} --output={log_file} "
            f"--wrap='cd {WORKDIR} && python {SCRIPT_PATH} --stage label_calculation --subset_dir {subset_dir} "
            f"--outpath {final_output_dir} --inpath_MD_simulations {args.inpath_MD_simulations} "
            f"--path_discarded {os.path.join(args.outpath, 'discarded')}'"
        )
        logging.info(f"Submitting job: {cmd}")
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout = proc.stdout.strip()
        job_id = stdout.split()[-1]
        job_ids.append(job_id)
    
    wait_for_slurm_jobs(job_ids)
    merge_label_calculation_outputs(subset_dirs, final_output_dir)

def merge_label_calculation_outputs(subset_dirs, final_output_dir):
    merged_labels = {}
    for subset_dir in subset_dirs:
        subset_output = os.path.join(subset_dir, "labels")
        label_file = os.path.join(subset_output, "labels.json")
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                labels = json.load(f)
            merged_labels.update(labels)
    final_label_file = os.path.join(final_output_dir, "labels.json")
    with open(final_label_file, "w") as f:
        json.dump(merged_labels, f, indent=2)
        
def worker_label_calculation(args):
    worker_output = os.path.join(args.subset_dir, "labels")
    os.makedirs(worker_output, exist_ok=True)
    class Dummy:
        pass
    dummy = Dummy()
    dummy.inpath_MD_simulations = args.inpath_MD_simulations
    dummy.plot_hist = args.plot_hist
    run_label_calculation(dummy, args.subset_dir, args.subset_dir, worker_output, args.path_discarded)

# -- Residue Feature Calculation --
def parallel_residue_features(args, n_jobs=10):
    """
    Splits the "cleaned" PDB files into subsets and creates symlinks for each subset.
    The symlinks are created using absolute paths to ensure that downstream programs can open them.
    """
    import glob
    input_dir = os.path.join(args.outpath, "cleaned")
    final_output_dir = os.path.join(args.outpath, "residue_features")
    os.makedirs(final_output_dir, exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(input_dir, "*.pdb")))
    subsets = split_list(file_list, n_jobs)
    subset_dirs = []
    base_subset_dir = os.path.join(args.outpath, "resfeat_subsets")
    os.makedirs(base_subset_dir, exist_ok=True)
    for i, subset in enumerate(subsets):
        subset_dir = os.path.join(base_subset_dir, f"subset_{i}")
        os.makedirs(subset_dir, exist_ok=True)
        for filepath in subset:
            fname = os.path.basename(filepath)
            dst_file = os.path.join(subset_dir, fname)
            abs_filepath = os.path.abspath(filepath)
            if not os.path.exists(dst_file):
                os.symlink(abs_filepath, dst_file)
        subset_dirs.append(subset_dir)
    
    processes = []
    for subset_dir in subset_dirs:
        job_name = f"resfeat_{os.path.basename(subset_dir)}"
        log_file = os.path.join(subset_dir, "job.log")
        cmd = (
            f"srun --job-name={job_name} --output={log_file} "
            f"python {SCRIPT_PATH} --stage residue_features --subset_dir {subset_dir} "
            f"--outpath {final_output_dir} --path_discarded {os.path.join(args.outpath, 'discarded')}"
        )
        logging.info(f"Launching job: {cmd}")
        proc = subprocess.Popen(cmd, shell=True)
        processes.append(proc)
    
    for proc in processes:
        proc.wait()
    
    merge_residue_features_outputs(subset_dirs, final_output_dir)

def farallel_residue_features(args, n_jobs=10):
    """
    Splits the "cleaned" PDB files into subsets and creates symlinks for each subset.
    The symlinks are created using absolute paths to ensure that downstream programs can open them.
    """
    import glob
    input_dir = os.path.join(args.outpath, "cleaned")
    final_output_dir = os.path.join(args.outpath, "residue_features")
    os.makedirs(final_output_dir, exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(input_dir, "*.pdb")))
    subsets = split_list(file_list, n_jobs)
    subset_dirs = []
    base_subset_dir = os.path.join(args.outpath, "resfeat_subsets")
    os.makedirs(base_subset_dir, exist_ok=True)
    for i, subset in enumerate(subsets):
        subset_dir = os.path.join(base_subset_dir, f"subset_{i}")
        os.makedirs(subset_dir, exist_ok=True)
        for filepath in subset:
            fname = os.path.basename(filepath)
            dst_file = os.path.join(subset_dir, fname)
            abs_filepath = os.path.abspath(filepath)
            if not os.path.exists(dst_file):
                os.symlink(abs_filepath, dst_file)
        subset_dirs.append(subset_dir)
    job_ids = []
    for subset_dir in subset_dirs:
        job_name = f"resfeat_{os.path.basename(subset_dir)}"
        log_file = os.path.join(subset_dir, "job.log")
        cmd = (
            f"sbatch --job-name={job_name} --output={log_file} "
            f"--wrap='cd {WORKDIR} && python {SCRIPT_PATH} --stage residue_features --subset_dir {subset_dir} "
            f"--outpath {final_output_dir} --path_discarded {os.path.join(args.outpath, 'discarded')}'"
        )
        logging.info(f"Submitting job: {cmd}")
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout = proc.stdout.strip()
        job_id = stdout.split()[-1]
        job_ids.append(job_id)
    wait_for_slurm_jobs(job_ids)
    merge_residue_features_outputs(subset_dirs, final_output_dir)

def worker_residue_features(args):
    worker_output = os.path.join(args.subset_dir, "residue_features")
    os.makedirs(worker_output, exist_ok=True)
    run_calculate_residue_features(args.subset_dir, worker_output, args.path_discarded)

def merge_residue_features_outputs(subset_dirs, final_output_dir):
    merged_features = {}
    for subset_dir in subset_dirs:
        subset_output = os.path.join(subset_dir, "residue_features")
        feat_file = os.path.join(subset_output, "residue_features.json")
        if os.path.exists(feat_file):
            with open(feat_file, "r") as f:
                features = json.load(f)
            merged_features.update(features)
    final_feat_file = os.path.join(final_output_dir, "residue_features.json")
    with open(final_feat_file, "w") as f:
        json.dump(merged_features, f, indent=2)

# -- SaProt Embeddings (Parallelized) --
def parallel_saprot_embeddings(args, n_jobs=10):
    import glob
    input_dir = os.path.join(args.outpath, "cleaned")
    final_output_dir = os.path.join(args.outpath, "saprot_embeddings")
    os.makedirs(final_output_dir, exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(input_dir, "*.pdb")))
    subsets = split_list(file_list, n_jobs)
    subset_dirs = []
    base_subset_dir = os.path.join(args.outpath, "saprot_subsets")
    os.makedirs(base_subset_dir, exist_ok=True)
    for i, subset in enumerate(subsets):
        subset_dir = os.path.join(base_subset_dir, f"subset_{i}")
        os.makedirs(subset_dir, exist_ok=True)
        for filepath in subset:
            fname = os.path.basename(filepath)
            dst_file = os.path.join(subset_dir, fname)
            if not os.path.exists(dst_file):
                os.symlink(os.path.abspath(filepath), dst_file)
        subset_dirs.append(subset_dir)
    
    processes = []
    for subset_dir in subset_dirs:
        job_name = f"saprot_{os.path.basename(subset_dir)}"
        log_file = os.path.join(subset_dir, "job.log")
        cmd = (
            f"srun --job-name={job_name} --output={log_file} "
            f"python {SCRIPT_PATH} --stage saprot_embeddings --subset_dir {subset_dir} "
            f"--outpath {final_output_dir} --path_discarded {os.path.join(args.outpath, 'discarded')}"
        )
        logging.info(f"Launching job: {cmd}")
        proc = subprocess.Popen(cmd, shell=True)
        processes.append(proc)
    
    for proc in processes:
        proc.wait()
    
    merge_saprot_embeddings_outputs(subset_dirs, final_output_dir)

def farallel_saprot_embeddings(args, n_jobs=10):
    import glob
    input_dir = os.path.join(args.outpath, "cleaned")
    final_output_dir = os.path.join(args.outpath, "saprot_embeddings")
    os.makedirs(final_output_dir, exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(input_dir, "*.pdb")))
    subsets = split_list(file_list, n_jobs)
    subset_dirs = []
    base_subset_dir = os.path.join(args.outpath, "saprot_subsets")
    os.makedirs(base_subset_dir, exist_ok=True)
    for i, subset in enumerate(subsets):
        subset_dir = os.path.join(base_subset_dir, f"subset_{i}")
        os.makedirs(subset_dir, exist_ok=True)
        for filepath in subset:
            fname = os.path.basename(filepath)
            dst_file = os.path.join(subset_dir, fname)
            if not os.path.exists(dst_file):
                os.symlink(os.path.abspath(filepath), dst_file)
        subset_dirs.append(subset_dir)
    job_ids = []
    for subset_dir in subset_dirs:
        job_name = f"saprot_{os.path.basename(subset_dir)}"
        log_file = os.path.join(subset_dir, "job.log")
        cmd = (
            f"sbatch --job-name={job_name} --output={log_file} "
            f"--wrap='cd {WORKDIR} && python {SCRIPT_PATH} --stage saprot_embeddings --subset_dir {subset_dir} "
            f"--outpath {final_output_dir} --path_discarded {os.path.join(args.outpath, 'discarded')}'"
        )
        logging.info(f"Submitting job: {cmd}")
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout = proc.stdout.strip()
        job_id = stdout.split()[-1]
        job_ids.append(job_id)
    wait_for_slurm_jobs(job_ids)
    merge_saprot_embeddings_outputs(subset_dirs, final_output_dir)

def worker_saprot_embeddings(args):
    worker_output = os.path.join(args.subset_dir, "saprot_embeddings")
    os.makedirs(worker_output, exist_ok=True)
    run_calculate_saprot_embeddings(args.subset_dir, worker_output, args.path_discarded)

def merge_saprot_embeddings_outputs(subset_dirs, final_output_dir):
    final_file = os.path.join(final_output_dir, "saprot_650M.h5")
    import h5py
    with h5py.File(final_file, "w") as hf_final:
        for subset_dir in subset_dirs:
            worker_file = os.path.join(subset_dir, "saprot_embeddings", "saprot_650M.h5")
            if os.path.exists(worker_file):
                with h5py.File(worker_file, "r") as hf_worker:
                    for key in hf_worker.keys():
                        hf_final.copy(hf_worker[key], key)

################ MAIN FUNCTION ################

@func_timer
def main(args):
    print(f"SCRIPT_PATH: {SCRIPT_PATH}")
    print(f"WORKDIR: {WORKDIR}") 
    
    if args.stage:
        if args.stage == "loop_extraction":
            worker_loop_extraction(args)
        elif args.stage == "label_calculation":
            worker_label_calculation(args)
        elif args.stage == "residue_features":
            worker_residue_features(args)
        elif args.stage == "saprot_embeddings":
            worker_saprot_embeddings(args)
        sys.exit(0)

    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    path_discarded = os.path.join(args.outpath, "discarded")
    os.makedirs(path_discarded, exist_ok=True)
    
    # Step 1: Harmonization (parallel)
    outpath_harmonization = os.path.join(args.outpath, "harmonized")
    if not os.path.exists(outpath_harmonization):
        logging.info(f"STEP 1: HARMONIZATION (parallel: {args.parallel})")
        run_harmonization(args, outpath_harmonization, path_discarded)
        # To use the parallel wrapper, uncomment the next line:
        # parallel_harmonization(args, n_jobs=args.n_jobs)
            
    # Step 2: TM-score filtering (sequential)
    logging.info("\nSTEP 2: FILTERING AF3 MODELS REGARDING TM-SCORE")
    run_TM_score_filtering(0.6, outpath_harmonization, path_discarded)    
  
    # Step 3: Loop Extraction (parallel)
    outpath_extraction = os.path.join(args.outpath, "extracted_loops")
    if not os.path.exists(outpath_extraction):
        logging.info(f"\nSTEP 3: LOOP EXTRACTION (parallel: {args.parallel})")
        if args.parallel:
            parallel_loop_extraction(args, n_jobs=args.n_jobs)
        else:
            run_loop_extraction(outpath_harmonization, outpath_extraction, path_discarded) 
            
    # Step 4: Label Calculation (parallel)
    outpath_labels = os.path.join(args.outpath, "labels")
    if not os.path.exists(outpath_labels):
        logging.info(f"\nSTEP 4: LABEL CALCULATION (parallel: {args.parallel})")
        if args.parallel:
            parallel_label_calculation(args, n_jobs=args.n_jobs)
        else:
            run_label_calculation(args, outpath_harmonization, outpath_extraction, outpath_labels, path_discarded) 
            
    # Step 5: Rosetta cleaning (sequential)
    outpath_cleaned = os.path.join(args.outpath, "cleaned")
    if not os.path.exists(outpath_cleaned):         
        os.makedirs(outpath_cleaned, exist_ok=True)
        logging.info("\nSTEP 5: CLEANING AF3 .PDB FILES WITH ROSETTA")
        for path_pdb in glob.glob(os.path.join(outpath_harmonization, "AF3_models", "*.pdb")):
            rosetta_clean(path_pdb, outpath_cleaned)
    
    # Step 6: Residue Feature Calculation (parallel)
    outpath_residue_features = os.path.join(args.outpath, "residue_features")
    if not os.path.exists(os.path.join(outpath_residue_features, "residue_features.json")):
        logging.info(f"\nSTEP 6: CALCULATING RESIDUE FEATURES (parallel: {args.parallel})")
        if args.parallel:
            parallel_residue_features(args, n_jobs=args.n_jobs)
        else:
            run_calculate_residue_features(outpath_cleaned, outpath_residue_features, path_discarded)
    
    # Step 7: SaProt Embeddings (parallel)
    outpath_saprot = os.path.join(args.outpath, "saprot_embeddings")
    if not os.path.exists(os.path.join(outpath_saprot, "saprot_650M.h5")):
        logging.info(f"\nSTEP 7: CALCULATING SAPROT SEQUENCE EMBEDDINGS (parallel: {args.parallel})")
        if args.parallel:
            parallel_saprot_embeddings(args, n_jobs=args.n_jobs)
        else:
            run_calculate_saprot_embeddings(outpath_cleaned, outpath_saprot, path_discarded)
            
    # Step 8: Filter discarded identifiers
    logging.info("\nSTEP 8: FILTERING DISCARDED IDENTIFIERS FROM FINAL FILES")
    filter_discarded(
        path_discarded=path_discarded,
        path_extracted_loops=outpath_extraction,
        path_labels_json=os.path.join(outpath_labels, "labels.json"),
        path_residue_features_json=os.path.join(outpath_residue_features, "residue_features.json"),
        path_saprot_h5=os.path.join(outpath_saprot, "saprot_650M.h5")
    )

    # Step 9: Create final dataset (sequential)
    dirname_dataset = f"DS_{os.path.basename(args.outpath)}"
    outpath_dataset = os.path.join(args.outpath, dirname_dataset)
    if not os.path.exists(outpath_dataset):
        logging.info("STEP 9: CREATING DATASET")
        run_create_dataset(args, outpath_extraction, outpath_labels, outpath_dataset, outpath_residue_features, outpath_saprot)
    
    logging.info("Pipeline Done.")

################ ENTRY POINT ################

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-AF3_LiMD-dp-pipeline-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser()
    DEFAULT_PATH = "/home/florian_wieser/projects/ELEN/elen_training/data_preparation/AF3_LiMD/AF_LiMD_200/harmonized/fix"
    parser.add_argument('--inpath_AF', default=f"{DEFAULT_PATH}/AF3_models")
    parser.add_argument('--inpath_natives', default=f"{DEFAULT_PATH}/natives")
    parser.add_argument('--inpath_MD', default=f"{DEFAULT_PATH}/MD_frames")
    parser.add_argument("--inpath_MD_simulations", type=str, default=f"{DEFAULT_PATH}/MD_simulations",
                        help="Input directory for MD simulation folders.")
    parser.add_argument('--outpath', default=f"{DEFAULT_PATH}/DS_prep_test")
    parser.add_argument('--parallel', action='store_true', default=False, help="Parallelize harmonization and #TODO")
    parser.add_argument('--plot_hist', action='store_true', default=False, help='Plot histograms of labels and dataset.')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing output.')
    parser.add_argument("--path_discarded", type=str, default=f"discarded", help="Output directory for failed PDB files.")
    parser.add_argument('--n_jobs', type=int, default=64, help='Number of parallel jobs to submit per stage.')
    parser.add_argument('--stage', type=str, choices=["harmonization", "loop_extraction", "label_calculation", "residue_features", "saprot_embeddings"],
                        help="If set, run the specified stage in worker mode on the provided subset.")
    parser.add_argument('--subset_dir', type=str, help="Subset directory to process when in worker mode.")
    
    args = parser.parse_args()
    main(args)
