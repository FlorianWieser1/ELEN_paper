#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python
#SBATCH -J make_ds
#SBATCH -o make_ds.log
#SBATCH -e make_ds.err
import os
import sys
import glob
import json
import shutil
import random
import logging
import argparse as ap
import atom3d.datasets.datasets as da
from elen.shared_utils.plot_utils import plot_histogram, plot_violinplot, concatenate_plots
from elen.shared_utils.utils_others import func_timer

### HELPERS
def get_identifiers(pdb_files):
    identifiers = {}
    for pdb_file in pdb_files:
        identifier = os.path.basename(pdb_file)[:4]
        identifiers.setdefault(identifier, []).append(pdb_file)
    return identifiers

def copy_pdbs(target_dir, ids, cutoff):
    path_output = os.path.join(args.outpath, "pdbs", target_dir)
    os.makedirs(path_output, exist_ok=True)
    copied_files_counter = 0
    for id in ids:
        for path_pdb in glob.glob(os.path.join(args.inpath, f"{id}*.pdb")):
            if copied_files_counter >= cutoff:
                break
            shutil.copy(path_pdb, path_output)
            copied_files_counter += 1

def plot_distributions(data, plot_type, outpath_plots):
    logging.info(f"Plotting {plot_type}.")
    plots_per_folder = []
    for folder in ["train", "val", "test"]:
        plots_per_label = []
        dict_max_labels = {"rmsd": 20, "lddt": 1.0, "CAD": 1.0}
        for label in ["rmsd", "lddt", "CAD"]:
            labels_pool = []
            for path_pdb in glob.glob(os.path.join(args.outpath, "pdbs", folder, "*.pdb")):
                labels_pool.extend(data[os.path.basename(path_pdb)][label])
            outpath_plot_label = os.path.join(outpath_plots, f"{label}_{folder}_{plot_type}.png")
            if plot_type == "histogram":
                plots_per_label.append(plot_histogram(labels_pool, label, "black", dict_max_labels[label], outpath_plot_label))
            elif plot_type == "violinplot":
                plots_per_label.append(plot_violinplot(labels_pool, label, "black", outpath_plot_label))
        outpath_plot_folder = os.path.join(outpath_plots, f"{folder}_{plot_type}s.png")
        plots_per_folder.append(concatenate_plots(plots_per_label, "-append", outpath_plot_folder))
    outpath_plot_final = os.path.join(args.outpath, "pdbs", f"{plot_type}s.png")
    concatenate_plots(plots_per_folder, "+append", outpath_plot_final)

def get_id_list_for_folder(cutoff, nr_models_per_identifier):
    ids = []
    nr_models_sum = 0
    for id, count in list(nr_models_per_identifier.items()):
        nr_models_sum += count
        ids.append(id)
        if nr_models_sum >= cutoff:
            break
    for id in ids:
        nr_models_per_identifier.pop(id, None)
    return ids, nr_models_sum, nr_models_per_identifier

###############################################################################
@func_timer
def main(args):
    if os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)

    if args.make_dataset:
        # Get list of all input .pdb files
        pdb_files = glob.glob(os.path.join(args.inpath, "*.pdb"))
       
        max_train_size = int(0.8 * len(pdb_files)) if args.max_sized_dataset else args.train_size
        
        for train_size in range(max_train_size, 1, -1):
            logging.debug(f"train_size {train_size}")
            val_size = round(train_size * args.val_mult)
            test_size = round(train_size * args.test_mult) if args.max_sized_dataset else args.test_size
            total_size = train_size + val_size + test_size

            logging.info(f"train, val, test, total nr_pdbs: {train_size, val_size, test_size, total_size, len(pdb_files)}")
            if args.train_size:
                assert total_size < len(pdb_files), (
                    f"Not enough .pdb files ({total_size}) for demanded dataset sizes"
                )
            # Create dictionary with identifiers as keys and list of corresponding files as values
            identifiers = get_identifiers(pdb_files)
            logging.debug(f"identifiers {identifiers}")
            dict_models_per_id = {id: len(files) for id, files in identifiers.items()}
            keys = list(dict_models_per_id.keys())
            random.shuffle(keys)
            dict_models_per_id = {key: dict_models_per_id[key] for key in keys}
            dict_models_per_id = dict(sorted(dict_models_per_id.items(), key=lambda item: item[1], reverse=True))
            
            logging.debug(f"dict_models_per_id {dict_models_per_id}")
            counter = 0                  
            ids_train, max_size, dict_models_per_id = get_id_list_for_folder(train_size, dict_models_per_id)
            counter += max_size
            ids_val, max_size, dict_models_per_id = get_id_list_for_folder(val_size, dict_models_per_id)
            counter += max_size
            ids_test, max_size, dict_models_per_id = get_id_list_for_folder(test_size, dict_models_per_id)
            counter += max_size
            
            logging.info(f"counter {counter}, len(pdb_files) {len(pdb_files)}")
            if counter <= len(pdb_files) or not args.max_sized_dataset:
                logging.debug(f"train_ids {ids_train}")
                logging.debug(f"val_ids {ids_val}")
                logging.debug(f"test_ids {ids_test}")

                if os.path.exists(args.outpath):
                    shutil.rmtree(args.outpath)
                os.makedirs(args.outpath, exist_ok=True)
                os.makedirs(os.path.join(args.outpath, "pdbs"), exist_ok=True)
                copy_pdbs("train", ids_train, train_size)
                copy_pdbs("val", ids_val, val_size)
                copy_pdbs("test", ids_test, test_size)
                logging.info("Dataset curation done.")
                break

    if args.lmdb:
        logging.info("Creating lmdb datasets.")
        outpath_lmdb = os.path.join(args.outpath, "lmdbs")
        os.makedirs(outpath_lmdb, exist_ok=True)
        for folder in ['train', 'val', 'test']:
            path_dir = os.path.join(args.outpath, "pdbs", folder)
            dataset = da.load_dataset(path_dir, 'pdb')
            outpath_dir = os.path.join(outpath_lmdb, folder)
            da.make_lmdb_dataset(dataset, outpath_dir)
        
    if args.plot_hist:
        outpath_plots = os.path.join(args.outpath, "pdbs", "plots")
        os.makedirs(outpath_plots, exist_ok=True)
        with open(args.path_labels, "r") as file:
            data = json.load(file)
        plot_distributions(data, "histogram", outpath_plots)
        plot_distributions(data, "violinplot", outpath_plots)
    logging.info("Done.")

###############################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-AF3_LiMD-create_dataset-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    parser = ap.ArgumentParser()
    default_path = "/home/florian_wieser/testbox/filter_ds/dir_1/out_loops/outtest"
    parser.add_argument('--inpath', type=str, default=default_path)
    parser.add_argument('--outpath', type=str, default="DS_EL_AFLi_MD_200")
    parser.add_argument('--make_dataset', action='store_true', default=True, help='Pick dataset.')
    parser.add_argument('--max_sized_dataset', action='store_true', default=False, help='Get biggest dataset possible from inpath.')
    parser.add_argument('--val_mult', type=float, default=0.2)
    parser.add_argument('--test_mult', type=float, default=0.1)
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--lmdb', action='store_true', default=False, help='Wrap .pdb files to lmdb (atom3D fileformat).')
    parser.add_argument('--plot_hist', action='store_true', default=False, help='Plot a histogram for all three datasets.')
    parser.add_argument('--path_labels', type=str, default="labels_clean.json")
    args = parser.parse_args()
    if not args.max_sized_dataset and args.train_size is None:
        logging.error("Error: either --max_sized_dataset, or --train_size X must be set.")
        sys.exit(1)
    main(args)
