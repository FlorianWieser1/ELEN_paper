#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J ELEN_inference
#SBATCH -o ELEN_inference_%j.log
#SBATCH -e ELEN_inference_%j.err
#SBATCH --gres gpu:1
#SBATCH --tasks-per-node 1

#TODO refactor

# ========== Standard Library ==========
import argparse as ap
import glob
import json
import logging
import os
import shutil
import subprocess
import sys

# ========== Third-Party Packages ==========
import pandas as pd
from tabulate import tabulate
import torch
import torch_geometric
import pytorch_lightning as pl
import atom3d.datasets as da

# ========== Local/Project-Specific ==========
from elen.config import PATH_DP_SCRIPTS, PATH_ELEN_MODELS, PATH_ROSETTA_TOOLS
from elen.data_preparation.utils_extraction import clean_pdb, extract_loops, extract_residues
from elen.data_preparation.utils_features import calculate_residue_features_pyrosetta_rs, add_features_to_df
from elen.inference.utils_inference import (
    add_combined_scores_to_dict,
    check_features_exist,
    custom_collate_fn,
    find_pdb_files,
    flatten_predictions,
    load_and_modify_hyperparameters,
    process_loop_residue_data,
    process_pdb_files_LP,
    process_pdb_files_RP,
    process_residue_data,
    split_into_chain,
    convert_numpy
)
import elen.training.data as d
import elen.training.model as m
from elen.shared_utils.utils_io import load_from_json
from elen.training.utils_training import seed_everything

###############################################################################
def run_inference(
    inpath,
    outpath,
    path_elen_models,
    elen_models,
    feature_mode="full", 
    pocket_type="RP",
    elen_score_to_pdb="lddt_cad",
    use_labels=False,
    num_workers=4,
    batch_size=8,
    ss_frag="None",
    ss_frag_size=4,
    loop_max_size=10,
    nr_residues=40,
    overwrite=False,
    print_activations=False,
    print_features=False,
    path_saprot_embeddings_CL=None,
    **kwargs
):
    """
    Main ELEN inference pipeline.

    Handles PDB input preparation, feature extraction, model inference, and
    output serialization for given protein models.

    Parameters
    ----------
    inpath : str
        Input directory containing .pdb files.
    outpath : str
        Output directory.
    path_elen_models : str
        Directory with ELEN model checkpoints.
    elen_models : list[str]
        List of ELEN model checkpoint names (without .ckpt extension).
    pocket_type : str
        Extraction type: 'LP' (loops) or 'RP' (residues).
    elen_score_to_pdb : str
        Which score to map to .pdb B-factors.
    use_labels : bool
        If True, use labeled data (for validation).
    num_workers : int
        Number of DataLoader workers.
    batch_size : int
        DataLoader batch size.
    ss_frag : str
        Secondary structure fragment type.
    ss_frag_size : int
        Size of secondary structure fragments.
    loop_max_size : int
        Max loop length for extraction.
    nr_residues : int
        Number of residues for extraction.
    overwrite : bool
        Overwrite output directory if exists.
    print_activations : bool
        If True, save model activations.
    **kwargs
        Additional arguments (not used).
    """
    # Prepare output directories
    if os.path.exists(outpath) and overwrite:
        shutil.rmtree(outpath)
    os.makedirs(outpath, exist_ok=True)
    if elen_models is None or len(elen_models) == 0:
        feature_mode_models = {
            "full": ["ELEN_full"],     # model that uses all features
            "no_saprot": ["ELEN_NoSeq"],   # model that uses only per-residue features, no sequence embeddings
            "saprot_only": ["ELEN_SeqOnly"], # model that uses only sequence embeddings
            "geom_only": ["ELEN_GeomOnly"],  # model that uses no additional features
        }
        elen_models = feature_mode_models[feature_mode]
        logging.info(f"Selected default ELEN model(s) for feature_mode '{feature_mode}': {elen_models}")
    else:
        logging.info(f"User provided ELEN model(s): {elen_models}")    

    path_common_input_data = os.path.join(outpath, "elen_data")
    os.makedirs(path_common_input_data, exist_ok=True)
    dir_extracted = "extracted_loops" if pocket_type == "LP" else "extracted_residues"
    path_extracted = os.path.join(path_common_input_data, dir_extracted)
    path_pdbs_split = os.path.join(path_common_input_data, "split_pdbs")
    path_pdbs_prepared = os.path.join(path_common_input_data, "prepared_pdbs")
    path_features = os.path.join(path_common_input_data, "features")
    path_pdbs_cleaned = os.path.join(path_common_input_data, "cleaned")
    path_saprot_embeddings = ""
    
    # ========== PREPARE COMMON INPUT DATA ==========
    # 1. Split original PDB files into chains (to preserve residue numbering)
    paths_pdbs = find_pdb_files(inpath)
    if not os.path.exists(path_pdbs_split):
        logging.info("Saving original .pdb residue numbering.")
        os.makedirs(path_pdbs_split, exist_ok=True)
        for path_pdb in paths_pdbs:
            try:
                split_into_chain(path_pdb, path_pdbs_split)
            except Exception as e:
                logging.error(f"Error splitting pdb {path_pdb}: {str(e)}")

    # 2. Clean input PDB files
    if not os.path.exists(path_pdbs_cleaned):
        logging.info("Cleaning input .pdb files.")
        os.makedirs(path_pdbs_cleaned, exist_ok=True)
        for path_pdb in paths_pdbs:
            try:
                clean_pdb(path_pdb, path_pdbs_cleaned)
            except Exception as e:
                logging.error(f"Error cleaning pdb {path_pdb}: {str(e)}")

    # 3. Split cleaned PDB files into chains
    paths_pdbs = glob.glob(os.path.join(path_pdbs_cleaned, "*.pdb"))
    if not os.path.exists(path_pdbs_prepared):
        logging.info("Splitting input .pdb files.")
        os.makedirs(path_pdbs_prepared, exist_ok=True)
        for path_pdb in paths_pdbs:
            try:
                split_into_chain(path_pdb, path_pdbs_prepared)
            except Exception as e:
                logging.error(f"Error splitting pdb {path_pdb}: {str(e)}")

    # ========== EXTRACT LOOP POCKETS / RESIDUES ==========
    paths_pdbs = glob.glob(os.path.join(path_pdbs_prepared, "*.pdb"))
    if not os.path.exists(path_extracted):
        logging.info("Extracting pockets/residues.")
        valid_extraction = []
        for path_pdb in paths_pdbs:
            try:
                if pocket_type == 'LP':
                    extract_loops(path_pdb, path_common_input_data, ss_frag, ss_frag_size, loop_max_size, nr_residues)
                elif pocket_type == 'RP':
                    extract_residues(path_pdb, path_common_input_data, nr_residues)
                valid_extraction.append(path_pdb)
            except Exception as e:
                logging.error(f"Error extracting {pocket_type} from pdb {path_pdb}: {str(e)}")
        # Only proceed with files that were successfully extracted
        paths_pdbs = valid_extraction

    if elen_models in [["ELEN_full"], ["ELEN_NoSeq"]]:
        # ========== COMPUTE FEATURES ==========
        os.makedirs(path_features, exist_ok=True)
        if not os.path.exists(os.path.join(path_features, "residue_features.json")):
            logging.info("Calculating residue features.")
            #calculate_residue_features(paths_pdbs, path_features, None)
            calculate_residue_features_pyrosetta_rs(paths_pdbs, path_features, None)
        
    if elen_models in [["ELEN_full"], ["ELEN_SeqOnly"]]:
        saprot_model = "saprot_650M"
        # calculate sequence embeddings only if they are not provided by CL
        if path_saprot_embeddings_CL is not None and os.path.exists(path_saprot_embeddings_CL):
            logging.info(f"Using provided SaProt embeddings from {path_saprot_embeddings_CL}")
            path_saprot_embeddings = path_saprot_embeddings_CL
        else:  
            logging.info(f"Computing sequence embeddings using SaProt model {saprot_model}.")
            path_saprot_embeddings = os.path.join(path_features, f"{saprot_model}.h5")
            if not os.path.exists(path_saprot_embeddings):
                logging.info("Calculating sequence embeddings.")
                # Run external embedding script
                result = subprocess.run([
                    f"{PATH_DP_SCRIPTS}/compute_SaProt_embeddings.py",
                    "--inpath_models", path_pdbs_prepared,
                    "--outpath", path_features,
                    "--saprot_model", saprot_model,
                    "--write_json"
                ], capture_output=True)
                if result.returncode != 0:
                    logging.error(f"SaProt embedding script failed:\n{result.stderr.decode()}")
                    raise RuntimeError("SaProt embedding computation failed.")
    
    #check_features_exist(path_extracted, paths_pdbs, path_features, path_saprot_embeddings)

    # ========== PREPARE AND RUN ELEN MODEL ==========
    for elen_model in elen_models:
        path_elen_model = os.path.join(path_elen_models, f"{elen_model}.ckpt")
        outpath_elen_model = os.path.join(outpath, f"elen_results_{elen_model}")
        os.makedirs(outpath_elen_model, exist_ok=True)
        path_json = os.path.join(outpath_elen_model, f"elen_scores_{pocket_type}.json")
        if not os.path.exists(path_json):
            logging.info(f"Preparing ELEN model {elen_model}")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(path_elen_model, map_location=torch.device(device))
            # load hyperparameters of model from wandb
            hparams_ckpt = load_and_modify_hyperparameters(checkpoint, inpath, use_labels, outpath)
            seed_everything(hparams_ckpt.random_seed)
            transform = d.EDN_Transform(hparams_ckpt, path_features, path_saprot_embeddings, use_labels, pocket_type)
            logging.info("Setup dataloaders...")
            dataset = da.load_dataset(path_extracted, "pdb", transform=transform)

            if batch_size is not None:
                batch_size_final = batch_size
            elif torch.cuda.is_available():
                batch_size_final = hparams_ckpt.batch_size
            else:
                batch_size_final = 1

            dataloader = torch_geometric.loader.DataLoader(
                dataset,
                batch_size=batch_size_final,
                num_workers=num_workers,
                shuffle=False,
                collate_fn=custom_collate_fn,
            )
            logging.info("Loading model weights...")
            edn_pl = m.EDN_PL.load_from_checkpoint(path_elen_model, strict=False)
            trainer = pl.Trainer.from_argparse_args(hparams_ckpt, accelerator=device, devices=1)

            # PREDICTION
            logging.info("Running prediction...")
            _ = trainer.test(edn_pl, dataloader, verbose=True)

            # save ELEN predictions
            dict_pred = edn_pl.predictions
            dict_pred = add_combined_scores_to_dict(dict_pred)
            rows = flatten_predictions(dict_pred)
            df = pd.DataFrame(rows)
            df.to_json(path_json)

            # save activation values
            if print_activations:
                activations = edn_pl.final_activations_dict
                outpath_activations = os.path.join(outpath_elen_model, f"activations_{elen_model}.json")
                with open(outpath_activations, "w") as f:
                    json.dump(activations, f, default=convert_numpy)
        else:
            data_dict = load_from_json(path_json)
            df = pd.DataFrame(data_dict)
            
        # ========== PROCESS AND WRITE OUTPUT ==========
        if pocket_type == "LP":
            df_scores = process_loop_residue_data(path_extracted, elen_score_to_pdb, df)
        elif pocket_type == "RP":
            df_scores = process_residue_data(path_extracted, elen_score_to_pdb, df)
        
        logging.info("ELEN scores:")
        if print_features: 
            dict_features = load_from_json(os.path.join(path_features, "residue_features.json"))
            df_features = add_features_to_df(dict_features)
            df_features["res_id"] = df_features["res_id"].apply(lambda x: int(x.split("_")[1]))
            df_scores = pd.merge(df_scores, df_features, on=["fname_pdb", "res_id"], how="left")
        print(tabulate(df_scores, headers="keys", tablefmt="psql", showindex=False))
        df_scores.to_csv(os.path.join(outpath_elen_model, f"elen_scores_{pocket_type}.csv"), index=False)

        logging.info("Writing scores to .pdb files.")
        if pocket_type == "LP":
            process_pdb_files_LP(df_scores, outpath_elen_model, path_pdbs_prepared, path_pdbs_split, pocket_type)
        elif pocket_type == "RP":
            process_pdb_files_RP(df_scores, outpath_elen_model, path_pdbs_prepared, path_pdbs_split, pocket_type)

        logging.info("Done.")


###############################################################################
if __name__ == "__main__":
    """
    Command-line interface for running ELEN inference.
    Sets up argument parsing, logging, and invokes the main inference routine.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    parser = ap.ArgumentParser(
        description="ELEN inference: runs model(s) on provided PDB files, extracts features, and writes output scores."
    )
    parser.add_argument('--inpath', type=str, default="input", help="Input directory containing .pdb file(s).")
    parser.add_argument('--outpath', type=str, default="out", help="Output directory.")
    parser.add_argument('--feature_mode', default='full', choices=['full', 'no_saprot', 'geom_only', 'saprot_only'],
                                                                help="Type of input features to use for ELEN model selection.")
    parser.add_argument('--path_elen_models', type=str, default=f"{PATH_ELEN_MODELS}", help="Path to ELEN model checkpoints.")
    parser.add_argument('--path_saprot_embeddings', type=str, help="Path to .h5 file containg sequence embeddings from SaProt.")
    parser.add_argument('--elen_models', nargs='+', default=None, help="ELEN model checkpoint names (no extension).")
    parser.add_argument("--pocket_type", type=str, default="RP", choices=["LP", "RP"], help="Extraction type: loop pocket (LP) or residue pocket (RP).")
    parser.add_argument("--elen_score_to_pdb", type=str, default="lddt_cad", help="Score to write to .pdb B-factor.")
    parser.add_argument("--use_labels", type=bool, default=False, help="Whether to use labels (validation mode).")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--ss_frag", type=str, default="None", choices=["None", "helix", "sheet"], help="Loop extraction: type of secondary structure fragment.")
    parser.add_argument("--ss_frag_size", type=int, default=4, help="Loop extraction: fragment size.")
    parser.add_argument("--loop_max_size", type=int, default=10, help="Loop extraction: max loop size.")
    parser.add_argument("--nr_residues", type=int, default=40, help="Extraction: number of residues to use.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing run/output directory.")
    parser.add_argument("--print_activations", action="store_true", default=False, help="Save model activations.")
    parser.add_argument("--print_features", action="store_true", default=False, help="Also output per-residue features to .csv/.json.")
    
    # Add model-specific and trainer arguments
    parser = m.EDN_PL.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    
    run_inference(
        inpath=args.inpath,
        outpath=args.outpath,
        path_elen_models=args.path_elen_models,
        elen_models=args.elen_models,
        feature_mode=args.feature_mode,
        pocket_type=args.pocket_type,
        elen_score_to_pdb=args.elen_score_to_pdb,
        use_labels=args.use_labels,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        ss_frag=args.ss_frag,
        ss_frag_size=args.ss_frag_size,
        loop_max_size=args.loop_max_size,
        nr_residues=args.nr_residues,
        overwrite=args.overwrite,
        print_activations=args.print_activations,
        print_features=args.print_features,
        path_saprot_embeddings=args.path_saprot_embeddings
    )
