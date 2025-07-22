#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J ELEN_inference
#SBATCH -o ELEN_inference_%j.log
#SBATCH -e ELEN_inference_%j.err
#SBATCH --gres gpu:1
#SBATCH --tasks-per-node 1

#TODO refactor imported files

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
from elen.inference.utils_inference import (
    add_combined_scores_to_dict,
    custom_collate_fn,
    find_pdb_files,
    flatten_predictions,
    load_and_modify_hyperparameters,
    process_loop_residue_data,
    process_residue_data,
    process_pdb_files,
    split_into_chain,
    convert_numpy
)
import elen.training.data as d
import elen.training.model as m
from elen.shared_utils.utils_io import load_from_json
from elen.shared_utils.utils_extraction import clean_pdb, extract_loops, extract_residues
from elen.shared_utils.utils_features import calculate_residue_features_pyrosetta_rs, add_features_to_df
from elen.training.utils_training import seed_everything

###############################################################################
def run_inference(
    input_dir,
    output_dir,
    elen_models_dir,
    elen_models,
    feature_mode="full",
    saprot_embeddings_file=None,
    pocket_type="RP",
    ss_frag="None",
    ss_frag_size=4,
    loop_max_size=10,
    nr_residues=40,
    elen_score="lddt_cad",
    save_activations=False,
    save_features=False,
    overwrite=False,
    use_labels=False,
    num_workers=4,
    batch_size=8,
    **kwargs
):
    """
    Main ELEN inference pipeline.

    Runs ELEN model inference on PDB structure files in a specified directory,
    handling feature extraction, model prediction, and output serialization.

    Parameters
    ----------
    input_dir : str
        Path to directory containing input .pdb files.
    output_dir : str
        Path to directory for saving output files.
    elen_models_dir : str
        Directory containing ELEN model checkpoint files.
    elen_models : list[str]
        Names of ELEN model checkpoints (no file extension).
    feature_mode : str, optional
        Feature mode for model selection. Options: 'full', 'no_saprot', 'geom_only', 'saprot_only'. Default: 'full'.
    saprot_embeddings_file : str, optional
        Path to .h5 file with SaProt sequence embeddings, if using SaProt-based features.
    pocket_type : str, optional
        Region extraction type: 'RP' (residue pocket) or 'LP' (loop pocket). Default: 'RP'.
    ss_frag : str, optional
        Secondary structure fragment type for loop pocket extraction. Options: 'None', 'helix', 'sheet'. Default: 'None'.
    ss_frag_size : int, optional
        Size of secondary structure fragments. Default: 4.
    loop_max_size : int, optional
        Maximum size of loops for extraction. Default: 10.
    nr_residues : int, optional
        Number of residues to extract per pocket/region. Default: 40.
    elen_score : str, optional
        Predicted metric to assign to PDB B-factor. Options: 'lddt', 'cad-score', 'rmsd', or combinations (e.g. 'lddt_cad'). Default: 'lddt_cad'.
    save_activations : bool, optional
        If True, saves intermediate model activations. Default: False.
    save_features : bool, optional
        If True, outputs per-residue features to .csv/.json. Default: False.
    overwrite : bool, optional
        If True, overwrites existing output directory. Default: False.
    use_labels : bool, optional
        If True, uses labels for validation/benchmarking. Default: False.
    num_workers : int, optional
        Number of DataLoader workers for parallel data loading. Default: 4.
    batch_size : int, optional
        Batch size for inference. Default: 8.
    **kwargs
        Additional keyword arguments for compatibility.
    """
    
    # Prepare output directories
    if os.path.exists(output_dir) and overwrite:
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Select ELEN models based on feature mode
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

    path_common_input_data = os.path.join(output_dir, "elen_data")
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
    paths_pdbs = find_pdb_files(input_dir)
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
        
    # ========== COMPUTE FEATURES ==========
    if elen_models in [["ELEN_full"], ["ELEN_NoSeq"]]:
        os.makedirs(path_features, exist_ok=True)
        if not os.path.exists(os.path.join(path_features, "residue_features.json")):
            logging.info("Calculating residue features.")
            calculate_residue_features_pyrosetta_rs(paths_pdbs, path_features, None)
    if elen_models in [["ELEN_full"], ["ELEN_SeqOnly"]]:
        saprot_model = "saprot_650M"
        # calculate sequence embeddings only if they are not provided by CL
        if saprot_embeddings_file is not None and os.path.exists(saprot_embeddings_file):
            logging.info(f"Using provided SaProt embeddings from {saprot_embeddings_file}")
            path_saprot_embeddings = saprot_embeddings_file
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
    
    # ========== PREPARE AND RUN ELEN MODEL ==========
    for elen_model in elen_models:
        path_elen_model = os.path.join(elen_models_dir, f"{elen_model}.ckpt")
        outpath_elen_model = os.path.join(output_dir, f"elen_results_{elen_model}")
        os.makedirs(outpath_elen_model, exist_ok=True)
        path_json = os.path.join(outpath_elen_model, f"elen_scores_{pocket_type}.json")
        if not os.path.exists(path_json):
            logging.info(f"Preparing ELEN model {elen_model}")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(path_elen_model, map_location=torch.device(device))
            # load hyperparameters of model from wandb
            hparams_ckpt = load_and_modify_hyperparameters(checkpoint, input_dir, use_labels, output_dir)
            seed_everything(hparams_ckpt.random_seed)
            transform = d.EDN_Transform(hparams_ckpt, path_features, path_saprot_embeddings, use_labels, pocket_type, feature_mode)
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
            trainer = pl.Trainer.from_argparse_args(hparams_ckpt, accelerator=device, devices=1, logger=False)

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
            if save_activations:
                activations = edn_pl.final_activations_dict
                outpath_activations = os.path.join(outpath_elen_model, f"activations_{elen_model}.json")
                with open(outpath_activations, "w") as f:
                    json.dump(activations, f, default=convert_numpy)
        else:
            data_dict = load_from_json(path_json)
            df = pd.DataFrame(data_dict)
            
        # ========== PROCESS AND WRITE OUTPUT ==========
        if pocket_type == "LP":
            df_scores = process_loop_residue_data(path_extracted, elen_score, df)
        elif pocket_type == "RP":
            df_scores = process_residue_data(path_extracted, elen_score, df)
        
        logging.info("ELEN scores:")
        if save_features: 
            dict_features = load_from_json(os.path.join(path_features, "residue_features.json"))
            df_features = add_features_to_df(dict_features)
            df_features["res_id"] = df_features["res_id"].apply(lambda x: int(x.split("_")[1]))
            df_scores = pd.merge(df_scores, df_features, on=["fname_pdb", "res_id"], how="left")
        print(tabulate(df_scores, headers="keys", tablefmt="psql", showindex=False))
        df_scores.to_csv(os.path.join(outpath_elen_model, f"elen_scores_{pocket_type}.csv"), index=False)

        logging.info("Writing scores to .pdb files.")
        if pocket_type == "LP":
            process_pdb_files(df_scores, outpath_elen_model, path_pdbs_prepared, path_pdbs_split, pocket_type, False)
        elif pocket_type == "RP":
            process_pdb_files(df_scores, outpath_elen_model, path_pdbs_prepared, path_pdbs_split, pocket_type, True)

        logging.info("Done.")


###############################################################################
if __name__ == "__main__":
    """
    Command-line interface for running ELEN inference.
    This interface allows users to select models, feature modes, and output options for structure-based per-residue model quality assessment.
    """

    # Set up logging for informative console output
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    parser = ap.ArgumentParser(
        description=(
            "Run ELEN inference on one or more PDB files. "
            "Extracts features, predicts per-residue quality scores, and writes results to output files."
        )
    )
    parser.add_argument(
        '--input_dir', type=str, default="input_pdbs",
        help="Input directory containing one or more PDB files for inference (default: %(default)s)."
    )
    parser.add_argument(
        '--output_dir', type=str, default="elen_output",
        help="Directory where output files (scores, features, etc.) will be saved (default: %(default)s)."
    )
    parser.add_argument(
        '--feature_mode', default='full', choices=['full', 'no_saprot', 'geom_only', 'saprot_only'],
        help=(
            "Type of input features for model selection: "
            "'full' (all features), "
            "'no_saprot' (exclude SaProt embeddings), "
            "'geom_only' (geometry features only), "
            "'saprot_only' (SaProt embeddings only). "
            "Choose based on available data and application."
        )
    )
    parser.add_argument(
        '--elen_models_dir', type=str, default=f"{PATH_ELEN_MODELS}",
        help="Directory containing ELEN model checkpoints. Should include all model weights to be used for inference."
    )
    parser.add_argument(
        '--saprot_embeddings_file', type=str,
        help="Path to .h5 file with precomputed SaProt sequence embeddings. Required if using SaProt-based features."
    )
    parser.add_argument(
        '--elen_models', nargs='+', default=None,
        help="Names of ELEN model checkpoint files (no file extension) to use for inference. Multiple models can be specified."
    )
    parser.add_argument(
        '--elen_score', type=str, default="lddt_cad",
        help=(
            "Predicted metric to write to the PDB B-factor column. "
            "Options: 'lddt', 'cad-score', 'rmsd', or combinations such as 'lddt_cad'."
        )
    )
    parser.add_argument(
        '--pocket_type', type=str, default="RP", choices=["LP", "RP"],
        help=(
            "Type of residue selection for feature extraction: "
            "'RP' (residue pocket, default) or "
            "'LP' (loop pocket, for loop-centric analysis)."
        )
    )
    # Loop dimension settings (relevant for pocket_type='LP')
    parser.add_argument(
        '--ss_frag', type=str, default="None", choices=["None", "helix", "sheet"],
        help="For loop pocket extraction: type of secondary structure fragment to consider ('helix', 'sheet', or 'None')."
    )
    parser.add_argument(
        '--ss_frag_size', type=int, default=4,
        help="Size of secondary structure fragments to extract for loop pocket analysis (default: %(default)s)."
    )
    parser.add_argument(
        '--loop_max_size', type=int, default=10,
        help="Maximum allowed loop size for extraction (default: %(default)s)."
    )
    parser.add_argument(
        '--nr_residues', type=int, default=40,
        help="Number of residues to include in each extracted pocket (default: %(default)s)."
    )
    parser.add_argument(
        '--save_activations', action="store_true", default=False,
        help="If set, saves intermediate model activations for further analysis or debugging."
    )
    parser.add_argument(
        '--save_features', action="store_true", default=False,
        help="If set, saves extracted per-residue features to CSV/JSON for external use."
    )
    parser.add_argument(
        '--overwrite', action="store_true", default=False,
        help="If set, overwrites existing files in the output directory."
    )
    parser.add_argument(
        '--use_labels', type=bool, default=False,
        help="If set to True, uses ground-truth labels for validation (relevant for training)."
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help="Number of worker processes for parallel data loading (default: %(default)s)."
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help="Batch size for inference (default: %(default)s)."
    )

    # Add model-specific and PyTorch Lightning trainer arguments
    parser = m.EDN_PL.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Run inference with arguments in a logical and documented order
    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        elen_models_dir=args.elen_models_dir,
        elen_models=args.elen_models,
        feature_mode=args.feature_mode,
        saprot_embeddings_file=args.saprot_embeddings_file,
        pocket_type=args.pocket_type,
        ss_frag=args.ss_frag,
        ss_frag_size=args.ss_frag_size,
        loop_max_size=args.loop_max_size,
        nr_residues=args.nr_residues,
        elen_score=args.elen_score,
        save_activations=args.save_activations,
        save_features=args.save_features,
        overwrite=args.overwrite,
        use_labels=args.use_labels,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
