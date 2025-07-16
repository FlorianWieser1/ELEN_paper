#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#TODO 
# go through min max part
# check if actually only rmsd is transformed
# test again

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
import argparse as ap
import logging
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import json 
import torch_geometric
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning.loggers as log
import atom3d.datasets as da 
from elen.shared_utils.utils_trafo_labels import get_scaled_labels_and_min_max_scale
from elen.training.utils_training import seed_everything
from elen.training.utils_training import initialize_training_script
from elen.shared_utils.utils_plot import log_plots
from elen.shared_utils.utils_others import func_timer
from elen.training.utils_training import transform_and_plot_label
from elen.training.utils_training import log_metrics
import elen.training.data as d
import elen.training.model as m
from elen.config import PATH_PROJECT
logger = logging.getLogger("lightning")   
from torch_geometric.data import Batch

# TODO try to integrate labels into lmdbs

###############################################################################
# Define a custom collate function that filters out any None datapoints.
def custom_collate_fn(batch):
    filtered_batch = [data for data in batch if data is not None]
    if len(filtered_batch) == 0:
        raise ValueError("All datapoints in batch were None!")
    return Batch.from_data_list(filtered_batch)

###############################################################################
@func_timer
def main(hparams):
    
    logger.info(f"Set random seed to {hparams.random_seed:}...")
    if hparams.random_seed == 123: logger.warning("WARNING RANDOM SEED IS FIXED! INTENDED?")
    seed_everything(hparams.random_seed)    
    #torch.set_float32_matmul_precision('medium') # activate for production run
    
    # initialize script depending if its run via slurm, locally or restarted 
    hparams, keyword_args, edn_pl = initialize_training_script(hparams, logger)
    
    os.makedirs(hparams.output_dir, exist_ok=True)
    outpath = os.path.join(hparams.output_dir, f"results_{wandb.run.id}")
    os.makedirs(outpath, exist_ok=True)
   
    logger.info(f"Load and scale labels...")
    # Open and load the JSON file
    path_scales_json = os.path.join(hparams.train_dir, "../../scales.json") 
    with open(path_scales_json, "r") as f:
        data_scales = json.load(f)
    edn_pl.min_scale = [data_scales['lddt']['min'], data_scales['CAD']['min'], data_scales['rmsd']['min']]
    edn_pl.max_scale = [data_scales['lddt']['max'], data_scales['CAD']['max'], data_scales['rmsd']['max']]
    
    logger.info(f"Transform input data...")
    transform = d.EDN_Transform(hparams, f"{hparams.train_dir}/../..", hparams.use_labels, hparams.prediction_type) 

    logger.info(f"Creating dataloaders...")
    train_dataset = da.load_dataset(hparams.train_dir, hparams.filetype, transform=transform)
   
    shuffle = False if hparams.random_seed == 123 else True # if random seed is fixed also fix shuffling (for debugging)
    train_dataloader = torch_geometric.loader.DataLoader(train_dataset, batch_size=hparams.batch_size, 
                                                       num_workers=hparams.num_workers, shuffle=shuffle, collate_fn=custom_collate_fn)
    val_dataset = da.load_dataset(hparams.val_dir, hparams.filetype, transform=transform)
    val_dataloader = torch_geometric.loader.DataLoader(val_dataset, batch_size=hparams.batch_size, 
                                                     num_workers=hparams.num_workers, shuffle=False, collate_fn=custom_collate_fn)
    test_dataset = da.load_dataset(hparams.test_dir, hparams.filetype, transform=transform)
    test_dataloader = torch_geometric.loader.DataLoader(test_dataset, batch_size=hparams.batch_size, 
                                                      num_workers=hparams.num_workers, shuffle=False, collate_fn=custom_collate_fn)

    logger.info(f"Setting up trainer...")
    wandb_logger = log.WandbLogger(save_dir=hparams.output_dir, project="edn-experiment", name=wandb.run.id, 
                                   log_model=True, mode="online")
    tb_logger = pl.loggers.TensorBoardLogger(hparams.output_dir, name="tensorboard", version="")
    
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(hparams.output_dir,          
        'checkpoints'), filename=f"{wandb.run.id}_{{epoch}}", every_n_epochs=1, save_top_k=-1)
    #early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
    
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    num_devices = torch.cuda.device_count()
    num_devices = 1 if num_devices == 0 else num_devices
    strategy = "ddp" if num_devices > 1 else None
    trainer = pl.Trainer.from_argparse_args(hparams, devices=num_devices, strategy=strategy, num_nodes=hparams.num_nodes, 
                                            logger=[wandb_logger, tb_logger], accelerator=device, callbacks=[checkpoint_callback])
    
    # TRAINING
    logger.info(f"Running training on {hparams.train_dir:} with val {hparams.val_dir:}...")
    _ = trainer.fit(edn_pl, train_dataloader, val_dataloader, **keyword_args)
  
    # TESTING    
    trainer.test(edn_pl, test_dataloader, verbose=True)
        
    serializable_dict = {key: value.tolist() if isinstance(value, np.ndarray) else value
                     for key, value in edn_pl.final_activations_dict.items()}
    
    with open(os.path.join(hparams.output_dir, "activations.json"), "w") as f:
        json.dump(serializable_dict, f, indent=4)
        
    accumulated_df = pd.DataFrame()
    labels = [label for label in [hparams.label_1, hparams.label_2, hparams.label_3] if label is not None]
    print("\t R \t spear \t r2 \t MAE \t var_out")
    for label, min_val, max_val in zip(labels, edn_pl.min_scale, edn_pl.max_scale):
        print(f"label, min_val, max_val: {label, min_val, max_val}")
        df = transform_and_plot_label(label, edn_pl.predictions, min_val, max_val, hparams.label_scale_type, outpath)
        accumulated_df = pd.concat([accumulated_df, df], axis=1)
        log_metrics(label, accumulated_df)
       
    print(accumulated_df)    
    table = wandb.Table(dataframe=accumulated_df)
    wandb.log({'predictions': table})
    accumulated_df.to_csv(os.path.join(outpath, "predictions.csv"))
    
    log_plots(outpath, "_corr.png", "target-correlation")
    wandb.finish()
    print("Done.")

###############################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='ELEN-%(levelname)s(%(asctime)s): %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S')
    #DATASET = "RP_500k"
    #DATASET = "LP_20"
    DATASET = "LP_20_labels_included"
    #DATASET = "RP_1M_labelled"
    parser = ap.ArgumentParser()
    #parser.add_argument('--dataset', type=str, default=f"{PATH_PROJECT}/../elen_training/datasets/{DATASET}/lmdbs/train")
    parser.add_argument('--train_dir', type=str, default=f"{PATH_PROJECT}/../elen_training/datasets/{DATASET}/lmdbs/train")
    parser.add_argument('--val_dir', type=str, default=f"{PATH_PROJECT}/../elen_training/datasets/{DATASET}/lmdbs/val")
    parser.add_argument('--test_dir', type=str, default=f"{PATH_PROJECT}/../elen_training/datasets/{DATASET}/lmdbs/test")
    parser.add_argument('--checkpoint', type=str, default=None)
    #parser.add_argument('--yaml', type=str, default="yaml/LP_20_acluster.yaml")
    parser.add_argument('--yaml', type=str, default=None)
    parser.add_argument('--wandb', type=str, default='disabled', choices=['online', 'offline', 'disabled'])
    parser.add_argument('-out', '--output_dir', type=str, default='out')
    parser.add_argument('--filetype', type=str, default='lmdb', choices=['lmdb', 'pdb'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=int(np.random.randint(1, 10e6)))
    #parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--use_labels', type=bool, default=True)
    parser.add_argument('--prediction_type', type=str, default="LP", choices=["LP", "RP"])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lmax', type=int, default=2, choices=[2, 3])
    parser.add_argument('--layer_size', type=int, default=40, choices=[20, 40, 60])
    parser.add_argument('--label_1', type=str, default="lddt", choices=["rmsd", "lddt", "CAD"])
    parser.add_argument('--label_2', type=str, default="CAD", choices=["rmsd", "lddt", "CAD"])
    parser.add_argument('--label_3', type=str, default="rmsd", choices=["rmsd", "lddt", "CAD"])
    parser.add_argument('--label_scale_type', type=str, default="normalization", 
                        choices=["none", "normalization", "standardization", "centralization"])
    parser.add_argument('--hr_atomlevel_features', nargs='+', default=["none"], 
                        choices=["none", "hbonds", "charges", "b-factor"])
    parser.add_argument('--atomlevel_features', nargs='+', default=["secondary_structure", "sequence", "sasa", "sap-score", "energies", "hbonds", "saprot_650M"], 
                        choices=["none", "hbonds", "sasa", "energies", "sequence", "secondary_structure", 
                                 "sap-score", "esm2_8M", "esm2_35M", "saprot_35M", "saprot_650M"])
    parser.add_argument('--reslevel_features', nargs='+', default=["secondary_structure", "sequence", "sasa", "sap-score", "energies", "hbonds", "saprot_650M"], 
                        choices=["none", "hbonds", "sasa", "energies", "sequence", "secondary_structure", 
                                 "sap-score", "esm2_8M", "esm2_35M" "saprot_35M", "saprot_650M"])
    parser.add_argument('--dense_layer_scale', type=float, default=1, choices=[0.5, 1, 2])
    parser.add_argument('--reslevel_feature_scale_type', type=str, default="normalization", 
                        choices=['none', 'standardization', 'normalization', 'centralization'])
    parser.add_argument('--reslevel_feature_scale_factor', type=int, default=1)
    parser.add_argument('--atomlevel_feature_scale_factor', type=int, default=1)
    parser.add_argument('--num_nearest_neighbors', type=int, default=40)
    parser.add_argument('--GRM_r1', type=int, default=10)
    parser.add_argument('--GRM_nr1', type=int, default=20)
    parser.add_argument('--GRM_r2', type=int, default=20)
    parser.add_argument('--GRM_nr2', type=int, default=40)
    parser.add_argument('--batch_norm', type=str, default="after", choices=["both", "before", "after"])
    parser.add_argument('--skip_connections', type=bool, default=False)
    parser.add_argument('--activation', type=str, default="elu", choices=['relu', 'leaky_relu', 'elu', 'softmax', 'swish', 'mish'])
    parser.add_argument('--optimizer', type=str, default="nadam", choices=['adam', 'adamw', 'rmsprop', 'nadam'])
    parser.add_argument('--weight_decay', type=float, default=0, choices=[0.0, 0.001, 0.0001])
    parser.add_argument('--loss_type', type=str, default="huber", choices=['huber', 'mae', 'mse', 'smooth_l1', 'cosine_similarity'])
    parser.add_argument('--huber_delta', type=float, default=50.0)
    
    parser = m.EDN_PL.add_model_specific_args(parser) # add model specific args
    parser = pl.Trainer.add_argparse_args(parser) # add trainer args
    hparams = parser.parse_args()
    main(hparams)
