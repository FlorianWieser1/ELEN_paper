import numpy as np
import pandas as pd
import logging
import pathlib
import sys, os, glob
import argparse as ap
import atom3d.datasets as da
import pytorch_lightning as pl
import pytorch_lightning.loggers as log
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch_geometric
import wandb
import torch
import json
PROJECT_PATH = os.environ.get('PROJECT_PATH')
DATA_PATH = os.environ.get('DATA_PATH')
sys.path.append(f"{PROJECT_PATH}/geometricDL/edn/edn_multi_labels_pr")
sys.path.append(f"{PROJECT_PATH}/geometricDL/edn/shared_utils")
import data_LM_dev as d
import model_heads as m
from plot_utils import plot_target_corr, log_plots
from multilabel_utils import scale_local_labels, scale_local_labels_from_json, scale_local_labels_back, calculate_regression_metrics
from shared_utils import func_timer
root_dir = pathlib.Path(__file__).parent.parent.absolute()
logger = logging.getLogger("lightning")
# TODO make random seed random again for real sweep

@func_timer
def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default=DATA_PATH + "/pdbs/train")
    parser.add_argument('--val_dir', type=str, default=DATA_PATH + "/pdbs/val")
    parser.add_argument('--test_dir', type=str, default=DATA_PATH + "/pdbs/test")
    parser.add_argument('--exp_dir', type=str, default=DATA_PATH + "/pdbs/test")
    parser.add_argument('-out', '--output_dir', type=str, default='out')
    parser.add_argument('-f', '--filetype', type=str, default='pdb', choices=['lmdb', 'pdb', 'silent'])
    parser.add_argument('--num_workers', type=int, default=0)
    #parser.add_argument('--random_seed', '-seed', type=int, default=123)
    parser.add_argument('--random_seed', '-seed', type=int, default=int(np.random.randint(1, 10e6)))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lmax', type=int, default=2, choices=[2, 3])
    parser.add_argument('--LM', type=int, default=8, choices=[8, 35, 150, 650])
    parser.add_argument('--LM_level', type=int, default=0, choices=[0, 1, 2, 3]) # 0: no LM, 1: atm LM, 2: res LM, 3: both
    parser.add_argument('--layer_size', type=int, default=40, choices=[20, 40, 60])
    parser.add_argument('--label_1', type=str, default=None, choices=["rmsd", "lddt", "CAD"])
    parser.add_argument('--label_2', type=str, default=None, choices=["rmsd", "lddt", "CAD"])
    parser.add_argument('--label_3', type=str, default=None, choices=["rmsd", "lddt", "CAD"])
    parser.add_argument('--wandb', action="store_true", default=False)
    parser.add_argument('--type_label_scale', type=str, default="normalization", choices=["normalization", "centralization"])
    parser.add_argument('--reslevel_features', nargs='+', default=["none"], 
                        choices=["none", "hbonds", "sasa", "energies", "sequence", "secondary_structure", "sap-score", "esm2_t6_8M_UR50D"])
    parser.add_argument('--dense_layer_scale', type=float, default=1, choices=[0.5, 1, 2])
    parser.add_argument('--stand_features', type=int, default=0)
    parser = m.EDN_PL.add_model_specific_args(parser) # add model specific args
    parser = pl.Trainer.add_argparse_args(parser) # add trainer args
    hparams = parser.parse_args()
   
    if hparams.wandb: 
        wandb.init(mode="online", project="edn-experiment")
        # overwrite hparams with wandb parameters
        hparams.learning_rate = wandb.config.learning_rate
        hparams.max_epochs = wandb.config.max_epochs
        hparams.lmax = wandb.config.lmax
        hparams.LM = wandb.config.LM
        hparams.LM_level = wandb.config.LM_level
        hparams.layer_size = wandb.config.layer_size
        hparams.label_1 = wandb.config.label_1
        if hasattr(wandb.config, 'label_2'):
            hparams.label_2 = wandb.config.label_2
        if hasattr(wandb.config, 'label_3'):
            hparams.label_3 = wandb.config.label_3
        hparams.batch_size = wandb.config.batch_size
        hparams.type_label_scale = wandb.config.type_label_scale
        hparams.reslevel_features = wandb.config.reslevel_features
        hparams.dense_layer_scale = wandb.config.dense_layer_scale
        hparams.stand_features = wandb.config.stand_features

        print(hparams.reslevel_features)
        print(type(hparams.reslevel_features))
        if isinstance(hparams.reslevel_features, str):
            hparams.reslevel_features = [hparams.reslevel_features]
        print(hparams.reslevel_features)
        print(type(hparams.reslevel_features))
    else:
        wandb.init(mode="disabled", project="edn-experiment")
        hparams.max_epochs = hparams.epochs
    
    logger.info(f"Set random seed to {hparams.random_seed:}...")
    pl.seed_everything(hparams.random_seed, workers=True)
    #torch.set_float32_matmul_precision('medium') 

    logger.info(f"Write output to {hparams.output_dir}")
    os.makedirs(hparams.output_dir, exist_ok=True)
    run_id = wandb.run.id
    
    outpath = os.path.join(hparams.output_dir, f"results_{run_id}")
    os.makedirs(outpath, exist_ok=True)

    with open(f"{hparams.train_dir}/../../labels.json") as file:
        data_labels = json.load(file)

    scaled_labels = []
    if hparams.label_1:
        scaled_labels_1, min_val_l1, max_val_l1 = scale_local_labels_from_json(hparams.label_1, data_labels, hparams.type_label_scale)
        scaled_labels.append(scaled_labels_1)
    if hparams.label_2:
        scaled_labels_2, min_val_l2, max_val_l2 = scale_local_labels_from_json(hparams.label_2, data_labels, hparams.type_label_scale)
        scaled_labels.append(scaled_labels_2)
    if hparams.label_3:
        scaled_labels_3, min_val_l3, max_val_l3 = scale_local_labels_from_json(hparams.label_3, data_labels, hparams.type_label_scale)
        scaled_labels.append(scaled_labels_3)
    transform = d.EDN_Transform(scaled_labels, hparams.LM, hparams.LM_level, True, num_nearest_neighbors=40)
    
 
    logger.info(f"Creating dataloaders...")
    train_dataset = da.load_dataset(hparams.train_dir, hparams.filetype, transform=transform)
    train_dataloader = torch_geometric.loader.DataLoader(train_dataset, batch_size=hparams.batch_size, 
                                                       num_workers=hparams.num_workers, shuffle=True) 
    val_dataset = da.load_dataset(hparams.val_dir, hparams.filetype, transform=transform)
    val_dataloader = torch_geometric.loader.DataLoader(val_dataset, batch_size=hparams.batch_size, 
                                                     num_workers=hparams.num_workers, shuffle=False)
    test_dataset = da.load_dataset(hparams.test_dir, hparams.filetype, transform=transform)
    test_dataloader = torch_geometric.loader.DataLoader(test_dataset, batch_size=hparams.batch_size, 
                                                      num_workers=hparams.num_workers, shuffle=False)

    # Initialize model
    logger.info("Initializing model...")
    edn_pl = m.EDN_PL(**vars(hparams))

    wandb_logger = log.WandbLogger(save_dir=hparams.output_dir, project="edn-experiment", name="model", 
                                   log_model=True, mode="online")

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=os.path.join(hparams.output_dir, 
                                          'checkpoints'), filename='edn-{epoch:03d}-{val_loss:.4f}')
    #early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
    
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer.from_argparse_args(hparams, devices=1, num_nodes=hparams.num_nodes, logger=wandb_logger, accelerator=device, 
                                            callbacks=[checkpoint_callback])#, val_check_interval=0.1)
    # TRAINING
    logger.info(f"Running training on {hparams.train_dir:} with val {hparams.val_dir:}...")
    _ = trainer.fit(edn_pl, train_dataloader, val_dataloader)
    
    # TESTING    
    trainer.test(edn_pl, test_dataloader, verbose=True)
    labels = [label for label in [hparams.label_1, hparams.label_2, hparams.label_3] if label is not None]
    
    min_scale = [min_val_l1]
    max_scale = [max_val_l1]
    if hparams.label_2:
        min_scale.append(min_val_l2)
        max_scale.append(max_val_l2)
    if hparams.label_3:
        min_scale.append(min_val_l3)
        max_scale.append(max_val_l3)

    accumulated_df = pd.DataFrame()
    for label, min, max in zip(labels, min_scale, max_scale):
        pred, targ = scale_local_labels_back(edn_pl.predictions, label, min, max, hparams.type_label_scale)
        df = pd.DataFrame({f"pred_{label}": pred, f"targ_{label}": targ})
        plot_target_corr(df, f"targ_{label}", f"pred_{label}", label, "red", os.path.join(outpath, f"{label}_corr.png"))
        accumulated_df = pd.concat([accumulated_df, df], axis=1)
    table = wandb.Table(dataframe=accumulated_df)
    wandb.log({'predictions: ': table})
    accumulated_df.to_csv(os.path.join(outpath, "predictions.csv"))

    print("\t R \t r2 \t MAE \t var_out")
    plot_target_corr(accumulated_df, f"targ_{hparams.label_1}", f"pred_{hparams.label_1}", hparams.label_1, "red", os.path.join(outpath, f"{hparams.label_1}_corr.png"))
    R_1, r2_1, mae_1, var_out_1 = calculate_regression_metrics(accumulated_df[f"pred_{hparams.label_1}"], accumulated_df[f"targ_{hparams.label_1}"])
    print(f"{hparams.label_1}\t{R_1:.3f}\t{r2_1:.3f}\t{mae_1:.3f}\t{var_out_1:.3f}")
    wandb.log({'R_1': R_1, 'r2_1': r2_1, 'mae_1': mae_1, 'var_out_1': var_out_1})
    if hparams.label_2:
        plot_target_corr(accumulated_df, f"targ_{hparams.label_2}", f"pred_{hparams.label_2}", hparams.label_2, "blue", os.path.join(outpath, f"{hparams.label_2}_corr.png"))
        R_2, r2_2, mae_2, var_out_2 = calculate_regression_metrics(accumulated_df[f"pred_{hparams.label_2}"], accumulated_df[f"targ_{hparams.label_2}"])
        print(f"{hparams.label_2}\t{R_2:.3f}\t{r2_2:.3f}\t{mae_2:.3f}\t{var_out_2:.3f}")
        wandb.log({'R_2': R_2, 'r2_2': r2_2, 'mae_2': mae_2, 'var_out_2': var_out_1})
    if hparams.label_3:
        plot_target_corr(accumulated_df, f"targ_{hparams.label_3}", f"pred_{hparams.label_3}", hparams.label_3, "magenta", os.path.join(outpath, f"{hparams.label_3}_corr.png"))
        R_3, r2_3, mae_3, var_out_3 = calculate_regression_metrics(accumulated_df[f"pred_{hparams.label_3}"], accumulated_df[f"targ_{hparams.label_3}"])
        print(f"{hparams.label_3}\t{R_3:.3f}\t{r2_3:.3f}\t{mae_3:.3f}\t{var_out_3:.3f}")
        wandb.log({'R_3': R_3, 'r2_3': r2_3, 'mae_3': mae_3, 'var_out_3': var_out_1})
    log_plots(outpath, "_corr.png", "target-correlation")    
    wandb.finish()
    print("Done.")
    """ 
    R_1, r2_1, mae_1, var_out_1 = calculate_regression_metrics(df[f"pred_{hparams.label_1}"], df[f"targ_{hparams.label_1}"])
    R_2, r2_2, mae_2, var_out_2 = calculate_regression_metrics(df[f"pred_{hparams.label_2}"], df[f"targ_{hparams.label_2}"])
    R_3, r2_3, mae_3, var_out_3 = calculate_regression_metrics(df[f"pred_{hparams.label_3}"], df[f"targ_{hparams.label_3}"])
    
    print("\t R \t r2 \t MAE \t var_out")
    print(f"{hparams.label_1}\t{R_1:.3f}\t{r2_1:.3f}\t{mae_1:.3f}\t{var_out_1:.3f}")
    print(f"{hparams.label_2}\t{R_2:.3f}\t{r2_2:.3f}\t{mae_2:.3f}\t{var_out_2:.3f}")
    print(f"{hparams.label_3}\t{R_3:.3f}\t{r2_3:.3f}\t{mae_3:.3f}\t{var_out_3:.3f}")
   
    wandb.log({'R_1': R_1, 'r2_1': r2_1, 'mae_1': mae_1, 'var_out_1': var_out_1, 
               'R_2': R_2, 'r2_2': r2_2, 'mae_2': mae_2, 'var_out_2': var_out_2,
               'R_3': R_3, 'r2_3': r2_3, 'mae_3': mae_3, 'var_out_3': var_out_3})

    wandb.finish()
    print("Done.")
    """ 
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s %(process)d: ' + '%(message)s',
                        level=logging.INFO)
    main()
