import argparse as ap
import collections as col
from functools import partial
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import torch_geometric as tg
import torch.nn.functional as F
from e3nn.kernel import Kernel
from e3nn.linear import Linear
from e3nn import o3
from e3nn.non_linearities.norm import Norm
from e3nn.non_linearities.nonlin import Nonlinearity
from e3nn.point.message_passing import Convolution
from e3nn.radial import GaussianRadialModel
import esm
import numpy as np
import pandas as pd
#from Bio import SeqIO, PDB
from shared_utils import scale_features
import sys
import os
import json
import re

import warnings
from Bio.PDB.PDBParser import PDBParser, PDBConstructionWarning
# Ignore PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
#from Bio.PDB import PDBParser
PROJECT_PATH = os.environ.get('PROJECT_PATH')
sys.path.append(f"{PROJECT_PATH}/geometricDL/edn/shared_utils")
from shared_utils import get_residue_numbers
from shared_utils import one_hot_encode_feature_list
from shared_utils import runtime
from helpers import load_from_json, load_from_hdf5
from multilabel_utils import Min_Max_scaler
import time

loss_reduction="mean"
loss_delta=1.00

class EDN_Model(nn.Module):

    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.activations = []
        self.activations_ten = torch.empty(0, dtype=torch.float32)
        self.features_json = load_from_json(f"{hparams.train_dir}/../../features.json")
        self.esm_h5 = load_from_hdf5(f"{hparams.train_dir}/../../esm.h5")

        lmax = hparams.lmax
        LM = hparams.LM    
        layer_size = hparams.layer_size
        hparams.LM_level = getattr(hparams, 'LM_level', 0)
         
        # Define the input and output representations
        Rs0_mult = 4
        if hparams.LM_level == 1 or hparams.LM_level == 3:
            if LM == 8:
                Rs0_mult += 320
            elif LM == 35:
                Rs0_mult += 480 
            elif LM == 150:
                Rs0_mult += 640
            elif LM == 650:
                Rs0_mult += 1280

        Rs0 = [(Rs0_mult, 0)]
        Rs1 = [(layer_size, 0)]
        Rs20 = [(layer_size, 0)]
        Rs21 = [(layer_size, 1)]
        Rs22 = [(layer_size, 2)]
        Rs23 = [(layer_size, 3)] if lmax == 3 else None
        Rs3 = [(layer_size, 0), (layer_size, 1), (layer_size, 2), (layer_size, 3)] if lmax == 3 else [(layer_size, 0), (layer_size, 1), (layer_size, 2)]
        Rs30 = [(layer_size, 0)]
        Rs31 = [(layer_size, 1)]
        Rs32 = [(layer_size, 2)]
        Rs33 = [(layer_size, 3)] if lmax == 3 else None

        # To account for multiple output paths of conv.
        Rs30_exp = [(4 * layer_size, 0)] if lmax == 3 else [(3 * layer_size, 0)] 
        Rs31_exp = [(9 * layer_size, 1)] if lmax == 3 else [(6 * layer_size, 1)]
        Rs32_exp = [(11 * layer_size, 2)] if lmax == 3 else [(6 * layer_size, 2)] 
        Rs33_exp = [(10 * layer_size, 3)] if lmax == 3 else None

        relu = torch.nn.ReLU()
        # Radial model:  R+ -> R^d
        RadialModel_1 = partial(GaussianRadialModel, max_radius=10.0, number_of_basis=20, h=12, L=1, act=relu)
        RadialModel_2 = partial(GaussianRadialModel, max_radius=20.0, number_of_basis=40, h=12, L=1, act=relu)

        ssp = ShiftedSoftplus()
        self.elu = torch.nn.ELU()

        # kernel: composed on a radial part that contains the learned
        # parameters and an angular part given by the spherical hamonics and
        # the Clebsch-Gordan coefficients
        selection_rule = partial(o3.selection_rule_in_out_sh, lmax=lmax)
        K1 = partial(Kernel, RadialModel=RadialModel_1, selection_rule=selection_rule)

        ### Layer 1
        self.lin1 = Linear(Rs0, Rs1)

        self.conv10 = Convolution(K1(Rs1, Rs20))
        self.conv11 = Convolution(K1(Rs1, Rs21))
        self.conv12 = Convolution(K1(Rs1, Rs22))
        self.conv13 = Convolution(K1(Rs1, Rs23)) if lmax == 3 else None

        self.norm = Norm()

        self.lin20 = Linear(Rs20, Rs20)
        self.lin21 = Linear(Rs21, Rs21)
        self.lin22 = Linear(Rs22, Rs22)
        self.lin23 = Linear(Rs23, Rs23) if lmax == 3 else None

        self.nonlin10 = Nonlinearity(Rs20, act=ssp)
        self.nonlin11 = Nonlinearity(Rs21, act=ssp)
        self.nonlin12 = Nonlinearity(Rs22, act=ssp)
        self.nonlin13 = Nonlinearity(Rs23, act=ssp) if lmax == 3 else None

        ### Layer 2
        self.lin30 = Linear(Rs20, Rs30)
        self.lin31 = Linear(Rs21, Rs31)
        self.lin32 = Linear(Rs22, Rs32)
        self.lin33 = Linear(Rs23, Rs33) if lmax == 3 else None


        def filterfn_def(x, f):
            return x == f

        self.conv2 = torch.nn.ModuleDict()
        for i in range(lmax + 1):
            for f in range(lmax + 1):
                for o in range(abs(f - i), min(i + f + 1, lmax + 1)):
                    filterfn = partial(filterfn_def, f=f)
                    selection_rule = partial(o3.selection_rule, lmax=lmax, lfilter=filterfn)
                    K = partial(Kernel, RadialModel=RadialModel_2, selection_rule=selection_rule)
                    self.conv2[str((i, f, o))] = Convolution(K([Rs3[i]], [Rs3[o]]))
                   
              
        reslevel_feature_dict = {'none': 0, 'hbonds': 1, 'sasa': 1, 'energies': 1, 'sap-score': 1, 'secondary_structure': 3, 'sequence': 20, 'esm2_t6_8M_UR50D': 320}
        lin40_dim = 120
        
        #print(f"lin type(hparams.reslevel_features) {type(hparams.reslevel_features)}")
        #print(f"lin hparams.reslevel_features {hparams.reslevel_features}")
        for reslevel_feature in hparams.reslevel_features:
            lin40_dim += reslevel_feature_dict[reslevel_feature]
        self.lin40 = Linear([(lin40_dim, 0)], Rs30)

        """
        if hparams.LM_level == 2 or hparams.LM_level == 3:
            LM_size_dict = { 0 : 120, 8 : 440, 35 : 600, 150 : 760, 650 : 1400}
            self.lin40 = Linear([(LM_size_dict[hparams.LM], 0)], Rs30)
        else:
            self.lin40 = Linear(Rs30_exp, Rs30)
        """
        
        #self.lin41 = Linear([(1040, 1)], Rs31)
        self.lin41 = Linear(Rs31_exp, Rs31)
        self.lin42 = Linear(Rs32_exp, Rs32)
        self.lin43 = Linear(Rs33_exp, Rs33) if lmax == 3 else None

        self.nonlin20 = Nonlinearity(Rs30, act=ssp)
        self.nonlin21 = Nonlinearity(Rs31, act=ssp)
        self.nonlin22 = Nonlinearity(Rs32, act=ssp)
        self.nonlin23 = Nonlinearity(Rs33, act=ssp) if lmax == 3 else None
        
        dim_out_dense1 = int(250 * hparams.dense_layer_scale)
        dim_out_dense2 = int(150 * hparams.dense_layer_scale)

        ### Final dense layers
        self.dense1_0 = torch.nn.Linear(layer_size, dim_out_dense1, bias=True)
        self.dense2_0 = torch.nn.Linear(dim_out_dense1, dim_out_dense2, bias=True)
        self.dense3_0 = torch.nn.Linear(dim_out_dense2, 1, bias=True)
        if hparams.label_2:
            self.dense1_1 = torch.nn.Linear(layer_size, dim_out_dense1, bias=True)
            self.dense2_1 = torch.nn.Linear(dim_out_dense1, dim_out_dense2, bias=True)
            self.dense3_1 = torch.nn.Linear(dim_out_dense2, 1, bias=True)
        if hparams.label_3:
            self.dense1_2 = torch.nn.Linear(layer_size, dim_out_dense1, bias=True)
            self.dense2_2 = torch.nn.Linear(dim_out_dense1, dim_out_dense2, bias=True)
            self.dense3_2 = torch.nn.Linear(dim_out_dense2, 1, bias=True)
          
        
    def forward(self, data, hparams):
        lmax = hparams.lmax
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        ### Layer 1
        out = self.lin1(data.x)

        out0 = self.conv10(out, edge_index, edge_attr)
        out1 = self.conv11(out, edge_index, edge_attr)
        out2 = self.conv12(out, edge_index, edge_attr)
        out3 = self.conv13(out, edge_index, edge_attr) if lmax == 3 else None

        out0 = self.norm(out0)
        out1 = self.norm(out1)
        out2 = self.norm(out2)
        out3 = self.norm(out3) if lmax == 3 else None

        out0 = self.lin20(out0)
        out1 = self.lin21(out1)
        out2 = self.lin22(out2)
        out3 = self.lin23(out3) if lmax == 3 else None

        out0 = self.nonlin10(out0)
        out1 = self.nonlin11(out1)
        out2 = self.nonlin12(out2)
        out3 = self.nonlin13(out3) if lmax == 3 else None

        ### Layer 2
        out0 = self.lin30(out0)
        out1 = self.lin31(out1)
        out2 = self.lin32(out2)
        out3 = self.lin33(out3) if lmax == 3 else None

        ins = {0: out0, 1: out1, 2: out2, 3: out3} if lmax == 3 else {0: out0, 1: out1, 2: out2}
        tmp = col.defaultdict(list)
        for i in range(lmax + 1):
            for f in range(lmax + 1):
                for o in range(abs(f - i), min(i + f + 1, lmax + 1)):
                    curr = self.conv2[str((i, f, o))](ins[i], edge_index, edge_attr)
                    tmp[o].append(curr)

        out0 = torch.cat(tmp[0], axis=1)
        out1 = torch.cat(tmp[1], axis=1)
        out2 = torch.cat(tmp[2], axis=1)
        out3 = torch.cat(tmp[3], axis=1) if lmax == 3 else None

        # all atoms -> CAs
        CA_sel = torch.nonzero(data['select_ca'].squeeze(dim=0)).squeeze(dim=1)
        out0 = torch.squeeze(out0[CA_sel])
        out1 = torch.squeeze(out1[CA_sel])
        out2 = torch.squeeze(out2[CA_sel])
        out3 = torch.squeeze(out3[CA_sel]) if lmax == 3 else None

        # add LM to residue level features
        if hparams.LM_level == 2 or hparams.LM_level ==3:
            # add LM embeddings to features
            stacked_LM_embeddings = get_sequence_embeddings_batch(hparams, data)
            stacked_LM_embeddings = scale_features(stacked_LM_embeddings)
            stacked_LM_embeddings = stacked_LM_embeddings.to(out0.device)
            out0 = torch.cat((out0, stacked_LM_embeddings), dim=1)
            #out1 = torch.cat((out1, stacked_tensor), dim=1)
            #out2 = torch.cat((out2, stacked_tensor), dim=1)
            #out3 = torch.cat((out3, stacked_tensor), dim=1) if lmax == 3 else None
            
        # Also need to update the nodes/edges indexing
        edge_index, edge_attr = tg.utils.subgraph(CA_sel, edge_index, edge_attr, relabel_nodes=True)
        batch = torch.squeeze(data.batch[CA_sel])
        out0 = self.norm(out0)
     
        for feature in hparams.reslevel_features:
            feature_list_stacked = []
            one_hot_dim = {"secondary_structure": 3, "sequence": 20}
            #print(f"feature {feature}")
            if feature == "none":
                continue
        
            if feature == "sequence" or feature == "secondary_structure":
                feature_array_stacked = np.empty((0, one_hot_dim[feature]), dtype=float)
            # put batch of features together
            for path_pdb in data.file_path:
                #print(f"path_pdb {path_pdb}")
                fname_pdb = os.path.basename(path_pdb)
                pattern = re.compile(r'(m[12345]).*?(\.pdb)')
                fname_pdb_json = re.sub(pattern, r'\1\2', fname_pdb)
                #pattern = r"_(\d+)_(HH|EH|HE|EE)\.pdb$"  # relax i guess
                #fname_pdb_json = fname_pdb[:6] + ".pdb"

                
                if not feature == "esm2_t6_8M_UR50D":
                    feature_list = self.features_json[fname_pdb_json][feature]
                elif feature == "esm2_t6_8M_UR50D":
                    feature_list = self.esm_h5[fname_pdb_json]
                    feature_list = feature_list.tolist()[0] # convert np.ndarray to list of lists
                # get list of residue numbers
                resnum_list = get_residue_numbers(path_pdb)            
                feature_list_filtered = [feature_list[i - 1] for i in resnum_list]
                #feature_list_filtered = [[item] for item in feature_list_filtered]
                
                if feature == "secondary_structure" or feature == "sequence":
                    feature_list_filtered = one_hot_encode_feature_list(feature, feature_list_filtered)
                    feature_array_stacked = np.concatenate((feature_array_stacked, feature_list_filtered), axis=0)
                else:
                    feature_list_stacked.extend(feature_list_filtered)

            # add feature batch to out0
            if feature == "secondary_structure" or feature == "sequence":
                feature_array_stacked = torch.tensor(feature_array_stacked)
                feature_array_stacked = feature_array_stacked.float()
                out0 = torch.cat((out0, feature_array_stacked.to(out0.device)), dim=1)
            else:
                feature_list_stacked = torch.tensor(feature_list_stacked).to(out0.device)
                if feature == "sasa" or feature == "energies" or feature == "sap-score" or feature == "hbonds":
                    feature_list_stacked = feature_list_stacked.to(out0.device).view(feature_list_stacked.shape[0], 1)
                if feature_list_stacked.dim() >= 3:
                    feature_list_stacked = feature_list_stacked.squeeze(1)
                if hparams.stand_features == 1:
                    feature_list_stacked = (feature_list_stacked - torch.mean(feature_list_stacked)) / torch.std(feature_list_stacked)
                out0 = torch.cat((out0, feature_list_stacked), dim=1)

        # concatenate scaled features scale to what?
        print(f"out0 {out0.shape}") 

        #out0 = self.norm(out0)
        out1 = self.norm(out1)
        out2 = self.norm(out2)
        out3 = self.norm(out3) if lmax == 3 else None
        
        out0 = self.lin40(out0)
        out1 = self.lin41(out1)
        out2 = self.lin42(out2)
        out3 = self.lin43(out3) if lmax == 3 else None

        out0_conv = self.nonlin20(out0)
        out1 = self.nonlin21(out1)
        out2 = self.nonlin22(out2)
        out3 = self.nonlin23(out3) if lmax == 3 else None

        # Per-channel mean.
        #out = scatter_mean(out0, batch, dim=0)
        # head for label 1
        out0 = self.dense1_0(out0_conv)
        out0 = self.elu(out0)
        out0 = self.dense2_0(out0)
        out0 = self.elu(out0)
        out0 = self.dense3_0(out0)
        out0 = torch.squeeze(out0, axis=1)
        
        if hparams.label_2:
            out1 = self.dense1_1(out0_conv)
            out1 = self.elu(out1)
            out1 = self.dense2_1(out1)
            out1 = self.elu(out1)
            out1 = self.dense3_1(out1)
            out1 = torch.squeeze(out1, axis=1)
        
        if hparams.label_3:
            out2 = self.dense1_2(out0_conv)
            out2 = self.elu(out2)
            out2 = self.dense2_2(out2)
            out2 = self.elu(out2)
            out2 = self.dense3_2(out2)
            out2 = torch.squeeze(out2, axis=1)
        
        if hparams.label_1 and not hparams.label_2 and not hparams.label_3:
            return out0
        if hparams.label_1 and hparams.label_2 and not hparams.label_3:
            return out0, out1
        if hparams.label_1 and hparams.label_2 and hparams.label_3:
            return out0, out1, out2


class EDN_PL(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ap.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

    def __init__(self, learning_rate=1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.predictions = col.defaultdict(lambda: col.defaultdict(lambda: {'pred': [], 'target': [], 'delta': []}))
        self.activations = []
        self.val_losses = []        
        self.net = EDN_Model(self.hparams, **self.hparams)

        
    def training_step(self, batch, _):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        y_hat = self(batch.to(device))
        
        loss1, loss2 = 0.0, 0.0 
        loss0 = torch.nn.functional.huber_loss(y_hat[0], batch.label_1.float())
        if self.hparams.label_2:
            loss1 = torch.nn.functional.huber_loss(y_hat[1], batch.label_2.float())
        if self.hparams.label_3:
            loss2 = torch.nn.functional.huber_loss(y_hat[2], batch.label_3.float())

        loss_total = loss0 + loss1 + loss2
        
        self.val_losses.append(loss_total)
        self.log('loss', loss_total, batch_size = self.hparams.batch_size)
        return {'loss': loss_total}

    
    def validation_step(self, batch, _):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        y_hat = self(batch.to(device))
        loss1, loss2 = 0.0, 0.0 
        loss0 = torch.nn.functional.huber_loss(y_hat[0], batch.label_1.float())
        if self.hparams.label_2:
            loss1 = torch.nn.functional.huber_loss(y_hat[1], batch.label_2.float())
        if self.hparams.label_3:
            loss2 = torch.nn.functional.huber_loss(y_hat[2], batch.label_3.float())

        loss_total = loss0 + loss1 + loss2
       
        self.val_losses.append(loss_total)
        self.log('val_loss', loss_total, batch_size = self.hparams.batch_size)
        return {'val_loss': loss_total}

        #def validation_epoch_end(self, outputs):
        #val_loss = torch.stack([x['val_loss_step'] for x in outputs]).mean()
        #self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        
    def on_epoch_end(self):
        val_loss_epoch_avg = torch.stack(self.val_losses).sum() / len(self.val_losses)
        self.log('val_loss_epoch', val_loss_epoch_avg)
        #print(self.val_losses)
        #print(val_loss_epoch_avg)
        
 
    def test_step(self, batch, _):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        y_hat = self(batch.to(device))
        
        loss1, loss2 = 0.0, 0.0 
        loss0 = torch.nn.functional.huber_loss(y_hat[0], batch.label_1.float())
        if self.hparams.label_2:
            loss1 = torch.nn.functional.huber_loss(y_hat[1], batch.label_2.float())
        if self.hparams.label_3:
            loss2 = torch.nn.functional.huber_loss(y_hat[2], batch.label_3.float())

        loss_total = loss0 + loss1 + loss2
        
        if self.hparams.label_1 and not self.hparams.label_2 and not self.hparams.label_3:
            pred_label_1_float = [float(value) for value in y_hat]
            targ_label_1_float = [float(value) for value in batch.label_1]
        if self.hparams.label_1 and self.hparams.label_2:
            pred_label_1_float = [float(value) for value in y_hat[0]]
            targ_label_1_float = [float(value) for value in batch.label_1]
            pred_label_2_float = [float(value) for value in y_hat[1]]
            targ_label_2_float = [float(value) for value in batch.label_2]
        if self.hparams.label_3:
            pred_label_3_float = [float(value) for value in y_hat[2]]
            targ_label_3_float = [float(value) for value in batch.label_3]
        
        batch_size = len(batch.id)
        part_length = len(pred_label_1_float) // batch_size

        for i in range(batch_size):
            # Calculate start and end index for slicing
            start_idx = i * part_length
            end_idx = (i + 1) * part_length
            # Append the sliced values to the corresponding id as key in the dictionary
            current_id = batch.id[i]
           
            # Slice the values list for the current part
            if self.hparams.label_1:
                pred_label_1 = pred_label_1_float[start_idx:end_idx]
                targ_label_1 = targ_label_1_float[start_idx:end_idx]
                delta_label_1 = [abs(a - b) for a, b in zip(pred_label_1, targ_label_1)]
                self.predictions[self.hparams.label_1][current_id]['pred'].extend(pred_label_1)
                self.predictions[self.hparams.label_1][current_id]['target'].extend(targ_label_1)
                self.predictions[self.hparams.label_1][current_id]['delta'].extend(delta_label_1)
           
            if self.hparams.label_2:
                pred_label_2 = pred_label_2_float[start_idx:end_idx]
                targ_label_2 = targ_label_2_float[start_idx:end_idx]
                delta_label_2 = [abs(a - b) for a, b in zip(pred_label_2, targ_label_2)]
                self.predictions[self.hparams.label_2][current_id]['pred'].extend(pred_label_2)
                self.predictions[self.hparams.label_2][current_id]['target'].extend(targ_label_2)
                self.predictions[self.hparams.label_2][current_id]['delta'].extend(delta_label_2)
                
            if self.hparams.label_3:
                pred_label_3 = pred_label_3_float[start_idx:end_idx]
                targ_label_3 = targ_label_3_float[start_idx:end_idx]
                delta_label_3 = [abs(a - b) for a, b in zip(pred_label_3, targ_label_3)]
                self.predictions[self.hparams.label_3][current_id]['pred'].extend(pred_label_3)
                self.predictions[self.hparams.label_3][current_id]['target'].extend(targ_label_3)
                self.predictions[self.hparams.label_3][current_id]['delta'].extend(delta_label_3)
        return {'test_loss': loss_total}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def forward(self, data):
        return self.net(data, self.hparams)


class ShiftedSoftplus:
    def __init__(self):
        self.shift = torch.nn.functional.softplus(torch.zeros(())).item()

    def __call__(self, x):
        return torch.nn.functional.softplus(x).sub(self.shift)

