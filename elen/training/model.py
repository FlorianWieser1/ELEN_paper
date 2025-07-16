import sys
import numpy as np
import argparse as ap
import collections as col
from functools import partial
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import torch_geometric as tg
from e3nn.kernel import Kernel
from e3nn.linear import Linear
from e3nn import o3
from e3nn.non_linearities.norm import Norm
from e3nn.non_linearities.nonlin import Nonlinearity
from e3nn.point.message_passing import Convolution
from e3nn.radial import GaussianRadialModel
from elen.training.utils_model import calculate_multilabel_loss
from elen.training.utils_model import split_activations_batch_into_dicts
from elen.training.utils_model import get_dimension
from elen.training.utils_model import Swish, Mish
import torch.nn.functional as F
import scipy.stats

class EDN_Model(nn.Module):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.use_labels = getattr(hparams, 'use_labels', False)
        self.min_scale = None
        self.max_scale = None
        self.activations_dict = {}
        # TODO testing dropout
        #self.dropout1 = Dropout(p=0.5, inplace=True)
        
        lmax = hparams.lmax
        layer_size = hparams.layer_size
        feature_dict = {'none': 0, 'hbonds': 1, 'sasa': 1, 'energies': 1, 'sap-score': 1, 
                        'secondary_structure': 3, 'sequence': 20, 'esm2_8M': 320, 'esm2_35M': 480, 
                        'esm2_650M': 1280, 'saprot_35M': 480, 'saprot_650M': 1280}
       
        # Define the input and output representations
        Rs0_mult = get_dimension(hparams, feature_dict)
        
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
        Rs30_exp_dim = 4 * layer_size if lmax == 3 else 3 * layer_size
        
        for residue_feature in hparams.reslevel_features:
            Rs30_exp_dim += feature_dict[residue_feature]
       
        Rs30_exp = [(Rs30_exp_dim, 0)]
        Rs31_exp = [(9 * layer_size, 1)] if lmax == 3 else [(6 * layer_size, 1)]
        Rs32_exp = [(11 * layer_size, 2)] if lmax == 3 else [(6 * layer_size, 2)] 
        Rs33_exp = [(10 * layer_size, 3)] if lmax == 3 else None
        
        dict_activations = {
            'relu' : torch.nn.ReLU(),
            'leaky_relu' : torch.nn.LeakyReLU(),
            'elu' : torch.nn.ELU(),
            'softmax' : torch.nn.Softmax(),
            'swish' : Swish(),
            'mish' : Mish(),
        }

        # hotfix to make inference script work with new sweep parameters        
        if not hasattr(hparams, "activation"):
            setattr(hparams, "activation", "relu")
        if not hasattr(hparams, "optimizer"):
            setattr(hparams, "optimizer", "adam")
        if not hasattr(hparams, "weight_decay"):
            setattr(hparams, "weight_decay", 0.0)
            
        relu = dict_activations[hparams.activation]
        
        # Radial model:  R+ -> R^d
        # standard GRM_r1 = 10, nr1 = 20
        # standard GRM_r2 = 20, nr2 = 40
        RadialModel_1 = partial(GaussianRadialModel, max_radius=hparams.GRM_r1, number_of_basis=hparams.GRM_nr1, h=12, L=1, act=relu)
        RadialModel_2 = partial(GaussianRadialModel, max_radius=hparams.GRM_r2, number_of_basis=hparams.GRM_nr2, h=12, L=1, act=relu)

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
        
        #if self.hparams.skip_connections == True:
        #    Rs20_skip = [(Rs20[0][0] * 2, 0)]
        #    Rs20 = Rs20_skip
            
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
                   
        # increase layer size based on active residue level features
        self.lin40 = Linear(Rs30_exp, Rs30)
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
        self.dense1 = torch.nn.Linear(layer_size, dim_out_dense1, bias=True)
        self.dense2 = torch.nn.Linear(dim_out_dense1, dim_out_dense2, bias=True)
        if hparams.label_1 and not hparams.label_2 and not hparams.label_3:
            self.dense3 = torch.nn.Linear(dim_out_dense2, 1, bias=True)
        if hparams.label_1 and hparams.label_2 and not hparams.label_3:
            self.dense3 = torch.nn.Linear(dim_out_dense2, 2, bias=True)
        if hparams.label_1 and hparams.label_2 and hparams.label_3:
            self.dense3 = torch.nn.Linear(dim_out_dense2, 3, bias=True)


    def forward(self, data, hparams, current_phase):
        lmax = hparams.lmax
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        ### Layer 1
        out = self.lin1(data.x.float())
        
        if hparams.skip_connections == True:
            out0 = self.conv10(out, edge_index, edge_attr) + out
            out1 = self.conv10(out, edge_index, edge_attr) + out
            out2 = self.conv10(out, edge_index, edge_attr) + out
        else:
            out0 = self.conv10(out, edge_index, edge_attr)
            out1 = self.conv10(out, edge_index, edge_attr)
            out2 = self.conv10(out, edge_index, edge_attr)
            
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
        #print(out0.shape)

        out0 = self.nonlin10(out0)
        out1 = self.nonlin11(out1)
        out2 = self.nonlin12(out2)
        out3 = self.nonlin13(out3) if lmax == 3 else None
        #print(out0.shape)
        
        ### Layer 2
        out0 = self.lin30(out0)
        out1 = self.lin31(out1)
        out2 = self.lin32(out2)
        out3 = self.lin33(out3) if lmax == 3 else None
        
        skip_out0 = out0
        skip_out1 = out1
        skip_out2 = out2
       
        ins = {0: out0, 1: out1, 2: out2, 3: out3} if lmax == 3 else {0: out0, 1: out1, 2: out2}
        tmp = col.defaultdict(list)
        for i in range(lmax + 1):
            for f in range(lmax + 1):
                for o in range(abs(f - i), min(i + f + 1, lmax + 1)):
                    curr = self.conv2[str((i, f, o))](ins[i], edge_index, edge_attr)
                    tmp[o].append(curr)
            
        self.adjust_dim_out0 = nn.Linear(40, 120)            
        self.adjust_dim_out1 = nn.Linear(120, 720)            
        self.adjust_dim_out2 = nn.Linear(200, 1200)            

        if hparams.skip_connections == True:
            skip_out0 = self.adjust_dim_out0(skip_out0)
            skip_out1 = self.adjust_dim_out1(skip_out1)
            skip_out2 = self.adjust_dim_out2(skip_out2)

            out0 = torch.cat(tmp[0], axis=1) + skip_out0
            out1 = torch.cat(tmp[1], axis=1) + skip_out1
            out2 = torch.cat(tmp[2], axis=1) + skip_out2
        else:
            out0 = torch.cat(tmp[0], axis=1)
            out1 = torch.cat(tmp[1], axis=1)
            out2 = torch.cat(tmp[2], axis=1)
           
        #out1 = torch.cat(tmp[1], axis=1)
        #out2 = torch.cat(tmp[2], axis=1)
        out3 = torch.cat(tmp[3], axis=1) if lmax == 3 else None
        
        
        #print("0000000000000000000000000000000")
        # all atoms -> CAs
        CA_sel = torch.nonzero(data['select_ca'].squeeze(dim=0)).squeeze(dim=1)
        out0 = torch.squeeze(out0[CA_sel])
        out1 = torch.squeeze(out1[CA_sel])
        out2 = torch.squeeze(out2[CA_sel])
        out3 = torch.squeeze(out3[CA_sel]) if lmax == 3 else None

        # Also need to update the nodes/edges indexing
        edge_index, edge_attr = tg.utils.subgraph(CA_sel, edge_index, edge_attr, relabel_nodes=True)
        batch = torch.squeeze(data.batch[CA_sel])
        
        if hparams.batch_norm == "before" or hparams.batch_norm == "both":
            out0 = self.norm(out0)
       
        # add residue level features
        if hparams.reslevel_features[0] != "none":
            reslevel_features_stacked = np.vstack(data.reslevel_features)
            out0 = torch.cat((out0, torch.tensor(reslevel_features_stacked).float().to(out0.device)), dim=1)
         
        if hparams.batch_norm == "after" or hparams.batch_norm == "both":
            out0 = self.norm(out0)
            
        out1 = self.norm(out1)
        out2 = self.norm(out2)
        out3 = self.norm(out3) if lmax == 3 else None

        out0 = self.lin40(out0)
        out1 = self.lin41(out1)
        out2 = self.lin42(out2)
        out3 = self.lin43(out3) if lmax == 3 else None

        out0 = self.nonlin20(out0)
        out1 = self.nonlin21(out1)
        out2 = self.nonlin22(out2)
        out3 = self.nonlin23(out3) if lmax == 3 else None
        #out0 = self.dropout1(out0)
       
        #test_input = torch.randn(10, 10)
        #output = self.dropout1(test_input)
        #print("Fraction of zeros:", (output == 0).float().mean().item())
        #print(output)
                 
        out = self.dense1(out0)
        out = self.elu(out)
        out = self.dense2(out)
        # collect activation values for UMAP feature depiction
        if current_phase == "test":
            try:
                activations = out.detach().cpu().numpy()
                num_activations = activations.shape[0]
                num_ids = len(data.id)
    
                if num_activations % num_ids != 0:
                    raise ValueError(f"[ERROR] Mismatch: {num_activations} activations vs {num_ids} IDs. "
                                     "Check your input data and CA selection.")
    
                self.activations_dict.update(split_activations_batch_into_dicts(activations, data.id))
            except Exception as e:
                print(f"[WARNING] Skipping activation saving due to error: {e}")

        out = self.elu(out)
        out = self.dense3(out)
        
        out = torch.squeeze(out, axis=1)
        if hparams.label_1 and not hparams.label_2 and not hparams.label_3:
            return out
        if hparams.label_1 and hparams.label_2 and not hparams.label_3:
            return out[:, 0], out[:, 1]
        if hparams.label_1 and hparams.label_2 and hparams.label_3:
            return out[:, 0], out[:, 1], out[:, 2]



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
        self.final_activations_dict = {}
        self.val_predictions = []
        self.val_labels = []
        #self.val_label2 = []
        #self.val_label3 = []
        self.current_phase = None
        self.net = EDN_Model(self.hparams, **self.hparams)


    def training_step(self, batch, _):
        self.current_phase = "train"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        y_hat = self(batch.to(device))
        loss = calculate_multilabel_loss(self, batch, y_hat) 
        self.log('loss', loss, batch_size = self.hparams.batch_size, sync_dist=True)
        return {'loss': loss}
    
    
    def validation_step(self, batch, _):
        self.current_phase = "val"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        y_hat = self(batch.to(device))

        loss = calculate_multilabel_loss(self, batch, y_hat) 
        #self.val_losses.append(loss)
        self.log('val_loss', loss, batch_size = self.hparams.batch_size, sync_dist=True)
       
        self.val_labels.extend(batch.label_1.cpu().numpy())
        self.val_labels.extend(batch.label_2.cpu().numpy())
        self.val_labels.extend(batch.label_3.cpu().numpy())
        
        for out_label in y_hat:
            self.val_predictions.extend(out_label.cpu().numpy())
        return {'val_loss': loss}


    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True)
        # Compute Spearman and Pearson correlations
        pearson_corr, _ = scipy.stats.pearsonr(self.val_predictions, self.val_labels)
        spearman_corr, _ = scipy.stats.spearmanr(self.val_predictions, self.val_labels)
       
        # Log metrics
        self.log('val_pearson', pearson_corr, prog_bar=True)
        self.log('val_spearman', spearman_corr, prog_bar=True)
        
            
    def test_step(self, batch, _):
        self.current_phase = "test"
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        y_hat = self(batch.to(device)) # call to forward pass
        
        self.final_activations_dict.update(self.net.activations_dict)
       
        loss = 0
        labels = [label for label in [self.hparams.label_1, self.hparams.label_2, self.hparams.label_3] if label is not None]
        for idx, (label, label_nr) in enumerate(zip(labels, ['label_1', 'label_2', 'label_3'])): 
            pred_float = [float(value) for value in y_hat[idx]]
            targ_float = [float(value) for value in getattr(batch, label_nr)]
        
            batch_size = len(batch.id)
            part_length = len(pred_float) // batch_size

            for i in range(batch_size):
                # Calculate start and end index for slicing
                start_idx = i * part_length
                end_idx = (i + 1) * part_length
                # Append the sliced values to the corresponding id as key in the dictionary
                current_id = batch.id[i]
                # Slice the values list for the current part
                pred = pred_float[start_idx:end_idx]
                targ = targ_float[start_idx:end_idx]
                delta = [abs(a - b) for a, b in zip(pred, targ)]
                self.predictions[label][current_id]['pred'].extend(pred)
                self.predictions[label][current_id]['target'].extend(targ)
                self.predictions[label][current_id]['delta'].extend(delta)
                
        return {'test_loss': loss}
    
    def configure_optimizers(self):
        dict_optimizers = {
            'adam' : torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay),
            'adamw' : torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay),
            'rmsprop' : torch.optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay),
            'nadam' : torch.optim.NAdam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay),
        }  
        optimizer = dict_optimizers[self.hparams.optimizer]
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        # This method is called when saving a checkpoint
        checkpoint['min_scale'] = self.min_scale
        checkpoint['max_scale'] = self.max_scale
        
    def on_load_checkpoint(self, checkpoint):
        # This method is called when loading a checkpoint
        self.min_scale = checkpoint['min_scale']
        self.max_scale = checkpoint['max_scale']
         
    def forward(self, data):
        return self.net(data, self.hparams, self.current_phase)

class ShiftedSoftplus:
    def __init__(self):
        self.shift = torch.nn.functional.softplus(torch.zeros(())).item()

    def __call__(self, x):
        return torch.nn.functional.softplus(x).sub(self.shift)
