#currently an empty file for model related utils
#more to come
import torch
import time
from becderqspr.e3nn import Network
from e3nn.io import CartesianTensor
import cmcrameri.cm as cm
import torch.nn as nn
import pandas as pd
import numpy as np

class E3NN(Network):
    def __init__(self, in_dim, emb_dim, num_layers, max_radius, num_neighbors, lmax=3, radial_layers=1, radial_neurons=100, loss_type='graph'):
         
        kwargs = {'reduce_output': False,
                  'irreps_in': str(emb_dim)+"x0e",
                  'irreps_out': str(CartesianTensor("ijk=ikj")),
                  #'irreps_out': str(9) + "x0e",
                  'irreps_node_attr': str(emb_dim)+"x0e",
                  'layers': num_layers,
                  'mul': 32,
                  'lmax': lmax,
                  'max_radius': max_radius,
                  'number_of_basis': 10,
                  'radial_layers': radial_layers,
                  'radial_neurons': radial_neurons,
                  'num_neighbors': num_neighbors
                 }
        super().__init__(**kwargs)
        
        # definitions
        self.cmap = cm.lipari
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.max_radius = max_radius
        self.num_neighbors = num_neighbors
        self.loss_type = loss_type
        self.model_name = 'becder_e' + str(emb_dim) + '_l' + str(num_layers)
        
        # embedding
        self.emb_x = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU()
        )
        
        self.emb_z = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.Tanh()
        )
    
    def transform(self, data):
        data['x'] = self.emb_x(data['x_in'])
        data['z'] = self.emb_z(data['z_in'])
        return super().forward(data)[0]
    
    def forward(self, data):
        x = self.transform(data)
        #batching
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)
        #y = torch_scatter.scatter_mean(x, batch, dim=0)
        #print(x)
        #print(x.shape)
        #print(torch_scatter.scatter_mean(x, batch, dim=0).shape)
        #x -= torch.mean(x, axis=0)
        #print(x.shape)
        #x -= torch.mean(x, axis=0)
        #x=x.reshape(batch_size,-1,9)
        return x
    
    
    def count_parameters(self): 
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

    def loss_b_graph(self, b_pred, b_true):
        b_pred=b_pred.reshape(b_true.shape)
        # print(b_pred.shape)
        # print(b_true.shape)
        #b_pred.reshape(b_true.shape)
        #return nn.MSELoss()(b_pred[-1,-4:], b_true[:,-4:])
        return nn.MSELoss()(b_pred, b_true)

    
    def loss_b_node(self, b_pred, b_true):
        b_true=b_true.reshape(b_pred.shape)
<<<<<<< HEAD
        b_pred_flat=b_pred.view([-1,*list(b_pred.shape[-2:])])
        b_true_flat=b_true.view([-1,*list(b_true.shape[-2:])])
=======
        b_pred_flat=b_pred.view([-1,*list(b_pred.shape[-1:])])
        b_true_flat=b_true.view([-1,*list(b_true.shape[-1:])])
>>>>>>> 578c0fd (update loss)
        # print(b_pred.shape)
        # print(b_true.shape)
        # print(b_pred.shape)
        # print(b_true.shape)
        #b_pred.reshape(b_true.shape)
        #return nn.MSELoss()(b_pred[-1,-4:], b_true[:,-4:])
        return nn.MSELoss()(b_pred_flat, b_true_flat)

    def compute_loss_b(self, pred, target, loss_type):
        if loss_type == "graph":
            return self.loss_b_graph(pred, target)
        elif loss_type == "node":
            return self.loss_b_node(pred, target)
    # def loss_raman(self, y_pred, y_true):
    #     return nn.MSELoss()(y_pred, y_true)
    
    
    def checkpoint(self, dataloader, device):
        self.eval()
        
        loss_cum = 0.
        #with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            d.pos.requires_grad = True
            y_b = self.forward(d)
            #print(y_bec.shape)
            loss_b = self.compute_loss_b(y_b, d.b, loss_type=self.loss_type).cpu()
            #loss_raman = self.loss_raman(y_raman_pred, d.raman).cpu()
            loss = loss_b
            
            loss_cum += loss.detach().item()
                
        return loss_cum/len(dataloader)

    
    def fit(self, opt, dataloader_train, dataloader_valid, history, s0, max_iter=10, device="cpu", scheduler=None):
        chkpt = 10

        for step in range(max_iter):
            self.train()

            loss_b = 0.
            loss_b_cum = 0.
            loss_cum = 0.
            start_time = time.time()

            for j, d in enumerate(dataloader_train):
                d.to(device)
                d.pos.requires_grad = True
                y_b = self.forward(d)
                #print(y_bec.shape)
                
                loss_b = self.compute_loss_b(y_b, d.b, loss_type=self.loss_type).cpu()
                #loss_raman = self.loss_raman(y_raman_pred, d.raman).cpu()
                loss = loss_b #+ loss_raman
                
                print(f"Iteration {step+1:5d}    batch {j+1:5d} / {len(dataloader_train):5d}   " +
                      f"batch loss = {loss.data:.4e}, becder. = {loss_b.data:.4e}", end="\r", flush=True)

                loss_b_cum += loss_b.detach().item()
                #loss_raman_cum += loss_raman.detach().item()
                loss_cum += loss.detach().item()
                
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            if scheduler is not None:
                scheduler.step()
            
            end_time = time.time()
            wall = end_time - start_time

            if (step+1)%chkpt == 0:
                loss_valid = self.checkpoint(dataloader_valid, device)
                loss_train = self.checkpoint(dataloader_train, device)
                print(f'\ntraining loss = {loss_train:.4e}\n')
                print(f'validation loss = {loss_valid:.4e}\n')
                # print(f"Iteration {step+1:5d}    batch {j+1:5d} / {len(dataloader_train):5d}   " +
                #       f"epoch loss = {loss_cum/len(dataloader_train):.4e}, becder. = {loss_b_cum/len(dataloader_train):.4e}")

                history.append({
                    'step': step + s0,
                    'wall': wall,
                    'batch': {
                        'loss': loss.item(),
                    },
                    'valid': {
                        'loss': loss_valid,
                    },
                     'train': {
                         'loss': loss_train,
                     },
                })

                yield {
                    'history': history,
                    'state': self.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler else None
                }

def print_text(ax, mat):
    for i in range(3):
        for j in range(3):
            text = ax.text(i,j, f"{mat[i, j]:.3f}",
                           color='tab:red', size=10, ha='center', va='center')

def visualize_output(entry: pd.Series, enn: E3NN, device: str):
    import torch_geometric as tg
    from e3nn.io import CartesianTensor
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    idx_to_plt = [20, 21, 22, 23] 
    x = tg.data.Batch.from_data_list([entry.data])
    x.pos.requires_grad = True
    b_pred = enn(x.to(device))
    b_pred = CartesianTensor("ij=ij").to_cartesian(b_pred.detach().cpu())
    b_pred = b_pred.reshape(-1,3,3)
    b_real = entry.becder.reshape(-1,3,3)
    
    fig, axs = plt.subplots(len(idx_to_plt) * 1,3, 
                           figsize=(4.5,2 * len(idx_to_plt)), 
                           gridspec_kw={'width_ratios': [1,1,0.07]}
                          )
    plt.subplots_adjust(wspace=0.1)
    
    vmax = np.abs(b_real[idx_to_plt]).max()
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    
    sm = mpl.cm.ScalarMappable(cmap=cm.vik_r, norm=norm)
    
    for i, idx in enumerate(idx_to_plt):
    
        axs[i, 0].imshow(b_real[idx], cmap=sm.cmap, norm=sm.norm)
        axs[i, 1].imshow(b_pred[idx], cmap=sm.cmap, norm=sm.norm)
        axs[i, 0].set_xticks([]); axs[i, 1].set_xticks([])
        axs[i, 0].set_yticks([]); axs[i, 1].set_yticks([])
        axs[i,2].set_visible(False)
        print_text(axs[i, 0], b_real[idx])
        print_text(axs[i, 1], b_pred[idx]-b_real[idx])
    
    #plot format
    axs[0, 0].set_title('True (red: value)')
    axs[0, 1].set_title('Pred. (red: error)')
    axs[0, 2].set_visible(True)
    plt.colorbar(sm, cax=axs[0, 2]);
    plt.tight_layout()
    fig.savefig('./images/example_becder_test_iloc0.png', bbox_inches='tight', transparent=False)
    
