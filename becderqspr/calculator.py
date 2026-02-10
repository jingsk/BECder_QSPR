# model
import torch
import torch_geometric as tg
# import torch_scatter
# import e3nn
# from e3nn import o3
from e3nn.io import CartesianTensor
# from e3nn.o3 import ReducedTensorProducts
# from typing import Dict, Union
#from becqsdr.data import train_valid_test_split
#from utils.e3nn import Network
from becqsdr.model import E3NN

import cmcrameri.cm as cm

# crystal structure data
from ase import Atom, Atoms
from ase.neighborlist import neighbor_list
#from ase.visualize.plot import plot_atoms

# data pre-processing and visualization
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import pandas as pd

# utilities
from tqdm import tqdm
# from becqsdr.data import load_db
# import yaml
# import wandb

import warnings
warnings.filterwarnings("ignore")
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
# tqdm.pandas(bar_format=bar_format)
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#db_file_name = 'data/bec_run_test.db'

# build data
def build_data(entry, am_onehot, type_encoding, type_onehot, r_max=3.5):

    symbols = list(entry.structure.symbols).copy()
    positions = torch.from_numpy(entry.structure.positions.copy())
    lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list("ijS", a=entry.structure, cutoff=r_max, self_interaction=True)
    
    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.from_numpy(edge_src)]
    edge_vec = (positions[torch.from_numpy(edge_dst)]
                - positions[torch.from_numpy(edge_src)]
                + torch.einsum('ni,nij->nj', torch.tensor(edge_shift, dtype=default_dtype), lattice[edge_batch]))

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)
    
    data = tg.data.Data(
        pos=positions, lattice=lattice, symbol=symbols,
        x_in=am_onehot[[type_encoding[specie] for specie in symbols]],   # atomic mass (node feature)
        z_in=type_onehot[[type_encoding[specie] for specie in symbols]], # atom type (node attribute)
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec, edge_len=edge_len,
        # y=CartesianTensor("ij=ji").from_cartesian(torch.from_numpy(entry.diel), rtp=ReducedTensorProducts('ij=ji', i='1o')).unsqueeze(0),
        # b=CartesianTensor("ij=ij").from_cartesian(torch.from_numpy(entry.bec)).unsqueeze(0)
    )

    return data


def load_build_data(frames, r_max):
    # load data
    import pandas as pd
    all_atoms = [atoms for atoms in frames]
    #all_bec = [row.data.bec for row in db.select(selection)]
    #all_diel = [row.data.diel for row in db.select(selection)]
    #energy = [row.energy for row in db.select(selection)]
    #forces = [row.forces for row in db.select(selection)]
    df = pd.DataFrame(data={'structure': all_atoms, 
                           },
                    )
    df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
    df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
    species = sorted(list(set(df['species'].sum())))
    
    #df, species = load_db(db_file_name, selection='has_bec')
    species = [Atom(k).number for k in species]
    Z_max = max([Atom(k).number for k in species])

    # one-hot encoding atom type and mass
    type_encoding = {}
    specie_am = []
    for Z in range(1, Z_max+1):
        specie = Atom(Z)
        type_encoding[specie.symbol] = Z - 1
        specie_am.append(specie.mass)

    type_onehot = torch.eye(len(type_encoding))
    am_onehot = torch.diag(torch.tensor(specie_am))
    load_df = False
    if load_df:
        df = pd.read_pickle("./df_data.pkl")
    else:
        #df['data'] = df.progress_apply(lambda x: build_data(x, am_onehot, type_encoding, type_onehot, r_max), axis=1)
        df['data'] = df.apply(lambda x: build_data(x, am_onehot, type_encoding, type_onehot, r_max), axis=1)
        df.to_pickle("./df_data.pkl")
    return df, Z_max

def load_model(args_enn, model_path, device):
    #run = wandb.init(config=config)
    lr = 5e-4

    enn = E3NN(**args_enn).to(device)
    opt = torch.optim.Adam(enn.parameters(), lr=lr)
    scheduler = None #torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)

    resume = True
        
    if resume:
        print(f'loading from {model_path}')
        saved = torch.load(model_path, map_location=device)
        enn.load_state_dict(saved['state'])
        opt.load_state_dict(saved['optimizer'])
        try:
            scheduler.load_state_dict(saved['scheduler'])
        except:
            scheduler = None
        history = saved['history']
        s0 = history[-1]['step'] + 1
        print(f'Starting from step {s0:d}')

    else:
        history = []
        s0 = 0

    return enn

def model_arg(r_max,Z_max):
    num_neighbors = 100
    emb_dim = 4
    num_layers = 3
    lmax = 4
    radial_layers = 3
    radial_neurons = 64

    args_enn = {'in_dim': Z_max,
                'emb_dim': emb_dim,
                'num_layers': num_layers,
                'max_radius': r_max,
                'num_neighbors': num_neighbors,
                'lmax': lmax,
                'radial_layers': radial_layers,
                'radial_neurons': radial_neurons,
            }
    return args_enn
#following suggestion to fix memory retention/inference slow down from https://discuss.pytorch.org/t/releasing-memory-after-running-a-pytorch-model-inference/175654/2
# no gradients / computation graph will be tracked, saving memory
@torch.no_grad()
def calculate_bec(atoms, enn):
    df, _ = load_build_data([
        atoms
    ],r_max=enn.max_radius)
    #print(f'Zmax={Z_max}')
    entry = df.iloc[0]
    x = entry.data
    x.pos.requires_grad = True

    bec_pred = enn(x.to(device))
    bec_pred = CartesianTensor("ij=ij").to_cartesian(bec_pred.detach().cpu())
    bec_pred = bec_pred.reshape(-1,3,3)
    return bec_pred

if __name__ =='__main__':

    r_max = 3 # cutoff radius
    Z_max = 34
    #sample usage

    args_enn = model_arg(r_max,Z_max)
    lr = 5e-4
    model_name = f"lr{lr}_num_neighbors{args_enn['num_neighbors']}_emb_dim{args_enn['emb_dim']}_num_layers{args_enn['num_layers']}_lmax{args_enn['lmax']}_radial_layers{args_enn['radial_layers']}_radial_neurons{args_enn['radial_neurons']}"
    model_path = f'/Users/ktrerayapiwat/jp/GeSe_bec/train/models/{model_name}.torch'
    enn = load_model(args_enn, model_path, device)

    from monochalcogenpy.build import unit_cell
    atoms = unit_cell(3.9,3.9,20)

    bec = calculate_bec(atoms, enn)
    print(bec)
