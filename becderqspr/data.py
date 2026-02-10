#currently an empty file for loading, wrangling data
import pandas as pd
from becderqspr.io import ase_db_to_df
import numpy as np

def load_db(filename, selection):
    # load data from a csv file and derive formula and species columns from structure
    df = ase_db_to_df(filename, selection)
    # try:
    #     # structure provided as Atoms object
    #     df['structure'] = df['structure'].apply(eval).progress_map(lambda x: Atoms.fromdict(x))
    
    # except:
    #     # no structure provided
    #     species = []

    # else:
    #     df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
    #     df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
    #     species = sorted(list(set(df['species'].sum())))
    
    df['formula'] = df['structure'].map(lambda x: x.get_chemical_formula())
    df['species'] = df['structure'].map(lambda x: list(set(x.get_chemical_symbols())))
    species = sorted(list(set(df['species'].sum())))

    #df['bec'] = df['bec'].apply(eval).apply(np.array)

    return df , species
def train_valid_test_split(data, valid_size, test_size, seed=12):
    size = len(data)
    rng = np.random.default_rng(seed)
    idx = np.arange(size)
    rng.shuffle(idx)
    idx_train = idx[:int(size*(1-valid_size-test_size))]
    idx_valid = idx[int(size*(1-valid_size-test_size)):int(size*(1-test_size))]
    idx_test = idx[int(size*(1-test_size)):]
    return idx_train, idx_valid, idx_test
