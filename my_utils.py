import pickle
import torch
import numpy as np
import os
from tqdm import tqdm


def remove_outer_axes(f, ax):
    ax = f.add_axes([0,0,1,1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return f,ax

def cache_data(cache_path: str, data_to_cache):
    os.makedirs('/'.join(cache_path.split('/')[:-1]), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data_to_cache, f)

def load_cached_data(cache_path: str):
    with open(cache_path, 'rb') as f:
        dat = pickle.load(f)
    return dat
