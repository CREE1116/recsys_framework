import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_ml1m():
    path = "/Users/leejongmin/code/recsys_framework/data/ml1m/ratings.dat"
    if not os.path.exists(path): return None
    df = pd.read_csv(path, sep="::", names=["user_id", "item_id", "rating", "timestamp"], engine='python')
    # Use 0-based indexing
    u_map = {v: i for i, v in enumerate(df['user_id'].unique())}
    i_map = {v: i for i, v in enumerate(df['item_id'].unique())}
    X = sp.csr_matrix((np.ones(len(df)), (df['user_id'].map(u_map), df['item_id'].map(i_map))))
    return X

def analyze_decay(name, X):
    if X is None: return
    print(f"\n--- {name} Analysis ---")
    
    # 1. Popularity Decay
    item_pops = np.array(X.sum(axis=0)).flatten()
    item_pops = np.sort(item_pops)[::-1]
    item_pops = item_pops[item_pops > 0]
    
    y_pop = np.log(item_pops)
    x_pop = np.log(np.arange(1, len(item_pops) + 1))
    p_slope, _ = np.polyfit(x_pop, y_pop, 1)
    print(f"Popularity Slope (p): {-p_slope:.4f}")
    
    # 2. Spectral Decay (Singular Values)
    k = min(500, X.shape[0]-1, X.shape[1]-1)
    print(f"Computing SVD (k={k})...")
    u, s, v = sp.linalg.svds(X.astype(float), k=k)
    s = s[::-1] # Descending
    s2 = s**2
    
    y_spec = np.log(s2)
    x_spec = np.log(np.arange(1, len(s2) + 1))
    spec_slope, _ = np.polyfit(x_spec, y_spec, 1)
    print(f"Spectral Slope (alpha): {-spec_slope:.4f}")
    
    # Check head vs tail slope for spectrum
    head_slope, _ = np.polyfit(x_spec[:50], y_spec[:50], 1)
    tail_slope, _ = np.polyfit(x_spec[50:], y_spec[50:], 1)
    print(f"Head Spectral Slope: {-head_slope:.4f}")
    print(f"Tail Spectral Slope: {-tail_slope:.4f}")

    # Theoretical Gammas
    # gamma = alpha - 1 (based on spectrum)
    # gamma = 2p - 1 (based on counts)
    print(f"Gamma (Spectral alpha-1): {-spec_slope - 1.0:.4f}")
    print(f"Gamma (Count 2p-1): {2*(-p_slope) - 1.0:.4f}")

if __name__ == "__main__":
    analyze_decay("MovieLens-1M", load_ml1m())
