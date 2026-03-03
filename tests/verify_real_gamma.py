import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
from src.models.csar.LIRALayer import estimate_mnar_gamma
import os

def load_ml1m():
    # Correct path from ml1m.yaml
    path = "/Users/leejongmin/code/recsys_framework/data/ml1m/ratings.dat"
    if not os.path.exists(path):
        print(f"File {path} not found. Skipping ML-1M test.")
        return None
    
    # Read with :: separator as per config
    df = pd.read_csv(path, sep="::", names=["user_id", "item_id", "rating", "timestamp"], engine='python')
    n_users = df['user_id'].max() + 1
    n_items = df['item_id'].max() + 1
    X = sp.csr_matrix((np.ones(len(df)), (df['user_id']-1, df['item_id']-1)), shape=(n_users, n_items))
    return X

def load_ml100k():
    path = "/Users/leejongmin/code/recsys_framework/data/ml100k/u.data"
    if not os.path.exists(path):
        print(f"File {path} not found. Skipping ML-100k test.")
        return None
    
    df = pd.read_csv(path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    n_users = df['user_id'].max() + 1
    n_items = df['item_id'].max() + 1
    X = sp.csr_matrix((np.ones(len(df)), (df['user_id']-1, df['item_id']-1)), shape=(n_users, n_items))
    return X

def verify_dataset(name, X):
    if X is None: return
    print(f"\n[Verification] {name} Gamma Estimation")
    
    # 1. Count-based Gamma
    gamma_count = estimate_mnar_gamma(X_sparse=X)
    print(f"{name} Count-based Gamma: {gamma_count:.4f}")
    
    # 2. Spectral-based Gamma (requires SVD)
    print(f"Computing SVD for {name} spectral gamma...")
    u, s, v = sp.linalg.svds(X.astype(float), k=min(200, X.shape[0]-1, X.shape[1]-1))
    s_torch = torch.from_numpy(s[::-1].copy()) # Sorted descending
    gamma_spec = estimate_mnar_gamma(X_sparse=X, singular_values=s_torch)
    print(f"{name} Spectral Gamma (k=200): {gamma_spec:.4f}")

def verify_real_data():
    verify_dataset("MovieLens-1M", load_ml1m())
    verify_dataset("MovieLens-100k", load_ml100k())

if __name__ == "__main__":
    verify_real_data()
