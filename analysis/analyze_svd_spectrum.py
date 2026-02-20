import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.sparse.linalg import svds

sys.path.append(os.getcwd())
from src.data_loader import DataLoader

def analyze_svd_spectrum(config_path='configs/dataset/ml1m.yaml'):
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure evaluation config exists to avoid KeyError in DataLoader
    if 'evaluation' not in config:
        config['evaluation'] = {
            'validation_method': 'sampled',
            'final_method': 'full'
        }
    
    # Check device
    device = config.get('device', 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading DataLoader...")
    dl = DataLoader(config)
    
    # Get Adjacency Matrix
    try:
        A = dl.train_matrix
    except AttributeError:
        # Fallback if not cached
        from scipy.sparse import csr_matrix
        train_df = dl.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        data = np.ones(len(rows))
        A = csr_matrix((data, (rows, cols)), shape=(dl.n_users, dl.n_items))
        A = A.astype(float)
        
    print(f"Adjacency Matrix Shape: {A.shape}")
    
    # Perform SVD
    # We want to see the spectrum. Let's compute a large number of SVs first.
    # Max possible rank is min(N, M). for ML-100k ~943 users, 1682 items.
    # Perform Full SVD using numpy (dense)
    # Since matrix is ~1000x1600, dense SVD is feasible.
    print(f"Converting to dense matrix for full spectrum analysis...")
    A_dense = A.toarray()
    
    print(f"Computing Full SVD using np.linalg.svd...")
    # full_matrices=False makes it return min(N,M) singular values
    u, s, vt = np.linalg.svd(A_dense, full_matrices=False)
    
    # s is already sorted in descending order by np.linalg.svd
    
    # Plot Spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(s, label='Singular Values')
    plt.yscale('log')
    plt.title('Singular Value Spectrum (Log Scale)')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.savefig('svd_spectrum.png')
    print("Saved svd_spectrum.png")
    
    # Cumulative Energy
    energy = s**2
    total_energy = np.sum(energy)
    cumulative_energy = np.cumsum(energy) / total_energy
    
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_energy, label='Cumulative Energy')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Energy')
    plt.axhline(y=0.9, color='g', linestyle='--', label='90% Energy')
    plt.title('Cumulative Energy Explained by Singular Values')
    plt.xlabel('Number of Components (K)')
    plt.ylabel('Cumulative Variance Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig('svd_energy.png')
    print("Saved svd_energy.png")
    
    # Find K for specific thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    print("\n[Optimal K Analysis]")
    print(f"Total Components: {len(s)}")
    print("-" * 30)
    for t in thresholds:
        k_needed = np.searchsorted(cumulative_energy, t) + 1
        print(f"Energy {t*100:>2.0f}% requires K = {k_needed}")
    print("-" * 30)

if __name__ == "__main__":
    analyze_svd_spectrum()
