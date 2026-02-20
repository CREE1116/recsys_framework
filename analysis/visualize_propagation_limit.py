import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, eye, diags
import argparse
import yaml
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data_loader import DataLoader
from src.models.csar.LIRA import LIRA

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def symmetric_normalize(adj):
    """Applies symmetric normalization: D^-0.5 * A * D^-0.5"""
    if hasattr(adj, 'toarray'):
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = diags(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    else:
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def compute_normalized_adjacency(X_sparse):
    """
    Compute Normalized Adjacency Matrix A_hat = D^-0.5 * A * D^-0.5
    where A = X^T @ X (Item-Item Co-occurrence) with diagonal removed
    """
    print("Computing Item-Item Adjacency Matrix (A = X^T @ X)...")
    X = X_sparse.copy()
    X.data = np.ones_like(X.data)
    A = X.T @ X
    A.setdiag(0)
    A.eliminate_zeros()
    return symmetric_normalize(A) 

def process_for_metric(mat):
    """
    Prepares matrix for comparison:
    1. Zero out diagonal (Requested: "Analyze excluding diagonal")
    2. No Max Scaling (Use SymNorm values directly)
    """
    if hasattr(mat, 'toarray'):
        mat = mat.toarray()
    mat = np.asarray(mat).copy()
    
    # Zero Diagonal
    np.fill_diagonal(mat, 0)
    
    return mat

def get_similarity(m1, m2):
    """Cosine similarity of flattened matrices"""
    return np.dot(m1.flatten(), m2.flatten()) / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, default='configs/dataset/ml1m.yaml')
    parser.add_argument('--lambda_val', type=float, default=100.0)
    parser.add_argument('--max_hops', type=int, default=4)
    parser.add_argument('--top_k_viz', type=int, default=100)
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading dataset from {args.dataset_config}...")
    config = load_config(args.dataset_config)
    config['model'] = {'name': 'lira', 'reg_lambda': args.lambda_val, 'normalize': True}
    config['device'] = 'cpu'
    
    if 'evaluation' not in config:
        config['evaluation'] = {'validation_method': 'sampled', 'final_method': 'full'}
    
    data_loader = DataLoader(config)
    
    # Initialize LIRA
    model = LIRA(config, data_loader)
    
    # Build S
    print(f"Building LIRA S matrix (lambda={args.lambda_val})...")
    model.lira_layer.build(model.train_matrix_csr)
    S_lira = model.lira_layer.S.detach().cpu().numpy()
    
    # 2. Compute Adjacency for Propagation
    X_sparse = model.train_matrix_csr
    A_norm = compute_normalized_adjacency(X_sparse) 
    
    # Top K
    item_freq = np.array(X_sparse.sum(axis=0)).flatten()
    top_k_indices = np.argsort(item_freq)[::-1][:args.top_k_viz]
    
    # LIRA S: Full Matrix for Similarity
    S_lira_full = process_for_metric(S_lira)
    
    # Top K for Visualization ONLY
    S_lira_sub = S_lira_full[np.ix_(top_k_indices, top_k_indices)]
    
    # Setup Plot
    # LIRA, Hop 1, Hop 2, ... Hop K
    n_plots = args.max_hops + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    
    sns.heatmap(S_lira_sub, ax=axes[0], cmap='viridis', cbar=False)
    axes[0].set_title(f"LIRA S (Off-Diag)\n(lambda={args.lambda_val})")
    axes[0].axis('off')
    
    print("Propagating (Pure Hops vs Damped Accumulation, Full Matrix)...")
    
    A_pow = A_norm
    
    # Initialize Accumulation 
    # Use full matrix for accumulation to be accurate
    if hasattr(A_norm, 'toarray'):
        A_accumulated = np.zeros_like(A_norm.toarray())
    else:
        # If sparse, keep sparse for accumulation then convert for metric?
        # A_norm is 2823x2823, manageable dense. 
        # But let's stick to whatever type A_norm is, assuming dense for now since we used process_for_metric on it
        # Actually process_for_metric converts to dense.
        # Let's initialize zero dense matrix based on A_norm shape
        shape = A_norm.shape
        A_accumulated = np.zeros(shape)

    alpha = 0.85
    metrics = {}
    
    # Loop Hops
    for k in range(1, args.max_hops + 1):
        if k > 1:
            A_pow = A_pow @ A_norm
            
        # 1. Pure Hop Comparison
        curr_hop_mat = process_for_metric(A_pow)
        sim_pure = get_similarity(S_lira_full, curr_hop_mat)
        
        # 2. Accumulated Damped Comparison
        # A_acc += alpha^k * A^k
        # Need to handle sparse A_pow if it is sparse
        if hasattr(A_pow, 'toarray'):
            A_pow_dense = A_pow.toarray()
        else:
            A_pow_dense = A_pow
            
        A_accumulated += (alpha ** k) * A_pow_dense
        
        curr_acc_mat = process_for_metric(A_accumulated)
        sim_acc = get_similarity(S_lira_full, curr_acc_mat)
        
        # Visualization (Subset - Pure vs Acc? Let's show Acc since Pure was shown before)
        # Or maybe split subplot top/bottom? 
        # For now, let's visualize the Accumulated one as that's the "new" thing user wants to see
        # But user script snippet didn't specify visualization, just printing.
        # I'll stick to visualizing Pure Hops (to keep previous style) or just update title.
        # Let's visualize Accumulated Damped as it's the more complex one.
        
        S_acc_sub = curr_acc_mat[np.ix_(top_k_indices, top_k_indices)]
        
        ax = axes[k]
        sns.heatmap(S_acc_sub, ax=ax, cmap='viridis', cbar=False)
        
        ax.set_title(f"Hop {k}\nPure: {sim_pure:.4f}\nAcc: {sim_acc:.4f}")
        ax.axis('off')
        
        metrics[f'hop_{k}_pure'] = float(sim_pure)
        metrics[f'hop_{k}_acc'] = float(sim_acc)
        print(f"Hop {k} | Pure Sim: {sim_pure:.4f} | Acc Sim: {sim_acc:.4f}")
        
    plt.tight_layout()
    output_path = os.path.join('analysis', 'propagation_limit.png')
    metrics_path = os.path.join('analysis', 'propagation_limit_metrics.json')
    os.makedirs('analysis', exist_ok=True)
    plt.savefig(output_path, dpi=300)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()
