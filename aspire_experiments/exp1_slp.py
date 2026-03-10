# Usage: uv run python aspire_experiments/exp1_slp.py --dataset ml1m --energy 0.95
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir

def run_slp(dataset_name, target_energy=0.95):
    print(f"Running Experiment 1: SLP Verification on {dataset_name} (Energy={target_energy})...")
    loader, R, S, V, config = get_loader_and_svd(dataset_name, target_energy=target_energy)
    
    # Item popularity
    p = np.array(R.sum(axis=0)).flatten()
    p_norm = p / (p.max() + 1e-9)  # Normalize
    P_diag = p_norm
    
    # M = V^T P V
    V_np = V.cpu().numpy()
    M = (V_np.T * P_diag) @ V_np
    
    D = np.diag(np.diag(M))
    off_diag = M - D
    
    # [Expert Feedback] Calculate epsilon as mean(abs(off-diag)) / mean(diag)
    diag_vals = np.diag(M)
    mask = ~np.eye(len(M), dtype=bool)
    epsilon = float(np.mean(np.abs(M[mask])) / (np.mean(diag_vals) + 1e-9))
    
    # Output setup
    out_dir = ensure_dir(f"aspire_experiments/output/slp/{config['dataset_name']}")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(M, cmap='viridis')
    plt.title(f"Spectral Popularity Matrix M (Rank K={V_np.shape[1]})\nDataset: {config['dataset_name']}, ε={epsilon:.4f}")
    plt.savefig(os.path.join(out_dir, "slp_heatmap.png"))
    plt.close()
    
    # Log results
    result = {
        "dataset": config['dataset_name'],
        "epsilon": float(epsilon),
        "rank_k": int(V.shape[1]),
        "m_shape": M.shape
    }
    with open(os.path.join(out_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"  SLP Epsilon: {epsilon:.4f}")
    print(f"  Result saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="Dataset name or path to yaml")
    parser.add_argument("--energy", type=float, default=0.95, help="Target energy for SVD rank")
    args = parser.parse_args()
    
    run_slp(args.dataset, args.energy)
