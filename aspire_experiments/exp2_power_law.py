# Usage: uv run python aspire_experiments/exp2_power_law.py --dataset ml1m --energy 0.95
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
import argparse

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir
from src.models.csar.ASPIRELayer import AspireEngine

def run_power_law(dataset_name, target_energy=0.95):
    print(f"Running Experiment 2: Power-law Coupling on {dataset_name} (Energy={target_energy})...")
    loader, R, S, V, config = get_loader_and_svd(dataset_name, target_energy=target_energy)
    
    item_pops = np.array(R.sum(axis=0)).flatten()
    s_np = S.cpu().numpy()
    
    # p_tilde = AspireEngine.compute_spp(V, item_pops)
    p_tilde = AspireEngine.compute_spp(V, item_pops)
    
    # Beta estimation (v13 logic)
    # Use return_line=True to get the exact fitted line from AspireEngine to avoid double fitting inconsistency
    beta, r2, y_pred = AspireEngine.estimate_beta(S, p_tilde, verbose=False, return_line=True)
    
    # Log-Log for visualization
    x = np.log(s_np + 1e-9)
    y = np.log(p_tilde + 1e-9)
    
    # Output setup
    out_dir = ensure_dir(f"aspire_experiments/output/powerlaw/{config['dataset_name']}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5, label='Data points')
    plt.plot(x, y_pred, color='red', label=f'Huber Fit (2β={2*beta:.3f}, β={beta:.4f})')
    plt.xlabel("log(σ_k)")
    plt.ylabel("log(p̃_k)")
    plt.title(f"Spectral Power-law Coupling\nDataset: {config['dataset_name']}, R²={r2:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "powerlaw_fit.png"))
    plt.close()
    
    # Log results
    result = {
        "dataset": config['dataset_name'],
        "beta": float(beta),
        "r2": float(r2),
    }
    with open(os.path.join(out_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"  Estimated Beta: {beta:.4f}, R²: {r2:.4f}")
    print(f"  Result saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="Dataset name or path to yaml")
    parser.add_argument("--energy", type=float, default=0.95, help="Target energy for SVD rank")
    args = parser.parse_args()
    
    run_power_law(args.dataset, args.energy)
