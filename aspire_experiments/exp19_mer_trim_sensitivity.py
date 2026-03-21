import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from aspire_experiments.exp_utils import get_loader_and_svd
from src.models.csar import beta_estimators
from src.models.csar.ASPIRELayer import AspireEngine

def run_mer_trim_sensitivity(dataset_name):
    print(f"\n--- [Exp 19] MER Trimming Sensitivity on {dataset_name} ---")
    
    # 1. Load Data & EVD
    loader, R, S, V, config = get_loader_and_svd(dataset_name)
    s_np = S.cpu().numpy()
    
    # 2. Compute SPP (p_tilde)
    item_pops = np.asarray(R.sum(axis=0)).flatten().astype(float)
    p_tilde = AspireEngine.compute_spp(V.cpu().numpy(), item_pops)
    
    # 3. Sweep Trimming
    trims = [0.0, 0.01, 0.02, 0.05, 0.10]
    results = []
    
    for trim in trims:
        # We need a trimmed version of MER.
        # Modified log_s/log_p with trimming
        log_s = np.log(np.maximum(s_np, 1e-12))
        log_p = np.log(np.maximum(p_tilde, 1e-12))
        
        # Apply trimming to both ends
        n = len(log_s)
        t = int(n * trim)
        if t > 0 and 2 * t < n - 2:
            s_trim = s_np[t:-t]
            p_trim = p_tilde[t:-t]
        else:
            s_trim = s_np
            p_trim = p_tilde
            
        # Re-run MER on trimmed data
        if len(s_trim) > 0:
            mer_beta, _ = beta_estimators.estimate_beta_mer(s_trim, p_trim)
        else:
            mer_beta = 0.0
        
        results.append({
            'Trim': trim,
            'MER_Beta': mer_beta
        })
        print(f"Trim {trim*100:2.0f}%: MER Beta = {mer_beta:.4f}")

    df = pd.DataFrame(results)
    
    # 4. Visualization of ER curves
    betas = np.linspace(0.0, 3.0, 100)
    plt.figure(figsize=(10, 6))
    
    for trim in [0.0, 0.05, 0.10]:
        n = len(s_np)
        t = int(n * trim)
        # Fix for t:-t indexing bug: ensure slice is valid
        if t > 0 and 2 * t < n: # Changed n-1 to n for correct slicing logic
            s_t = s_np[t:-t]
            p_t = p_tilde[t:-t]
        elif t == 0:
            s_t = s_np
            p_t = p_tilde
        else: # If trimming would result in empty or invalid slice, use full arrays
            s_t = s_np
            p_t = p_tilde
        
        log_s_t = np.log(np.maximum(s_t, 1e-12))
        log_p_t = np.log(np.maximum(p_t, 1e-12))
        
        ers = []
        if len(s_t) == 0:
            ers = [0.0] * len(betas)
        else:
            for b in betas:
                z = log_s_t - b * log_p_t
                zm = np.max(z)
                ss = np.exp(z - zm)
                er = (np.sum(ss)**2) / (np.sum(ss**2) + 1e-12)
                ers.append(er)
        
        plt.plot(betas, ers, label=f'ER Curve (Trim {trim*100:g}%)')

    plt.xlabel('Beta')
    plt.ylabel('Effective Rank')
    plt.title(f'MER Sensitivity to Trimming ({dataset_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = f"aspire_experiments/plots/exp19_mer_sensitivity_{dataset_name}.png"
    os.makedirs("aspire_experiments/plots", exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    args = parser.parse_args()
    run_mer_trim_sensitivity(args.dataset)
