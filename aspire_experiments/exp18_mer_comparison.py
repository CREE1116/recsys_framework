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

def run_mer_comparison(dataset_name):
    print(f"\n--- [Exp 18] MER vs OLS/LAD on {dataset_name} ---")
    
    # 1. Load Data & EVD
    loader, R, S, V, config = get_loader_and_svd(dataset_name)
    s_np = S.cpu().numpy()
    
    # 2. Compute SPP (p_tilde)
    item_pops = np.asarray(R.sum(axis=0)).flatten().astype(float)
    p_tilde = AspireEngine.compute_spp(V.cpu().numpy(), item_pops)
    
    # 3. Estimate Beta with various methods
    ols_beta, ols_r2 = beta_estimators.estimate_beta_ols(s_np, p_tilde, trim_tail=0.05)
    lad_beta, lad_r2 = beta_estimators.beta_lad(s_np, p_tilde, trim_tail=0.05)
    mer_beta, _ = beta_estimators.estimate_beta_mer(s_np, p_tilde)
    
    print(f"\n[Comparison for {dataset_name}]")
    print(f"OLS (Trim 5%): {ols_beta:.4f} (R2={ols_r2:.4f})")
    print(f"LAD (Trim 5%): {lad_beta:.4f} (R2={lad_r2:.4f})")
    print(f"MER (Full)   : {mer_beta:.4f}")

    # 4. Effective Rank Curve Analysis
    betas = np.linspace(0.0, 3.0, 100)
    er_values = []
    
    def get_er(beta_val):
        s_restored = s_np / (p_tilde ** beta_val + 1e-12)
        l1 = np.sum(s_restored) ** 2
        l2 = np.sum(s_restored ** 2)
        return l1 / (l2 + 1e-12)
        
    for b in betas:
        er_values.append(get_er(b))
        
    # 5. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(betas, er_values, label='Effective Rank (ER)', color='purple', linewidth=2)
    plt.axvline(mer_beta, color='red', linestyle='--', label=f'Optimal MER Beta ({mer_beta:.4f})')
    plt.axvline(ols_beta, color='blue', linestyle=':', label=f'OLS Beta ({ols_beta:.4f})')
    plt.axvline(lad_beta, color='green', linestyle='-.', label=f'LAD Beta ({lad_beta:.4f})')
    
    plt.xlabel('Beta Value')
    plt.ylabel('Effective Rank')
    plt.title(f'Effective Rank Optimization ({dataset_name})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = f"aspire_experiments/plots/exp18_mer_analysis_{dataset_name}.png"
    os.makedirs("aspire_experiments/plots", exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    args = parser.parse_args()
    run_mer_comparison(args.dataset)
