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

def run_trim_sensitivity(dataset_name):
    print(f"\n--- [Exp 17] OLS Trimming Sensitivity on {dataset_name} ---")
    
    # 1. Load Data & EVD using framework utility
    loader, R, S, V, config = get_loader_and_svd(dataset_name)
    s_np = S.cpu().numpy()
    
    # 2. Compute SPP (p_tilde)
    item_pops = np.asarray(R.sum(axis=0)).flatten().astype(float)
    p_tilde = AspireEngine.compute_spp(V.cpu().numpy(), item_pops)
    
    # 3. Sweep Trimming Levels
    trims = [0.0, 0.05, 0.10, 0.15, 0.20]
    results = []
    
    for trim in trims:
        ols_beta, ols_r2 = beta_estimators.estimate_beta_ols(s_np, p_tilde, trim_tail=trim)
        lad_beta, lad_r2 = beta_estimators.beta_lad(s_np, p_tilde, trim_tail=trim)
        
        results.append({
            'Trim_Per_Side': trim,
            'Total_Trim_Perc': 2 * trim * 100,
            'OLS_Beta': ols_beta,
            'OLS_R2': ols_r2,
            'LAD_Beta': lad_beta,
            'LAD_R2': lad_r2
        })
        print(f"Trim {trim*100:2.0f}% per side (Total {2*trim*100:2.0f}%): OLS={ols_beta:.4f} (R2={ols_r2:.4f}), LAD={lad_beta:.4f} (R2={lad_r2:.4f})")

    df = pd.DataFrame(results)
    
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(df['Trim_Per_Side'] * 100, df['OLS_Beta'], 'o-', label='OLS Beta', markersize=8)
    plt.plot(df['Trim_Per_Side'] * 100, df['LAD_Beta'], 's--', label='LAD Beta', markersize=8)
    plt.xlabel('Trimming Percentage (Per Side)')
    plt.ylabel('Estimated Beta (Raw Slope)')
    plt.title(f'Beta Sensitivity to Spectral Trimming ({dataset_name})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = f"aspire_experiments/plots/exp17_sensitivity_{dataset_name}.png"
    os.makedirs("aspire_experiments/plots", exist_ok=True)
    plt.savefig(save_path)
    print(f"\n[Summary for {dataset_name}]")
    print(df[['Trim_Per_Side', 'OLS_Beta', 'LAD_Beta']].to_string(index=False))
    print(f"\nPlot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    args = parser.parse_args()
    run_trim_sensitivity(args.dataset)
