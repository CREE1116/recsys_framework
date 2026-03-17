# Usage: uv run python aspire_experiments/exp2_power_law.py --dataset ml1m --energy 1.0
import os
import sys
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
import argparse

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir
from src.models.csar.ASPIRELayer import AspireEngine
from src.models.csar import beta_estimators

def run_power_law(dataset_name):
    print(f"Running Experiment 2: Power-law Coupling on {dataset_name} (Full Spectrum)...")
    loader, R, S, V, config = get_loader_and_svd(dataset_name)
    
    item_pops = np.array(R.sum(axis=0)).flatten()
    s_np = S.cpu().numpy()
    
    # p_tilde = AspireEngine.compute_spp(V, item_pops)
    p_tilde = AspireEngine.compute_spp(V, item_pops)
    
    # 1. OLS
    beta_ols, r2_ols = AspireEngine.estimate_beta(S, p_tilde, verbose=False, estimator_type="ols")
    
    # 2. Pure LAD
    beta_lad, r2_lad = AspireEngine.estimate_beta(S, p_tilde, verbose=False, estimator_type="lad")

    # 3. Pairwise
    beta_pair, r2_pair = AspireEngine.estimate_beta(S, p_tilde, verbose=False, estimator_type="pairwise")
    
    # Data's raw OLS slope for reference (Global)
    raw_slope = np.linalg.lstsq(np.column_stack([np.log(s_np + 1e-12), np.ones_like(s_np)]), np.log(p_tilde + 1e-12), rcond=None)[0][0]
    
    print(f"\n[Beta Fitting Comparison for {dataset_name}]")
    print(f"  Observed Slope: {raw_slope:.4f}")
    print(f"  OLS       : β={beta_ols:.4f}, R²={r2_ols:.4f}")
    print(f"  Pure LAD  : β={beta_lad:.4f}, R²={r2_lad:.4f}")
    print(f"  Pairwise  : β={beta_pair:.4f}, R²={r2_pair:.4f}")

    x = np.log(s_np + 1e-9)
    y = np.log(p_tilde + 1e-9)
    log_s, log_pt = x, y
    
    # Output setup
    out_dir = ensure_dir(f"aspire_experiments/output/powerlaw/{config['dataset_name']}")
    
    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, alpha=0.3, s=10, label='Data points (Raw SPP)')
    
    def plot_line(b, x_vals, y_vals, color, label):
        # Slope in log-log space is 2 * beta / (1 + beta)
        # Using the Corollary 1 relation: slope = 2*beta / (1+beta)
        slope = 2.0 * b / (1.0 + b)
        intercept = np.mean(y_vals) - slope * np.mean(x_vals)
        plt.plot(x_vals, slope * x_vals + intercept, color=color, label=label)

    plot_line(beta_ols, x, y, 'blue', f'OLS (β={beta_ols:.3f}, R²={r2_ols:.3f})')
    plot_line(beta_lad, x, y, 'green', f'Pure LAD (β={beta_lad:.3f}, R²={r2_lad:.3f})')
    plot_line(beta_pair, x, y, 'orange', f'Pairwise (β={beta_pair:.3f}, R²={r2_pair:.3f})')
    
    plt.xlabel("log(σ_k)")
    plt.ylabel("log(p̃_k)")
    plt.title(f"Spectral Power-law Analysis: {config['dataset_name']}\n(L1-Robust Estimation)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "powerlaw_comparison.png"))
    plt.close()
    
    # 메인 결과는 OLS로 기록하여 확인
    beta, r2 = beta_ols, r2_ols
    
    result = {
        "dataset": config['dataset_name'],
        "beta_ols": float(beta_ols),
        "beta_lad": float(beta_lad),
        "beta_pair": float(beta_pair),
        "r2_ols": float(r2_ols),
        "r2_lad": float(r2_lad)
    }
    with open(os.path.join(out_dir, "result.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
        
    # [NEW] Save detailed log-log data
    pd.DataFrame({
        "rank": np.arange(1, len(s_np) + 1),
        "singular_value": s_np,
        "log_singular_value": x,
        "spp_ptilde": p_tilde,
        "log_spp_ptilde": y
    }).to_csv(os.path.join(out_dir, "powerlaw_data.csv"), index=False)
        
    print(f"  Estimated Beta: {beta_ols:.4f}, R²: {r2_ols:.4f}")
    print(f"  Result saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="Dataset name or path to yaml")
    args = parser.parse_args()
    
    run_power_law(args.dataset)
