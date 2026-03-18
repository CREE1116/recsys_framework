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

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir, get_trimmed_data
from src.models.csar.ASPIRELayer import AspireEngine
from src.models.csar import beta_estimators

def run_power_law(dataset_name, seed=42):
    print(f"Running Experiment 2: Power-law Coupling on {dataset_name} (Full Spectrum, seed={seed})...")
    loader, R, S, V, config = get_loader_and_svd(dataset_name, seed=seed)
    
    item_pops = np.array(R.sum(axis=0)).flatten()
    s_np = S.cpu().numpy()
    
    # p_tilde = AspireEngine.compute_spp(V, item_pops)
    p_tilde = AspireEngine.compute_spp(V, item_pops)
    
    # Calculate aspect ratio Q for RMT
    M_full, N_full = R.shape
    Q_val = float(M_full / N_full)

    # 1. OLS
    beta_ols, r2_ols = AspireEngine.estimate_beta(S, p_tilde, verbose=False, estimator_type="ols")
    
    # 2. Pure LAD
    beta_lad, r2_lad = AspireEngine.estimate_beta(S, p_tilde, verbose=False, estimator_type="lad")
    
    # 3. Simple Slope (Global Untrimmed)
    beta_simple, r2_simple = AspireEngine.estimate_beta(S, p_tilde, verbose=False, estimator_type="simple_slope")

    # 4. Log-Derivative (Robust Exponent Estimation) - Multiple Quantiles
    beta_25, r2_25, _ = AspireEngine.estimate_beta(S, p_tilde, verbose=False, estimator_type="log_derivative", item_freq=item_pops, q=0.25)
    beta_50, r2_50, _ = AspireEngine.estimate_beta(S, p_tilde, verbose=False, estimator_type="log_derivative", item_freq=item_pops, q=0.50)
    beta_75, r2_75, _ = AspireEngine.estimate_beta(S, p_tilde, verbose=False, estimator_type="log_derivative", item_freq=item_pops, q=0.75)
    
    # Data's raw OLS slope for reference (Global)
    raw_slope = np.linalg.lstsq(np.column_stack([np.log(s_np + 1e-12), np.ones_like(s_np)]), np.log(p_tilde + 1e-12), rcond=None)[0][0]
    
    print(f"\n[Beta Fitting Comparison for {dataset_name}]")
    print(f"  Observed Slope: {raw_slope:.4f}")
    print(f"  OLS       : β={beta_ols:.4f}, R²={r2_ols:.4f}")
    print(f"  Pure LAD  : β={beta_lad:.4f}, R²={r2_lad:.4f}")
    print(f"  Simple Sl : β={beta_simple:.4f} (Global, Ident)")
    print(f"  Log-Deriv (q=0.25): β={beta_25:.4f}, R²={r2_25:.4f}")
    print(f"  Log-Deriv (q=0.50): β={beta_50:.4f}, R²={r2_50:.4f}")
    print(f"  Log-Deriv (q=0.75): β={beta_75:.4f}, R²={r2_75:.4f}")

    x = np.log(s_np + 1e-9)
    y = np.log(p_tilde + 1e-9)
    log_s, log_pt = x, y
    
    # Trimming for visualization (exclude 5% from both ends)
    x_trim, y_trim = get_trimmed_data(log_s, log_pt)
    
    # Output setup
    out_dir = ensure_dir(f"aspire_experiments/output/powerlaw/{config['dataset_name']}")
    
    plt.figure(figsize=(10, 7))
    plt.scatter(x_trim, y_trim, alpha=0.3, s=10, label='Data points (Trimmed 5%)')
    
    def plot_line(b, x_vals, y_vals, color, label, is_simple=False):
        val = b if np.isscalar(b) else np.mean(b)
        if is_simple:
            slope = val
        else:
            # v2 mapping for other legacy estimators in this script
            slope = 2.0 * val / (1.0 + val)
        intercept = np.mean(y_vals) - slope * np.mean(x_vals)
        plt.plot(x_vals, slope * x_vals + intercept, color=color, label=label)

    plot_line(beta_ols, x_trim, y_trim, 'blue', f'OLS (β={beta_ols:.3f})')
    plot_line(beta_lad, x_trim, y_trim, 'green', f'Pure LAD (β={beta_lad:.3f})')
    
    plot_line(beta_simple, x_trim, y_trim, 'orange', f'Simple Slope (β={beta_simple:.3f})', is_simple=True)
    plot_line(beta_25, x_trim, y_trim, 'cyan', f'Log-Deriv q=.25 (β={beta_25:.3f})')
    plot_line(beta_50, x_trim, y_trim, 'teal', f'Log-Deriv q=.50 (β={beta_50:.3f})')
    plot_line(beta_75, x_trim, y_trim, 'darkcyan', f'Log-Deriv q=.75 (β={beta_75:.3f})')
    
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
        "r2_ols": float(r2_ols),
        "r2_lad": float(r2_lad),
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
    parser.add_argument("--dataset", type=str, default="ml100k", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    dataset = args.dataset.replace("旋", "") # Clean up accidental char
    run_power_law(dataset, seed=args.seed)
