import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import json
import argparse
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import load_config, ensure_dir
from src.data_loader import DataLoader
from src.utils.gpu_accel import get_device, EVDCacheManager

def compute_spl_metrics(X_sparse, dataset_name, device, label="MNAR", energy_target=0.95):
    """
    SVD를 수행하고 Spectral Observation Probability (p_k)와 특이값(sigma_k)의 멱법칙 관계 분석
    """
    print(f"  Analyzing {label} spectrum...")
    
    # 1. EVD/SVD 계산
    manager = EVDCacheManager()
    U, S, V, total_energy = manager.get_evd(X_sparse, k=None, dataset_name=f"{dataset_name}_{label}")
    
    # S는 고윳값 (sigma^2)
    s_vals = S.cpu().numpy()
    sigma = np.sqrt(np.maximum(s_vals, 1e-12))
    V_np = V.cpu().numpy()
    
    # 2. Energy-based Truncation (95%)
    cum_energy = np.cumsum(s_vals)
    total_energy_val = s_vals.sum()
    k_95 = np.searchsorted(cum_energy, energy_target * total_energy_val) + 1
    
    sigma_k = sigma[:k_95]
    V_k = V_np[:, :k_95]
    
    # 3. p_k proxy 계산 (아이템 인기도를 스펙트럴 방향으로 집약)
    # n_i: 아이템별 상호작용 빈도 (popularity)
    n_i = np.array(X_sparse.sum(axis=0)).flatten()
    pi_i = n_i / (n_i.sum() + 1e-12)
    
    # p_k = Σ_i pi_i * V_{ik}^2
    # matrix form: diag(V_k.T @ diag(pi_i) @ V_k)
    p_k = np.sum(pi_i[:, np.newaxis] * (V_k**2), axis=0)
    
    # 4. Log-Log Regression
    log_sigma = np.log(sigma_k + 1e-12)
    log_p = np.log(p_k + 1e-12)
    
    # OLS Regression
    x = log_sigma.reshape(-1, 1)
    y = log_p
    reg = LinearRegression().fit(x, y)
    slope = reg.coef_[0]
    r_squared = reg.score(x, y)
    
    # Pearson R
    r_val, _ = pearsonr(log_sigma, log_p)
    
    return {
        "label": label,
        "k_95": int(k_95),
        "slope": float(slope),
        "r_squared": float(r_squared),
        "pearson_r": float(r_val),
        "log_sigma": log_sigma.tolist(),
        "log_p": log_p.tolist()
    }

def run_exp14(dataset_name="yahoo_r3"):
    print(f"Running Exp 14: Direct SPL Verification on {dataset_name} (MCAR vs MNAR)")
    
    # 1. Load Data
    config = load_config(dataset_name)
    loader = DataLoader(config)
    
    # MNAR (Train)
    rows_mnar = loader.train_df['user_id'].values
    cols_mnar = loader.train_df['item_id'].values
    X_mnar = csr_matrix((np.ones(len(rows_mnar)), (rows_mnar, cols_mnar)), shape=(loader.n_users, loader.n_items))
    
    # MCAR (Test)
    rows_mcar = loader.test_df['user_id'].values
    cols_mcar = loader.test_df['item_id'].values
    X_mcar = csr_matrix((np.ones(len(rows_mcar)), (rows_mcar, cols_mcar)), shape=(loader.n_users, loader.n_items))
    
    device = get_device()
    
    # 2. Compute Metrics
    res_mnar = compute_spl_metrics(X_mnar, dataset_name, device, label="MNAR")
    res_mcar = compute_spl_metrics(X_mcar, dataset_name, device, label="MCAR")
    
    # 3. Plotting
    out_dir = ensure_dir(f"aspire_experiments/output/exp14/{dataset_name}")
    
    plt.figure(figsize=(10, 6))
    
    # MNAR Plot
    plt.scatter(res_mnar["log_sigma"], res_mnar["log_p"], alpha=0.5, s=15, color='blue', label=f'MNAR (Slope: {res_mnar["slope"]:.3f}, $R^2$: {res_mnar["r_squared"]:.3f})')
    # OLS line for MNAR
    x_range = np.array([min(res_mnar["log_sigma"]), max(res_mnar["log_sigma"])])
    plt.plot(x_range, res_mnar["slope"] * (x_range - np.mean(res_mnar["log_sigma"])) + np.mean(res_mnar["log_p"]), 'b--', alpha=0.8)

    # MCAR Plot
    plt.scatter(res_mcar["log_sigma"], res_mcar["log_p"], alpha=0.5, s=15, color='red', label=f'MCAR (Slope: {res_mcar["slope"]:.3f}, $R^2$: {res_mcar["r_squared"]:.3f})')
    # OLS line for MCAR
    x_range_mcar = np.array([min(res_mcar["log_sigma"]), max(res_mcar["log_sigma"])])
    plt.plot(x_range_mcar, res_mcar["slope"] * (x_range_mcar - np.mean(res_mcar["log_sigma"])) + np.mean(res_mcar["log_p"]), 'r--', alpha=0.8)

    plt.title(f'SPL Verification: MNAR vs MCAR ({dataset_name})\n'
              f'Delta Slope (Bias Intensity): {res_mnar["slope"] - res_mcar["slope"]:.4f}')
    plt.xlabel(r'$\log \sigma_k$ (Singular Values)')
    plt.ylabel(r'$\log p_k$ (Spectral Popularity)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(out_dir, "spl_mnar_vs_mcar.png"), dpi=150)
    plt.close()
    
    # Save Results
    results = {
        "dataset": dataset_name,
        "mnar": res_mnar,
        "mcar": res_mcar,
        "delta_slope": res_mnar["slope"] - res_mcar["slope"]
    }
    
    # Remove large lists for JSON
    res_mnar.pop("log_sigma"); res_mnar.pop("log_p")
    res_mcar.pop("log_sigma"); res_mcar.pop("log_p")
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Exp 14 finished. MNAR Slope: {res_mnar['slope']:.4f}, MCAR Slope: {res_mcar['slope']:.4f}")
    print(f"  Delta Slope: {results['delta_slope']:.4f} (Larger value indicates stronger MNAR bias)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yahoo_r3", help="Dataset name")
    args = parser.parse_args()
    
    try:
        run_exp14(dataset_name=args.dataset)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
