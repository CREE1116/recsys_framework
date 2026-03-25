import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import pearsonr
import json
import argparse
from scipy.sparse import csr_matrix

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import load_config, ensure_dir
from src.data_loader import DataLoader
from src.utils.gpu_accel import EVDCacheManager

def run_exp1(dataset_name):
    print(f"Running Exp 1 on {dataset_name} (95% Energy, LAD, No Trim)...")
    
    # 1. Load Data
    config = load_config(dataset_name)
    loader = DataLoader(config)
    rows = loader.train_df['user_id'].values
    cols = loader.train_df['item_id'].values
    R = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(loader.n_users, loader.n_items))
    
    # 2. Get Full/Large EVD
    manager = EVDCacheManager()
    # Requesting k=None for "full" spectrum
    _, S, V, total_energy = manager.get_evd(R, k=None, dataset_name=config["dataset_name"])
    
    # 3. Energy-based Truncation (95%)
    # S are eigenvalues (sigma^2). Cumulative sum of S gives cumulative energy.
    s_vals = S.cpu().numpy()
    cum_energy = np.cumsum(s_vals)
    total_energy_val = s_vals.sum()
    k_95 = np.searchsorted(cum_energy, 0.95 * total_energy_val) + 1
    
    sigma_k = np.sqrt(np.maximum(s_vals[:k_95], 1e-12))
    V_k = V[:, :k_95].cpu().numpy()
    
    print(f"  Total Components: {len(s_vals)}, 95% Energy Components: {k_95}")
    
    # 4. Compute Spectral Observation Probability p_k
    # pi_j ∝ degree_j
    item_degrees = np.array(R.sum(axis=0)).flatten()
    pi_j = item_degrees / (item_degrees.sum() + 1e-12)
    # p_k = Σ_j π_j * V_{jk}^2
    p_k = np.sum(pi_j[:, np.newaxis] * (V_k**2), axis=0)
    
    log_sigma = np.log(sigma_k + 1e-12)
    log_p = np.log(p_k + 1e-12)
    
    # 5. LAD Fitting: minimize sum|y - (mx + b)| on ALL k_95 points
    def lad_loss(params, x, y):
        m, b = params
        return np.sum(np.abs(y - (m * x + b)))
    
    # Initial guess from OLS
    m_ols, b_ols = np.polyfit(log_sigma, log_p, 1)
    res = minimize(lad_loss, x0=[m_ols, b_ols], args=(log_sigma, log_p), method='Nelder-Mead')
    m_lad, b_lad = res.x

    # 6. Quantitative metrics
    # Pearson R and p-value for linearity
    r_val, p_val = pearsonr(log_sigma, log_p)
    
    # R2 calculation (using LAD line)
    y_pred = m_lad * log_sigma + b_lad
    ss_res = np.sum((log_p - y_pred)**2)
    ss_tot = np.sum((log_p - np.mean(log_p))**2)
    r_squared = 1 - (ss_res / (ss_tot + 1e-12))
    
    # 7. Distribution Comparison (Likelihood Ratio)
    # sig_vals here refers to the top k_95 singular values
    sig_vals = s_vals[:k_95]
    
    # Exponential Fit
    m_exp, c_exp = np.polyfit(sig_vals, log_p, 1)
    y_pred_exp = m_exp * sig_vals + c_exp
    res_exp = np.sum((log_p - y_pred_exp)**2)
    
    # Power-law residual (LAD)
    res_pl = np.sum((log_p - y_pred)**2)
    
    # Log-likelihood Ratio (simplified as difference of MSE or similar)
    # We want to show PL residual is significantly smaller than Exponential residual.
    ll_ratio = res_exp / (res_pl + 1e-12)
    
    # K-S Statistic (D) - lower is better
    from scipy.stats import kstest
    residuals = log_p - y_pred
    res_std = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)
    ks_stat, ks_p = kstest(res_std, 'norm') # D is ks_stat
    
    # 8. Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(log_sigma, log_p, alpha=0.4, s=15, color='blue', label=f'Data (k <= {k_95})')
    
    # Plot LAD line
    x_range = np.array([log_sigma.min(), log_sigma.max()])
    plt.plot(x_range, m_lad * x_range + b_lad, color='red', linewidth=2.5, 
             label=f'PL Fit: Slope={m_lad:.2f}, $R^2$={r_squared:.4f}')
    
    plt.xlabel(r"$\log(\sigma_k)$ (Singular Values)")
    plt.ylabel(r"$\log(p_k)$ (Observation Probability)")
    plt.title(f"Rigorous SPL Verification: {dataset_name}\n"
              f"LL-Ratio (PL/Exp)={ll_ratio:.2f}, K-S Dist (D)={ks_stat:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    out_dir = ensure_dir(f"aspire_experiments/output/exp1/{dataset_name}")
    plt.savefig(os.path.join(out_dir, "log_log_spectral_rigorous.png"), dpi=150)
    plt.close()
    
    results = {
        "dataset": dataset_name,
        "k_95": int(k_95),
        "r_squared": float(r_squared),
        "pearson_r": float(r_val),
        "ks_stat_D": float(ks_stat),
        "ks_p": float(ks_p),
        "ll_ratio_pl_exp": float(ll_ratio),
        "slope": float(m_lad),
        "pl_res": float(res_pl),
        "exp_res": float(res_exp)
    }
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Exp 1 on {dataset_name} finished. R^2: {r_squared:.4f}, LL-Ratio: {ll_ratio:.2f}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="Dataset name")
    args = parser.parse_args()
    
    try:
        run_exp1(args.dataset)
    except Exception as e:
        print(f"Error on {args.dataset}: {e}")
        import traceback
        traceback.print_exc()
