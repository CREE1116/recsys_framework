import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure root is in path
sys.path.append(os.getcwd())

from src.data_loader import DataLoader
from src.utils.gpu_accel import EVDCacheManager
from src.models.csar import beta_estimators
from aspire_experiments.exp_utils import load_config
from scipy.sparse import csr_matrix
from scipy.ndimage import gaussian_filter1d
from src.models.csar.beta_estimators import estimate_beta_ols

def run_robust_beta_analysis(dataset_name, device="auto"):
    print(f"\n[Exp07] Analyzing Robust Beta Estimation (Spectral Smoothing) for {dataset_name}...")
    
    # 1. Load Data & Compute/Load EVD
    config = load_config(dataset_name)
    data_loader = DataLoader(config)
    evd_manager = EVDCacheManager(device=device)
    
    rows = data_loader.train_df['user_id'].values
    cols = data_loader.train_df['item_id'].values
    vals = np.ones(len(rows))
    R = csr_matrix((vals, (rows, cols)), shape=(data_loader.n_users, data_loader.n_items))
    
    U, S, V, total_energy = evd_manager.get_evd(R, k=None, dataset_name=dataset_name)
    
    s_np = S.cpu().numpy()
    if hasattr(data_loader, 'item_popularity'):
        item_pop_np = np.asarray(data_loader.item_popularity, dtype=float)
    else:
        item_pop_np = np.asarray(R.sum(axis=0)).flatten()
        
    v_np = V.cpu().numpy()
    p_tilde = (v_np.T ** 2) @ item_pop_np
    p_tilde = np.clip(p_tilde, 1e-12, None)
    
    # 2. Scanning smooth_sigma
    sigma_values = [0.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    beta_results = []
    r2_results = []
    
    for sigma in sigma_values:
        # We use decoupling (scalar) for sensitivity analysis
        beta, r2 = estimate_beta_ols(s_np, p_tilde, smooth_sigma=sigma)
        beta_results.append(beta)
        r2_results.append(r2)
        print(f"  Sigma: {sigma:>4.1f} -> Beta: {beta:.4f} (R2={r2:.4f})")
    
    # 3. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Sensitivity Curve
    ax = axes[0, 0]
    ax.plot(sigma_values, beta_results, marker='o', color='purple', linewidth=2)
    ax.set_title("Beta Sensitivity to Spectral Smoothing", fontsize=14)
    ax.set_xlabel("Smoothing Sigma (Gaussian filter)", fontsize=12)
    ax.set_ylabel(r"Estimated Scalar $\beta$", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Signal Comparison (Log-S vs Log-P) for Sigma=0 vs Sigma=20
    ax = axes[0, 1]
    log_s_raw = np.log(s_np + 1e-12)
    log_p_raw = np.log(p_tilde + 1e-12)
    log_s_smooth = gaussian_filter1d(log_s_raw, sigma=20.0)
    log_p_smooth = gaussian_filter1d(log_p_raw, sigma=20.0)
    
    ax.plot(log_p_raw, log_s_raw, '.', color='gray', alpha=0.1, label='Raw Signals')
    ax.plot(log_p_smooth, log_s_smooth, 'r-', linewidth=1.5, label='Smoothed (Sigma=20)')
    ax.set_title("Spectral Signal Denoising", fontsize=14)
    ax.set_xlabel(r"$\log p_k$", fontsize=12)
    ax.set_ylabel(r"$\log \sigma_k$", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: Vector Beta (Smooth Vector) Comparison
    ax = axes[1, 0]
    # lambda_smooth=10.0 from previous experiment
    print("  Estimating smooth_vector with sigma=0...")
    bv_raw, _, _ = beta_estimators.smooth_estimate_vector_opt(s_np, p_tilde, lambda_smooth=10.0, smooth_sigma=0.0)
    print("  Estimating smooth_vector with sigma=20...")
    bv_smooth, _, _ = beta_estimators.smooth_estimate_vector_opt(s_np, p_tilde, lambda_smooth=10.0, smooth_sigma=20.0)
    
    k_idx = np.arange(len(bv_raw))
    ax.plot(k_idx, bv_raw, label=r'Beta Vector ($\sigma_{smooth}=0$)', alpha=0.5, color='orange')
    ax.plot(k_idx, bv_smooth, label=r'Beta Vector ($\sigma_{smooth}=20$)', color='red', linewidth=2)
    ax.set_title("Vector Beta with Spectral Smoothing", fontsize=14)
    ax.set_xlabel("Dimension (k)", fontsize=12)
    ax.set_ylabel(r"$\beta_k$", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Restoration Effect (Sigma=0 vs Sigma=20)
    ax = axes[1, 1]
    log_s_raw = np.log(s_np + 1e-12)
    log_p_raw = np.log(p_tilde + 1e-12)
    
    # Restore using scalar beta from sigma=0
    rest_raw = log_s_raw - beta_results[0] * log_p_raw
    # Restore using scalar beta from sigma=20
    # Note: we use RAW signals for the actual restoration plot to see if the restoration is better
    rest_smooth = log_s_raw - beta_results[3] * log_p_raw # sigma=10.0 or 20.0
    idx_target = 3 # sigma=10.0
    beta_target = beta_results[idx_target]
    
    ax.plot(k_idx, log_s_raw, label='Original', color='gray', alpha=0.3, linestyle='--')
    ax.plot(k_idx, rest_raw, label=f'Restored ($\sigma_{{sm}}=0$, $\\beta$={beta_results[0]:.2f})', alpha=0.6)
    ax.plot(k_idx, log_s_raw - beta_target * log_p_raw, label=f'Restored ($\sigma_{{sm}}={sigma_values[idx_target]}$, $\\beta$={beta_target:.2f})', color='blue', linewidth=1.5)
    
    ax.set_title("Restoration Comparison", fontsize=14)
    ax.set_xlabel("Dimension (k)", fontsize=12)
    ax.set_ylabel(r"$\log \sigma_k$", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_dir = "aspire_experiments/output/robust_beta"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp07_robust_{dataset_name}.png")
    plt.savefig(out_path, dpi=200)
    print(f"[Done] Analysis plot saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    run_robust_beta_analysis(args.dataset, args.device)
