import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure root is in path
sys.path.append(os.getcwd())

from src.data_loader import DataLoader
from src.utils.gpu_accel import EVDCacheManager
from src.models.csar.beta_estimators import estimate_beta_ols
from aspire_experiments.exp_utils import load_config
from scipy.sparse import csr_matrix

def run_beta_trimming_analysis(dataset_name, device="auto"):
    print(f"\n[Exp06] Analyzing Scalar Beta Trimming for {dataset_name}...")
    
    # 1. Load Data & Compute/Load EVD
    config = load_config(dataset_name)
    data_loader = DataLoader(config)
    evd_manager = EVDCacheManager(device=device)
    
    # Need to build R for EVD (interaction matrix)
    rows = data_loader.train_df['user_id'].values
    cols = data_loader.train_df['item_id'].values
    vals = np.ones(len(rows))
    R = csr_matrix((vals, (rows, cols)), shape=(data_loader.n_users, data_loader.n_items))
    
    U, S, V, total_energy = evd_manager.get_evd(R, k=None, dataset_name=dataset_name)
    
    s_np = S.cpu().numpy()
    
    # Get item popularity from DataLoader
    # Note: ensure Data Loader has item_popularity attribute
    if hasattr(data_loader, 'item_popularity'):
        item_pop_np = np.asarray(data_loader.item_popularity, dtype=float)
    else:
        # Fallback: compute from R
        item_pop_np = np.asarray(R.sum(axis=0)).flatten()
        
    v_np = V.cpu().numpy()
    p_tilde = (v_np.T ** 2) @ item_pop_np
    p_tilde = np.clip(p_tilde, 1e-12, None)
    
    # 2. Sequential Scanning of trim_tail
    trim_values = np.linspace(0.0, 0.4, 11) # 0.0, 0.04, ..., 0.4
    beta_results = []
    r2_results = []
    
    for trim in trim_values:
        # 1. Scalar Estimation
        beta_ols, r2_ols = estimate_beta_ols(s_np, p_tilde, trim_tail=trim)
        beta_results.append(beta_ols)
        r2_results.append(r2_ols)
        print(f"  Trim: {trim:.2f} -> Beta: {beta_ols:.4f} (R2={r2_ols:.4f})")
    
    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: sensitivity curve
    ax = axes[0]
    ax.plot(trim_values, beta_results, marker='o', color='teal', linewidth=2)
    ax.set_title("Beta Sensitivity to Tail Trimming", fontsize=14)
    ax.set_xlabel("Trim Proportion (Head & Tail, each)", fontsize=12)
    ax.set_ylabel(r"Estimated Scalar $\beta$ (Decoupling)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Spectral Restoration Comparison
    ax = axes[1]
    log_s = np.log(s_np + 1e-12)
    log_p = np.log(p_tilde + 1e-12)
    k_range = np.arange(len(log_s))
    
    # Original
    ax.plot(k_range, log_s, label='Original $\sigma_k$', color='gray', alpha=0.4, linestyle='--')
    
    # Trim=0.0
    beta_0 = beta_results[0]
    log_s_0 = log_s - beta_0 * log_p
    ax.plot(k_range, log_s_0, label=f'Restored (Trim=0.0, $\\beta$={beta_0:.2f})', alpha=0.7)
    
    # Trim=0.2
    idx_20 = len(trim_values) // 2 # ~0.2
    beta_20 = beta_results[idx_20]
    log_s_20 = log_s - beta_20 * log_p
    ax.plot(k_range, log_s_20, label=f'Restored (Trim={trim_values[idx_20]:.2f}, $\\beta$={beta_20:.2f})', alpha=0.7, linewidth=2)
    
    ax.set_title(f"Spectral Restoration Comparison ({dataset_name})", fontsize=14)
    ax.set_xlabel("Dimension (k)", fontsize=12)
    ax.set_ylabel(r"$\log \sigma_k$", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_dir = "aspire_experiments/output/trimming_analysis"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp06_trimming_{dataset_name}.png")
    plt.savefig(out_path, dpi=200)
    print(f"[Done] Analysis plot saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    run_beta_trimming_analysis(args.dataset, args.device)
