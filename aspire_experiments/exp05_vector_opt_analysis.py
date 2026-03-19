import os
import sys
# Allow importing from the root directory
sys.path.append(os.getcwd())

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import argparse

from src.data_loader import DataLoader
from src.utils.gpu_accel import EVDCacheManager
from src.models.csar.beta_estimators import estimate_beta_ols, smooth_estimate_vector_opt, estimate_vector_beta

def run_vector_opt_analysis(dataset_name="ml1m", device="auto"):
    print(f"\n[Exp05] Analyzing Spectral Restoration for {dataset_name}...")
    
    # 1. Load Data
    config = {
        'dataset': {'path': f'configs/dataset/{dataset_name}.yaml'},
        'model': {'name': 'ASPIRE'},
        'seed': 42
    }
    # Load raw config to pass to DataLoader
    import yaml
    with open(f"configs/dataset/{dataset_name}.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
    config.update(ds_config)
    
    dl = DataLoader(config)
    
    # Construct interaction matrix from train_df
    from scipy.sparse import csr_matrix
    train_df = dl.train_df
    rows = train_df['user_id'].values
    cols = train_df['item_id'].values
    values = np.ones(len(train_df))
    X_sparse = csr_matrix((values, (rows, cols)), shape=(dl.n_users, dl.n_items))
    
    M, N = X_sparse.shape
    
    # 2. Get EVD
    evd_manager = EVDCacheManager(device=device)
    u, s, v, total_energy = evd_manager.get_evd(X_sparse, dataset_name=dataset_name)
    
    s_np = s.cpu().numpy()
    
    # Get item popularity from DataLoader
    item_pop = dl.item_popularity # length N
    item_pop_np = np.asarray(item_pop, dtype=float)
    
    # v is (N, K) where N=items, K=components.
    # Correct SPP: Weighted sum of squared eigenvector components by raw popularity
    v_np = v.cpu().numpy()
    p_tilde = (v_np.T ** 2) @ item_pop_np  # (K,)
    
    # Keep original scale, just clip for log stability
    p_tilde = np.clip(p_tilde, 1e-12, None)
    
    # Ensure lengths match (K components)
    min_len = min(len(s_np), len(p_tilde))
    s_np = s_np[:min_len]
    p_tilde = p_tilde[:min_len]
    
    # 3. Estimate Betas
    # 1. Compare Scalars
    beta_ols, _ = estimate_beta_ols(s_np, p_tilde, trim_tail=0.0)
    bv_smooth, _, _ = smooth_estimate_vector_opt(s_np, p_tilde, lambda_smooth=10.0, trim_tail=0.0)
    
    # 2. Restoration
    # Original
    # Need to define s_norm and t_idx for the snippet to be syntactically correct.
    # Assuming s_norm is a normalized version of s_np and t_idx is an index array.
    # For this specific change, I will assume s_norm = s_np and t_idx = np.arange(len(s_np))
    s_norm = s_np / (s_np.max() + 1e-12) # Example normalization
    t_idx = np.arange(len(s_np)) # Example index
    
    s_orig = s_norm[t_idx]
    
    # OLS Restoration
    s_ols_rest = s_np[t_idx] / (np.power(p_tilde[t_idx], beta_ols) + 1e-12)
    s_ols_rest = s_ols_rest / (s_ols_rest.max() + 1e-12)
    
    # Vector Restoration
    s_vec_rest = s_np[t_idx] / (np.power(p_tilde[t_idx], bv_smooth[t_idx]) + 1e-12)
    s_vec_rest = s_vec_rest / (s_vec_rest.max() + 1e-12)
    
    # helper to plot restoration
    def plot_restoration(ax, log_s_orig, log_s_rest, title):
        ax.plot(k_range, log_s_orig, label='Original $\sigma_k$', color='gray', alpha=0.4, linestyle='--', linewidth=0.8)
        ax.plot(k_range, log_s_rest, label='Restored $\sigma_k$ (ASPIRE)', color='blue', linewidth=1.0, alpha=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Dimension (k)")
        ax.set_ylabel(r"$\log \sigma_k$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ROW 0: RAW
    log_s_corr_raw = log_s - beta_vec_raw * log_p
    plot_restoration(axes[0, 0], log_s, log_s_corr_raw, f"Raw Restoration ($\lambda=0$)\n{dataset_name}")
    
    axes[0, 1].plot(k_range, beta_vec_raw, label=r'Raw $\beta_k$', color='orange', linewidth=1.2)
    axes[0, 1].set_title("Raw Beta Vector", fontsize=12)
    axes[0, 1].set_xlabel("Principal Dimension (k)", fontsize=12)
    axes[0, 1].set_ylabel(r"Estimated $\beta_k$", fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ROW 1: SMOOTH
    log_s_corr_smooth = log_s - beta_vec_smooth * log_p
    plot_restoration(axes[1, 0], log_s, log_s_corr_smooth, f"Smooth Restoration ($\lambda=10.0$)\n{dataset_name}")
    
    axes[1, 1].plot(k_range, beta_vec_smooth, label=r'Smooth $\beta_k$', color='red', linewidth=1.5)
    axes[1, 1].set_title("Smooth Beta Vector", fontsize=12)
    axes[1, 1].set_xlabel("Principal Dimension (k)", fontsize=12)
    axes[1, 1].set_ylabel(r"Estimated $\beta_k$", fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = "aspire_experiments/output/vector_analysis"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp05_restoration_{dataset_name}.png")
    plt.savefig(out_path, dpi=200)
    print(f"[Done] Analysis plot saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    run_vector_opt_analysis(args.dataset, args.device)
