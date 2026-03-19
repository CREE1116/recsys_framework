import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.sparse import csr_matrix

# Ensure root is in path
sys.path.append(os.getcwd())

from src.data_loader import DataLoader
from src.utils.gpu_accel import EVDCacheManager
from src.models.csar.beta_estimators import estimate_beta_ols
from aspire_experiments.exp_utils import load_config

def calculate_gini(x):
    """Calculates Gini coefficient of a distribution."""
    # Handle non-positive values
    x = np.maximum(x, 1e-12)
    n = len(x)
    x = np.sort(x)
    index = np.arange(1, n + 1)
    return ((2 * index - n - 1) * x).sum() / (n * x.sum())

def run_debiasing_impact_analysis(dataset_name, device="auto"):
    print(f"\n[Exp08] Analyzing Debiasing Impact (Popularity Redistribution) for {dataset_name}...")
    
    # 1. Load Data & Compute EVD
    config = load_config(dataset_name)
    data_loader = DataLoader(config)
    evd_manager = EVDCacheManager(device=device)
    
    rows = data_loader.train_df['user_id'].values
    cols = data_loader.train_df['item_id'].values
    vals = np.ones(len(rows))
    R = csr_matrix((vals, (rows, cols)), shape=(data_loader.n_users, data_loader.n_items))
    
    U, S, V, _ = evd_manager.get_evd(R, k=None, dataset_name=dataset_name)
    
    s_np = S.cpu().numpy()
    if hasattr(data_loader, 'item_popularity'):
        item_pop_orig = np.asarray(data_loader.item_popularity, dtype=float)
    else:
        item_pop_orig = np.asarray(R.sum(axis=0)).flatten().astype(float)
        
    v_np = V.cpu().numpy()
    p_tilde = (v_np.T ** 2) @ item_pop_orig
    p_tilde = np.clip(p_tilde, 1e-12, None)
    
    # 2. ASPIRE (OLS)
    beta, _ = estimate_beta_ols(S, p_tilde)
    print(f"  Estimated Beta: {beta:.4f}")
    h = AspireEngine.apply_filter(S, 500.0, beta).to(device)
    
    # 4. Compute Restored Item Popularity
    # Original scores: P = U @ S @ V.T
    # Item pop (original SVD): 1_u.T @ P = (1_u.T @ U) @ S @ V.T
    # Restored scores: P_rest = U @ S_rest @ V.T
    # Item pop (restored): 1_u.T @ P_rest = (1_u.T @ U) @ S_rest @ V.T
    
    u_sum = U.sum(dim=0).cpu().numpy() # (K,)
    
    # Restored Singular Values
    s_rest = s_np * (p_tilde ** (-beta))
    
    # Restored Popularity Profile
    pop_rest = (u_sum * s_rest) @ v_np.T # (N_items,)
    # Non-negativity via Shifting (User Suggestion)
    pop_min = pop_rest.min()
    pop_rest = pop_rest - pop_min + 1e-12
    pop_rest = pop_rest / pop_rest.sum() * item_pop_orig.sum()
    
    # 5. Metrics
    gini_orig = calculate_gini(item_pop_orig)
    gini_rest = calculate_gini(pop_rest)
    
    print(f"  Gini Coefficient: {gini_orig:.4f} (Original) -> {gini_rest:.4f} (Restored)")
    
    # 6. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Popularity Curves (Sorted)
    ax = axes[0]
    p_orig_sorted = np.sort(item_pop_orig)[::-1]
    p_rest_sorted = np.sort(pop_rest)[::-1]
    
    ax.plot(p_orig_sorted, label=f'Original (Gini={gini_orig:.3f})', color='gray', linewidth=2)
    ax.plot(p_rest_sorted, label=f'Restored (Gini={gini_rest:.3f})', color='red', linewidth=2)
    ax.set_yscale('log')
    ax.set_title(f"Item Popularity Distribution ({dataset_name})", fontsize=14)
    ax.set_xlabel("Items (Sorted by Rank)", fontsize=12)
    ax.set_ylabel("Exposure Score (Log Scale)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Lorenz Curve style sharing
    ax = axes[1]
    def get_lorenz(x):
        x = np.sort(x)
        cumsum = np.cumsum(x)
        return cumsum / cumsum[-1]
    
    lorenz_orig = get_lorenz(item_pop_orig)
    lorenz_rest = get_lorenz(pop_rest)
    u_points = np.linspace(0, 1, len(lorenz_orig))
    
    ax.plot(u_points, lorenz_orig, label='Original', color='gray')
    ax.plot(u_points, lorenz_rest, label='Restored (ASPIRE)', color='red', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Equality')
    
    ax.set_title("Exposure Lorenz Curve", fontsize=14)
    ax.set_xlabel("Cumulative Share of Items", fontsize=12)
    ax.set_ylabel("Cumulative Share of Exposure", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_dir = "aspire_experiments/output/debiasing_impact"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exp08_impact_{dataset_name}.png")
    plt.savefig(out_path, dpi=200)
    print(f"[Done] Analysis plot saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    run_debiasing_impact_analysis(args.dataset, args.device)
