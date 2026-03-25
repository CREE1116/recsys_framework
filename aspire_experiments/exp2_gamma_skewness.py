import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from tqdm import tqdm
from scipy.sparse import csr_matrix

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import load_config, ensure_dir, get_eval_config
from aspire_experiments.proof_models import ASPIRE_Test
from src.data_loader import DataLoader
from src.evaluation import evaluate_metrics

def estimate_zeta(loader):
    """Estimate Zipf index zeta from item popularity."""
    item_pop = loader.item_popularity
    sorted_pop = np.sort(item_pop)[::-1]
    sorted_pop = sorted_pop[sorted_pop > 0]
    ranks = np.arange(1, len(sorted_pop) + 1)
    
    log_r = np.log(ranks)
    log_p = np.log(sorted_pop)
    m, _ = np.polyfit(log_r, log_p, 1)
    return -m

def run_gap_analysis(dataset_name):
    print(f"Running Exp 2: Spectral Symmetry & Gap Analysis on {dataset_name}...")
    config = load_config(dataset_name)
    loader = DataLoader(config)
    
    # 1. Theoretical Gamma (Bridge Lemma)
    zeta = estimate_zeta(loader)
    gamma_theory = 2.0 - zeta
    
    # 2. Get Singular Values (Original)
    # We'll use the pre-calculated EVD/SVD for speed if possible, 
    # but here let's just use the loader's svd logic via R_sparse.
    rows = loader.train_df['user_id'].values
    cols = loader.train_df['item_id'].values
    R = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(loader.n_users, loader.n_items))
    
    # Compute gram matrix for singular values (sigma^2)
    if R.shape[1] <= R.shape[0]:
        G = (R.T @ R).toarray().astype(np.float32)
    else:
        G = (R @ R.T).toarray().astype(np.float32)
    
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.sort(eigvals)[::-1]
    sig_orig = np.sqrt(np.maximum(eigvals, 1e-12))
    
    # Take top 95% energy for analysis (same as Exp 1)
    energy = np.cumsum(sig_orig**2) / np.sum(sig_orig**2)
    k_95 = np.searchsorted(energy, 0.95) + 1
    sig_orig_top = sig_orig[:k_95]
    ranks = np.arange(1, k_95 + 1)
    
    # 3. Spectral Correction
    # Filter: sigma_new = sigma_old / (sigma_old^gamma) = sigma_old^(1-gamma)
    gamma_best = 0.1 # Known empirical best
    sig_theory = sig_orig_top**(1.0 - gamma_theory)
    sig_best = sig_orig_top**(1.0 - gamma_best)
    
    # 4. Measure Symmetry (Slope in log-log space)
    def get_slope(s):
        log_r = np.log(np.arange(1, len(s) + 1))
        log_s = np.log(s + 1e-12)
        m, _ = np.polyfit(log_r, log_s, 1)
        return m

    slope_orig = get_slope(sig_orig_top)
    slope_theory = get_slope(sig_theory)
    slope_best = get_slope(sig_best)

    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, sig_orig_top / sig_orig_top[0], label=f"Original (slope={slope_orig:.2f})", alpha=0.7)
    plt.plot(ranks, sig_theory / sig_theory[0], label=f"Theory-Corrected (\u03b3={gamma_theory:.2f}, slope={slope_theory:.2f})", linewidth=2)
    plt.plot(ranks, sig_best / sig_best[0], label=f"Empirical-Best (\u03b3={gamma_best:.2f}, slope={slope_best:.2f})", linestyle='--')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Normalized Singular Value')
    plt.title(f'Spectral Symmetry Gap Analysis: {dataset_name}')
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    
    out_dir = ensure_dir(f"aspire_experiments/output/exp2/{dataset_name}")
    plt.savefig(os.path.join(out_dir, "spectral_symmetry_gap.png"), dpi=150)
    plt.close()
    
    results = {
        "dataset": dataset_name,
        "zeta": float(zeta),
        "gamma_theory": float(gamma_theory),
        "gamma_best": float(gamma_best),
        "slopes": {
            "original": float(slope_orig),
            "theory": float(slope_theory),
            "best": float(slope_best)
        },
        "analysis": (
            f"The theoretical gamma ({gamma_theory:.2f}) aims for perfect spectral flatness (slope ~ 0). "
            f"However, the empirical best ({gamma_best:.2f}) maintains some bias (slope {slope_best:.2f}). "
            "This suggests that in extremely sparse data, complete restoration of tail signals "
            "introduces more noise than useful collaborative information."
        )
    }
    
    with open(os.path.join(out_dir, "gap_analysis_spectral.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Exp 2 Spectral Gap Analysis finished. Theory Slope: {slope_theory:.4f}, Best Slope: {slope_best:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k", help="Dataset name")
    args = parser.parse_args()
    
    run_gap_analysis(args.dataset)
