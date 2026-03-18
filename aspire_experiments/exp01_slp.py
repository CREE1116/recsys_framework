# Usage: uv run python aspire_experiments/exp1_slp.py --dataset ml1m
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir

def run_slp(dataset_name, seed=42):
    print(f"Running Experiment 1: SLP Verification on {dataset_name} (Full Spectrum, seed={seed})...")
    loader, R, S, V, config = get_loader_and_svd(dataset_name, seed=seed)
    
    # Item popularity
    p = np.array(R.sum(axis=0)).flatten()
    p_norm = p / (p.max() + 1e-9)  # Normalize
    P_diag = p_norm
    
    # M = V^T P V
    V_np = V.cpu().numpy()
    p_vec = p_norm
    full_k = V_np.shape[1]
    
    # Calculate for multiple K as requested
    rho_results = {}
    ks_to_test = [50, 100, 200, 500]
    # Filter out K larger than full_k
    ks_to_test = [k for k in ks_to_test if k <= full_k]
    if full_k not in ks_to_test:
        ks_to_test.append(full_k)
    
    for k in sorted(ks_to_test):
        Vk = V_np[:, :k]
        VTPV_k = (Vk.T * p_vec) @ Vk
        
        diag_k_vals = np.diag(VTPV_k)
        diag_k = np.diag(diag_k_vals)
        offdiag_k = VTPV_k - diag_k
        
        rho_val = float(np.linalg.norm(offdiag_k, 'fro') / (np.linalg.norm(diag_k, 'fro') + 1e-12))
        
        # Mean ratio for this k
        mask_k = ~np.eye(k, dtype=bool)
        m_ratio = float(np.mean(np.abs(offdiag_k[mask_k])) / (np.mean(np.abs(diag_k_vals)) + 1e-12))
        
        rho_results[f"K={k}"] = {"rho": rho_val, "mean_ratio": m_ratio}
        print(f"  K={k:3d}: ρ = {rho_val:.4f}, Mean Ratio = {m_ratio:.4f}")

    # Use full rank results for main summary
    main_rho = rho_results[f"K={full_k}"]["rho"]
    main_epsilon = rho_results[f"K={full_k}"]["mean_ratio"]
    
    # M for visualization (full rank)
    M = (V_np.T * p_vec) @ V_np
    diag_vals = np.diag(M)

    # Output setup
    out_dir = ensure_dir(f"aspire_experiments/output/slp/{config['dataset_name']}")
    
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(M, cmap='viridis')
    plt.title(f"Spectral Popularity Matrix M (Full Rank K={full_k})\nDataset: {config['dataset_name']}, ρ={main_rho:.4f}")
    plt.savefig(os.path.join(out_dir, "slp_heatmap.png"))
    plt.close()
    
    # Log results
    result = {
        "dataset": config['dataset_name'],
        "epsilon": main_epsilon, 
        "rho_frob": main_rho,
        "rank_k": full_k,
        "m_shape": M.shape,
        "multi_k_rho": rho_results
    }
    with open(os.path.join(out_dir, "result.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
        
    # [NEW] Save detailed M data
    pd.DataFrame(M).to_csv(os.path.join(out_dir, "spectral_popularity_matrix.csv"), index=False)
    pd.DataFrame({"index": np.arange(len(diag_vals)), "diag_value": diag_vals}).to_csv(os.path.join(out_dir, "m_diagonal.csv"), index=False)
    
    print(f"  SLP Mean Ratio (Full): {main_epsilon:.4f}")
    print(f"  SLP Rho (Frob, Full): {main_rho:.4f}")
    print(f"  Result saved to {out_dir}")

if __name__ == "__main__":
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    run_slp(args.dataset, seed=args.seed)
