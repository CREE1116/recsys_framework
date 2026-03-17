# Usage: uv run python aspire_experiments/exp14_spectral_restoration.py --dataset ml1m
import os
import sys
import json
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir
from src.models.csar.ASPIRELayer import AspireEngine
from src.models.csar import beta_estimators

def run_restoration(dataset_name, alpha_reg=500.0):
    print(f"\n[Restoration] Analyzing {dataset_name} (Full Spectrum)...")
    loader, R, S, V, config = get_loader_and_svd(dataset_name)
    
    item_freq = np.array(R.sum(axis=0)).flatten().astype(float)
    s_np = S.cpu().numpy()
    p_tilde = AspireEngine.compute_spp(V, item_freq)
    
    # Estimate Beta & Gamma
    beta, r2_beta = beta_estimators.beta_lad(S, p_tilde)
    gamma = 2.0 / (1.0 + beta)
    
    # Original Spectrum
    k = len(s_np)
    ranks = np.arange(1, k + 1)
    log_ranks = np.log(ranks)
    log_s = np.log(np.clip(s_np, 1e-12, None))
    
    # Restored Spectrum (Scaling effect)
    # We apply the ASPIRE filter: h(s) = s^gamma / (s^gamma + alpha)
    # Effectively, for large s, s_hat = s * (s^gamma / s^gamma) = s?
    # Wait, the theory of POWER-LAW restoration is usually about the power adjustment.
    # Actually, ASPIRE v3 focuses on the " Wiener shrinkage + scale correction "
    
    h = AspireEngine.apply_filter(S, alpha_reg, beta).cpu().numpy()
    s_restored = s_np * h
    log_s_restored = np.log(np.clip(s_restored, 1e-12, None))

    # Visualization
    out_dir = ensure_dir(f"aspire_experiments/output/spectral_restoration/{dataset_name}")
    
    plt.figure(figsize=(10, 7))
    plt.scatter(log_ranks, log_s, color='gray', alpha=0.3, s=10, label='Original Spectrum (Observed)')
    plt.scatter(log_ranks, log_s_restored, color='red', alpha=0.5, s=10, label=f'Restored Spectrum (β={beta:.3f}, γ={gamma:.3f})')
    
    # Fit line to head of original to show "ideal" (undistorted) trend
    head_k = max(20, k // 10)
    A_head = np.column_stack([log_ranks[:head_k], np.ones(head_k)])
    slope_head, ic_head = np.linalg.lstsq(A_head, log_s[:head_k], rcond=None)[0]
    plt.plot(log_ranks, slope_head * log_ranks + ic_head, 'k--', alpha=0.8, label='Head Trend (Ideal Power-law)')
    
    plt.xlabel("log Rank (k)")
    plt.ylabel("log Singular Value (σ_k)")
    plt.title(f"Spectral Restoration Analysis: {dataset_name}\nRestoring Power-law Linearity via ASPIRE Filter")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(out_dir, "spectral_restoration.png"), dpi=150)
    plt.close()
    
    # Tail Lifting Analysis
    tail_k = k // 2
    avg_original_tail = np.mean(log_s[tail_k:])
    avg_restored_tail = np.mean(log_s_restored[tail_k:])
    lift = avg_restored_tail - avg_original_tail
    
    print(f"  Beta  : {beta:.4f}")
    print(f"  Gamma : {gamma:.4f}")
    print(f"  Tail Lift (log scale avg): {lift:.4f}")
    
    # Check linearity (R2) of restored tail vs original tail
    def get_r2(x, y):
        A = np.column_stack([x, np.ones_like(x)])
        res = np.linalg.lstsq(A, y, rcond=None)
        ss_res = res[1][0] if len(res[1]) > 0 else 0
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / (ss_tot + 1e-12)

    r2_orig = get_r2(log_ranks, log_s)
    r2_rest = get_r2(log_ranks, log_s_restored)
    print(f"  R² Original: {r2_orig:.4f}")
    print(f"  R² Restored: {r2_rest:.4f}")
    
    # [NEW] Save detailed spectral data
    pd.DataFrame({
        "rank": ranks,
        "original_s": s_np,
        "restored_s": s_restored,
        "h_filter": h
    }).to_csv(os.path.join(out_dir, "spectral_data.csv"), index=False)

    result = {
        "dataset": dataset_name,
        "beta": float(beta),
        "gamma": float(gamma),
        "r2_original": float(r2_orig),
        "r2_restored": float(r2_rest),
        "tail_lift": float(lift)
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m")
    parser.add_argument("--alpha", type=float, default=590.0) # Approx best for ml1m
    args = parser.parse_args()
    
    run_restoration(args.dataset, args.alpha)

if __name__ == "__main__":
    main()
