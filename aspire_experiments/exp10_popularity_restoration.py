# Usage: uv run python aspire_experiments/exp10_popularity_restoration.py --dataset ml1m --energy 0.99
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

def run_popularity_restoration(dataset_name, target_energy=0.99):
    print(f"\n[Exp 10] Theoretical Spectral Restoration Visualization on {dataset_name}...")
    
    loader, R, S, V, config = get_loader_and_svd(dataset_name, target_energy=target_energy)
    
    item_pops = np.array(R.sum(axis=0)).flatten().astype(float)
    # Normalize popularity to avoid extreme log values
    item_pops = item_pops / (item_pops.max() + 1e-12)
    
    s_np = S.cpu().numpy()
    # Normalize S such that sigma_1 = 1 for better exponent visualization
    s_norm = s_np / (s_np.max() + 1e-12)
    
    p_tilde = AspireEngine.compute_spp(V, item_pops)
    p_tilde = p_tilde / (p_tilde.max() + 1e-12)
    
    # 1. Estimate Beta (Theoretical Bias Factor)
    # a = slope_n / slope_s = 1 + beta
    # beta = a - 1
    beta, r2_beta = beta_estimators.beta_lad(S, p_tilde)
    gamma = 2.0 / (1.0 + beta)
    
    # 2. THEORETICAL RESTORATION
    # spectral bias says sigma_k = sigma^*_k * p_k^0.5 (approximately)
    # restoration: sigma^*_k = sigma_k / p_k^0.5
    
    # We use the estimated beta to be more precise: 
    # log p = a*log sigma => log sigma = (1/a)*log p
    # To decouple, we need sigma_rest = sigma / p^0.5
    # (Since sigma_obs = sigma_true * p^0.5 under Assumption A)
    rest_exponent = 0.5
    s_restored_decoupled = s_np / (np.power(p_tilde, rest_exponent) + 1e-12)
    s_restored_decoupled /= s_restored_decoupled.max()
    
    # Also show the ASPIRE filter effect (shrinkage + scale correction)
    alpha_viz = 500.0
    h = AspireEngine.apply_filter(S, alpha_viz, beta).cpu().numpy()
    s_aspire = s_np * h
    
    # 3. Visualization
    out_dir = ensure_dir(f"aspire_experiments/output/popularity_restoration/{dataset_name}")
    
    log_pt = np.log10(np.clip(p_tilde, 1e-12, None))
    log_s_orig = np.log10(np.clip(s_norm, 1e-12, None))
    log_s_rest = np.log10(np.clip(s_restored_decoupled, 1e-12, None))
    log_s_aspire = np.log10(np.clip(s_aspire / s_aspire.max(), 1e-12, None))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # PLOT 1: De-coupling Analysis (Singular Value vs Spectral Propensity)
    ax = axes[0]
    ax.scatter(log_pt, log_s_orig, color='blue', alpha=0.3, s=15, label='Original (Biased Coupling)')
    ax.scatter(log_pt, log_s_rest, color='green', alpha=0.3, s=15, label='Theoretical Decoupled ($\sigma/p^{0.5}$)')
    
    # Linear Fit for original coupling
    mask = log_pt > np.percentile(log_pt, 10)
    z_o = np.polyfit(log_pt[mask], log_s_orig[mask], 1)
    ax.plot(log_pt, np.poly1d(z_o)(log_pt), "b--", alpha=0.8, label=f'Orig Slope: {z_o[0]:.3f}')
    
    # Linear Fit for restored coupling
    z_r = np.polyfit(log_pt[mask], log_s_rest[mask], 1)
    ax.plot(log_pt, np.poly1d(z_r)(log_pt), "g--", alpha=0.8, label=f'Decoupled Slope: {z_r[0]:.3f} (→0)')
    
    ax.set_xlabel(r"log10(Spectral Propensity $\tilde{p}_k$)")
    ax.set_ylabel(r"log10(Singular Value $\sigma_k$)")
    ax.set_title("Spectral Bias De-coupling\n(Relationship between singular energy and direction popularity)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    # PLOT 2: Power-law Restoration (Scree Plot: Rank vs Singular Value)
    ax = axes[1]
    ranks = np.arange(1, len(s_np)+1)
    log_r = np.log10(ranks)
    
    # Zipf-like plot
    ax.plot(log_r, log_s_orig, color='blue', alpha=0.4, label='Observed (Distorted Tail)')
    ax.plot(log_r, log_s_aspire, color='red', alpha=0.6, label='ASPIRE Filtered (Wiener Shrinkage)')
    ax.plot(log_r, log_s_rest, color='green', alpha=0.8, label='Theoretical Restoration (Tail Lifted)')
    
    # Theoretical trend from Head (Head is usually less distorted relative to True index)
    head_k = max(20, len(s_np)//10)
    z_h = np.polyfit(log_r[:head_k], log_s_orig[:head_k], 1)
    alpha_h = abs(z_h[0])
    ax.plot(log_r, np.poly1d(z_h)(log_r), "k--", alpha=0.5, label='Theoretical Power-law (Head Trend)')
    
    ax.set_xlabel("log10(Rank $k$)")
    ax.set_ylabel(r"log10(Singular Value $\sigma_k$)")
    ax.set_title("Spectral Power-law Restoration\n(Lifting the tail to match the signal's power-law index)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    plt.suptitle(f"ASPIRE Theoretical Restoration Visualization ({dataset_name})\n"
                 f"Bias Factor β={beta:.3f} | True Alpha ≈ {alpha_h/(1+beta):.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spectral_restoration_comparison.png"), dpi=150)
    plt.close()
    
    print(f"  [Done] Visualization saved to {out_dir}")
    
    # Final check: Correlation decrease
    corr_orig = np.corrcoef(log_pt, log_s_orig)[0, 1]
    corr_rest = np.corrcoef(log_pt, log_s_rest)[0, 1]
    
    result = {
        "dataset": dataset_name,
        "beta": float(beta),
        "correlation_original": float(corr_orig),
        "correlation_restored": float(corr_rest),
        "slope_original": float(z_o[0]),
        "slope_restored": float(z_r[0])
    }
    
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
        
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--energy", type=float, default=0.99)
    args = parser.parse_args()
    
    run_popularity_restoration(args.dataset, args.energy)
