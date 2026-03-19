import os
import sys
import json
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir, get_trimmed_data
from src.models.csar.ASPIRELayer import AspireEngine
from src.models.csar import beta_estimators

def run_popularity_restoration(dataset_name):
    print(f"\n[Exp 04] Spectral Restoration Comparison (All Estimators) on {dataset_name}...")
    
    loader, R, S, V, config = get_loader_and_svd(dataset_name)
    
    item_pops = np.array(R.sum(axis=0)).flatten().astype(float)
    item_pops = item_pops / (item_pops.max() + 1e-12)
    s_np = S.cpu().numpy()
    s_norm = s_np / (s_np.max() + 1e-12)
    p_tilde = AspireEngine.compute_spp(V, item_pops)
    p_tilde = p_tilde / (p_tilde.max() + 1e-12)
    
    # Trim for visualization
    indices = np.arange(len(p_tilde))
    t_idx, _ = get_trimmed_data(indices, indices)
    
    log_pt = np.log10(np.clip(p_tilde[t_idx], 1e-12, None))
    log_s_orig = np.log10(np.clip(s_norm[t_idx], 1e-12, None))
    
    # 1. Get all estimators
    estimates = beta_estimators.estimate_all(s_np, p_tilde, item_freq=item_pops, n_items=R.shape[0])
    methods = ["ols", "lad", "blue", "smooth_vector", "max_median", "iso_pop_no_detrend"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    all_metrics = []
    
    for i, method in enumerate(methods):
        ax = axes[i]
        if method not in estimates:
            ax.set_title(f"Method {method} not found")
            continue
            
        beta_val, r2 = estimates[method][0], estimates[method][1]
        
        # Restoration
        if isinstance(beta_val, np.ndarray):
            # For vector_opt, use the actual per-dimension beta
            beta_active = beta_val[t_idx]
            s_rest = s_np[t_idx] / (np.power(p_tilde[t_idx], beta_active) + 1e-12)
            mean_beta = np.mean(beta_val)
        else:
            s_rest = s_np[t_idx] / (np.power(p_tilde[t_idx], beta_val) + 1e-12)
            mean_beta = beta_val
            
        s_rest = s_rest / (s_rest.max() + 1e-12)
        log_s_rest = np.log10(np.clip(s_rest, 1e-12, None))
        
        # Plotting
        ax.scatter(log_pt, log_s_orig, color='blue', alpha=0.3, s=10, label='Original')
        ax.scatter(log_pt, log_s_rest, color='green', alpha=0.3, s=10, label='Restored')
        
        # Fitting lines
        z_o = np.polyfit(log_pt, log_s_orig, 1)
        z_r = np.polyfit(log_pt, log_s_rest, 1)
        
        ax.plot(log_pt, np.poly1d(z_o)(log_pt), "b--", alpha=0.8, label=f'Orig Slope: {z_o[0]:.3f}')
        ax.plot(log_pt, np.poly1d(z_r)(log_pt), "g--", alpha=0.8, label=f'Rest Slope: {z_r[0]:.3f}')
        
        ax.set_title(f"Method: {method.upper()}\n(Beta={mean_beta:.3f}, Corr_rest={np.corrcoef(log_pt, log_s_rest)[0, 1]:.3f})")
        ax.set_xlabel("log10(Spectral Propensity)")
        ax.set_ylabel("log10(Singular Value)")
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        all_metrics.append({
            "method": method,
            "beta": float(mean_beta),
            "orig_slope": float(z_o[0]),
            "rest_slope": float(z_r[0])
        })

    plt.suptitle(f"Spectral Restoration Comparison across Estimators ({dataset_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_dir = ensure_dir(f"aspire_experiments/output/popularity_restoration/{dataset_name}")
    plt.savefig(os.path.join(out_dir, "multi_estimator_restoration.png"), dpi=150)
    plt.close()
    
    with open(os.path.join(out_dir, "multi_estimator_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"  [Done] Visualization saved to {out_dir}")
    return all_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    args = parser.parse_args()
    
    run_popularity_restoration(args.dataset)
