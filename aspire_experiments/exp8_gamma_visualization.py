import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir

def visualize_gamma_inflation(dataset_name="ml100k"):
    print(f"\n{'='*60}")
    print(f"EXP8: Gamma-only Spectral Inflation Visualization ({dataset_name})")
    print(f"{'='*60}")
    
    # Load original singular values
    loader, R, S, V, config = get_loader_and_svd(dataset_name, k=None)
    
    # Normalize S (Relative scale)
    s_max = S[0].item()
    s_tilde = S.cpu().numpy() / s_max
    
    # Calculate Item Popularity Correlation
    # Popularity vector (item freq)
    item_freq = np.array(R.sum(axis=0)).flatten()
    # Normalize popularity for correlation
    item_freq_norm = (item_freq - item_freq.mean()) / (item_freq.std() + 1e-10)
    
    # Correlations for top 500 components (or all if less)
    n_plot = min(500, len(S))
    corrs = []
    V_np = V.cpu().numpy() # [items, components]
    for k in range(n_plot):
        v_k = V_np[:, k]
        v_k_norm = (v_k - v_k.mean()) / (v_k.std() + 1e-10)
        corr = np.mean(v_k_norm * item_freq_norm)
        corrs.append(abs(corr)) # Use absolute to show association strength
        
    plt.figure(figsize=(18, 5))
    
    # Subplot 1: Real Spectrum & Popularity Correlation
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(range(n_plot), S.cpu().numpy()[:n_plot], color='blue', alpha=0.6, label="Singular Value ($s_k$)")
    ax1.set_xlabel("Component Index ($k$)")
    ax1.set_ylabel("Singular Value", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.bar(range(n_plot), corrs, color='orange', alpha=0.3, label="|Corr(v_k, Pop)|")
    ax2.set_ylabel("Abs Correlation with Popularity", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_title(f"Spectrum vs Popularity Correlation ({dataset_name})")
    
    # Subplot 2: Spectral Inflation (s^gamma Transformation)
    plt.subplot(1, 3, 2)
    betas = [1, 2, 4, 9] # beta=9 => gamma=0.1
    for b in betas:
        g = 1.0 / (1.0 + b)
        plt.plot(range(n_plot), s_tilde[:n_plot]**g, label=rf"$\beta$={b} ($\gamma$={g:.2f})")
    
    # Original Reference (beta=0 => gamma=1.0)
    plt.plot(range(n_plot), s_tilde[:n_plot], color='black', linewidth=3, alpha=0.8, label=r"$\beta$=0 ($\gamma$=1.0)")
    
    plt.axhline(1.0, color='red', linestyle='--', alpha=0.3, label="Cut-off Reference (1.0)")
    plt.xlabel(r"Component Index ($k$)")
    plt.ylabel(r"Normalized Signal ($\tilde{s}_k^\gamma$)")
    plt.title(r"Spectral Inflation ($s \rightarrow s^\gamma$)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Subplot 3: Filter Response h(k) = s^g / (1 + s^g)
    plt.subplot(1, 3, 3)
    for b in betas + [0]:
        g = 1.0 / (1.0 + b)
        h = (s_tilde**g) / (1.0 + s_tilde**g)
        style = '-' if b != 0 else '--'
        color = None if b != 0 else 'black'
        plt.plot(range(n_plot), h[:n_plot], label=rf"$\beta$={b}", linestyle=style, color=color)
        
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label="h=0.5 (Signal Peak)")
    plt.xlabel(r"Component Index ($k$)")
    plt.ylabel(r"Filter Coefficient $h_k$")
    plt.title(r"Filter Coefficients ($h_k = \frac{\tilde{s}^\gamma}{1 + \tilde{s}^\gamma}$)")
    plt.legend()
    plt.grid(alpha=0.3)
    
    output_dir = ensure_dir(f"aspire_experiments/output/exp8/{dataset_name}")
    save_path = os.path.join(output_dir, "gamma_visualization.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n[DONE] Visualization saved to {save_path}")
    
    # Print Inflation Statistics (Real Data Distribution)
    print("\n[Real Data Statistics] Mean Energy Inflation & Spectral Entropy:")
    base_mean = np.mean(s_tilde)
    gamma_stats = []
    
    # Analyze with g=2.0 (Wiener Filter) too
    for b in sorted(betas + [0, -0.5]): # beta=-0.5 => gamma=2.0
        g = 1.0 / (1.0 + b)
        inflated_s = s_tilde**g
        h = inflated_s / (1.0 + inflated_s + 1e-10)
        
        # Calculate Effective Energy: (s * h)^2
        eff_energy = (s_tilde * h)**2
        # Normalize to probability distribution for Entropy
        p_k = eff_energy / (np.sum(eff_energy) + 1e-12)
        # Spectral Entropy
        entropy = -np.sum(p_k * np.log(p_k + 1e-12))
        # Normalized Entropy (0 to 1)
        norm_entropy = entropy / np.log(len(s_tilde))
        
        inflated_mean = np.mean(inflated_s)
        factor = inflated_mean / base_mean
        
        mode_label = f"Beta {b}" if b >= 0 else "Gamma 2.0 (Wiener)"
        print(f"  {mode_label:18} | Factor: {factor:6.2f}x | Entropy: {norm_entropy:.4f}")
        
        gamma_stats.append({
            "beta": float(b),
            "gamma": float(g),
            "inflation_factor": float(factor),
            "spectral_entropy": float(norm_entropy)
        })
    
    # Save to JSON
    import json
    stats_path = os.path.join(output_dir, "results.json")
    with open(stats_path, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "base_mean": float(base_mean),
            "beta_stats": gamma_stats,
            "timestamp": datetime.now().isoformat()
        }, f, indent=4)
    print(f"Stats JSON saved to {stats_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k", help="Dataset name (ml100k, ml1m, steam, etc.)")
    args = parser.parse_args()
    
    visualize_gamma_inflation(args.dataset)
