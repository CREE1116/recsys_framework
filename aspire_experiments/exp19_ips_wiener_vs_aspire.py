import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.metrics import mean_squared_error

sys.path.append(os.getcwd())
try:
    from aspire_experiments.exp_utils import ensure_dir
except ImportError:
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)
        return path

def run_exp19_spectral(num_items=1500, latent_dim=30, true_bias=1.0, noise_level=0.2):
    print(f"Running Exp 19: IPS+Wiener vs ASPIRE (Spectral Space Comparison)")
    
    np.random.seed(42)
    # 1. Base Unbiased Space (True Spectrum)
    V_true = np.random.randn(num_items, latent_dim)
    C_true = V_true @ V_true.T
    
    vals_true = eigh(C_true, eigvals_only=True)[::-1]
    lam_true = np.maximum(vals_true, 0)
    
    # Normalize true spectrum to max=1.0 for shape comparison
    lam_true_norm = lam_true / (lam_true[0] + 1e-12)
    
    # 2. Bias and Observation Noise
    item_pop = np.random.pareto(a=1.5, size=num_items) + 1.0
    item_pop = np.sort(item_pop)[::-1]
    
    D_tau = np.diag( (item_pop / item_pop.mean()) ** true_bias )
    E_noise = np.random.randn(num_items, num_items)
    E_noise = (E_noise + E_noise.T) * 0.5 * noise_level * np.std(C_true)
    
    C_obs = D_tau @ C_true @ D_tau + E_noise
    
    # 3. Method A: IPS + Wiener
    # In IPS, because of tiny propensities at the tail, D_inv blows up.
    D_inv = np.diag( 1.0 / (np.diag(D_tau) + 1e-5) ) # Clipping added to prevent infinity
    C_ips = D_inv @ C_obs @ D_inv
    
    vals_ips = eigh(C_ips, eigvals_only=True)[::-1]
    lam_ips = np.maximum(vals_ips, 0)
    
    alphas = np.logspace(-2, 6, 50)
    ips_errors = []
    
    # Metric: Normalized Mean Absolute Error of the Spectral shape
    def spectral_shape_error(lam_pred, lam_target):
        p_norm = lam_pred / (lam_pred[0] + 1e-12)
        return np.mean(np.abs(p_norm - lam_target))
        
    for alpha in alphas:
        h_wiener = lam_ips / (lam_ips + alpha)
        lam_ips_filtered = lam_ips * h_wiener
        err = spectral_shape_error(lam_ips_filtered, lam_true_norm)
        ips_errors.append(err)
        
    best_ips_err = min(ips_errors)
    best_alpha = alphas[np.argmin(ips_errors)]
    
    # Compute the best IPS+Wiener spectrum for plotting
    h_wiener_best = lam_ips / (lam_ips + best_alpha)
    lam_ips_best = lam_ips * h_wiener_best
    lam_ips_best_norm = lam_ips_best / (lam_ips_best[0] + 1e-12)
    
    # 4. Method B: ASPIRE Spectral Penalty
    vals_obs = eigh(C_obs, eigvals_only=True)[::-1]
    lam_obs = np.maximum(vals_obs, 0)
    lam_max = float(lam_obs[0])
    
    gammas = np.linspace(0.0, 3.0, 50)
    aspire_errors = []
    
    for gamma in gammas:
        gamma = float(gamma)
        lam_gamma = lam_obs ** gamma
        lam_max_gamma = lam_max ** gamma
        h_aspire = lam_gamma / (lam_gamma + lam_max_gamma + 1e-10)
        
        lam_aspire_filtered = lam_obs * h_aspire
        err = spectral_shape_error(lam_aspire_filtered, lam_true_norm)
        aspire_errors.append(err)
        
    best_aspire_err = min(aspire_errors)
    best_gamma = gammas[np.argmin(aspire_errors)]
    
    # Compute the best ASPIRE spectrum for plotting
    h_aspire_best = (lam_obs ** best_gamma) / ((lam_obs ** best_gamma) + (lam_max ** best_gamma) + 1e-10)
    lam_aspire_best = lam_obs * h_aspire_best
    lam_aspire_best_norm = lam_aspire_best / (lam_aspire_best[0] + 1e-12)
    
    print(f" [IPS+Wiener] Best Spectral Error: {best_ips_err:.4f} at alpha={best_alpha:.2e}")
    print(f" [ASPIRE]     Best Spectral Error: {best_aspire_err:.4f} at gamma={best_gamma:.2f}")
    
    # 5. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left Plot: Spectral Error Sweep
    axes[0].plot(alphas, ips_errors, color='red', label='IPS + Wiener')
    axes[0].plot(best_alpha, best_ips_err, 'ro', markersize=8)
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r"Wiener Penalty ($\alpha$)")
    axes[0].set_ylabel("Spectral Shape Error (MAE, Lower is better)")
    axes[0].set_title("IPS+Wiener Spectral Recovery Error")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    ax0_twin = axes[0].twiny()
    ax0_twin.plot(gammas, aspire_errors, color='blue', label='ASPIRE')
    ax0_twin.plot(best_gamma, best_aspire_err, 'bo', markersize=8)
    ax0_twin.set_xlabel(r"ASPIRE Penalty ($\gamma$)")
    ax0_twin.legend(loc='upper right')
    
    # Right Plot: The Recovered Spectrum Shapes (Log Scale)
    top_components = min(100, num_items)
    x_rank = np.arange(1, top_components + 1)
    
    lam_obs_norm = lam_obs / (lam_obs[0] + 1e-12)
    
    axes[1].plot(x_rank, lam_true_norm[:top_components], 'k-', linewidth=3, label='True Unbiased Spectrum', zorder=5)
    axes[1].plot(x_rank, lam_obs_norm[:top_components], 'gray', linestyle=':', label='Biased Spectrum (No Filter)')
    axes[1].plot(x_rank, lam_ips_best_norm[:top_components], 'r--', label=rf'Best IPS+Wiener ($\alpha$={best_alpha:.1e})')
    axes[1].plot(x_rank, lam_aspire_best_norm[:top_components], 'b-.', label=rf'Best ASPIRE ($\gamma$={best_gamma:.2f})')
    
    axes[1].set_yscale('log')
    axes[1].set_xlabel("Principal Component Rank (k)")
    axes[1].set_ylabel("Normalized Eigenvalue (Energy)")
    axes[1].set_title("Recovered Spectral Shape (Top 100 Components)")
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    
    out_dir = ensure_dir("aspire_experiments/output/exp19")
    plot_path = os.path.join(out_dir, f"spectral_recovery_noise_{noise_level}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    results = {
        "noise_level": noise_level,
        "best_ips_err": float(best_ips_err),
        "best_ips_alpha": float(best_alpha),
        "best_aspire_err": float(best_aspire_err),
        "best_aspire_gamma": float(best_gamma)
    }
    
    json_path = os.path.join(out_dir, f"spectral_results_noise_{noise_level}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Exp 19 Spectral Analysis finished. Results saved to {out_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_level", type=float, default=0.2)
    args = parser.parse_args()
    
    run_exp19_spectral(noise_level=args.noise_level)
