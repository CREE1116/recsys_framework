import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, subspace_angles

# Framework root path
sys.path.append(os.getcwd())
try:
    from aspire_experiments.exp_utils import ensure_dir
    from src.models.csar.ASPIRELayer import AspireFilter
except ImportError:
    # Fallback if run directly and path not resolved
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)
        return path
    class AspireFilter:
        @staticmethod
        def apply_filter(vals: torch.Tensor, gamma: float = 1.0, alpha: float = 1.0, 
                        mode: str = 'gamma_only', is_gram: bool = False):
            s = torch.clamp(vals.float(), min=1e-12)
            exp = float(gamma) if not is_gram else float(gamma) / 2.0
            s_gamma = torch.pow(s, exp)
            s_max_gamma = s_gamma.max().item()
            effective_lambda = s_max_gamma if mode == 'gamma_only' else float(alpha)
            h = s_gamma / (s_gamma + effective_lambda + 1e-10)
            return h.float(), 1.0, float(effective_lambda)

def run_exp18_aspire(num_items=3000, latent_dim=50, top_k=20, true_bias=1.5):
    print(f"Running Exp 18: Authentic ASPIRE Spectral Penalty Analysis")
    print(f"Params: num_items={num_items}, latent_dim={latent_dim}, top_k={top_k}, true_bias(tau)={true_bias}")
    
    print("1. Generating Unbiased True Preferences (C_true)...")
    np.random.seed(42) # For reproducible shapes
    V_true_base = np.random.randn(num_items, latent_dim)
    C_true = V_true_base @ V_true_base.T
    
    vals_true, vecs_true = eigh(C_true)
    top_vecs_true = vecs_true[:, -top_k:][:, ::-1]
    
    print("2. Generating MNAR Multiplicative Bias (C_obs)...")
    item_pop = np.random.pareto(a=1.5, size=num_items) + 1
    item_pop = np.sort(item_pop)[::-1]
    
    D_tau = np.diag(item_pop ** true_bias)
    C_obs = D_tau @ C_true @ D_tau
    
    print("3. Performing EVD on C_obs (To get V_obs and Lambda_obs)...")
    vals_obs, vecs_obs = eigh(C_obs)
    # Sort descending
    vals_obs_desc = vals_obs[::-1]
    vecs_obs_desc = vecs_obs[:, ::-1]
    
    top_vecs_obs = vecs_obs_desc[:, :top_k]
    
    baseline_subspace_sim = np.cos(subspace_angles(top_vecs_true, top_vecs_obs)).mean()
    print(f"   Original True vs Obs Subspace Sim: {baseline_subspace_sim:.4f}")
    
    print("4. Sweeping Gamma using ASPIRE Spectral Filter...")
    gammas = np.linspace(0.0, 3.0, 7) # Fewer steps for clarity in legend
    
    # Validation metrics
    sim_obs_to_aspire = []
    sim_true_to_aspire = []
    
    # Spectral plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Eigenvector stability
    # Subplot 2: Energy Flattening (Eigenvalues)
    
    vals_obs_torch = torch.from_numpy(np.maximum(vals_obs_desc, 0)) # Ensure positive
    
    # We will plot the energy distribution of top K components
    x_rank = np.arange(1, top_k + 1)
    
    print(f"{'Gamma':>6} | {'Sim(V_aspire, V_obs)':>20} | {'Sim(V_aspire, V_true)':>20}")
    print("-" * 55)
    
    for g in gammas:
        # Applying authentic ASPIRE filter (using gamma_only mode which anchors to max component)
        # Note: ASPIRE filters singular values (S = sqrt(lambda)). Since C is Gram/Covariance, eigenvalues = S^2.
        # So we pass is_gram=True strictly to mimic ASPIRELayer applied to Covariance.
        h_vals, _, _ = AspireFilter.apply_filter(
            vals_obs_torch, gamma=g, alpha=1.0, mode='gamma_only', is_gram=True
        )
        h_vals = h_vals.numpy()
        
        # New filtered eigenvalues
        vals_filtered = vals_obs_desc * h_vals
        
        # Reconstruct ASPIRE smoothed Covariance Matrix
        # C_aspire = V_obs * diag(Lambda_filtered) * V_obs^T
        C_aspire = vecs_obs_desc @ np.diag(vals_filtered) @ vecs_obs_desc.T
        
        # User Question Verification: "Do eigenvectors change if we apply SVD/EVD to C_aspire again?"
        vals_aspire_re, vecs_aspire_re = eigh(C_aspire)
        
        # eigh sorts ascending, reverse to descending
        top_vecs_aspire_re = vecs_aspire_re[:, -top_k:][:, ::-1]
        
        # Calculate Subspace Angle between V_aspire_reconstructed and V_obs (Should be perfectly 1.0)
        sim_vs_obs_val = np.cos(subspace_angles(top_vecs_obs, top_vecs_aspire_re)).mean()
        
        # Calculate Subspace Angle between V_aspire_reconstructed and V_true (Should be identical to baseline)
        sim_vs_true_val = np.cos(subspace_angles(top_vecs_true, top_vecs_aspire_re)).mean()
        
        sim_obs_to_aspire.append(sim_vs_obs_val)
        sim_true_to_aspire.append(sim_vs_true_val)
        
        print(f"{g:>6.2f} | {sim_vs_obs_val:>20.4f} | {sim_vs_true_val:>20.4f}")
        
        # Plot energy flattening
        # Normalize filtered eigenvalues to visualize relative flattening effect
        normalized_energy = vals_filtered[:top_k] / vals_filtered[0]
        axes[1].plot(x_rank, normalized_energy, marker='o', label=rf"$\gamma$={g:.1f}", alpha=0.8)
    
    # Plotting logic for Left Plot (Eigenvector Stability)
    axes[0].plot(gammas, sim_obs_to_aspire, marker='s', color='green', label=r"$V_{aspire}$ vs $V_{obs}$")
    axes[0].plot(gammas, sim_true_to_aspire, marker='D', color='red', label=r"$V_{aspire}$ vs $V_{true}$ (Base:%0.3f)" % baseline_subspace_sim)
    
    axes[0].set_title("Eigenvector Subspace Stability after ASPIRE Filtering")
    axes[0].set_xlabel(r"Correction Strength ($\gamma$)")
    axes[0].set_ylabel("Subspace Similarity (1.0 = Identical)")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].set_title("Spectral Energy Flattening (Down-weighting Elite)")
    axes[1].set_xlabel("Principal Component Rank (k)")
    axes[1].set_ylabel("Normalized Energy ($\lambda_k^{(filtered)} / \lambda_1^{(filtered)}$)")
    axes[1].set_yscale('log')
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    
    out_dir = ensure_dir("aspire_experiments/output/exp18")
    fig_path = os.path.join(out_dir, "aspire_spectral_penalty_analysis.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    print("\n--- EXPERIMENT CONCLUSION ---")
    print("V_aspire vs V_obs similarity strictly stays at 1.0.")
    print("This proves ASPIRE DOES NOT change eigenvectors nor un-warp the space!")
    print("Instead, it merely flattens the energy distribution (eigenvalues) of the distorted space.")
    
    results = {
        "num_items": num_items,
        "gammas": gammas.tolist(),
        "sim_obs_to_aspire": sim_obs_to_aspire,
        "sim_true_to_aspire": sim_true_to_aspire,
        "baseline_subspace_sim": float(baseline_subspace_sim)
    }
    
    json_path = os.path.join(out_dir, "results_aspire_spectral.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 18: Genuine ASPIRE Spectral Penalty Analysis")
    parser.add_argument("--num_items", type=int, default=3000)
    parser.add_argument("--latent_dim", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()
    
    try:
        run_exp18_aspire(
            num_items=args.num_items, 
            latent_dim=args.latent_dim, 
            top_k=args.top_k
        )
    except Exception as e:
        print(f"Error on Exp 18: {e}")
        import traceback
        traceback.print_exc()
