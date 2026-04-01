import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, subspace_angles

# Framework root path
sys.path.append(os.getcwd())
try:
    from aspire_experiments.exp_utils import ensure_dir
except ImportError:
    # Fallback if run directly and path not resolved
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)
        return path

def run_exp17(num_items=5000, latent_dim=100, top_k=50, bias_strength=2.0):
    print(f"Running Exp 17: Eigenvector Stability under MNAR Multiplicative Bias...")
    print(f"Params: num_items={num_items}, latent_dim={latent_dim}, top_k={top_k}, bias_strength={bias_strength}")
    
    print("1. Generating True Preference Matrix with Power-law Popularity...")
    # Assume item popularity follows a power-law (Pareto distribution)
    item_pop = np.random.pareto(a=1.5, size=num_items) + 1
    item_pop = np.sort(item_pop)[::-1] # Hub items first (Descending order)
    
    # Generate base latent factor matrix weighted by popularity
    V_true_base = np.random.randn(num_items, latent_dim)
    V_true_base = V_true_base * np.sqrt(item_pop)[:, np.newaxis]
    
    # True covariance matrix C_true
    C_true = V_true_base @ V_true_base.T
    
    print("2. Generating MNAR Multiplicative Bias Matrix D...")
    # Multiplicative exposure bias proportional to popularity
    propensity = item_pop ** bias_strength 
    D = np.diag(propensity)
    
    print("3. Generating Observed Covariance Matrix C_obs (C_obs = D * C_true * D)...")
    C_obs = D @ C_true @ D
    
    print("4. Performing Eigenvalue Decomposition (EVD)...")
    vals_true, vecs_true = eigh(C_true)
    vals_obs, vecs_obs = eigh(C_obs)
    
    # Extract Top-K eigenvectors (descending order)
    top_vecs_true = vecs_true[:, -top_k:][:, ::-1]
    top_vecs_obs = vecs_obs[:, -top_k:][:, ::-1]
    
    print("5. Calculating Cosine Similarity and Subspace Angles...")
    # 1) Cosine similarity for 1:1 matching eigenvectors
    cosine_sims = np.abs(np.sum(top_vecs_true * top_vecs_obs, axis=0))
    
    # 2) Subspace preservation
    angles = subspace_angles(top_vecs_true, top_vecs_obs)
    subspace_sim = np.cos(angles) 
    
    mean_cos_top10 = np.mean(cosine_sims[:10])
    mean_sub_top10 = np.mean(subspace_sim[:10])
    
    print(f"Mean Cosine Sim (Top 10): {mean_cos_top10:.4f}")
    print(f"Mean Subspace Sim (Top 10): {mean_sub_top10:.4f}")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(range(1, top_k + 1), cosine_sims, marker='o', linestyle='-', color='b')
    axes[0].set_title(f"Top-{top_k} Eigenvector Cosine Similarity")
    axes[0].set_xlabel("Eigenvector Index (k)")
    axes[0].set_ylabel("Absolute Cosine Similarity")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True)
    
    axes[1].plot(range(1, len(subspace_sim) + 1), subspace_sim, marker='s', linestyle='-', color='r')
    axes[1].set_title(f"Subspace Similarity (cos(theta))")
    axes[1].set_xlabel("Principal Angle Index")
    axes[1].set_ylabel("Similarity (1 = Identical)")
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Save results
    out_dir = ensure_dir("aspire_experiments/output/exp17")
    plt.savefig(os.path.join(out_dir, f"eigenvector_stability_bias_{bias_strength}.png"), dpi=150)
    plt.close()
    
    results = {
        "num_items": num_items,
        "latent_dim": latent_dim,
        "top_k": top_k,
        "bias_strength": float(bias_strength),
        "mean_cos_sim_top10": float(mean_cos_top10),
        "mean_subspace_sim_top10": float(mean_sub_top10),
        "cosine_sims": cosine_sims.tolist(),
        "subspace_sims": subspace_sim.tolist()
    }
    
    with open(os.path.join(out_dir, f"results_bias_{bias_strength}.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Exp 17 finished. Results saved to {out_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 17: Eigenvector Stability under MNAR Multiplicative Bias")
    parser.add_argument("--num_items", type=int, default=3000, help="Number of items")
    parser.add_argument("--latent_dim", type=int, default=50, help="Latent dimension size")
    parser.add_argument("--top_k", type=int, default=20, help="Top K eigenvectors to analyze")
    parser.add_argument("--bias_strength", type=float, default=1.5, help="Strength of MNAR popularity bias")
    args = parser.parse_args()
    
    try:
        run_exp17(
            num_items=args.num_items, 
            latent_dim=args.latent_dim, 
            top_k=args.top_k, 
            bias_strength=args.bias_strength
        )
    except Exception as e:
        print(f"Error on Exp 17: {e}")
        import traceback
        traceback.print_exc()
