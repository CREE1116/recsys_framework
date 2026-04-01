import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.stats import entropy

sys.path.append(os.getcwd())
try:
    from aspire_experiments.exp_utils import ensure_dir
except ImportError:
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)
        return path

def ndcg_at_k(preds, targets, mask_train, k=20):
    preds = preds.copy()
    preds[mask_train > 0] = -np.inf
    sorted_idx = np.argsort(-preds, axis=1)[:, :k]
    ndcg_list = []
    for u in range(preds.shape[0]):
        actual_top = targets[u]
        num_hits = actual_top.sum()
        if num_hits == 0: continue
        ideal_hits = min(int(num_hits), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        if idcg == 0: continue
        dcg = 0.0
        for i, item_idx in enumerate(sorted_idx[u]):
            if actual_top[item_idx] > 0:
                dcg += 1.0 / np.log2(i + 2)
        ndcg_list.append(dcg / idcg)
    return np.mean(ndcg_list) if len(ndcg_list) > 0 else 0.0

def find_knee_point(x, y, curve_type='convex'):
    """
    Finds the knee/elbow point of a curve by measuring the max distance to the connecting line.
    curve_type: 'convex' (e.g. exponential decay), 'concave' (e.g. logarithmic growth)
    """
    x = np.array(x)
    y = np.array(y)
    
    # Normalize
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
    
    # Line connecting first and last point
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    
    distances = []
    for i in range(len(x)):
        p0 = np.array([x_norm[i], y_norm[i]])
        # Distance from point to line formulation
        num = np.abs((p2[1]-p1[1])*p0[0] - (p2[0]-p1[0])*p0[1] + p2[0]*p1[1] - p2[1]*p1[0])
        den = np.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)
        distances.append(num/den)
        
    distances = np.array(distances)
    best_idx = np.argmax(distances)
    return x[best_idx], best_idx

def estimate_powerlaw_gamma(item_freqs):
    """
    Estimates the Zipf/Pareto exponent alpha from log-log item frequencies.
    """
    # Exclude tail sparsity noise by taking top 10% to 50%
    sorted_freqs = np.sort(item_freqs)[::-1]
    sorted_freqs = sorted_freqs[sorted_freqs > 0]
    if len(sorted_freqs) < 10: return 1.0
    
    start_idx = max(int(len(sorted_freqs)*0.05), 1)
    end_idx = int(len(sorted_freqs)*0.3)
    
    Y = np.log(sorted_freqs[start_idx:end_idx])
    X = np.log(np.arange(start_idx, end_idx) + 1)
    
    # y = -alpha * x + c
    slope, _ = np.polyfit(X, Y, 1)
    return abs(slope)

def run_exp21(num_users=2000, num_items=800, latent_dim=30, tau=1.5):
    print(f"Running Exp 21: Zero-cost Theoretical Gamma Estimation vs HPO")
    np.random.seed(42)
    
    # 1. Dataset Generation (Like Exp 20)
    U_true = np.random.randn(num_users, latent_dim)
    V_true = np.random.randn(num_items, latent_dim)
    R_dense = U_true @ V_true.T
    
    R_true = np.zeros_like(R_dense)
    for u in range(num_users):
        idx = np.argsort(-R_dense[u])[:30]
        R_true[u, idx] = 1.0
        
    item_pop = np.random.pareto(a=1.5, size=num_items) + 1.0
    item_pop = np.sort(item_pop)[::-1]
    prob_keep = (item_pop / item_pop.max()) ** tau
    
    R_train = np.copy(R_true)
    for i in range(num_items):
        mask = np.random.rand(num_users) > prob_keep[i]
        R_train[mask, i] = 0.0
        
    R_test = R_true - R_train
    active_users = (R_test.sum(axis=1) > 0) & (R_train.sum(axis=1) > 0)
    R_train = R_train[active_users]
    R_test = R_test[active_users]
    
    # 2. HPO Baseline (Ground Truth NDCG Peak)
    print("Executing HPO Sweep to find Ground Truth Peak...")
    gammas = np.linspace(0.0, 3.0, 60)
    C_obs = R_train.T @ R_train
    vals_obs, vecs_obs = eigh(C_obs)
    vals_obs = np.maximum(vals_obs, 0)
    lam_max = float(vals_obs.max())
    
    ndcgs = []
    ratios = []
    entropies = []
    
    for g in gammas:
        h = (vals_obs ** g) / (vals_obs ** g + lam_max ** g + 1e-10)
        lam_filter = vals_obs * h
        
        # NDCG calculation
        C_aspire = vecs_obs @ np.diag(lam_filter) @ vecs_obs.T
        Pred = R_train @ C_aspire
        ndcg = ndcg_at_k(Pred, R_test, R_train, k=20)
        ndcgs.append(ndcg)
        
        # Theoretical Heuristics Metrics
        ratio_max_energy = lam_filter.max() / (lam_filter.sum() + 1e-10)
        ratios.append(ratio_max_energy)
        
        p_dist = lam_filter / (lam_filter.sum() + 1e-10)
        entropies.append(entropy(p_dist))

    # Real Optimal Gamma
    best_ndcg = max(ndcgs)
    best_gamma_hpo = gammas[np.argmax(ndcgs)]
    print(f" -> HPO Maximum NDCG: {best_ndcg:.4f} at Gamma = {best_gamma_hpo:.2f}")
    
    # 3. Automatic Estimators
    print("\nEvaluating Theoretical Zero-cost Estimators...")
    
    # Estimator 1: Power-Law Tail Index
    item_freqs = R_train.sum(axis=0)
    est_gamma_powerlaw = estimate_powerlaw_gamma(item_freqs)
    # Clip to valid range
    est_gamma_powerlaw = np.clip(est_gamma_powerlaw, 0.0, 3.0)
    print(f" -> [Estimator 1] Power-Law Tail Index : {est_gamma_powerlaw:.2f}")
    
    # Estimator 2: Spectral Energy Ratio Knee Point
    # The ratio curve decays exponentially/convexly. We find the knee.
    est_gamma_elbow, idx_elb = find_knee_point(gammas, ratios, curve_type='convex')
    print(f" -> [Estimator 2] Spectral Ratio Elbow : {est_gamma_elbow:.2f}")
    
    # Estimator 3: Spectral Entropy Max Curvature (Knee Point)
    # Entropy grows concavely. We find the knee.
    est_gamma_entropy, idx_ent = find_knee_point(gammas, entropies, curve_type='concave')
    print(f" -> [Estimator 3] Spectral Entropy Knee: {est_gamma_entropy:.2f}")

    # 4. Visualization & Reporting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot True NDCG Objective Function
    color = 'tab:gray'
    ax1.set_xlabel(r"ASPIRE Penalty ($\gamma$)")
    ax1.set_ylabel('Ground Truth NDCG@20', color='black', fontweight='bold')
    ax1.plot(gammas, ndcgs, color='black', linewidth=3, label='NDCG Validation Curve (HPO target)')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Theoretical Est Points vertical lines
    ax1.axvline(best_gamma_hpo, color='black', linestyle='--', linewidth=2, label=f'True Optimal Peak ({best_gamma_hpo:.2f})')
    
    colors = ['tab:red', 'tab:green', 'tab:blue']
    ax1.axvline(est_gamma_powerlaw, color=colors[0], linestyle='-.', linewidth=2, label=f'Est 1: Power-Law ({est_gamma_powerlaw:.2f})')
    ax1.axvline(est_gamma_elbow, color=colors[1], linestyle='-.', linewidth=2, label=f'Est 2: Energy Elbow ({est_gamma_elbow:.2f})')
    ax1.axvline(est_gamma_entropy, color=colors[2], linestyle='-.', linewidth=2, label=f'Est 3: Entropy Knee ({est_gamma_entropy:.2f})')

    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    
    # Sub axes for tracking heuristics visually
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Heuristic Metric Target', color=color)  
    ax2.plot(gammas, ratios, color='tab:green', linestyle=':', label='Max Energy Ratio')
    ax2.plot(gammas, (entropies - np.min(entropies)) / (np.max(entropies) - np.min(entropies) + 1e-9), color='tab:blue', linestyle=':', alpha=0.5, label='Normalized Entropy')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Zero-cost Gamma Estimation vs Actual Performance (NDCG) Search")
    fig.tight_layout()
    
    out_dir = ensure_dir("aspire_experiments/output/exp21")
    plot_path = os.path.join(out_dir, f"gamma_estimation_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    results = {
        "best_gamma_hpo": float(best_gamma_hpo),
        "est_gamma_powerlaw": float(est_gamma_powerlaw),
        "est_gamma_elbow": float(est_gamma_elbow),
        "est_gamma_entropy": float(est_gamma_entropy)
    }
    json_path = os.path.join(out_dir, f"estimation_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Exp 21 finished. Plots saved to {out_dir}")
    return results

if __name__ == "__main__":
    run_exp21()
