import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

sys.path.append(os.getcwd())
try:
    from aspire_experiments.exp_utils import ensure_dir
except ImportError:
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)
        return path

def ndcg_at_k(preds, targets, mask_train, k=20):
    """
    preds: (num_users, num_items) array of predicted scores
    targets: (num_users, num_items) array of true binary interactions
    mask_train: (num_users, num_items) array of observed training interactions
    """
    # Exclude training items from evaluation
    preds = preds.copy()
    preds[mask_train > 0] = -np.inf
    
    # Get top K indices for each user
    sorted_idx = np.argsort(-preds, axis=1)[:, :k]
    
    ndcg_list = []
    for u in range(preds.shape[0]):
        actual_top = targets[u]
        num_hits = actual_top.sum()
        if num_hits == 0:
            continue
            
        ideal_hits = min(int(num_hits), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        
        if idcg == 0:
            continue
            
        dcg = 0.0
        for i, item_idx in enumerate(sorted_idx[u]):
            if actual_top[item_idx] > 0:
                dcg += 1.0 / np.log2(i + 2)
                
        ndcg_list.append(dcg / idcg)
        
    return np.mean(ndcg_list) if len(ndcg_list) > 0 else 0.0

def run_exp20_proxy_ndcg(num_users=2000, num_items=800, latent_dim=30, top_k=20, tau=1.5):
    print(f"Running Exp 20: Proxy IPS vs ASPIRE Ranking Performance (NDCG@{top_k})")
    print(f"Params: users={num_users}, items={num_items}, tau={tau}")
    
    np.random.seed(42)
    # 1. Generate True Preferences
    U_true = np.random.randn(num_users, latent_dim)
    V_true = np.random.randn(num_items, latent_dim)
    R_dense = U_true @ V_true.T
    
    R_true = np.zeros_like(R_dense)
    items_per_user = 30
    for u in range(num_users):
        idx = np.argsort(-R_dense[u])[:items_per_user]
        R_true[u, idx] = 1.0
        
    # 2. MNAR Exposure Bias (Popularity)
    item_pop = np.random.pareto(a=1.5, size=num_items) + 1.0
    item_pop = np.sort(item_pop)[::-1]
    
    # Exposure probabilities proportional to true popularity
    prob_keep = (item_pop / item_pop.max()) ** tau
    
    R_train = np.copy(R_true)
    for i in range(num_items):
        mask = np.random.rand(num_users) > prob_keep[i]
        R_train[mask, i] = 0.0
        
    R_test = R_true - R_train
    
    active_users = (R_test.sum(axis=1) > 0) & (R_train.sum(axis=1) > 0)
    R_train = R_train[active_users]
    R_test = R_test[active_users]
    num_users_eval = R_train.shape[0]
    print(f"Valid users for evaluation (train>0, test>0): {num_users_eval}")
    
    # 3. Compute Proxy Propensity Frequency from Observation
    freq = R_train.sum(axis=0)
    # Add-1 Smoothing to prevent absolute zero inverse scaling
    P_proxy = (freq + 1.0) / (freq.max() + 1.0)
    
    # ---------- Method A: Proxy IPS SVD ----------
    print("Evaluating Proxy IPS Model...")
    ips_gammas = np.linspace(0.0, 1.5, 30) # Typically IPS beta sweeps [0, 1]
    ips_ndcgs = []
    
    for g_ips in ips_gammas:
        # 1. Scale User-Item matrix directly: R_scaled = R * D_proxy_inv
        D_proxy_inv = 1.0 / (P_proxy ** g_ips)
        R_train_scaled = R_train * D_proxy_inv[np.newaxis, :]
        
        # 2. Extract Latent factors (SVD equivalent via Covariance)
        C_ips = R_train_scaled.T @ R_train_scaled
        vals_ips, vecs_ips = eigh(C_ips)
        # Reconstruct with full or partial spectrum
        # Adding small Ridge/Wiener reg 1e-3 to prevent extreme explosion locally
        vals_filtered = np.maximum(vals_ips, 0)
        
        C_ips_recon = vecs_ips @ np.diag(vals_filtered) @ vecs_ips.T
        Pred_ips = R_train_scaled @ C_ips_recon
        
        ndcg = ndcg_at_k(Pred_ips, R_test, R_train, k=top_k)
        ips_ndcgs.append(ndcg)

    best_ips_ndcg = max(ips_ndcgs)
    best_ips_gamma = ips_gammas[np.argmax(ips_ndcgs)]
    
    # ---------- Method B: ASPIRE ----------
    print("Evaluating ASPIRE Spectral Penalty Model...")
    aspire_gammas = np.linspace(0.0, 3.0, 30)
    aspire_ndcgs = []
    
    # 1. Base C_obs EVD
    C_obs = R_train.T @ R_train
    vals_obs, vecs_obs = eigh(C_obs)
    vals_obs = np.maximum(vals_obs, 0)
    lam_max = float(vals_obs.max())
    
    for g_asp in aspire_gammas:
        h_aspire = (vals_obs ** g_asp) / (vals_obs ** g_asp + lam_max ** g_asp + 1e-10)
        lam_filtered = vals_obs * h_aspire
        
        # Reconstruct 
        C_aspire = vecs_obs @ np.diag(lam_filtered) @ vecs_obs.T
        
        # ASPIRE prediction does NOT divide the input row by D_proxy_inv
        # It just uses standard raw inputs
        Pred_aspire = R_train @ C_aspire
        
        ndcg = ndcg_at_k(Pred_aspire, R_test, R_train, k=top_k)
        aspire_ndcgs.append(ndcg)
        
    best_aspire_ndcg = max(aspire_ndcgs)
    best_aspire_gamma = aspire_gammas[np.argmax(aspire_ndcgs)]
    
    print(f" [Proxy IPS] Best NDCG@{top_k}: {best_ips_ndcg:.5f} at gamma={best_ips_gamma:.2f}")
    print(f" [ASPIRE]    Best NDCG@{top_k}: {best_aspire_ndcg:.5f} at gamma={best_aspire_gamma:.2f}")
    
    # ---------- Visualization ----------
    plt.figure(figsize=(10, 6))
    
    # Plot IPS
    plt.plot(ips_gammas, ips_ndcgs, 'r-s', linewidth=2, label='Proxy IPS (Empirical Frequency)')
    plt.plot(best_ips_gamma, best_ips_ndcg, 'r*', markersize=15, markeredgecolor='black')
    
    # Plot ASPIRE
    plt.plot(aspire_gammas, aspire_ndcgs, 'b-o', linewidth=2, label='ASPIRE (Spectral Penalty)')
    plt.plot(best_aspire_gamma, best_aspire_ndcg, 'b*', markersize=15, markeredgecolor='black')
    
    # Baseline
    plt.axhline(y=ips_ndcgs[0], color='gray', linestyle='--', label='Baseline SVD (No Correction)')
    
    plt.title(f"End-to-End Ranking Performance: Proxy IPS vs ASPIRE (NDCG@{top_k})")
    plt.xlabel(r"Correction Strength Sweep (IPS $\gamma_{ips}$ / ASPIRE $\gamma_{aspire}$)")
    plt.ylabel(f"NDCG@{top_k} (Test Set Recovery)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    
    out_dir = ensure_dir("aspire_experiments/output/exp20")
    plot_path = os.path.join(out_dir, f"ranking_ndcg_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    results = {
        "num_valid_users": int(num_users_eval),
        "best_ips_ndcg": float(best_ips_ndcg),
        "best_ips_gamma": float(best_ips_gamma),
        "ips_ndcgs": [float(v) for v in ips_ndcgs],
        "best_aspire_ndcg": float(best_aspire_ndcg),
        "best_aspire_gamma": float(best_aspire_gamma),
        "aspire_ndcgs": [float(v) for v in aspire_ndcgs]
    }
    
    json_path = os.path.join(out_dir, f"ranking_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Exp 20 finished. Results saved to {out_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_users", type=int, default=2000)
    parser.add_argument("--num_items", type=int, default=800)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()
    
    run_exp20_proxy_ndcg(num_users=args.num_users, num_items=args.num_items, top_k=args.top_k)
