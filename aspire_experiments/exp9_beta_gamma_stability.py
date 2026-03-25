import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from tqdm import tqdm
from scipy.sparse import csr_matrix

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import load_config, ensure_dir, AspireHPO, get_eval_config
from aspire_experiments.proof_models import ASPIRE_Test
from src.data_loader import DataLoader
from src.evaluation import evaluate_metrics

from scipy.optimize import minimize

def fit_beta(s_vals, k_start, total_energy_sum):
    """Fit power-law slope using OLS, Head-Weighted, and Tail-Weighted OLS."""
    log_r = np.log(np.arange(k_start, len(s_vals) + k_start))
    log_s = np.log(s_vals + 1e-12)
    
    # 1. Unweighted OLS
    m_ols_unweighted, b_ols_unweighted = np.polyfit(log_r, log_s, 1)
    beta_ols = -2.0 * m_ols_unweighted
    
    # 2. Head-Weighted (Energy Fraction)
    w_head = (s_vals**2) / (total_energy_sum + 1e-12)
    def head_loss(p):
        return np.sum(w_head * (log_s - (p[0] * log_r + p[1]))**2)
    
    res_head = minimize(head_loss, x0=[m_ols_unweighted, b_ols_unweighted])
    beta_head = -2.0 * res_head.x[0]
    
    # 3. Tail-Weighted (Inverse Magnitude)
    # We use inverse magnitude to prioritize the smallest values
    w_tail = 1.0 / (s_vals + 1e-6)
    w_tail /= w_tail.sum()
    def tail_loss(p):
        return np.sum(w_tail * (log_s - (p[0] * log_r + p[1]))**2)
    
    res_tail = minimize(tail_loss, x0=[m_ols_unweighted, b_ols_unweighted])
    beta_tail = -2.0 * res_tail.x[0]
    
    return beta_ols, beta_head, beta_tail

def run_exp9(dataset_name, n_trials=30):
    print(f"Running Exp 9: Gamma Stability Analysis on {dataset_name}...")
    config = load_config(dataset_name)
    loader = DataLoader(config)
    eval_cfg = get_eval_config(loader, {"top_k": [20], "metrics": ["NDCG"]})
    test_loader = loader.get_final_loader(batch_size=2048)
    
    # Get Singular Values for Theoretical Gamma
    rows, cols = loader.train_df['user_id'].values, loader.train_df['item_id'].values
    R = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(loader.n_users, loader.n_items))
    if R.shape[1] <= R.shape[0]:
        G = (R.T @ R).toarray().astype(np.float32)
    else:
        G = (R @ R.T).toarray().astype(np.float32)
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.sort(eigvals)[::-1]
    sig_all = np.sqrt(np.maximum(eigvals, 1e-12))
    total_energy_sum = np.sum(sig_all**2)
    
    # Metrics
    results = []
    
    for skip_n in range(6):
        print(f"\n[Skip Top-{skip_n}] Running HPO for Gamma...")
        
        def objective(params):
            m_cfg = {
                "name": "aspire_test", 
                "gamma": params['gamma'], 
                "skip_top_k": skip_n,
                "filter_mode": "gamma_only", 
                "target_energy": 1.0
            }
            cfg = {**config, 'model': m_cfg, 'device': 'auto'}
            model = ASPIRE_Test(cfg, loader)
            metrics = evaluate_metrics(model, loader, eval_cfg, model.device, test_loader)
            return metrics["NDCG@20"]

        hpo = AspireHPO([{'name': 'gamma', 'type': 'float', 'range': '0.0 2.0'}], n_trials=n_trials, patience=15)
        best_p, best_score = hpo.search(objective, study_name=f"Exp8_Skip{skip_n}_{dataset_name}")
        
        gamma_best = best_p['gamma']
        
        # Theoretical fit (OLS, Head, Tail)
        energy_cum = np.cumsum(sig_all**2) / total_energy_sum
        k_95 = np.searchsorted(energy_cum, 0.95) + 1
        sig_clip = sig_all[skip_n:k_95]
        beta_ols, beta_head, beta_tail = fit_beta(sig_clip, skip_n + 1, total_energy_sum)
        
        # Convert to Gamma
        gamma_ols = 2.0 / (1.0 + beta_ols)
        gamma_head = 2.0 / (1.0 + beta_head)
        gamma_tail = 2.0 / (1.0 + beta_tail)
        
        results.append({
            "skip_n": skip_n,
            "gamma_best": float(gamma_best),
            "gamma_ols": float(gamma_ols),
            "gamma_head": float(gamma_head),
            "gamma_tail": float(gamma_tail),
            "ndcg": float(best_score)
        })
        print(f"  Result: Skip={skip_n}, G_best={gamma_best:.4f}, G_ols={gamma_ols:.4f}, G_head={gamma_head:.4f}, G_tail={gamma_tail:.4f}")

    # Plotting
    skips = [r["skip_n"] for r in results]
    g_emp = [r["gamma_best"] for r in results]
    g_ols = [r["gamma_ols"] for r in results]
    g_head = [r["gamma_head"] for r in results]
    g_tail = [r["gamma_tail"] for r in results]
    
    plt.figure(figsize=(12, 7))
    plt.plot(skips, g_emp, 'o-', label='Empirical Gamma (HPO-opt)', markersize=8, color='black', linewidth=3)
    plt.plot(skips, g_ols, 's--', label='Theoretical Gamma (OLS-Unweighted)', markersize=8, alpha=0.6)
    plt.plot(skips, g_head, '^:', label='Theoretical Gamma (Head-Weighted)', markersize=8, color='red')
    plt.plot(skips, g_tail, 'v:', label='Theoretical Gamma (Tail-Weighted)', markersize=8, color='blue', alpha=0.7)
    
    plt.xlabel('Number of Top Components Skipped (N)')
    plt.ylabel('Gamma (\u03b3)')
    plt.title(f'Gamma Stability: Weighting Comparison ({dataset_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_dir = ensure_dir(f"aspire_experiments/output/exp9/{dataset_name}")
    plt.savefig(os.path.join(out_dir, "gamma_stability_weighting_compare.png"), dpi=200)
    plt.close()
    
    with open(os.path.join(out_dir, "stability_results_gamma_weighted.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nExp 9 finished on {dataset_name}. Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    run_exp9(args.dataset, n_trials=args.trials)
