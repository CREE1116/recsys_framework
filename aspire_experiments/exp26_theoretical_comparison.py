import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import kstest
from sklearn.linear_model import LinearRegression

# Add root directory to sys.path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir

# --------------------------------------------------------------------------------
# 1. Helper Functions
# --------------------------------------------------------------------------------

def ndcg_at_k(preds, targets, mask_train, k=20):
    """Simple NDCG implementation for NumPy matrices"""
    preds = preds.copy()
    preds[mask_train > 0] = -np.inf
    sorted_idx = np.argsort(-preds, axis=1)[:, :k]
    ndcg_list = []
    
    for u in range(preds.shape[0]):
        actual_top = targets[u].toarray().flatten() if hasattr(targets[u], 'toarray') else targets[u]
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

def estimate_power_law_tau(counts):
    """Clauset et al. (2009) MLE for power-law exponent tau"""
    x = counts[counts > 0]
    x_min = np.min(x)
    n = len(x)
    tau = 1 + n / np.sum(np.log(x / x_min))
    return tau

def gavish_donoho_threshold(sigma, U, I):
    """Gavish-Donoho (2014) Bulk Edge Estimator for unknown noise level"""
    beta = min(U, I) / max(U, I)
    # Omega(beta) approximation
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    sigma_median = np.median(sigma)
    threshold = omega * sigma_median
    k_bulk = np.sum(sigma > threshold)
    return k_bulk, threshold

# --------------------------------------------------------------------------------
# 2. Three Theoretical Modes
# --------------------------------------------------------------------------------

def mode_wiener_fixed_point(sigma, V_full, R_train, max_iter=15):
    """
    Mode 1: Wiener Fixed-Point with Popularity Projection (p_k)
    """
    # Calculate item popularity counts and normalize
    item_counts = np.array(R_train.sum(axis=0)).flatten()
    pi_i = item_counts / (item_counts.sum() + 1e-10)
    
    # Project popularity energy onto item-subspace components: pk = sum(pi_i * v_ik^2)
    # V_full is (I, N)
    pk = np.sum(pi_i[:, None] * (V_full**2), axis=0) # (N,)
    
    sigma_1 = sigma[0]
    gamma = 2.0
    valid_range = min(1000, len(sigma))
    log_pk = np.log(pk[:valid_range] + 1e-12).reshape(-1, 1)
    
    history = []
    for i in range(max_iter):
        prev_gamma = gamma
        
        # h_k = sigma_k^gamma / (sigma_k^gamma + sigma_1^gamma)
        h = (sigma**gamma) / (sigma**gamma + sigma_1**gamma + 1e-12)
        sigma_new = sigma * h
        
        # Fit log pk vs log sigma_new
        log_s = np.log(sigma_new[:valid_range] + 1e-12).reshape(-1, 1)
        reg = LinearRegression().fit(log_s, log_pk)
        b = float(reg.coef_.item())
        
        # Update gamma = 2 / (1 + |b|)
        # The relationship: energy decay beta = |b|. We use positive b definition.
        beta = abs(b)
        gamma = 2.0 / (1.0 + beta + 1e-10)
        
        diff = abs(gamma - prev_gamma)
        history.append({'iter': i+1, 'gamma': gamma, 'beta': beta})
        if diff < 1e-5: break
        
    return gamma, history

def mode_rmt_marchenko_pastur(sigma, U, I):
    """
    Mode 2: Marchenko-Pastur Fitting based on RMT
    """
    k_bulk, threshold = gavish_donoho_threshold(sigma, U, I)
    tail_sigmas = sigma[k_bulk:]
    
    if len(tail_sigmas) < 10:
        return 1.0, [] # Fallback
        
    def objective(g):
        transformed = tail_sigmas**g
        # We want the transformed tail to follow a flat/isotropic distribution (MP limit)
        # Here we simplify: minimize the variance of the transformed tail relative to its scale
        # or use KS-test against uniform for the cumulative energy if scaled.
        # A simpler robust way: maximize entropy or minimize coefficient of variation
        return np.std(transformed) / (np.mean(transformed) + 1e-10)

    res = minimize(objective, x0=[1.0], bounds=[(0.1, 3.0)])
    gamma_rmt = float(res.x[0])
    
    return gamma_rmt, [{'method': 'gavish-donoho', 'k_bulk': k_bulk}]

def mode_graph_theory_chung(R_train):
    """
    Mode 3: Chung-Lu Power-Law Mapping
    """
    item_counts = np.array(R_train.sum(axis=0)).flatten()
    tau = estimate_power_law_tau(item_counts)
    
    # User's formula: b = 2(tau - 1) / (tau - 2)
    # This b is usually the eigenvalue decay slope.
    # beta = b
    # gamma = 2 / (1 + beta)
    b_chung = 2 * (tau - 1) / (max(tau - 2, 0.1))
    gamma_graph = 2.0 / (1.0 + b_chung + 1e-10)
    
    return gamma_graph, [{'tau': tau, 'b_predicted': b_chung}]

# --------------------------------------------------------------------------------
# 3. Main Experiment Execution
# --------------------------------------------------------------------------------

def run_theoretical_comparison(dataset='ml100k'):
    print(f"\n" + "="*60)
    print(f"Exp 26: Theoretical ASPIRE-Zero Comparison")
    print(f"Dataset: {dataset}")
    print(f"="*60)
    
    # 1. Load Data
    loader, R_train_sparse, S_full_torch, V_full_torch, _ = get_loader_and_svd(dataset, k=None)
    R_train = R_train_sparse.toarray()
    sigma = np.sqrt(S_full_torch.detach().cpu().numpy())
    V_full = V_full_torch.detach().cpu().numpy()
    
    # Test Data
    test_df = loader.test_df
    R_test = np.zeros_like(R_train)
    for row in test_df.itertuples():
        R_test[row.user_id, row.item_id] = 1.0
        
    results = {}
    
    # --- Execute Mode 1 ---
    print("\n[Mode 1] Wiener Fixed-Point (p_k projection)...")
    gamma_fp, hist_fp = mode_wiener_fixed_point(sigma, V_full, R_train)
    results['fixed_point'] = {'gamma': gamma_fp, 'desc': 'Self-consistent p_k'}
    
    # --- Execute Mode 2 ---
    print("[Mode 2] RMT Marchenko-Pastur Fitting...")
    gamma_rmt, hist_rmt = mode_rmt_marchenko_pastur(sigma, *R_train.shape)
    results['rmt_mp'] = {'gamma': gamma_rmt, 'desc': 'Gavish-Donoho Noise Bulk'}
    
    # --- Execute Mode 3 ---
    print("[Mode 3] Graph-Theory (Chung-Lu Mapping)...")
    gamma_graph, hist_graph = mode_graph_theory_chung(R_train)
    results['graph_theory'] = {'gamma': gamma_graph, 'desc': 'Degree Power-law MLE'}
    
    # 2. Evaluation
    U, I = R_train.shape
    M = R_train.sum()
    sigma_rmt = np.sqrt(M / min(U, I))
    
    eval_results = []
    
    for mode, data in results.items():
        g = data['gamma']
        # We use a consistent RMT noise base for evaluation if possible, 
        # or Mode 1's sigma_1 anchor if that's more consistent. 
        # Here we use RMT base as it's more universal across datasets.
        lam = sigma_rmt**g
        
        h = (sigma**g) / (sigma**g + lam + 1e-12)
        K_eval = min(1000, len(h))
        W = V_full[:, :K_eval] @ np.diag(h[:K_eval]) @ V_full[:, :K_eval].T
        
        ndcg = ndcg_at_k(R_train @ W, R_test, R_train, k=20)
        data['ndcg'] = ndcg
        print(f"  -> Mode: {mode:12s} | Gamma: {g:.4f} | NDCG@20: {ndcg:.4f}")
        eval_results.append((mode, g, ndcg))

    # 3. Visualization
    out_dir = ensure_dir("aspire_experiments/output/exp26")
    plt.figure(figsize=(10, 6))
    
    for mode, g, ndcg in eval_results:
        h_curve = (sigma**g) / (sigma**g + sigma_rmt**g + 1e-12)
        plt.plot(h_curve[:500], label=f"{mode} ($\gamma$={g:.3f}, NDCG={ndcg:.3f})")
    
    plt.title(f"Wiener Filter Shape Comparison ({dataset})")
    plt.xlabel("Component Rank")
    plt.ylabel("Filter Gain (h)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(out_dir, f"filter_comparison_{dataset}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nComparison plot saved to {plot_path}")
    
    return results

if __name__ == "__main__":
    datasets = ['ml100k', 'ml1m', 'steam', 'yahoo_r3']
    all_summary = {}
    
    for ds in datasets:
        try:
            res = run_theoretical_comparison(ds)
            all_summary[ds] = res
        except Exception as e:
            print(f"Error processing {ds}: {e}")
            
    # Final Table Output
    print("\n\n" + "="*80)
    print(f"{'Dataset':<10} | {'Mode':<15} | {'Gamma':<10} | {'NDCG@20':<10}")
    print("-" * 80)
    for ds, modes in all_summary.items():
        for mname, mdata in modes.items():
            print(f"{ds:<10} | {mname:<15} | {mdata['gamma']:<10.4f} | {mdata['ndcg']:<10.4f}")
    print("="*80)
