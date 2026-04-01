import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Add root directory to sys.path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir

def ndcg_at_k(preds, targets, mask_train, k=10):
    """Simple NDCG implementation for NumPy matrices"""
    preds = preds.copy()
    preds[mask_train > 0] = -np.inf
    sorted_idx = np.argsort(-preds, axis=1)[:, :k]
    ndcg_list = []
    
    # Process u-i targets (sparse or dense)
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

def aspire_zero_fixed_point_iteration(S_full, R_train_shape, max_iter=10):
    """
    User's Original Idea: Self-Consistent Fixed-Point Iteration
    S_full: Full eigenvalues (sigma^2)
    """
    sigma = np.sqrt(S_full)
    sigma_1 = sigma[0]
    gamma = 2.0 # Initialization (MCAR Assumption)
    
    # We use a reasonable rank K for slope measurement (e.g., top 10% or first scores of components)
    # However, the user said "drop optimal K", so we use a stable signal range.
    K_limit = min(500, len(sigma)) # Keep it stable within the main spectrum
    ranks = np.arange(1, K_limit + 1)
    log_ranks = np.log(ranks).reshape(-1, 1)
    
    history = []
    
    print(f"Starting Fixed-Point Iteration (sigma_1 fixed)...")
    for i in range(max_iter):
        # 1. Filtered Spectrum: s_k = sigma_k^gamma / (sigma_k^gamma + sigma_1^gamma)
        s = (sigma**gamma) / (sigma**gamma + sigma_1**gamma + 1e-10)
        
        # 2. Corrected Spectrum: sigma_new = sigma * s
        sigma_new = sigma * s
        
        # 3. Estimate new slope b via OLS on corrected spectrum
        # We estimate b from the log-log linear region
        log_sigmas_new = np.log(sigma_new[:K_limit] + 1e-10)
        reg = LinearRegression().fit(log_ranks, log_sigmas_new)
        b = -reg.coef_[0]
        
        # 4. Update Gamma: gamma = 2 / (1 + b)
        prev_gamma = gamma
        gamma = 2.0 / (1.0 + b + 1e-10)
        
        diff = abs(gamma - prev_gamma)
        history.append({'iter': i+1, 'gamma': float(gamma), 'b': float(b)})
        print(f"Iteration {i+1:2d}: Gamma = {gamma:.4f}, b = {b:.4f}, diff = {diff:.6f}")
        
        if diff < 1e-4:
            print(" -> Converged.")
            break
            
    return gamma, history

def run_exp22(dataset='ml100k'):
    print(f"\n==================================================")
    print(f"Exp 22: Self-Consistent Fixed-Point ASPIRE-Zero")
    print(f"Dataset: {dataset}")
    print(f"==================================================")
    
    # 1. Load Data
    loader, R_train_sparse, S_full_torch, V_full_torch, config = get_loader_and_svd(dataset, k=None)
    R_train = R_train_sparse.toarray()
    S_full = S_full_torch.detach().cpu().numpy()
    V_full = V_full_torch.detach().cpu().numpy()
    
    # Prepare Test Data
    test_df = loader.test_df
    R_test = np.zeros_like(R_train)
    for row in test_df.itertuples():
        R_test[row.user_id, row.item_id] = 1.0
    
    # 2. Execution: Fixed-Point Iteration
    gamma_zero, history = aspire_zero_fixed_point_iteration(S_full, R_train.shape)
    
    # 3. Final Filtering with Converged Gamma
    sigma = np.sqrt(S_full)
    sigma_1 = sigma[0]
    # Filter: s_k = sigma_k^gamma / (sigma_k^gamma + sigma_1^gamma)
    s = (sigma**gamma_zero) / (sigma**gamma_zero + sigma_1**gamma_zero + 1e-10)
    
    # Reconstruction Matrix (Using EVD/SVD approach)
    # W = V diag(s) V.T
    # We restrict to a reasonable rank for efficiency (e.g., 500)
    K_eval = min(1000, len(s))
    W = V_full[:, :K_eval] @ np.diag(s[:K_eval]) @ V_full[:, :K_eval].T
    
    ndcg_zero = ndcg_at_k(R_train @ W, R_test, R_train, k=10)
    
    print(f"\n==================================================")
    print(f"Final Result:")
    print(f" -> Converged Gamma : {gamma_zero:.4f}")
    print(f" -> Resulting NDCG  : {ndcg_zero:.4f}")
    print(f"==================================================")
    
    # 4. Visualization (Convergence Path)
    out_dir = ensure_dir("aspire_experiments/output/exp22")
    iters = [h['iter'] for h in history]
    gammas = [h['gamma'] for h in history]
    
    plt.figure(figsize=(8, 5))
    plt.plot(iters, gammas, marker='o', linestyle='-', color='purple')
    plt.title(f"Gamma Convergence Path ({dataset})")
    plt.xlabel("Iteration")
    plt.ylabel("Gamma (γ)")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(out_dir, f"fixed_point_convergence_{dataset}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Convergence Plot saved to {plot_path}")
    
    return gamma_zero, ndcg_zero

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml100k')
    args = parser.parse_args()
    
    run_exp22(args.dataset)
