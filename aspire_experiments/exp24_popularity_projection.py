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

def run_exp24_fixed_point_pk(S_full, V_full, R_train, max_iter=15):
    """
    Exp 24: Popularity Projection p_k based Fixed-Point Iteration
    """
    sigma = np.sqrt(S_full)
    sigma_1 = sigma[0]
    
    # 1. Calculate Item Popularity pi_i
    n_i = np.array(R_train.sum(axis=0)).flatten()
    pi_i = n_i / (n_i.sum() + 1e-12)
    
    # 2. Spectral Projection p_k
    # We restrict to a stable range K_limit (e.g. 500)
    K_limit = min(500, len(sigma))
    V_sub = V_full[:, :K_limit]
    p_k = np.sum(pi_i[:, None] * (V_sub**2), axis=0) # Variance in each dimension
    log_p_k = np.log(p_k + 1e-12)
    
    gamma = 2.0 # Initialization
    history = []
    
    print(f"Starting Popularity-Projected Fixed-Point Iteration (SPL)...")
    
    for i in range(max_iter):
        prev_gamma = gamma
        
        # Current filter s_k based on current gamma
        s = (sigma[:K_limit]**gamma) / (sigma[:K_limit]**gamma + sigma_1**gamma + 1e-12)
        sigma_new = sigma[:K_limit] * s
        
        # Log-reg: log p_k vs log sigma_new
        log_sigma_new = np.log(sigma_new + 1e-12)
        
        # reg = LinearRegression().fit(log_sigma_new.reshape(-1, 1), log_p_k)
        # b = reg.coef_[0]
        
        # Manual OLS for stability
        # log p_k = a + b * log sigma_new
        X = log_sigma_new.reshape(-1, 1)
        reg = LinearRegression().fit(X, log_p_k)
        b = reg.coef_[0]
        
        # Update gamma = 2 / (1 + b)
        gamma = 2.0 / (1.0 + b + 1e-12)
        
        diff = abs(gamma - prev_gamma)
        history.append({'iter': i+1, 'gamma': float(gamma), 'b': float(b)})
        print(f"Iteration {i+1:2d}: Gamma={gamma:.4f}, b={b:.4f}, diff={diff:.6f}")
        
        if diff < 1e-4 and i > 0:
            print(" -> Converged.")
            break
            
    return gamma, history, p_k, sigma[:K_limit]

def run_exp24(dataset='ml100k'):
    print(f"\n==================================================")
    print(f"Exp 24: Popularity-Projected ASPIRE (SPL)")
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
        
    # 2. Execution
    gamma_zero, history, p_k, sigma_sub = run_exp24_fixed_point_pk(S_full, V_full, R_train)
    
    # 3. Final Evaluation
    sigma = np.sqrt(S_full)
    sigma_1 = sigma[0]
    s = (sigma**gamma_zero) / (sigma**gamma_zero + sigma_1**gamma_zero + 1e-12)
    K_eval = min(1000, len(s))
    W = V_full[:, :K_eval] @ np.diag(s[:K_eval]) @ V_full[:, :K_eval].T
    
    ndcg_res = ndcg_at_k(R_train @ W, R_test, R_train, k=10)
    
    print(f"\n==================================================")
    print(f"Final Outcome (SPL-Based):")
    print(f" -> Converged Gamma  : {gamma_zero:.4f}")
    print(f" -> Final Slope b     : {history[-1]['b']:.4f}")
    print(f" -> Resulting NDCG   : {ndcg_res:.4f}")
    print(f"==================================================")
    
    # 4. Visualization
    out_dir = ensure_dir("aspire_experiments/output/exp24")
    
    # Plot 1: Convergence
    iters = [h['iter'] for h in history]
    gammas = [h['gamma'] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(iters, gammas, marker='o', color='brown')
    plt.title(f"SPL-Based Gamma Convergence ({dataset})")
    plt.xlabel("Iteration")
    plt.ylabel("Gamma (γ)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"spl_convergence_{dataset}.png"), dpi=150)
    
    # Plot 2: log p_k vs log sigma (SPL Fit)
    plt.figure(figsize=(8, 5))
    plt.scatter(np.log(sigma_sub), np.log(p_k + 1e-12), alpha=0.5, s=10, color='orange')
    plt.title(f"Spectral Power Law Fit ({dataset})")
    plt.xlabel(r"log($\sigma_k$)")
    plt.ylabel(r"log($p_k$)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"spl_fit_{dataset}.png"), dpi=150)
    
    print(f"Plots saved to {out_dir}")
    
    return gamma_zero, ndcg_res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml100k')
    args = parser.parse_args()
    
    run_exp24(args.dataset)
