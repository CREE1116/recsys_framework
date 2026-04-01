import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
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

def calculate_sure(lam, S_gamma):
    """
    Stein's Unbiased Risk Estimator for Wiener Filter
    S_gamma: sigma_k^gamma
    """
    denom = S_gamma + lam + 1e-12
    h = S_gamma / denom
    term1 = ((1 - h)**2 * S_gamma).sum()
    term2 = 2 * lam * (h / denom).sum()
    return term1 + term2

def robust_slope_lad(log_ranks, log_sigmas):
    """
    LAD (Least Absolute Deviations) Regression for Power-Law Slope
    Minimizes L1 loss
    """
    def l1_loss(params):
        a, b = params
        preds = a + b * log_ranks.flatten()
        return np.sum(np.abs(log_sigmas - preds))
    
    # Init with OLS
    reg = LinearRegression().fit(log_ranks, log_sigmas)
    init_params = [reg.intercept_, reg.coef_[0]]
    
    res = minimize(l1_loss, init_params, method='Nelder-Mead')
    return -res.x[1] # Return b (slope)

def double_fixed_point_iteration(S_full, shape, max_iter=15, tol=1e-4):
    """
    Exp 23: Joint estimation of lambda (SURE) and gamma (Fixed-Point)
    """
    sigma = np.sqrt(S_full)
    sigma_1 = sigma[0]
    
    # Initialization
    gamma = 2.0
    lam = sigma_1**gamma # Start with high noise floor
    
    # Range for slope measurement
    K_limit = min(500, len(sigma))
    ranks = np.arange(1, K_limit + 1)
    log_ranks = np.log(ranks).reshape(-1, 1)
    
    history = []
    print(f"Starting SURE-based Double Fixed-Point Iteration...")
    
    for i in range(max_iter):
        prev_gamma = gamma
        prev_lam = lam
        
        # --- Step 1: λ Update (SURE Minimization) ---
        # Fix gamma, optimize lambda
        S_gamma = sigma**gamma
        res_lam = minimize_scalar(
            calculate_sure, 
            bounds=(1e-6, sigma_1**gamma * 2), 
            args=(S_gamma,), 
            method='bounded'
        )
        lam = res_lam.x
        
        # --- Step 2: γ Update (Fixed-Point) ---
        # Fix lambda, calculate s_k and update gamma
        s = (sigma**gamma) / (sigma**gamma + lam + 1e-12)
        sigma_new = sigma * s
        
        # Robust slope measurement (LAD)
        log_sigmas_new = np.log(sigma_new[:K_limit] + 1e-12)
        b = robust_slope_lad(log_ranks, log_sigmas_new)
        
        gamma = 2.0 / (1.0 + b + 1e-12)
        
        diff_g = abs(gamma - prev_gamma)
        diff_l = abs(lam - prev_lam) / (prev_lam + 1e-12)
        
        history.append({'iter': i+1, 'gamma': float(gamma), 'lambda': float(lam)})
        print(f"Iteration {i+1:2d}: Gamma={gamma:.4f}, Lambda={lam:.2e}, diff_g={diff_g:.6f}")
        
        if diff_g < tol and i > 1:
            print(" -> Converged.")
            break
            
    return gamma, lam, history

def run_exp23(dataset='ml100k'):
    print(f"\n==================================================")
    print(f"Exp 23: SURE-based Double Fixed-Point ASPIRE")
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
    gamma_zero, lam_zero, history = double_fixed_point_iteration(S_full, R_train.shape)
    
    # 3. Final Evaluation
    sigma = np.sqrt(S_full)
    s = (sigma**gamma_zero) / (sigma**gamma_zero + lam_zero + 1e-12)
    K_eval = min(1000, len(s))
    W = V_full[:, :K_eval] @ np.diag(s[:K_eval]) @ V_full[:, :K_eval].T
    
    ndcg_res = ndcg_at_k(R_train @ W, R_test, R_train, k=10)
    
    print(f"\n==================================================")
    print(f"Final Result (SURE-Optimized):")
    print(f" -> Converged Gamma  : {gamma_zero:.4f}")
    print(f" -> Optimized Lambda : {lam_zero:.4e}")
    print(f" -> Resulting NDCG   : {ndcg_res:.4f}")
    print(f"==================================================")
    
    # 4. Visualization
    out_dir = ensure_dir("aspire_experiments/output/exp23")
    iters = [h['iter'] for h in history]
    gammas = [h['gamma'] for h in history]
    lambdas = [h['lambda'] for h in history]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Gamma (γ)', color='purple')
    ax1.plot(iters, gammas, marker='o', color='purple', label='Gamma path')
    ax1.tick_params(axis='y', labelcolor='purple')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Lambda (λ)', color='green')
    ax2.plot(iters, lambdas, marker='s', color='green', linestyle='--', label='Lambda path')
    ax2.tick_params(axis='y', labelcolor='green')
    
    plt.title(f"SURE Double Fixed-Point Trace ({dataset})")
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(out_dir, f"sure_convergence_{dataset}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Convergence Plot saved to {plot_path}")
    
    return gamma_zero, lam_zero, ndcg_res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml100k')
    args = parser.parse_args()
    
    run_exp23(args.dataset)
