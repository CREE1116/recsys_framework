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

def run_exp25_rmt_fixed_point(S_full, R_train, max_iter=15):
    """
    Exp 25: RMT Noise Boundary + Self-Consistent Fixed-Point Iteration
    """
    sigma = np.sqrt(S_full)
    U, I = R_train.shape
    M = R_train.sum() # Total interactions
    
    # RMT Noise Edge sigma_rmt = sqrt(M / min(U, I))
    sigma_rmt = np.sqrt(M / min(U, I))
    print(f"RMT Noise Base (sigma_rmt): {sigma_rmt:.4f}")
    
    gamma = 2.0 # Initialization
    
    # We restrict to a stable range K_limit (e.g. 500)
    K_limit = min(500, len(sigma))
    ranks = np.arange(1, K_limit + 1)
    log_ranks = np.log(ranks).reshape(-1, 1)
    
    history = []
    print(f"Starting RMT-Consistent Fixed-Point Iteration...")
    
    for i in range(max_iter):
        prev_gamma = gamma
        
        # 1. Update lambda = sigma_rmt^gamma
        lam = sigma_rmt**gamma
        
        # 2. Filter h_k = sigma_k^gamma / (sigma_k^gamma + lambda)
        s = (sigma**gamma) / (sigma**gamma + lam + 1e-12)
        
        # 3. Corrected Spectrum: sigma_new = sigma * s
        sigma_new = sigma * s
        
        # 4. Estimate new slope b on corrected spectrum
        log_sigmas_new = np.log(sigma_new[:K_limit] + 1e-12)
        reg = LinearRegression().fit(log_ranks, log_sigmas_new)
        b = -reg.coef_[0]
        
        # 5. Update gamma = 2 / (1 + b)
        gamma = 2.0 / (1.0 + b + 1e-12)
        
        diff = abs(gamma - prev_gamma)
        history.append({'iter': i+1, 'gamma': float(gamma), 'b': float(b), 'lambda': float(lam)})
        print(f"Iteration {i+1:2d}: Gamma={gamma:.4f}, Lambda={lam:.2e}, b={b:.4f}, diff={diff:.6f}")
        
        if diff < 1e-4 and i > 0:
            print(" -> Converged.")
            break
            
    return gamma, sigma_rmt**gamma, history

def run_exp25(dataset='ml100k'):
    print(f"\n==================================================")
    print(f"Exp 25: RMT-Consistent Fixed-Point ASPIRE")
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
    gamma_zero, lam_zero, history = run_exp25_rmt_fixed_point(S_full, R_train)
    
    # 3. Final Evaluation
    sigma = np.sqrt(S_full)
    s = (sigma**gamma_zero) / (sigma**gamma_zero + lam_zero + 1e-12)
    K_eval = min(1000, len(s))
    W = V_full[:, :K_eval] @ np.diag(s[:K_eval]) @ V_full[:, :K_eval].T
    
    ndcg_res = ndcg_at_k(R_train @ W, R_test, R_train, k=10)
    
    print(f"\n==================================================")
    print(f"Final Result (RMT-Consistent):")
    print(f" -> Converged Gamma  : {gamma_zero:.4f}")
    print(f" -> Optimized Lambda : {lam_zero:.4e}")
    print(f" -> Resulting NDCG   : {ndcg_res:.4f}")
    print(f"==================================================")
    
    # 4. Visualization
    out_dir = ensure_dir("aspire_experiments/output/exp25")
    iters = [h['iter'] for h in history]
    gammas = [h['gamma'] for h in history]
    lambdas = [h['lambda'] for h in history]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Gamma (γ)', color='purple')
    ax1.plot(iters, gammas, marker='o', color='purple', label='Gamma path')
    ax1.tick_params(axis='y', labelcolor='purple')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Lambda (λ) [RMT-based]', color='green')
    ax2.plot(iters, lambdas, marker='s', color='green', linestyle='--', label='Lambda path')
    ax2.tick_params(axis='y', labelcolor='green')
    
    plt.title(f"RMT-Consistent Fixed-Point Trace ({dataset})")
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(out_dir, f"rmt_convergence_{dataset}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Convergence Plot saved to {plot_path}")
    
    return gamma_zero, lam_zero, ndcg_res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml100k')
    args = parser.parse_args()
    
    run_exp25(args.dataset)
