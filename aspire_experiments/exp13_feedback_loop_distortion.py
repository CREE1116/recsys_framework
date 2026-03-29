import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import json
import argparse
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import ensure_dir
from src.utils.gpu_accel import get_device, gpu_gram_solve

def fit_lad_r2(x, y, fit_intercept=True):
    """LAD 피팅 및 표준 OLS R2 계산"""
    def lad_loss(params, x, y):
        if fit_intercept:
            m, b = params
            return np.sum(np.abs(y - (m * x + b)))
        else:
            m = params[0]
            return np.sum(np.abs(y - (m * x)))
            
    if fit_intercept:
        m_ols, b_ols = np.polyfit(x, y, 1)
        res = minimize(lad_loss, x0=[m_ols, b_ols], args=(x, y), method='Nelder-Mead')
        m_lad, b_lad = res.x
        y_pred = m_lad * x + b_lad
    else:
        m_ols = np.sum(x * y) / np.sum(x**2)
        res = minimize(lad_loss, x0=[m_ols], args=(x, y), method='Nelder-Mead')
        m_lad = res.x[0]
        b_lad = 0
        y_pred = m_lad * x

    return m_lad, b_lad, r2_score(y, y_pred)

def train_ease_iterative(X_sparse, reg=500.0, device='auto'):
    """EASE 모델 학습 (Closed-form)"""
    P = gpu_gram_solve(X_sparse, reg, device=device, return_tensor=True)
    diag = torch.diagonal(P)
    B = -P / diag.view(1, -1)
    B.fill_diagonal_(0)
    return B

def get_singular_values_and_pk(X_sparse, k=500, device='auto'):
    """특이값과 p_k(스펙트럴 인기도) 추출"""
    dev = get_device(device)
    X_dense = torch.from_numpy(X_sparse.toarray().astype(np.float32)).to(dev)
    U, S, V = torch.linalg.svd(X_dense, full_matrices=False)
    
    k = min(k, len(S))
    sigma_k = S[:k].cpu().numpy()
    V_k = V[:k, :].t().cpu().numpy() 
    
    n_i = np.array(X_sparse.sum(axis=0)).flatten()
    pi_i = n_i / (n_i.sum() + 1e-12)
    p_k = np.sum(pi_i[:, np.newaxis] * (V_k**2), axis=0)
    
    return sigma_k, p_k

def run_exp13(args):
    print(f"Running Exp 13: Feedback Loop Simulation (RANDOM {args.num_users}x{args.num_items}, Density: {args.density})")
    nnz = int(args.num_users * args.num_items * args.density)
    rows = np.random.randint(0, args.num_users, nnz)
    cols = np.random.randint(0, args.num_items, nnz)
    X_current = csr_matrix((np.ones(nnz, dtype=np.float32), (rows, cols)), shape=(args.num_users, args.num_items))
    X_current.data = np.ones_like(X_current.data)
    
    num_users, num_items = X_current.shape
    device = get_device()
    k_max = min(num_users, num_items, 500)
    
    # History tracking
    s_history = []
    p_history = []
    
    # Iter 0
    s0, p0 = get_singular_values_and_pk(X_current, k=k_max, device=device)
    s_history.append(s0); p_history.append(p0)
    
    for t in range(1, args.iter + 1):
        print(f"  Iteration {t}/{args.iter}...")
        B = train_ease_iterative(X_current, reg=args.reg, device=device)
        
        # 1. Scores & Masking
        X_dense = torch.from_numpy(X_current.toarray().astype(np.float32)).to(device)
        Scores = X_dense @ B
        mask = X_dense > 0.0
        Scores[mask] = -1e9
        
        # 2. Sampling (Top-K then Softmax within it)
        Scores_cpu = Scores.cpu().numpy()
        del B, X_dense, Scores, mask
        
        new_rows, new_cols = [], []
        for u in range(num_users):
            user_scores = Scores_cpu[u]
            top_indices = np.argsort(user_scores)[-args.top_k:][::-1]
            top_scores = user_scores[top_indices]
            
            top_scores = top_scores - np.max(top_scores)
            probs = np.exp(top_scores / args.temp)
            probs /= np.sum(probs) + 1e-12
            
            clicked = np.random.choice(top_indices, size=min(args.num_clicks, args.top_k), replace=False, p=probs)
            for item_id in clicked:
                new_rows.append(u); new_cols.append(item_id)
        
        # 3. Update Matrix
        X_new = csr_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=(num_users, num_items))
        X_current = X_current + X_new
        X_current.data = np.ones_like(X_current.data)
        
        # 4. Record Spectrum
        st, pt = get_singular_values_and_pk(X_current, k=k_max, device=device)
        s_history.append(st); p_history.append(pt)
        
    # --- Analysis & Plotting ---
    out_dir = ensure_dir(f"aspire_experiments/output/exp13/random_simulation")
    s_0_orig = s_history[0]
    valid_mask = s_0_orig > 1e-6
    s_0 = s_0_orig[valid_mask]
    sigma_max_0 = s_0[0]
    log_s_0 = np.log(s_0 / sigma_max_0)
    
    plt.figure(figsize=(15, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 1, args.iter))
    
    # Left: Distortion
    plt.subplot(1, 2, 1)
    distortion_res = []
    for t in range(1, args.iter + 1):
        s_t = s_history[t][valid_mask]
        # [FIXED NORMALIZATION] Use sigma_max_0 as reference
        log_s_t = np.log(s_t / sigma_max_0)
        # [FIT INTERCEPT] To properly capture slope (> 1.0) when top values amplify
        m, b, r2 = fit_lad_r2(log_s_0, log_s_t, fit_intercept=True)
        distortion_res.append({"iter": t, "slope": float(m), "intercept": float(b), "r2": float(r2)})
        plt.scatter(log_s_0, log_s_t, alpha=0.3, s=10, color=colors[t-1], label=f'Iter {t} (Slope: {m:.3f}, $R^2$: {r2:.3f})')
        plt.plot(log_s_0, m * log_s_0 + b, color=colors[t-1], alpha=0.5)
    
    plt.plot([min(log_s_0), 0], [min(log_s_0), 0], 'k--', label='Baseline')
    plt.title('Multiplicative Distortion Evolution (Fixed Norm + Intercept)')
    plt.xlabel('log(sigma_0 / sigma_0[0])'); plt.ylabel('log(sigma_t / sigma_0[0])')
    plt.legend(); plt.grid(True, alpha=0.2)
    
    # Right: SPL
    plt.subplot(1, 2, 2)
    spl_res = []
    for t in range(args.iter + 1):
        log_st = np.log(s_history[t][valid_mask])
        log_pt = np.log(p_history[t][valid_mask] + 1e-12)
        m, b, r2 = fit_lad_r2(log_st, log_pt, fit_intercept=True)
        pr, _ = pearsonr(log_st, log_pt)
        spl_res.append({"iter": t, "slope": float(m), "r2": float(r2), "pearson": float(pr)})
        color = 'gray' if t == 0 else colors[t-1]
        plt.scatter(log_st, log_pt, alpha=0.3, s=10, color=color, label=f'Iter {t} ($R^2$: {r2:.3f})')
        plt.plot(log_st, m * log_st + b, color=color, alpha=0.5)
        
    plt.title('SPL Relationship Evolution'); plt.xlabel('log(sigma_k)'); plt.ylabel('log(p_k)')
    plt.legend(); plt.grid(True, alpha=0.2); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feedback_spl_evolution.png"), dpi=150); plt.close()
    
    # Save JSON
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "distortion": distortion_res, "spl": spl_res}, f, indent=4)
    print(f"Exp 13 Done. Final SPL R2: {spl_res[-1]['r2']:.4f}, Distortion Slope: {distortion_res[-1]['slope']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--num_clicks", type=int, default=2)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--reg", type=float, default=500.0)
    parser.add_argument("--num_users", type=int, default=1000)
    parser.add_argument("--num_items", type=int, default=1000)
    parser.add_argument("--density", type=float, default=0.05)
    run_exp13(parser.parse_args())
