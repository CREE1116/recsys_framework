import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import json
import argparse
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import load_config, ensure_dir
from src.data_loader import DataLoader
from src.utils.gpu_accel import get_device, gpu_gram_solve, EVDCacheManager

def fit_spl_lad(sigma, pk):
    """LAD 피팅을 통한 SPL 분석 (log p_k = m * log sigma_k + b)"""
    x = np.log(sigma + 1e-12)
    y = np.log(pk + 1e-12)
    
    def lad_loss(params, x, y):
        m, b = params
        return np.sum(np.abs(y - (m * x + b)))
    
    # Initial guess
    m_ols, b_ols = np.polyfit(x, y, 1)
    res = minimize(lad_loss, x0=[m_ols, b_ols], args=(x, y), method='Nelder-Mead')
    m_lad, b_lad = res.x
    y_pred = m_lad * x + b_lad
    
    return {
        "slope": float(m_lad),
        "intercept": float(b_lad),
        "r2": float(r2_score(y, y_pred)),
        "x": x,
        "y": y
    }

def get_sigma_and_pk(X_sparse, device, dataset_name, label, k=500):
    """특이값과 스펙트럴 인기도(p_k) 추출"""
    manager = EVDCacheManager()
    _, S, V, _ = manager.get_evd(X_sparse, k=k, dataset_name=f"{dataset_name}_{label}")
    
    s_vals = S.cpu().numpy()
    sigma_k = np.sqrt(np.maximum(s_vals, 1e-12))
    V_k = V.cpu().numpy()
    
    # p_k = Σ_i π_i * V_{ik}^2
    n_i = np.array(X_sparse.sum(axis=0)).flatten()
    pi_i = n_i / (n_i.sum() + 1e-12)
    p_k = np.sum(pi_i[:, np.newaxis] * (V_k**2), axis=0)
    
    return sigma_k, p_k

def run_simulation(args, device):
    """패널 A: 시뮬레이션 (무작위 -> SPL 출현)"""
    print(f"Running Simulation (Panel A)...")
    nnz = int(args.num_users * args.num_items * args.density)
    rows = np.random.randint(0, args.num_users, nnz)
    cols = np.random.randint(0, args.num_items, nnz)
    X = csr_matrix((np.ones(nnz, dtype=np.float32), (rows, cols)), shape=(args.num_users, args.num_items))
    X.data = np.ones_like(X.data)
    
    history = []
    # Iter 0
    s0, p0 = get_sigma_and_pk(X, device, "sim", "iter0")
    history.append(fit_spl_lad(s0, p0))
    
    for t in range(1, args.iter + 1):
        # EASE 학습
        from aspire_experiments.exp13_feedback_loop_distortion import train_ease_iterative
        B = train_ease_iterative(X, reg=500.0, device=device)
        
        # 추천 및 샘플링 (Top-K Softmax)
        X_dense = torch.from_numpy(X.toarray().astype(np.float32)).to(device)
        Scores = X_dense @ B
        Scores[X_dense > 0] = -1e9
        Scores_cpu = Scores.cpu().numpy()
        del B, X_dense, Scores
        
        new_rows, new_cols = [], []
        for u in range(args.num_users):
            user_scores = Scores_cpu[u]
            top_idx = np.argsort(user_scores)[-args.top_k:][::-1]
            top_scores = user_scores[top_idx]
            top_scores -= np.max(top_scores)
            probs = np.exp(top_scores / args.temp)
            probs /= (np.sum(probs) + 1e-12)
            
            clicked = np.random.choice(top_idx, size=args.num_clicks, replace=False, p=probs)
            for item_id in clicked:
                new_rows.append(u); new_cols.append(item_id)
        
        X_new = csr_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=(args.num_users, args.num_items))
        X = X + X_new
        X.data = np.ones_like(X.data)
        
        if t == args.iter:
            st, pt = get_sigma_and_pk(X, device, "sim", f"iter{t}")
            history.append(fit_spl_lad(st, pt))
            
    return history

def run_real_data(device):
    """패널 B: 실제 데이터 (Yahoo R3 MNAR vs MCAR)"""
    print(f"Running Real Data (Panel B: Yahoo R3)...")
    config = load_config("yahoo_r3")
    loader = DataLoader(config)
    
    X_mnar = csr_matrix((np.ones(len(loader.train_df)), (loader.train_df['user_id'], loader.train_df['item_id'])), 
                        shape=(loader.n_users, loader.n_items))
    X_mcar = csr_matrix((np.ones(len(loader.test_df)), (loader.test_df['user_id'], loader.test_df['item_id'])), 
                        shape=(loader.n_users, loader.n_items))
    
    s_mnar, p_mnar = get_sigma_and_pk(X_mnar, device, "yahoo", "mnar")
    s_mcar, p_mcar = get_sigma_and_pk(X_mcar, device, "yahoo", "mcar")
    
    return fit_spl_lad(s_mnar, p_mnar), fit_spl_lad(s_mcar, p_mcar)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--num_clicks", type=int, default=2)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--num_users", type=int, default=5000)
    parser.add_argument("--num_items", type=int, default=5000)
    parser.add_argument("--density", type=float, default=0.01)
    args = parser.parse_args()
    
    device = get_device()
    out_dir = ensure_dir("aspire_experiments/output/exp15")
    
    # 1. 시뮬레이션 수행
    sim_history = run_simulation(args, device)
    
    # 2. 실제 데이터 수행
    mnar_res, mcar_res = run_real_data(device)
    
    # 3. 통합 시각화 (2-Panel)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 패널 A: 시뮬레이션
    ax = axes[0]
    res0, rest = sim_history[0], sim_history[-1]
    ax.scatter(res0["x"], res0["y"], color='gray', alpha=0.3, s=10, label=f'Iter 0 (Initial)')
    ax.plot(res0["x"], res0["slope"]*res0["x"] + res0["intercept"], color='gray', linestyle='--')
    
    ax.scatter(rest["x"], rest["y"], color='orange', alpha=0.4, s=15, label=f'Iter {args.iter} (Final)')
    ax.plot(rest["x"], rest["slope"]*rest["x"] + rest["intercept"], color='orange', linewidth=2, 
            label=f'Slope: {rest["slope"]:.2f}, $R^2$: {rest["r2"]:.2f}')
    
    ax.set_title("(a) Simulation: Emergence of SPL via Loop", fontsize=13)
    ax.set_xlabel(r"$\log \sigma_k$ (Singular Values)", fontsize=11)
    ax.set_ylabel(r"$\log p_k$ (Spectral Popularity)", fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.2)

    # 패널 B: 실제 데이터
    ax = axes[1]
    ax.scatter(mcar_res["x"], mcar_res["y"], color='red', alpha=0.3, s=10, label='MCAR (Test)')
    ax.plot(mcar_res["x"], mcar_res["slope"]*mcar_res["x"] + mcar_res["intercept"], color='red', linestyle='--')
    
    ax.scatter(mnar_res["x"], mnar_res["y"], color='blue', alpha=0.4, s=15, label='MNAR (Train)')
    ax.plot(mnar_res["x"], mnar_res["slope"]*mnar_res["x"] + mnar_res["intercept"], color='blue', linewidth=2,
            label=f'Slope: {mnar_res["slope"]:.2f}, $R^2$: {mnar_res["r2"]:.2f}')
            
    ax.set_title("(b) Real Data (Yahoo! R3): MNAR vs MCAR", fontsize=13)
    ax.set_xlabel(r"$\log \sigma_k$ (Singular Values)", fontsize=11)
    ax.set_ylabel(r"$\log p_k$ (Spectral Popularity)", fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spl_proof_master.png"), dpi=200)
    plt.close()

    # JSON 기록 (대용량 좌표 리스트 제거)
    for res in sim_history:
        res.pop("x", None); res.pop("y", None)
    mnar_res.pop("x", None); mnar_res.pop("y", None)
    mcar_res.pop("x", None); mcar_res.pop("y", None)
    
    final_results = {
        "config": vars(args),
        "simulation": sim_history,
        "yahoo_r3": {
            "mnar": mnar_res,
            "mcar": mcar_res
        }
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=4)
    
    print(f"Master proof completed. Figure and JSON saved to: {out_dir}")

if __name__ == "__main__":
    main()
