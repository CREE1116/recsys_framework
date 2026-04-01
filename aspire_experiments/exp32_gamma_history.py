import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

from src.utils.gpu_accel import EVDCacheManager
from src.data_loader import DataLoader

def run_gamma_chain_iteration(dataset_name):
    print(f"\n{'='*60}")
    print(f"ASPIRE-Zero: Self-Consistent Slope-Beta Chain Iteration")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    # 1. Load Data & Get Spectrum
    config = {
        'ml1m': {
            'dataset_name': "ml-1m", 'data_path': "./data/ml1m/ratings.dat", 'separator': "::", 'columns': ["user_id", "item_id", "rating", "timestamp"],
            'rating_threshold': 0, 'min_user_interactions': 5, 'min_item_interactions': 5, 'split_method': "temporal_ratio", 'train_ratio': 0.8, 'valid_ratio': 0.1, 'data_cache_path': "./data_cache/"
        },
        'yahoo_r3': {
            'dataset_name': "yahoo_r3", 'train_file': "./data/yahooR3/processed/train_implicit_th0.txt", 'test_file': "./data/yahooR3/processed/test_implicit_th0.txt",
            'separator': "\t", 'columns': ["user_id", "item_id", "rating"], 'rating_threshold': 0, 'split_method': "presplit", 'data_cache_path': "./data_cache/"
        }
    }.get(dataset_name, {})
    
    dl = DataLoader(config)
    manager = EVDCacheManager(device='cpu')
    X_sparse = csr_matrix((np.ones(len(dl.train_df)), (dl.train_df['user_id'], dl.train_df['item_id'])), 
                          shape=(dl.n_users, dl.n_items))
    _, s, v, _ = manager.get_evd(X_sparse, k=None, dataset_name=dataset_name)
    
    Lambda_obs = s.cpu().numpy() ** 2
    log_rank = np.log(np.arange(1, len(Lambda_obs) + 1))
    
    def compute_slope(log_lam, log_r):
        return np.diff(log_lam) / (np.diff(log_r) + 1e-12)

    # 2. Iterate for Self-Consistent Beta
    beta = 0.0
    gamma_history = []
    tol = 1e-5
    max_iter = 50
    
    # Pre-find Plateau Region in observed spectrum
    n = len(Lambda_obs)
    start_idx, end_idx = int(n*0.02), int(n*0.5)
    raw_slope = compute_slope(np.log(Lambda_obs + 1e-12), log_rank)
    curvature = np.abs(np.diff(raw_slope))
    peaks = np.argsort(curvature[start_idx:end_idx])[-2:] + start_idx
    k1, k2 = np.sort(peaks)
    
    print(f"[Plateau] Fixed for iteration: [{k1} ~ {k2}]")
    print(f"[Iteration Start]")
    
    for i in range(max_iter):
        # 1. 보정 스펙트럼
        Lambda_corr = Lambda_obs ** (1.0 / (1.0 + beta))
        log_lambda_corr = np.log(Lambda_corr + 1e-12)
        
        # 2. slope (bulk 대역 median 기울기 측정)
        d = compute_slope(log_lambda_corr, log_rank)
        s_region = d[k1:k2]
        b_new = abs(np.median(s_region)) # b는 기울기의 크기 (양수화)
        
        # 3. gamma 계산 (User formula: 2 / (1 + b))
        gamma = 2.0 / (1.0 + b_new)
        gamma_history.append(gamma)
        
        # 4. beta 업데이트
        beta_new = b_new
        
        print(f"  Iter {i:2d}: beta={beta:.4f}, slope(b)={b_new:.4f}, gamma={gamma:.4f}")
        
        if abs(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new

    gamma_star = np.mean(gamma_history)
    print(f"\n[Convergence]")
    print(f"  -> Final Beta*  : {beta:.4f}")
    print(f"  -> Final Gamma* : {gamma_star:.4f} (Mean of History)")

    # 3. Visualization
    plt.figure(figsize=(10, 6))
    ranks = np.arange(1, len(Lambda_obs) + 1)
    Lambda_final = Lambda_obs ** (1.0 / (1.0 + beta))
    
    plt.loglog(ranks, Lambda_obs, label="Observed ($\Lambda_{obs}$)", alpha=0.3)
    plt.loglog(ranks, Lambda_final, label=f"Self-Consistent ($\Lambda_{{corr}}$, $\\beta={beta:.2f}$)", color='red')
    plt.axvline(k1, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(k2, color='gray', linestyle='--', alpha=0.5)
    plt.title(f"ASPIRE-Zero: Chain Iteration ({dataset_name})")
    plt.legend()
    
    save_path = f"aspire_experiments/output/exp32/chain_iter_{dataset_name}.png"
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"  -> Plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m")
    args = parser.parse_args()
    run_gamma_chain_iteration(args.dataset)
