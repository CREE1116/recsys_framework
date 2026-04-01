import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

from src.utils.gpu_accel import EVDCacheManager
from src.data_loader import DataLoader
from src.evaluation import get_ndcg, get_recall

def run_final_aspire_zero(dataset_name):
    print(f"\n{'='*60}")
    print(f"ASPIRE-Zero: Final Self-Consistent Beta Saturation")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    # 1. Load Data
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
    
    # 2. Step 1: 스펙트럼 (Full EVD)
    manager = EVDCacheManager(device='cpu')
    X_sparse = csr_matrix((np.ones(len(dl.train_df)), (dl.train_df['user_id'], dl.train_df['item_id'])), 
                          shape=(dl.n_users, dl.n_items))
    _, s, v, _ = manager.get_evd(X_sparse, k=None, dataset_name=dataset_name)
    
    Lambda_obs = s.cpu().numpy() ** 2 # Eigenvalues (Power space)
    V = v.cpu().numpy()
    
    # Step 2 & 3: 준비
    log_rank = np.log(np.arange(1, len(Lambda_obs) + 1))
    
    def compute_slope(log_lam, log_r):
        # np.diff(y) / np.diff(x)
        return np.diff(log_lam) / (np.diff(log_r) + 1e-12)

    # Step 4: Bulk 구간 찾기 (Plateau Detection)
    # Numerical noise 방지를 위해 앞/뒤 마진(Margin)을 두고 탐색
    n = len(Lambda_obs)
    start_idx = int(n * 0.02) # 상위 2% 이후부터
    end_idx = int(n * 0.5)    # 중위 50% 이전까지
    
    raw_slope = compute_slope(np.log(Lambda_obs + 1e-12), log_rank)
    curvature = np.abs(np.diff(raw_slope))
    
    # Relevance Range 내에서 가장 큰 곡률 변화 2개 찾기
    curv_in_range = curvature[start_idx:end_idx]
    peaks = np.argsort(curv_in_range)[-2:] + start_idx
    k1, k2 = np.sort(peaks)
    
    print(f"[Step 4] Robust Plateau Found: [{k1} ~ {k2}]")

    # Step 5: flatness 측정 (개선도 측정용)
    def flatness_metric(slope, k1, k2):
        s = slope[k1:k2]
        if len(s) == 0: return 1e10
        return np.var(s) # 분산 하나만 써도 평탄도 측정에 충분

    # 3. Step 2: Beta Self-Consistent Iteration
    beta = 0.0
    step = 0.1
    tol = 1e-5 # 개선도 임계치
    max_iter = 200
    
    prev_f = float('inf')

    print(f"[Step 2] Iterating for Flatness Improvement Saturation...")
    for i in range(max_iter):
        # 1. 보정 스펙트럼
        Lambda_corr = Lambda_obs ** (1 / (1 + beta))
        log_lambda_corr = np.log(Lambda_corr + 1e-12)
        
        # 2. slope (Step 3)
        slope = compute_slope(log_lambda_corr, log_rank)
        
        # 3. bulk 찾기 (Step 4 - Robust Margin 적용)
        curr_n = len(Lambda_obs)
        curr_start = int(curr_n * 0.02)
        curr_end = int(curr_n * 0.5)
        curvature = np.abs(np.diff(slope))
        curv_in_range = curvature[curr_start:curr_end]
        peaks = np.argsort(curv_in_range)[-2:] + curr_start
        k1, k2 = np.sort(peaks)
        
        # 4. flatness 측정 (Step 5)
        f = flatness_metric(slope, k1, k2)
        
        # 5. stopping condition (핵심: Improvement 기준)
        if prev_f - f < tol:
            print(f"  [Stop] Improvement saturated at Iter {i:3d}: imp={prev_f - f:.8f}")
            break
            
        prev_f = f
        beta += step
        if i % 10 == 0:
            print(f"  Iter {i:3d}: beta={beta:.2f}, slope_var={f:.6f}")

    beta_star = beta
    print(f"[Convergence] Final Beta* = {beta_star:.4f} (Final Var={f:.6e})")

    # 4. Step 3 & 4: Wiener 필터 및 추천 생성
    # lam (HCR base) - From HPO or heuristically 10.0 for this test
    # (Note: In a real system, 'lam' is the only HPO parameter)
    lam_hpo = 50.0 if dataset_name == "ml1m" else 10.0 
    
    Lambda_corr_final = Lambda_obs ** (1 / (1 + beta))
    h = Lambda_corr_final / (Lambda_corr_final + lam_hpo)
    
    print(f"[Results]")
    print(f"  -> Total Compression: {1/(1+beta):.4f}")
    print(f"  -> Noise Floor (lam): {lam_hpo}")

    # 5. Visualization
    plt.figure(figsize=(10, 6))
    ranks = np.arange(1, len(Lambda_obs) + 1)
    plt.loglog(ranks, Lambda_obs, label="Observed ($\Lambda_{obs}$)", alpha=0.3)
    plt.loglog(ranks, Lambda_corr_final, label=f"Flattened ($\Lambda_{{corr}}$, $\\beta={beta:.2f}$)", color='red')
    plt.axvline(k1, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(k2, color='gray', linestyle='--', alpha=0.5)
    plt.title(f"ASPIRE-Zero: Self-Consistent Flatness ({dataset_name})")
    plt.legend()
    
    save_path = f"aspire_experiments/output/exp31/self_consistent_{dataset_name}.png"
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"  -> Plot saved to: {save_path}")

    # 6. 본질 요약
    print(f"\n[Essence]")
    print(f"  \"스펙트럼의 bulk가 white noise(slope=0)처럼 보일 때까지 power distortion을 제거했다.\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m")
    args = parser.parse_args()
    run_final_aspire_zero(args.dataset)
