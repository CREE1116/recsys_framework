import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

from scipy.sparse import csr_matrix
from src.utils.gpu_accel import EVDCacheManager
from src.data_loader import DataLoader
from src.evaluation import get_ndcg

def pure_wiener_self_consistency(lambda_obs: torch.Tensor, lambda_ext: float, 
                                 max_iter: int = 50, omega_start: float = 0.05, omega_end: float = 0.5):
    """
    Algorithm 1: Self-Consistent Wiener Optimality Recovery (Pure PyTorch)
    목표: Bulk 구간에서 필터 통과 전(tau)과 통과 후(s)의 OLS 기울기 차이가 0이 되는 beta 탐색
    """
    n_components = len(lambda_obs)
    start_idx = int(n_components * omega_start)
    end_idx = int(n_components * omega_end)
    
    # 1. X축 준비 (Bulk 영역의 log(rank)) - 루프 외부에서 1회만 계산
    ranks = torch.arange(start_idx + 1, end_idx + 1, dtype=torch.float32, device=lambda_obs.device)
    x = torch.log(ranks)
    x_mean = torch.mean(x)
    x_centered = x - x_mean
    x_var = torch.sum(x_centered ** 2) + 1e-12 # OLS 분모
    
    beta_min = torch.tensor(0.0, device=lambda_obs.device)
    beta_max = torch.tensor(10.0, device=lambda_obs.device) # Extended range
    
    for t in range(max_iter):
        beta_mid = (beta_min + beta_max) / 2.0
        gamma = 1.0 / (1.0 + beta_mid)
        
        # (1) Undistortion
        tau = torch.pow(lambda_obs, gamma)
        
        # (2) Wiener Filtering
        h = tau / (tau + lambda_ext + 1e-12)
        s = tau * h
        
        # 2. 기울기 측정 (OLS)
        # 2-1. 필터 통과 전 (tau) 기울기 측정
        y_tau = torch.log(tau[start_idx : end_idx] + 1e-12)
        y_tau_centered = y_tau - torch.mean(y_tau)
        slope_tau = torch.sum(x_centered * y_tau_centered) / x_var
        
        # 2-2. 필터 통과 후 (s) 기울기 측정
        y_s = torch.log(s[start_idx : end_idx] + 1e-12)
        y_s_centered = y_s - torch.mean(y_s)
        slope_s = torch.sum(x_centered * y_s_centered) / x_var
        
        # 3. 목적 함수: 기울기 변화량 (Self-Consistency Error)
        error = slope_tau - slope_s 
        
        if error < 0:
            beta_max = beta_mid  # beta가 너무 커서 과보정됨, 줄여야 함
        else:
            beta_min = beta_mid  # beta가 너무 작아서 덜 보정됨, 키워야 함
            
        if (beta_max - beta_min) < 1e-6:
            break
            
    # 최종 결과 반환
    beta_star = (beta_min + beta_max) / 2.0
    gamma_star = 1.0 / (1.0 + beta_star)
    tau_final = torch.pow(lambda_obs, gamma_star)
    h_final = tau_final / (tau_final + lambda_ext + 1e-12)
    s_final = tau_final * h_final
    
    return beta_star.item(), gamma_star.item(), tau_final, h_final, s_final

def run_self_consistency_test(dataset_name):
    print(f"\n" + "="*70)
    print(f"Algorithm: Self-Consistent Wiener Test (Slope Invariance)")
    print(f"Dataset: {dataset_name}")
    print(f"="*70)

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
    manager = EVDCacheManager(device='cpu')
    X_sparse = csr_matrix((np.ones(len(dl.train_df)), (dl.train_df['user_id'], dl.train_df['item_id'])), 
                          shape=(dl.n_users, dl.n_items))
    _, s, v, _ = manager.get_evd(X_sparse, k=None, dataset_name=dataset_name)
    
    lambda_obs = (s ** 2).clone().detach()
    sort_idx = torch.argsort(lambda_obs, descending=True)
    lambda_obs = lambda_obs[sort_idx]
    v_tensor = v[:, sort_idx]
    
    # Noise Floor
    lambda_ext = 50.0 if dataset_name == 'ml1m' else 10.0
    
    # 2. Run Self-Consistency Engine
    beta_star, gamma_star, tau_final, h_final, s_final = pure_wiener_self_consistency(
        lambda_obs, lambda_ext
    )
    
    print(f"\n[Final Results]")
    print(f"  -> β* : {beta_star:.6f}")
    print(f"  -> γ* : {gamma_star:.6f}")
    print(f"  -> λ_ext : {lambda_ext:.1f}")

    # 3. Evaluation
    test_users = dl.test_df['user_id'].unique()
    X_test_users = torch.from_numpy(X_sparse[test_users].toarray()).float()
    XV = torch.mm(X_test_users, v_tensor)
    scores = torch.mm(XV * h_final, v_tensor.t()).numpy()
    
    # Masking
    user_train_history = dl.train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    for i, u_id in enumerate(test_users):
        history = list(user_train_history.get(u_id, set()))
        if history:
            scores[i, history] = -1e9
            
    # Metrics
    ground_truth = dl.test_df.groupby('user_id')['item_id'].apply(list).to_dict()
    top_indices = np.argsort(scores, axis=1)[:, -20:][:, ::-1]
    
    ndcgs = []
    for i, u_id in enumerate(test_users):
        ndcgs.append(get_ndcg(top_indices[i], ground_truth.get(u_id, [])))
        
    print(f"  -> NDCG@20: {np.mean(ndcgs):.4f}")

    # 4. Save Plot
    plt.figure(figsize=(10, 6))
    plt.loglog(lambda_obs.numpy(), label="Observed", alpha=0.3)
    plt.loglog(tau_final.numpy(), label=f"Undistorted (Beta={beta_star:.3f})", color='blue')
    plt.loglog(s_final.numpy(), label="Filtered (Self-Consistent)", color='red')
    plt.axhline(lambda_ext, color='gray', linestyle='--', label=f"Noise (ext={lambda_ext})")
    plt.title(f"Self-Consistent Wiener Test: {dataset_name}")
    plt.legend()
    save_path = f"aspire_experiments/output/exp34/self_consistency_{dataset_name}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"  -> Plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m")
    args = parser.parse_args()
    run_self_consistency_test(args.dataset)
