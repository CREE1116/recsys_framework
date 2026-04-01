import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

from scipy.sparse import csr_matrix
from src.utils.gpu_accel import EVDCacheManager
from src.data_loader import DataLoader
from src.evaluation import get_ndcg

def run_wiener_optimality_test(dataset_name):
    print(f"\n{'='*60}")
    print(f"ASPIRE-Zero: Self-Consistent Chain Iteration Engine")
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
    manager = EVDCacheManager(device='cpu')
    X_sparse = csr_matrix((np.ones(len(dl.train_df)), (dl.train_df['user_id'], dl.train_df['item_id'])), 
                          shape=(dl.n_users, dl.n_items))
    _, s, v, _ = manager.get_evd(X_sparse, k=None, dataset_name=dataset_name)
    
    # Eigenvalue Base
    lambda_obs = s.cpu().numpy() ** 2
    sort_idx = np.argsort(lambda_obs)[::-1]
    lambda_obs = lambda_obs[sort_idx]
    v_np = v.cpu().numpy()[:, sort_idx]
    
    n_components = len(lambda_obs)
    log_rank = np.log(np.arange(1, n_components + 1))
    
    # 2. Strong Fixed-Point: Chain Iteration for Fast Convergence (Eigenvalue Power Base)
    print(f"[Engine] Strong Chain Iteration (beta = |b|)...")
    
    lambda_ext = 50.0 if dataset_name == 'ml1m' else 10.0
    start_bulk, end_bulk = int(n_components * 0.02), int(n_components * 0.5)
    
    gamma = 1.0 # Initialize on λ base
    prev_gamma = -1.0
    T = 20 
    
    for t in range(T):
        tau = lambda_obs ** gamma
        
        # Calculate Slope
        log_tau = np.log(tau + 1e-12)
        g_tau = np.diff(log_tau) / (np.diff(log_rank) + 1e-12)
        current_slope = np.median(g_tau[start_bulk:end_bulk])
        
        # Fixed point from Physical Equilibrium: beta = |current_slope|
        beta = abs(current_slope)
        gamma_new = 1.0 / (1.0 + beta)
        
        # Damping for stability
        gamma = 0.5 * gamma + 0.5 * gamma_new
        
        print(f"  Iter {t+1:2d}: gamma={gamma:.6f}, measured_slope={current_slope:.6f}")
        
        if abs(gamma - prev_gamma) < 1e-5:
            break
        prev_gamma = gamma

    gamma_star = gamma
    tau_final = lambda_obs ** gamma_star
    h_final = tau_final / (tau_final + lambda_ext + 1e-12)
    
    print(f"\n[Final Results]")
    print(f"  -> β* : {beta_star:.6f}")
    print(f"  -> γ* : {gamma_star:.6f}")

    # 3. Evaluation
    test_users = dl.test_df['user_id'].unique()
    X_test_users = torch.from_numpy(X_sparse[test_users].toarray()).float()
    
    XV = torch.mm(X_test_users, torch.from_numpy(v_np).float())
    scores = torch.mm(XV * torch.from_numpy(h_final).float(), torch.from_numpy(v_np).float().t()).numpy()
    
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
    plt.loglog(lambda_obs, label="Observed", alpha=0.3)
    plt.loglog(tau_final, label=f"Undistorted (Beta={beta_star:.3f})", color='blue')
    plt.loglog(tau_final * h_final, label="Filtered", color='red')
    plt.title(f"ASPIRE-Zero (Chain Iteration): {dataset_name}")
    plt.legend()
    save_path = f"aspire_experiments/output/exp33/chain_iteration_{dataset_name}.png"
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"  -> Plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m")
    args = parser.parse_args()
    run_wiener_optimality_test(args.dataset)
