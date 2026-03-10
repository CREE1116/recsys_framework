import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.sparse import csr_matrix
from sklearn.linear_model import HuberRegressor

# Framework root path
sys.path.append(os.getcwd())

from src.data_loader import DataLoader
from src.utils.gpu_accel import SVDCacheManager
from src.models.csar.ASPIRELayer import AspireEngine
from aspire_experiments.exp_utils import get_loader_and_svd, ensure_dir

def setup_output_dirs(dataset_name, seed, base_dir="aspire_experiments/output"):
    paths = {}
    for exp in ["slp", "powerlaw", "tracking"]:
        path = os.path.join(base_dir, exp, dataset_name, f"seed_{seed}")
        os.makedirs(path, exist_ok=True)
        paths[exp] = path
    return paths

def get_interaction_matrix(loader):
    """DataLoader에서 interaction matrix R 생성"""
    n_users = loader.n_users
    n_items = loader.n_items
    train_df = loader.train_df
    
    rows = train_df['user_id'].values
    cols = train_df['item_id'].values
    vals = np.ones(len(rows))
    
    R = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    return R

def experiment_1_slp(R, V, dataset_name, out_dir):
    print(f"  Running Experiment 1: SLP Verification...")
    # Item popularity
    p = np.array(R.sum(axis=0)).flatten()
    p_norm = p / (p.max() + 1e-9)
    P_diag = p_norm
    
    # M = V^T P V
    V_np = V.cpu().numpy()
    M = (V_np.T * P_diag) @ V_np
    
    diag_vals = np.diag(M)
    mask = ~np.eye(len(M), dtype=bool)
    epsilon = float(np.mean(np.abs(M[mask])) / (np.mean(diag_vals) + 1e-9))
    
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(M, cmap='viridis')
    plt.title(f"Spectral Popularity Matrix M (Rank K={V_np.shape[1]})\nDataset: {dataset_name}, ε={epsilon:.4f}")
    plt.savefig(os.path.join(out_dir, "slp_heatmap.png"))
    plt.close()
    
    result = {
        "dataset": dataset_name,
        "epsilon": float(epsilon),
        "rank_k": int(V.shape[1])
    }
    with open(os.path.join(out_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"    SLP Epsilon: {epsilon:.4f}")
    return result

def experiment_2_powerlaw(S, V, item_pops, dataset_name, out_dir):
    print(f"  Running Experiment 2: Power-law Coupling...")
    s_np = S.cpu().numpy()
    
    p_tilde = AspireEngine.compute_spp(V, item_pops)
    
    # Beta estimation
    beta, r2 = AspireEngine.estimate_beta(S, p_tilde, verbose=False)
    
    # Log-Log fitting for visualization
    x = np.log(s_np + 1e-9)
    y = np.log(p_tilde + 1e-9)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5, label='Data points')
    
    # Linear fit for plotting
    hub = HuberRegressor()
    hub.fit(x.reshape(-1, 1), y)
    y_pred = hub.predict(x.reshape(-1, 1))
    
    plt.plot(x, y_pred, color='red', label=f'Huber Fit (slope={hub.coef_[0]:.3f}, beta={beta:.4f})')
    plt.xlabel("log(σ_k)")
    plt.ylabel("log(p̃_k)")
    plt.title(f"Spectral Power-law Coupling\nDataset: {dataset_name}, R²={r2:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "powerlaw_fit.png"))
    plt.close()
    
    result = {
        "dataset": dataset_name,
        "beta": float(beta),
        "r2": float(r2),
        "slope": float(hub.coef_[0])
    }
    with open(os.path.join(out_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"    Estimated Beta: {beta:.4f}, R²: {r2:.4f}")
    return result

def experiment_3_tracking(R, K, dataset_name, out_dir, remove_levels=[0.0, 0.2, 0.4, 0.6, 0.8]):
    print(f"  Running Experiment 3: Beta-MNAR Tracking...")
    
    n_users, n_items = R.shape
    popularity = np.array(R.sum(axis=0)).flatten()
    item_rank = np.argsort(popularity)
    
    # Tail 40% items
    tail_threshold = int(n_items * 0.4)
    tail_items = item_rank[:tail_threshold]
    
    mnar_betas = []
    mcar_betas = []
    
    for r in remove_levels:
        # MNAR: Remove from tail items
        R_mnar = R.copy().tolil()
        for item in tail_items:
            indices = R_mnar.getcol(item).nonzero()[0]
            if len(indices) > 0:
                remove_n = int(len(indices) * r)
                if remove_n > 0:
                    remove_idx = np.random.choice(indices, remove_n, replace=False)
                    for ridx in remove_idx:
                        R_mnar[ridx, item] = 0
        
        # MCAR: Remove randomly from all interactions
        R_mcar = R.copy().tolil()
        total_interactions = R.nnz
        remove_n_total = int(total_interactions * (r * 0.4)) # Adjusted to match scale
        all_rows, all_cols = R.nonzero()
        remove_indices = np.random.choice(total_interactions, remove_n_total, replace=False)
        for idx in remove_indices:
            R_mcar[all_rows[idx], all_cols[idx]] = 0
            
        # Estimate beta for both
        def quick_beta(R_mod):
            R_mod_csr = R_mod.tocsr()
            # Fast SVD
            from scipy.sparse.linalg import svds
            u, s, vt = svds(R_mod_csr.astype(float), k=K)
            idx = np.argsort(s)[::-1]
            s, vt = s[idx], vt[idx, :]
            V_mod = torch.from_numpy(vt.T.copy()).float()
            S_mod = torch.from_numpy(s.copy()).float()
            item_pops = np.array(R_mod_csr.sum(axis=0)).flatten()
            p_tilde = AspireEngine.compute_spp(V_mod, item_pops)
            beta, _ = AspireEngine.estimate_beta(S_mod, p_tilde, verbose=False)
            return beta

        beta_mnar = quick_beta(R_mnar)
        beta_mcar = quick_beta(R_mcar)
        
        mnar_betas.append(beta_mnar)
        mcar_betas.append(beta_mcar)
        print(f"    Ratio {r:.1f}: MNAR_beta={beta_mnar:.4f}, MCAR_beta={beta_mcar:.4f}")

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.plot(remove_levels, mnar_betas, 'o-', label='MNAR (Tail removal)', color='blue')
    plt.plot(remove_levels, mcar_betas, 's-', label='MCAR (Random removal)', color='gray')
    plt.xlabel("Removal Ratio")
    plt.ylabel("Estimated Beta")
    plt.title(f"Beta-MNAR Tracking\nDataset: {dataset_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, "beta_tracking.png"))
    plt.close()
    
    result = {
        "remove_levels": remove_levels,
        "mnar_betas": [float(b) for b in mnar_betas],
        "mcar_betas": [float(b) for b in mcar_betas]
    }
    with open(os.path.join(out_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=4)
        
    return result

def main():
    parser = argparse.ArgumentParser(description="Run ASPIRE Theory Experiments")
    parser.add_argument("--datasets", nargs='+', default=["ml100k"], help="Dataset names (space-separated)")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42], help="Random seeds (space-separated)")
    parser.add_argument("--energy", type=float, default=0.95, help="Target energy for SVD rank")
    parser.add_argument("--k", type=int, default=None, help="Fixed rank K (optional, overrides energy)")
    args = parser.parse_args()

    datasets = args.datasets
    seeds = args.seeds

    for dataset in datasets:
        print(f"\nProcessing Dataset: {dataset}")
        
        for seed in seeds:
            print(f" Seed: {seed}")
            # Set seeds
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # Load data and SVD
            try:
                loader, R, S, V, config = get_loader_and_svd(dataset, k=args.k, target_energy=args.energy)
            except Exception as e:
                print(f"  Error loading dataset {dataset}: {e}")
                continue

            item_pops = np.array(R.sum(axis=0)).flatten()
            exp_paths = setup_output_dirs(dataset, seed)

            # Run Experiments
            experiment_1_slp(R, V, dataset, exp_paths["slp"])
            experiment_2_powerlaw(S, V, item_pops, dataset, exp_paths["powerlaw"])
            experiment_3_tracking(R, K=min(50, V.shape[1]), dataset_name=dataset, out_dir=exp_paths["tracking"])

    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    main()
