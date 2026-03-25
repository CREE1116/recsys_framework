import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import time
from scipy.sparse import csr_matrix

# Framework root path
sys.path.append(os.getcwd())

from aspire_experiments.exp_utils import load_config, ensure_dir, get_eval_config
from src.data_loader import DataLoader
from src.evaluation import evaluate_metrics

class LiteSVDASPIRE:
    """Minimal SVD-based ASPIRE for timing."""
    def __init__(self, n_users, n_items, k=None, gamma=1.0, target_energy=1.0):
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.gamma = gamma
        self.target_energy = target_energy
        self.V = None
        self.filter_diag = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def eval(self):
        pass

    def fit(self, R_sparse):
        # 1. EVD of Gram Matrix (X^T X)
        X = R_sparse
        XTX = (X.T @ X).toarray()
        S, V = np.linalg.eigh(XTX)
        
        # Sort descending
        idx = np.argsort(S)[::-1]
        S, V = S[idx], V[:, idx]
        
        # Truncate by k or energy
        if self.k is not None:
             S, V = S[:self.k], V[:, :self.k]
        elif self.target_energy < 1.0:
            cumsum = np.cumsum(S)
            k_val = np.where(cumsum / (cumsum[-1] + 1e-12) >= self.target_energy)[0][0] + 1
            S, V = S[:k_val], V[:, :k_val]
        
        # 2. Filter
        s_gamma = np.power(np.maximum(S, 1e-12), self.gamma / 2.0)
        effective_lambda = s_gamma.max()
        h = s_gamma / (s_gamma + effective_lambda + 1e-10)
        
        self.V = torch.from_numpy(V).float().to(self.device)
        self.filter_diag = torch.from_numpy(h).float().to(self.device)
        self.train_matrix = R_sparse

    def forward(self, users):
        batch_users = users.cpu().numpy()
        X_batch = torch.from_numpy(self.train_matrix[batch_users].toarray()).float().to(self.device)
        XV = X_batch @ self.V
        return (XV * self.filter_diag) @ self.V.t()

class LiteChebyASPIRE:
    """Minimal ChebyASPIRE for timing."""
    def __init__(self, n_users, n_items, degree=20, gamma=1.0):
        self.n_users = n_users
        self.n_items = n_items
        self.degree = degree
        self.gamma = gamma
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.X_sparse = None
        self.Xt_sparse = None
        self.coeffs = None
        self.t_mid = 0
        self.t_half = 0

    def eval(self):
        pass

    def fit(self, R_sparse):
        # 1. Sparse Matrix Setup
        coo = R_sparse.tocoo()
        indices = torch.stack([torch.from_numpy(coo.row).long(), torch.from_numpy(coo.col).long()])
        values = torch.from_numpy(coo.data).float()
        self.X_sparse = torch.sparse_coo_tensor(indices, values, R_sparse.shape).coalesce().to(self.device)
        self.Xt_sparse = self.X_sparse.t().coalesce()
        
        # 2. Estimate lambda_max (Power Iteration)
        v = torch.randn(R_sparse.shape[1], 1, device=self.device)
        for _ in range(10): # 10 iterations enough for rough estimate
            v = torch.sparse.mm(self.Xt_sparse, torch.sparse.mm(self.X_sparse, v))
            lam = torch.norm(v)
            v /= lam
        lam_max = float(lam) * 1.01
        self.t_mid = self.t_half = lam_max / 2.0
        
        # 3. Compute Coefficients
        K = self.degree
        j = np.arange(K + 1)
        theta = np.pi * (j + 0.5) / (K + 1)
        lam_nodes = self.t_mid + self.t_half * np.cos(theta)
        
        s_gamma = np.power(lam_nodes, self.gamma / 2.0)
        f_nodes = s_gamma / (s_gamma + s_gamma.max() + 1e-10)
        
        self.coeffs = np.zeros(K + 1)
        for k_idx in range(K + 1): # Renamed k to k_idx to avoid conflict with function parameter k
            self.coeffs[k_idx] = (2.0 / (K + 1)) * np.sum(f_nodes * np.cos(k_idx * theta))
        self.coeffs[0] /= 2.0
        self.coeffs = torch.from_numpy(self.coeffs).float().to(self.device)
        self.train_matrix = R_sparse

    def forward(self, users):
        batch_users = users.cpu().numpy()
        X_batch = torch.from_numpy(self.train_matrix[batch_users].toarray()).float().to(self.device)
        
        # Chebyshev Recurrence
        T_prev = X_batch.t()
        # (L @ T_prev - mid * T_prev) / half
        temp = torch.sparse.mm(self.Xt_sparse, torch.sparse.mm(self.X_sparse, T_prev))
        T_curr = (temp - self.t_mid * T_prev) / self.t_half
        
        W = self.coeffs[0] * T_prev + self.coeffs[1] * T_curr
        for k_idx in range(2, self.degree + 1): # Renamed k to k_idx to avoid conflict with function parameter k
            temp = torch.sparse.mm(self.Xt_sparse, torch.sparse.mm(self.X_sparse, T_curr))
            T_next = 2.0 * (temp - self.t_mid * T_curr) / self.t_half - T_prev
            W += self.coeffs[k_idx] * T_next
            T_prev, T_curr = T_curr, T_next
        
        return W.t()

def run_exp4(dataset_name, k=None):
    print(f"Running Exp 4 on {dataset_name} (Lite Models)...")
    config = load_config(dataset_name)
    loader = DataLoader(config)
    
    # Prep sparse matrix
    train_df = loader.train_df
    R_sparse = csr_matrix((np.ones(len(train_df)), (train_df['user_id'].values, train_df['item_id'].values)), 
                          shape=(loader.n_users, loader.n_items))
    
    results = []
    eval_cfg = get_eval_config(loader, {"top_k": [20], "metrics": ["NDCG"]})
    test_loader = loader.get_final_loader(batch_size=2048)
    
    import resource
    def get_peak_mem():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024) # MB (on Mac, ru_maxrss is bytes)

    # 1. Full SVD Baseline (or Truncated k)
    print(f"  Testing SVD ASPIRE baseline (k={k})...")
    t0 = time.time()
    mem0 = get_peak_mem()
    svd_model = LiteSVDASPIRE(loader.n_users, loader.n_items, k=k, gamma=1.0, target_energy=1.0)
    svd_model.fit(R_sparse)
    svd_build_time = time.time() - t0
    svd_peak_mem = get_peak_mem() - mem0
    
    t0 = time.time()
    metrics = evaluate_metrics(svd_model, loader, eval_cfg, svd_model.device, test_loader, is_final=True)
    svd_eval_time = time.time() - t0
    
    results.append({
        "N": "Full SVD",
        "NDCG@20": metrics['NDCG@20'],
        "build_time": svd_build_time,
        "eval_time": svd_eval_time,
        "peak_mem_mb": svd_peak_mem
    })

    # 2. Chebyshev Sweep
    n_values = [1, 3, 5, 10, 20, 30, 40]
    for n in n_values:
        print(f"  Testing Chebyshev Order N={n}...")
        t0 = time.time()
        mem0 = get_peak_mem()
        model = LiteChebyASPIRE(loader.n_users, loader.n_items, degree=n, gamma=1.0)
        model.fit(R_sparse)
        build_time = time.time() - t0
        peak_mem = get_peak_mem() - mem0
        
        t0 = time.time()
        metrics = evaluate_metrics(model, loader, eval_cfg, model.device, test_loader, is_final=True)
        eval_time = time.time() - t0
        
        results.append({
            "N": n,
            "NDCG@20": metrics['NDCG@20'],
            "build_time": build_time,
            "eval_time": eval_time,
            "peak_mem_mb": peak_mem
        })

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x_labels = [str(r['N']) for r in results]
    ndcg = [r['NDCG@20'] for r in results]
    build_times = [r['build_time'] for r in results]
    eval_times = [r['eval_time'] for r in results]
    total_times = [r['build_time'] + r['eval_time'] for r in results]
    
    x = np.arange(len(x_labels))
    ax1.set_xlabel('Model Configuration')
    ax1.set_ylabel('NDCG@20', color='blue')
    ax1.plot(x, ndcg, marker='o', color='blue', linewidth=2, label='NDCG@20')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Time (s)', color='red')
    ax2.bar(x, total_times, alpha=0.3, color='red', label='Total Time')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title(f"ChebyASPIRE vs Full EVD Efficiency: {dataset_name}")
    fig.tight_layout()
    
    out_dir = ensure_dir(f"aspire_experiments/output/exp4/{dataset_name}")
    plt.savefig(os.path.join(out_dir, "cheby_tradeoff_plot.png"), dpi=150)
    plt.close()
    
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"k": k if k is not None else min(10000, loader.n_items), "results": results}, f, indent=4)
        
    print(f"Exp 4 on {dataset_name} finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ml1m", help="Dataset name")
    parser.add_argument("--k", type=int, default=None, help="Rank k for SVD baseline")
    args = parser.parse_args()
    run_exp4(args.dataset, k=args.k)
