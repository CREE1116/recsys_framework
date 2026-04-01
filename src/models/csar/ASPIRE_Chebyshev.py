import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from src.models.base_model import BaseModel

class ASPIRE_Chebyshev(BaseModel):
    """
    ASPIRE-Zero [EVD-free] Version:
    - Gamma Inference: Fast Lanczos (Top-50 Eigens)
    - Filtering: Chebyshev Polynomial Approximation (O(nnz))
    """

    def __init__(self, config, data_loader):
        super(ASPIRE_Chebyshev, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_config = config.get('model', {})
        self.K = model_config.get('cheb_order', 10)
        self.n_lanczos = model_config.get('n_lanczos', 50)
        self.max_iter = model_config.get('max_iter', 20)
        self.tol = 1e-5
        self.lambda_base_config = model_config.get('lambda_base', 'auto')

        # From Trainer/DataLoader
        # From Trainer/DataLoader
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        self._build(self.train_matrix_csr, config.get('dataset_name', 'unknown'))

    def _build_sparse_matrix(self, data_loader):
        df = data_loader.train_df
        return csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df['user_id'].values, df['item_id'].values)),
            shape=(self.n_users, self.n_items)
        )

    def _find_plateau(self, lambda_obs):
        """[EVD-free] 상위 고윳값 내에서 곡률 기반 벌크 구간 탐색"""
        log_lambda = np.log(lambda_obs + 1e-12)
        log_rank = np.log(np.arange(1, len(lambda_obs) + 1))
        
        n = len(lambda_obs)
        # Lanczos 상위 K개 내에서 탐색 범위를 5%~95%로 설정
        start_idx = max(1, int(n * 0.05))
        end_idx = int(n * 0.95)
        
        slope = np.diff(log_lambda) / (np.diff(log_rank) + 1e-12)
        curvature = np.abs(np.diff(slope))
        
        curv_in_range = curvature[start_idx:end_idx]
        if len(curv_in_range) < 2: # 너무 작은 경우 대비
             return 0, n-1
             
        peaks = np.argsort(curv_in_range)[-2:] + start_idx
        k1, k2 = np.sort(peaks)
        return k1, k2

    def _infer_gamma_fixed_point(self, lambda_obs, k1, k2, anchor_ext):
        """EVD-free Fixed-Point Iteration Engine"""
        ranks = np.arange(k1 + 1, k2 + 1, dtype=np.float64)
        x = np.log(ranks)
        x_centered = x - np.mean(x)
        x_var = np.sum(x_centered ** 2) + 1e-12
        
        gamma = 1.0
        for i in range(self.max_iter):
            prev_gamma = gamma
            tau = lambda_obs ** gamma
            h = tau / (tau + anchor_ext + 1e-12)
            s = tau * h
            
            log_s = np.log(s[k1:k2] + 1e-12)
            y_centered = log_s - np.mean(log_s)
            b = np.sum(x_centered * y_centered) / x_var
            
            gamma = 1.0 / (1.0 + abs(b))
            gamma = 0.5 * prev_gamma + 0.5 * gamma # Damping
            
            if abs(gamma - prev_gamma) < self.tol:
                break
        return gamma, abs(b)

    def _get_cheb_coeffs(self, gamma, lam_ext, lam_max):
        """
        Compute Chebyshev coefficients for f(lambda) = lambda^gamma / (lambda^gamma + lam_ext)
        Mapped to [-1, 1] domain.
        """
        n = self.K + 1
        # Chebyshev nodes in [-1, 1]
        nodes = np.cos(np.pi * (np.arange(n) + 0.5) / n)
        # Map to [0, lam_max]
        lams = (nodes + 1) * lam_max / 2.0
        
        # Target function
        tau = lams ** gamma
        f_vals = tau / (tau + lam_ext + 1e-12)
        
        coeffs = np.zeros(n)
        for k in range(n):
            coeffs[k] = (2.0 / n) * np.sum(f_vals * np.cos(k * np.pi * (np.arange(n) + 0.5) / n))
        return coeffs

    @torch.no_grad()
    def _build(self, X_sparse, dataset_name):
        # 1. Fast Lanczos for Top-K Eigenvalues
        # Using scipy eigsh on X.T @ X
        self._log(f"Running Fast Lanczos (K={self.n_lanczos})...")
        # Ensure we work on float64 for stability in eigenvalue calc
        X_f64 = X_sparse.astype(np.float64)
        Gram_op = X_f64.T @ X_f64
        lambdas, _ = eigsh(Gram_op, k=self.n_lanczos, which='LM')
        lambda_top_k = np.sort(lambdas)[::-1]
        
        self.lam_max = float(lambda_top_k[0])
        
        # 2. Noise Floor (RMT Based or Config)
        if self.lambda_base_config == 'auto':
            nnz = X_sparse.nnz
            U, I = X_sparse.shape
            self.lam_ext = np.sqrt(nnz / min(U, I))
        else:
            self.lam_ext = float(self.lambda_base_config)
            
        # 3. Automated Gamma Inference (Curvature-based)
        k1, k2 = self._find_plateau(lambda_top_k)
        self.gamma, self.beta = self._infer_gamma_fixed_point(lambda_top_k, k1, k2, self.lam_ext)
        
        # 4. Chebyshev Precomputation
        self._log(f"Inferred Gamma: {self.gamma:.4f} | Beta: {self.beta:.4f}")
        cheb_coeffs_np = self._get_cheb_coeffs(self.gamma, self.lam_ext, self.lam_max)
        self.register_buffer("cheb_coeffs", torch.from_numpy(cheb_coeffs_np).float().to(self.device))
        
        # 5. Prepare Train Matrix for GPU (Sparse for efficiency)
        indices = torch.LongTensor(np.vstack(X_sparse.nonzero()))
        values = torch.FloatTensor(X_sparse.data)
        self.X_train_sparse = torch.sparse_coo_tensor(indices, values, X_sparse.shape).to(self.device)

        # 6. EVD-free Analysis Dashboard
        self._save_spectral_analysis(lambda_top_k)

        self._log(f"EVD-free Build Complete (Cheb Order: {self.K}, lam_max: {self.lam_max:.2f})")

    def _apply_cheb_filter(self, X_batch):
        """
        Recursive Chebyshev: W_k = 2 * W_{k-1} * L_sc - W_{k-2}
        L_sc = (2/lam_max) * (X_train.T @ X_train) - I
        """
        # If batch is dense
        W0 = X_batch
        
        # L_sc product helper
        # W @ L_sc = (2/lam_max) * (W @ X_train.T @ X_train) - W
        def L_sc_prod(W):
            return (2.0 / self.lam_max) * (W @ self.X_train_sparse.t() @ self.X_train_sparse) - W

        W1 = L_sc_prod(W0)
        
        # Chebyshev expansion: res = c0/2 * W0 + c1 * W1 + ...
        res = (self.cheb_coeffs[0] / 2.0) * W0 + self.cheb_coeffs[1] * W1
        
        W_prev2 = W0
        W_prev1 = W1
        
        for k in range(2, self.K + 1):
            W_curr = 2.0 * L_sc_prod(W_prev1) - W_prev2
            res += self.cheb_coeffs[k] * W_curr
            W_prev2 = W_prev1
            W_prev1 = W_curr
        
        return res

    def forward(self, users):
        user_ids = users.cpu().numpy()
        # Fetch rows from train matrix
        # For simplicity in GPU-forward, if small enough, use sliced dense
        # But we need Matrix-free approach.
        
        # X_batch is current interactions for these users
        # For efficiency, we can use the same sparse matrix
        # indices_batch = [self.X_train_sparse[u] for u in user_ids]
        
        # For Torch compatibility, let's use dense slicing (CPU-based first or selective GPU)
        X_batch = self.X_train_sparse.to_dense()[users] # [B, I]
        
        # Apply Chebyshev Filter iteratively
        X_filtered = self._apply_cheb_filter(X_batch)
        
        return X_filtered

    def predict_for_pairs(self, user_ids, item_ids):
        # Similar logic for pairs
        users_unique = torch.unique(user_ids)
        X_batch = self.X_train_sparse.to_dense()[users_unique]
        X_filtered_all = self._apply_cheb_filter(X_batch)
        
        # Index back
        user_map = {u.item(): i for i, u in enumerate(users_unique)}
        mapped_idx = [user_map[u.item()] for u in user_ids]
        
        all_scores = X_filtered_all[mapped_idx, item_ids]
        return all_scores

    def get_final_item_embeddings(self):
        # Chebyshev base doesn't have explicit embeddings, it's a matrix operator
        return None

    def _save_spectral_analysis(self, lambda_top_k):
        """
        EVD-free Spectral Analysis: Using Lanczos Top-K and Analytical Cheb Response
        """
        output_dir = os.path.join(self.output_path, "spectral_analysis")
        os.makedirs(output_dir, exist_ok=True)
        dataset_name = self.config.get('dataset_name', 'unknown')
        
        # 1. Target function vs Cheb Approximation
        test_lams = np.linspace(0, self.lam_max, 500)
        # Target
        tau_target = test_lams ** self.gamma
        h_target = tau_target / (tau_target + self.lam_ext + 1e-12)
        
        # Cheb approx response
        nodes = (2.0 * test_lams / self.lam_max) - 1.0 # Map to [-1, 1]
        cheb_vals = np.zeros_like(test_lams)
        def T_k(n, x): return np.cos(n * np.arccos(x))
        for k in range(self.K + 1):
            cheb_vals += self.cheb_coeffs[k].item() * T_k(k, nodes)
        cheb_vals[0] -= self.cheb_coeffs[0].item() / 2.0 # Adjust for c0/2
        
        # Plotting
        plt.style.use('seaborn-v0_8-muted') if 'seaborn-v0_8-muted' in plt.style.available else plt.style.use('ggplot')
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"EVD-free ASPIRE-Zero Analysis: {dataset_name.upper()}\n"
                     rf"(Lanczos $\gamma={self.gamma:.4f}$, Chebyshev Order={self.K})", 
                     fontsize=14, fontweight='bold')

        # [Panel 1] Top-K Spectrum & Noise Floor
        ax = axes[0]
        ranks = np.arange(1, len(lambda_top_k) + 1)
        ax.scatter(ranks, lambda_top_k, color='#3498db', s=30, label='Top-K Eigenvalues (Lanczos)')
        ax.axhline(self.lam_ext, color='#e74c3c', linestyle='--', label=f'RMT Noise Floor ({self.lam_ext:.2f})')
        ax.set_yscale('log')
        ax.set_title("Spectral Head & Noise Floor", fontsize=12, fontweight='bold')
        ax.set_xlabel("Rank (k)", fontsize=10)
        ax.set_ylabel("Eigenvalue Scale (log)", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # [Panel 2] Filter Frequency Response
        ax = axes[1]
        ax.plot(test_lams, h_target, 'k--', alpha=0.5, label='Target Filter')
        ax.plot(test_lams, cheb_vals, color='#2ecc71', linewidth=2, label=f'Chebyshev Order {self.K}')
        ax.fill_between(test_lams, cheb_vals, alpha=0.1, color='#2ecc71')
        ax.set_title("Filter Frequency Response", fontsize=12, fontweight='bold')
        ax.set_xlabel("Eigenvalue ($\lambda$)", fontsize=10)
        ax.set_ylabel("Filter Gain $h(\lambda)$", fontsize=10)
        ax.set_ylim(-0.1, 1.2)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dashboard.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Metadata JSON (Unified with ASPIRE_Zero structure)
        metadata = {
            "config": {
                "dataset": dataset_name,
                "gamma_final": float(self.gamma),
                "beta_final": float(self.beta),
                "lambda_ext": float(self.lam_ext)
            },
            "spectral_stats": {
                "lambda_obs_max": float(self.lam_max),
                "lambda_obs_mean": float(lambda_top_k.mean()), # Head mean
                "signal_plateau_range": [0, len(lambda_top_k)]
            },
            "filter_stats": {
                "h_max": float(cheb_vals.max()),
                "h_min": float(cheb_vals.min()),
                "h_mean": float(cheb_vals.mean())
            }
        }
        with open(os.path.join(output_dir, "analysis.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        self._log(f"EVD-free diagnostics saved to: {output_dir}")

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
