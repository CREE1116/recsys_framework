import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from src.models.base_model import BaseModel
from src.utils.gpu_accel import EVDCacheManager

class ASPIRE_Chebyshev(BaseModel):
    """
    ASPIRE-Chebyshev: Primitive Fixed-Point Engine (The Original)
    """

    def __init__(self, config, data_loader):
        super(ASPIRE_Chebyshev, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_config = config.get('model', {})
        self.max_iter  = model_config.get('max_iter', 20)
        self.k         = model_config.get('k', 500)
        self.n_cheb    = model_config.get('n_cheb', 10)
        self.tol       = 1e-4
        self.lambda_base_config = model_config.get('lambda_base', 'auto')

        self.register_buffer("cheb_coeffs", torch.empty(0))
        self.V = None
        self.h = None
        
        self.gamma      = 1.0
        self.beta       = 0.0

        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        self._build(self.train_matrix_csr, config.get('dataset_name', 'unknown'))

    def _build_sparse_matrix(self, data_loader):
        df = data_loader.train_df
        return csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df['user_id'].values, df['item_id'].values)),
            shape=(self.n_users, self.n_items)
        )

        n = len(lambda_obs)
        
        # 2% ~ 50% 구간 제약 적용
        start_idx = max(5, int(n * 0.02))
        end_idx = int(n * 0.5)
        
        slope = np.diff(log_lambda) / (np.diff(log_rank) + 1e-12)
        curvature = np.abs(np.diff(slope))
        
        curv_in_range = curvature[start_idx:end_idx]
        
        # 전역 극대(Max)와 전역 극소(Min)를 기점으로 Plateau 구간 획정
        k_max = np.argmax(curv_in_range) + start_idx
        k_min = np.argmin(curv_in_range) + start_idx
        
        k1, k2 = np.sort([k_max, k_min])
        
        if k2 - k1 < 10:
            k2 = min(k1 + 20, n - 2)
            
        return int(k1), int(k2)

    def _infer_gamma_cheb(self, lambda_obs, k1, k2, anchor_ext):
        """Standard ASPIRE Fixed-point engine for Chebyshev"""
        ranks = np.arange(k1 + 1, k2 + 1, dtype=np.float64)
        x = np.log(ranks)
        x_centered = x - np.mean(x)
        x_var = np.sum(x_centered ** 2) + 1e-12
        
        gamma = 1.0
        b_final = 0.0
        for i in range(self.max_iter):
            prev_gamma = gamma
            tau = lambda_obs ** gamma
            h = tau / (tau + anchor_ext + 1e-12)
            s = tau * h
            
            log_s = np.log(s[k1:k2] + 1e-12)
            y_centered = log_s - np.mean(log_s)
            b = np.sum(x_centered * y_centered) / x_var
            
            gamma = 1.0 / (1.0 + abs(b))
            b_final = abs(b)
            if abs(gamma - prev_gamma) < self.tol:
                break
        return gamma, b_final

    @torch.no_grad()
    def _build(self, X_sparse, dataset_name):
        manager = EVDCacheManager(device=self.device.type)
        u_t, s_t, v_t, lam_max = manager.get_evd(X_sparse, k=self.k, dataset_name=dataset_name)

        lambda_obs = s_t.cpu().numpy() ** 2
        sort_idx = np.argsort(lambda_obs)[::-1]
        lambda_obs = lambda_obs[sort_idx]
        v_np = v_t.cpu().numpy()[:, sort_idx]
        
        nnz = X_sparse.nnz
        U_dim, I_dim = X_sparse.shape
        self.lam_ext = np.sqrt(nnz / min(U_dim, I_dim))
        self.lam_max = float(lam_max) if lam_max else lambda_obs[0]
        
        # 1. Plateau & Gamma (The Primitive Engine)
        self.gamma, k1, k2 = self._infer_gamma_fixed_point(lambda_obs, self.lam_ext)
        
        # 2. Chebyshev Coefficients precomputation
        cheb_coeffs_np = self._get_cheb_coeffs(self.gamma, self.lam_ext, self.lam_max)
        self.register_buffer("cheb_coeffs", torch.from_numpy(cheb_coeffs_np).float().to(self.device))
        
        # Subspace fallback build
        tau_final = lambda_obs ** self.gamma
        h_np = tau_final / (tau_final + self.lam_ext + 1e-12)
        
        self.h = torch.from_numpy(h_np.astype(np.float32)).to(self.device)
        self.V = torch.from_numpy(v_np).float().to(self.device)

        # 3. Diagnostics
        self._save_spectral_analysis(lambda_obs, tau_final, h_np, tau_final * h_np, k1, k2, self.lam_ext)
        print(f"[ASPIRE-Chebyshev] Primitive Engine Built | Plateau: [{k1}~{k2}] | Gamma*: {self.gamma:.4f}")

    def _get_cheb_coeffs(self, gamma, lam_ext, lam_max):
        from scipy.integrate import quad
        def f(lam):
            sigma = np.sqrt(lam)
            sigma_1 = np.sqrt(lam_max)
            return (sigma**gamma) / (sigma**gamma + sigma_1**gamma + 1e-10)
        def integrand(theta, j):
            lam = 0.5 * lam_max * (np.cos(theta) + 1.0)
            return f(lam) * np.cos(j * theta)
        coeffs = []
        for j in range(self.n_cheb + 1):
            val, _ = quad(integrand, 0, np.pi, args=(j,))
            c = (2.0 if j > 0 else 1.0) / np.pi * val
            coeffs.append(c)
        return np.array(coeffs)

    def _save_spectral_analysis(self, lambda_obs, tau, h, s, k1, k2, anchor_ext):
        import matplotlib.pyplot as plt
        output_dir = os.path.join(self.output_path, "spectral_analysis")
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-muted') if 'seaborn-v0_8-muted' in plt.style.available else plt.style.use('ggplot')
        ranks = np.arange(1, len(lambda_obs) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"ASPIRE-Chebyshev Primitive Dashboard (γ={self.gamma:.4f})")
        axes[0].loglog(ranks, lambda_obs, alpha=0.3)
        axes[0].loglog(ranks, tau, color='green', linewidth=2)
        axes[1].plot(ranks, h, color='orange')
        axes[2].loglog(ranks, s, color='purple')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dashboard.png"))
        plt.savefig("spectral_audit.png")
        plt.close()

    def forward(self, users):
        if torch.is_tensor(users) and users.dtype in (torch.int64, torch.long):
            batch = users.cpu().numpy()
            X_u = torch.from_numpy(self.train_matrix_csr[batch].toarray()).float().to(self.device)
        else:
            X_u = users.float().to(self.device)
        latent = X_u @ self.V
        weighted_latent = latent * self.h.unsqueeze(0)
        scores = weighted_latent @ self.V.t()
        return scores

    def predict_for_pairs(self, users, items):
        batch = users.cpu().numpy()
        X_u = torch.from_numpy(self.train_matrix_csr[batch].toarray()).float().to(self.device)
        latent = X_u @ self.V
        weighted_latent = latent * self.h.unsqueeze(0)
        item_vecs = self.V[items]
        return (weighted_latent * item_vecs).sum(dim=1).cpu().numpy()

    def calc_loss(self, batch):
        return (torch.tensor(0.0, device=self.device),), {}

    def get_final_item_embeddings(self):
        return self.V

    def score(self, users):
        return self.forward(users)
