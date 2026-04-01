import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from src.models.base_model import BaseModel
from src.utils.gpu_accel import EVDCacheManager

class ASPIRE_Naked(BaseModel):
    """
    ASPIRE-Naked: Primitive Fixed-Point Engine
    
    [핵심 메커니즘]
    - 안전장치(Damping, OLS, Derivative Plateau)를 모두 제거.
    - 오직 두 지점의 로그 기울기만을 이용한 무관성 고정점 반복.
    """

    def __init__(self, config, data_loader):
        super(ASPIRE_Naked, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_config = config.get('model', {})
        self.max_iter  = model_config.get('max_iter', 20)
        self.k         = model_config.get('k', 500)
        self.tol       = 1e-4
        self.lambda_base_config = model_config.get('lambda_base', 'auto')

        # Buffers
        self.register_buffer("V_raw",       torch.empty(0, 0))
        self.register_buffer("filter_diag", torch.empty(0))

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

    def _find_plateau(self, lambda_obs):
        """신호 대역(Signal Plateau)의 경계를 곡률 피크 기반으로 자동 탐색"""
        log_lambda = np.log(lambda_obs + 1e-12)
        log_rank = np.log(np.arange(1, len(lambda_obs) + 1))
        
        n = len(lambda_obs)
        start_idx = int(n * 0.02)
        end_idx = int(n * 0.5)
        
        slope = np.diff(log_lambda) / (np.diff(log_rank) + 1e-12)
        curvature = np.abs(np.diff(slope))
        
        curv_in_range = curvature[start_idx:end_idx]
        peaks = np.argsort(curv_in_range)[-2:] + start_idx
        k1, k2 = np.sort(peaks)
        return k1, k2

    def _infer_gamma_naked(self, lambda_obs, k1, k2, anchor_ext):
        """Standard ASPIRE-Zero Fixed-point engine for Naked"""
        n_components = len(lambda_obs)
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
            # No Damping: Direct Fixed-Point Update
            
            if abs(gamma - prev_gamma) < self.tol:
                break
        return gamma, abs(b)

    @torch.no_grad()
    def _build(self, X_sparse, dataset_name):
        manager = EVDCacheManager(device=self.device.type)
        _, s, v, _ = manager.get_evd(X_sparse, k=None, dataset_name=dataset_name)

        # 1. Eigenvalues Base
        lambda_obs = s.cpu().numpy() ** 2
        sort_idx = np.argsort(lambda_obs)[::-1]
        lambda_obs = lambda_obs[sort_idx]
        v_np = v.cpu().numpy()[:, sort_idx]
        
        # 2. Noise Floor
        nnz = X_sparse.nnz
        U, I = X_sparse.shape
        
        if self.lambda_base_config == 'auto':
            anchor_ext = np.sqrt(nnz / min(U, I))
        else:
            anchor_ext = float(self.lambda_base_config)
        
        # 3. Plateau Detection (Legacy Curvature)
        k1, k2 = self._find_plateau(lambda_obs)
        
        # 4. Fixed-Point Iteration Engine
        self.gamma, self.beta = self._infer_gamma_naked(lambda_obs, k1, k2, anchor_ext)
        
        # 5. Final Filter
        tau_final = lambda_obs ** self.gamma
        h_np = tau_final / (tau_final + anchor_ext + 1e-12)
        
        self.register_buffer("V_raw",       torch.from_numpy(v_np).float().to(self.device))
        self.register_buffer("filter_diag", torch.from_numpy(h_np).float().to(self.device))

        # 6. Save Analysis Dashboard
        self._save_spectral_analysis(lambda_obs, tau_final, h_np, tau_final * h_np, k1, k2, anchor_ext)

        self._log(
            f"Naked Engine Built (Legacy) | Plateau: [{k1}~{k2}] | Gamma*: {self.gamma:.4f} | λ_ext: {anchor_ext:.4f}"
        )

    def _save_spectral_analysis(self, lambda_obs, tau, h, s, k1, k2, anchor_ext):
        """
        ASPIRE-Naked Integrated Spectral Dashboard
        """
        output_dir = os.path.join(self.output_path, "spectral_analysis")
        os.makedirs(output_dir, exist_ok=True)
        dataset_name = self.config.get('dataset_name', 'unknown')
        
        # Metadata JSON
        metadata = {
            "config": {
                "dataset": dataset_name,
                "gamma_final": float(self.gamma),
                "beta_final": float(self.beta),
                "lambda_ext": float(anchor_ext)
            },
            "spectral_stats": {
                "lambda_obs_max": float(lambda_obs.max()),
                "signal_plateau_range": [int(k1), int(k2)]
            }
        }
        with open(os.path.join(output_dir, "analysis.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        # 3-Panel Plotting
        plt.style.use('seaborn-v0_8-muted') if 'seaborn-v0_8-muted' in plt.style.available else plt.style.use('ggplot')
        
        n = len(lambda_obs)
        ranks = np.arange(1, n + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"ASPIRE-Naked Spectral Analysis: {dataset_name.upper()}\n"
                     rf"(Physical Equilibrium: $\gamma^*={self.gamma:.4f}$, $\beta={self.beta:.4f}$)", 
                     fontsize=15, fontweight='bold', y=1.02)
        
        # [Panel 1] Recovery
        ax = axes[0]
        ax.loglog(ranks, lambda_obs, label=r'Observed', color='#3498db', alpha=0.3)
        ax.loglog(ranks, tau, label=r'Undistorted ($\tau$)', color='#2ecc71', linewidth=2)
        ax.axvspan(ranks[k1], ranks[k2], color='yellow', alpha=0.1, label="Plateau")
        ax.set_title("Spectral Recovery", fontsize=12, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
        # [Panel 2] Filter
        ax = axes[1]
        ax.plot(ranks, h, color='#e67e22', linewidth=2, label=r'Wiener Filter')
        ax.fill_between(ranks, h, color='#e67e22', alpha=0.1)
        ax.set_title(f"Filter Shape ($\lambda_{{ext}}={anchor_ext:.2f}$)", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # [Panel 3] Restoration
        ax = axes[2]
        ax.loglog(ranks, s, color='#9b59b6', label=r'Filtered Signal $s$')
        bulk_ranks = ranks[k1:k2]
        bulk_s = s[k1:k2]
        z = np.polyfit(np.log(bulk_ranks), np.log(bulk_s + 1e-12), 1)
        ax.loglog(bulk_ranks, np.exp(z[0]*np.log(bulk_ranks) + z[1]), 'k--', alpha=0.8, label=f"Fit Slope: {z[0]:.4f}")
        ax.set_title("Self-Consistent Equilibrium", fontsize=12, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dashboard.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        self._log(f"Spectral diagnostics saved to: {output_dir}")

    @torch.no_grad()
    def forward(self, users):
        batch = users.cpu().numpy()
        X_u   = torch.from_numpy(self.train_matrix_csr[batch].toarray()).float().to(self.device)
        XV    = torch.mm(X_u, self.V_raw)
        scores = torch.mm(XV * self.filter_diag, self.V_raw.t())
        return scores

    def predict_for_pairs(self, users, items):
        batch  = users.cpu().numpy()
        X_u    = torch.from_numpy(self.train_matrix_csr[batch].toarray()).float().to(self.device)
        XV     = torch.mm(X_u, self.V_raw)
        scores = torch.mm(XV * self.filter_diag, self.V_raw.t())
        return scores.gather(1, items.unsqueeze(1)).squeeze(1)

    def get_final_item_embeddings(self):
        return self.V_raw

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
