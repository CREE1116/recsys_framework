import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from src.models.base_model import BaseModel
from src.utils.gpu_accel import EVDCacheManager

class ASPIRE_Zero(BaseModel):
    """
    ASPIRE-Zero: Self-Consistent Fixed-Point Iteration
    
    [핵심 메커니즘]
    - γ를 자기 일관성(Self-consistency) 조건에 의해 반복적으로 최적화.
    - 필터링된 스펙트럼 s의 기울기 b가 곧 보정 지수 beta가 되도록 함.
    - γ = 2 / (1 + β)
    """

    def __init__(self, config, data_loader):
        super(ASPIRE_Zero, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_config = config.get('model', {})
        self.k         = model_config.get('k', None)
        self.max_iter  = model_config.get('max_iter', 20)
        self.tol       = model_config.get('tol', 1e-4)
        self.visualize = model_config.get('visualize', True)
        self.lambda_base_config = model_config.get('lambda_base', 'auto')

        # Buffers
        self.register_buffer("V_raw",       torch.empty(0, 0))
        self.register_buffer("filter_diag", torch.empty(0))

        self.gamma      = 2.0
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

    def _infer_gamma_fixed_point(self, lambda_obs, k1, k2, anchor_ext):
        """
        Self-Consistent Fixed-Point Iteration Engine (Eigenvalue Power Base)
        """
        n_components = len(lambda_obs)
        ranks = np.arange(k1 + 1, k2 + 1, dtype=np.float64)
        x = np.log(ranks)
        x_centered = x - np.mean(x)
        x_var = np.sum(x_centered ** 2) + 1e-12
        
        # 1. Initialize gamma = 1.0
        gamma = 1.0
        
        for i in range(self.max_iter):
            prev_gamma = gamma
            
            # (1) Wiener Filtering on λ spectrum directly
            tau = lambda_obs ** gamma
            h = tau / (tau + anchor_ext + 1e-12)
            s = tau * h # Filtered Power Spectrum
            
            # (2) Measure New Slope b from s
            log_s = np.log(s[k1:k2] + 1e-12)
            y_centered = log_s - np.mean(log_s)
            b = np.sum(x_centered * y_centered) / x_var
            
            # (3) Update beta & gamma
            # β = |b|, γ = 1 / (1 + β)
            beta_new = abs(b)
            gamma = 1.0 / (1.0 + beta_new)
            
            # Damping for stability
            gamma = 0.5 * prev_gamma + 0.5 * gamma
            
            if abs(gamma - prev_gamma) < self.tol:
                break
            prev_gamma = gamma
                
        return gamma, abs(b)

    def _save_spectral_analysis(self, lambda_obs, tau, h, s, k1, k2, anchor_ext):
        """
        ASPIRE-Zero Integrated Self-Analysis (Pro Version):
        Saves professional spectral dashboard and detailed JSON metadata.
        """
        output_dir = os.path.join(self.output_path, "spectral_analysis")
        os.makedirs(output_dir, exist_ok=True)
        dataset_name = self.config.get('dataset_name', 'unknown')
        
        # 1. Save Detailed JSON Metadata (aspire_visualizer style)
        metadata = {
            "config": {
                "dataset": dataset_name,
                "gamma_final": float(self.gamma),
                "beta_final": float(self.beta),
                "lambda_ext": float(anchor_ext)
            },
            "spectral_stats": {
                "lambda_obs_max": float(lambda_obs.max()),
                "lambda_obs_mean": float(lambda_obs.mean()),
                "signal_plateau_range": [int(k1), int(k2)]
            },
            "filter_stats": {
                "h_max": float(h.max()),
                "h_min": float(h.min()),
                "h_mean": float(h.mean())
            }
        }
        with open(os.path.join(output_dir, "analysis.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        # 2. Professional Plotting (3-Panel Analysis)
        plt.style.use('seaborn-v0_8-muted') if 'seaborn-v0_8-muted' in plt.style.available else plt.style.use('ggplot')
        
        n = len(lambda_obs)
        ranks = np.arange(1, n + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"ASPIRE-Zero Spectral Analysis: {dataset_name.upper()}\n"
                     rf"(Fixed-Point Equilibrium: $\gamma^*={self.gamma:.4f}$, $\beta={self.beta:.4f}$)", 
                     fontsize=15, fontweight='bold', y=1.02)
        
        # [Panel 1] Spectral Power-law (Log-Log)
        ax = axes[0]
        ax.loglog(ranks, lambda_obs, label=r'Observed ($\lambda_{obs}$)', color='#3498db', alpha=0.3)
        ax.loglog(ranks, tau, label=r'Undistorted ($\tau$)', color='#2ecc71', linewidth=2)
        ax.axvspan(ranks[k1], ranks[k2], color='yellow', alpha=0.1, label="Plateau (Bulk)")
        ax.set_title("Spectral Distortion Recovery", fontsize=12, fontweight='bold')
        ax.set_xlabel("Rank (k)", fontsize=10)
        ax.set_ylabel("Eigenvalue Scale", fontsize=10)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
        # [Panel 2] Filter Transfer Function
        ax = axes[1]
        ax.plot(ranks, h, color='#e67e22', linewidth=2, label=r'Wiener Filter $h(\lambda)$')
        ax.fill_between(ranks, h, color='#e67e22', alpha=0.1)
        ax.set_title(f"Filter Shape ($\lambda_{{ext}}={anchor_ext:.2f}$)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Rank (k)", fontsize=10)
        ax.set_ylabel("Filter Gain", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # [Panel 3] Spectral Restoration (Effective)
        ax = axes[2]
        ax.loglog(ranks, s, color='#9b59b6', label=r'Filtered Signal $s = \tau \cdot h$')
        # Regression for Flatness check
        bulk_ranks = ranks[k1:k2]
        bulk_s = s[k1:k2]
        z = np.polyfit(np.log(bulk_ranks), np.log(bulk_s + 1e-12), 1)
        ax.loglog(bulk_ranks, np.exp(z[0]*np.log(bulk_ranks) + z[1]), 'k--', alpha=0.8, label=f"Fit Slope: {z[0]:.4f}")
        ax.set_title("Self-Consistent Equilibrium", fontsize=12, fontweight='bold')
        ax.set_xlabel("Rank (k)", fontsize=10)
        ax.set_ylabel("Signal Intensity", fontsize=10)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dashboard.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self._log(f"Spectral diagnostics saved to: {output_dir}")

    @torch.no_grad()
    def _build(self, X_sparse, dataset_name):
        manager = EVDCacheManager(device=self.device.type)
        _, s, v, _ = manager.get_evd(X_sparse, k=self.k, dataset_name=dataset_name)

        # 1. Eigenvalues Base
        lambda_obs = s.cpu().numpy() ** 2
        sort_idx = np.argsort(lambda_obs)[::-1]
        lambda_obs = lambda_obs[sort_idx]
        v_np = v.cpu().numpy()[:, sort_idx]
        
        # 2. Noise Floor (RMT or Config)
        if self.lambda_base_config == 'auto':
            M = X_sparse.nnz
            U, I = X_sparse.shape
            anchor_ext = np.sqrt(M / min(U, I))
        else:
            anchor_ext = float(self.lambda_base_config)
        
        # 3. Plateau Detection
        k1, k2 = self._find_plateau(lambda_obs)
        
        # 4. Fixed-Point Iteration Engine
        self.gamma, self.beta = self._infer_gamma_fixed_point(lambda_obs, k1, k2, anchor_ext)
        
        # 5. Final Wiener Filter Build (λ base)
        tau_final = lambda_obs ** self.gamma
        h_np = tau_final / (tau_final + anchor_ext + 1e-12)
        
        self.register_buffer("V_raw", torch.from_numpy(v_np).float().to(self.device))
        self.register_buffer("filter_diag", torch.from_numpy(h_np).float().to(self.device))

        # 6. Save Analysis (Integrated Dashboard & JSON in output_path)
        self._save_spectral_analysis(lambda_obs, tau_final, h_np, tau_final * h_np, k1, k2, anchor_ext)

        self._log(
            f"Fixed-Point Engine | Plateau: [{k1}~{k2}] | Converged γ*: {self.gamma:.4f} | β: {self.beta:.4f} | λ_ext: {anchor_ext:.4f}"
        )

    @torch.no_grad()
    def predict_full(self, users, items=None):
        batch = users.cpu().numpy()
        X_u   = torch.from_numpy(self.train_matrix_csr[batch].toarray()).float().to(self.device)
        XV    = torch.mm(X_u, self.V_raw)
        scores = torch.mm(XV * self.filter_diag, self.V_raw.t())
        if items is not None: return scores.gather(1, items)
        return scores

    def forward(self, users, items=None):
        return self.predict_full(users, items)

    @torch.no_grad()
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

    def diagnostics(self):
        return {"gamma": self.gamma, "beta": self.beta}
