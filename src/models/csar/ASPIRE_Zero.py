import torch
import torch.nn as nn
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.optimize import minimize_scalar

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


    def _infer_gamma_consistency(self, lambda_obs, anchor_ext):
        """
        [Log-Space Wiener Self-Consistency Engine]
        위너 필터링의 물리적 고정점 조건 (s = tau)을 로그 공간에서 최적화.
        - s = lambda_obs * h (위너 필터링된 관측 신호)
        - tau = lambda_obs ** gamma (재구성된 신호)
        - Goal: Find gamma s.t. log(s) ≈ log(tau)
        """
        from scipy.optimize import minimize_scalar
        n = len(lambda_obs)
        eps = 1e-12
        
        # 유효 신호 영역 (Noise Floor 이상의 의미 있는 신호)
        # 전체의 80% 정도 혹은 lambda_ext 이상의 영역에 가중치를 둠
        w = np.exp(- (np.maximum(0, np.log(anchor_ext + eps) - np.log(lambda_obs + eps)))**2)
        w /= (w.sum() + eps)

        def objective(gamma):
            tau = lambda_obs ** gamma
            h = tau / (tau + anchor_ext + eps)
            s = lambda_obs * h
            
            # log-error: log(s) - log(tau) = log(s/tau) = log(lambda_obs / (tau + lambda))
            # 이 차이가 0에 가까울수록 lambda_obs ≈ tau + lambda 물리 법칙에 정합함.
            error = np.log(s + eps) - np.log(tau + eps)
            return np.sum(w * (error ** 2))

        # 정밀 최적화 수행
        res = minimize_scalar(objective, bounds=(0.1, 1.1), method='bounded')
        gamma = float(res.x)

        # [Final Diagnostics & Visualization]
        tau_final = lambda_obs ** gamma
        h_final = tau_final / (tau_final + anchor_ext + eps)
        s_final = tau_final * h_final
        log_k = np.log(np.arange(1, n + 1))
        log_ext = np.log(anchor_ext + eps)
        
        # 진단용 가중치 (Display beta) 
        w_f = np.exp(- (np.log(tau_final + eps) - log_ext)**2 )
        w_f /= (w_f.sum() + eps)
        
        # 진단용 기울기 산출
        x_m = np.sum(w_f * log_k)
        y_m = np.sum(w_f * np.log(s_final + eps))
        beta_diag = abs(np.sum(w_f * (log_k - x_m) * (np.log(s_final + eps) - y_m)) / (np.sum(w_f * (log_k - x_m)**2) + eps))
        
        # 시각화 대역 k1, k2 (Wiener Transition 구역)
        active_idx = np.where((h_final > 0.2) & (h_final * 1.0 < 0.8))[0]
        k1, k2 = (active_idx[0], active_idx[-1]) if len(active_idx) > 0 else (0, n-1)
        
        return gamma, float(beta_diag), k1, k2

    def _save_spectral_analysis(self, lambda_obs, tau, h, s, beta, k1, k2, anchor_ext):
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
        ax.axvspan(ranks[k1], ranks[k2], color='yellow', alpha=0.1, label="Transition Band")
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
        
        # Weighted OLS Fit line for visualization
        log_k_full = np.log(ranks)
        # Using converged beta and a reference point from the center of transition band
        ref_idx = (k1 + k2) // 2
        ref_y = np.log(s[ref_idx] + 1e-12)
        ref_x = log_k_full[ref_idx]
        fit_y = -beta * (log_k_full - ref_x) + ref_y
        
        ax.loglog(ranks[k1:k2], np.exp(fit_y[k1:k2]), 'k--', alpha=0.8, label=f"W-OLS Slope: {-beta:.4f}")
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
        
        # 3. Log-Space Wiener Self-Consistency 기반 Gamma 최적화
        self.gamma, self.beta, k1, k2 = self._infer_gamma_consistency(lambda_obs, anchor_ext)
        
        # 5. Final Wiener Filter Build (λ base)
        tau_final = lambda_obs ** self.gamma
        h_np = tau_final / (tau_final + anchor_ext + 1e-12)
        
        self.register_buffer("V_raw", torch.from_numpy(v_np).float().to(self.device))
        self.register_buffer("filter_diag", torch.from_numpy(h_np).float().to(self.device))

        # 6. Save Analysis (Integrated Dashboard & JSON in output_path)
        self._save_spectral_analysis(lambda_obs, tau_final, h_np, tau_final * h_np, self.beta, k1, k2, anchor_ext)

        self._log(
            f"Self-Consistency Engine | Band: [{k1}~{k2}] | Optimized γ*: {self.gamma:.4f} | β(diag): {self.beta:.4f} | λ_ext: {anchor_ext:.4f}"
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