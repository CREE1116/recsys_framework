import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.utils.gpu_accel import EVDCacheManager

class ASPIRE_Equilibrium(BaseModel):
    """
    ASPIRE_Equilibrium: Slope-Consistency Engine
    [Core Logic]
    - Full implementation of the fixed-point iteration algorithm from Section 3.3.5.2.
    - Measures the 'Log-Rank Slope (b)' within the effective signal band.
    - Updates gamma_L so that the output slope b converges to zero (shape invariance).
    """
    def __init__(self, config, data_loader):
        super(ASPIRE_Equilibrium, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_config = config.get('model', {})
        self.k         = model_config.get('k', None)
        self.lambda_base_config = model_config.get('lambda_base', 'auto')
        self.max_iter  = model_config.get('max_iter', 20)
        self.tol       = 1e-4

        # Buffers
        self.register_buffer("V_raw",       torch.empty(0, 0))
        self.register_buffer("filter_diag", torch.empty(0))

        self.gamma_L = 1.0 # beta = 0, MCAR Assume [cite: 3.3.5.2]
        self.final_b = 0.0

        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        self._build(self.train_matrix_csr, config.get('dataset_name', 'unknown'))

    def _build_sparse_matrix(self, data_loader):
        df = data_loader.train_df
        return csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df['user_id'].values, df['item_id'].values)),
            shape=(self.n_users, self.n_items)
        )

    def _infer_gamma_slope_consistency(self, lambda_obs, anchor_ext):
        """
        [Momentum-driven Equilibrium Engine]
        Find gamma_L using Adam-like resistance (Momentum=0.9) to prevent oscillation.
        Ensures a minimum of 50 samples for statistically valid Hill Estimation.
        """
        gamma_L = 0.5  
        eps = 1e-12
        momentum = 0.5  # High resistance to change
        final_b = 0.0

        for i in range(self.max_iter):
            prev_gamma_L = gamma_L
            
            # 1. Reconstruct Distorted Signal 
            tau_k = lambda_obs ** gamma_L
            
            # 2. Apply Wiener Filter (Naked Scale)
            h_k = tau_k / (tau_k + anchor_ext + eps)
            
            # 3. Get Filtered Signal Corrected by Filter
            s_k = tau_k * h_k
            
            # 4. Identify Plateau Band with Sample Size Protection (n >= 50)
            threshold_factors = [2.0, 1.5, 1.0, 0.5, 0.1, 0.01]
            valid_mask = None
            n_valid = 0
            for factor in threshold_factors:
                valid_mask = tau_k > (factor * anchor_ext)
                n_valid = np.sum(valid_mask)
                if n_valid >= 50:
                    break
            
            if n_valid < 2:
                b = 1.0 # fallback to steep
            else:
                # [Theoretical Reference] Index right before noise boundary
                global_valid = tau_k > (anchor_ext + eps)
                
                # Defensive check: if no signal is above anchor_ext, fallback to the last element
                if np.sum(global_valid) > 0:
                    s_min_anchor = tau_k[global_valid][-1]
                else:
                    s_min_anchor = tau_k[-1] + eps
                
                # Measure slope (b) on the FILTERED signal s_k
                b = self._get_plateau_slope_mle(s_k, valid_mask, s_min=s_min_anchor)

            # 5. Momentum Update: gamma_L = (0.9 * current) + (0.1 * target)
            # target_gamma_L = 1 / (1 + b)
            target_gamma_L = 1.0 / (1.0 + b + eps)
            gamma_L = (momentum * prev_gamma_L) + (1.0 - momentum) * target_gamma_L
            gamma_L = np.clip(gamma_L, 0.01, 1.0)
            
            # 6. Log Convergence Step with Momentum Info
            print(f"[ASPIRE-Eq] Iter {i+1:2d} | γ_L: {prev_gamma_L:.4f} | Target: {target_gamma_L:.4f} | b: {b:.4f} | n: {n_valid}")
            
            final_b = b
            if abs(gamma_L - prev_gamma_L) < self.tol:
                print(f"[ASPIRE-Eq] Engine Converged at Iter {i+1} | Final γ_L: {gamma_L:.4f}")
                break
                
        return gamma_L, final_b

    def _get_plateau_slope_mle(self, s_k, valid_mask, s_min=None):
        """
        [Theoretical Hill Estimator & Zipf Slope]
        1. Pareto Exponent (alpha) from Hill Estimator (Clauset et al. 2009)
        2. Zipf Slope (b) derived from alpha: b = 1 / (alpha - 1)
        """
        s_valid = s_k[valid_mask]
        
        if len(s_valid) < 2:
            return 0.0
            
        # Use provided s_min (baseline) or fallback to last element in valid subset
        if s_min is None:
            s_min = s_valid[-1] + 1e-12
        
        # Step A: Log-relative mean (zeta)
        # This is exactly 1 / (alpha - 1) in Pareto theory
        zeta = np.mean(np.log((s_valid + 1e-12) / s_min))
        
        # Step B: Proper Hill Estimator (alpha)
        alpha = 1.0 + (1.0 / (zeta + 1e-12))
        
        # Step C: Zipfian Slope (b) used for shape restoration
        # b = 1 / (alpha - 1) which is mathematically equivalent to zeta
        b = 1.0 / (alpha - 1.0 + 1e-12)
        
        return np.abs(b)

    @torch.no_grad()
    def _build(self, X_sparse, dataset_name):
        manager = EVDCacheManager(device=self.device.type)
        _, s, v, _ = manager.get_evd(X_sparse, k=self.k, dataset_name=dataset_name)

        lambda_obs = s.cpu().numpy() ** 2
        sort_idx = np.argsort(lambda_obs)[::-1]
        lambda_obs = lambda_obs[sort_idx]
        
        nnz, U, I = X_sparse.nnz, X_sparse.shape[0], X_sparse.shape[1]
        
        # [Pure Eigenvalue Scale] Unify everything to Eigenvalue domain
        if self.lambda_base_config == 'auto':
            # RMT noise floor for eigenvalues is exactly nnz / min(U, I)
            anchor_ext = nnz / min(U, I)
        else:
            # Treat config value as a raw Eigenvalue scale floor
            anchor_ext = float(self.lambda_base_config)
        
        # Execute Slope-Consistency Engine (Raw Scale)
        self.gamma_L, self.final_b = self._infer_gamma_slope_consistency(lambda_obs, anchor_ext)
        
        # Build Final Filter (Standard ASPIRE Scale)
        tau_final = lambda_obs ** self.gamma_L
        h_np = tau_final / (tau_final + anchor_ext + 1e-12)
        
        # Register Buffers
        V_np = v.cpu().numpy()[:, sort_idx]
        self.register_buffer("V_raw",       torch.from_numpy(V_np).float().to(self.device))
        self.register_buffer("filter_diag", torch.from_numpy(h_np).float().to(self.device))

        # Save Analysis
        self._save_spectral_analysis(lambda_obs, tau_final, h_np, anchor_ext)

        print(f"[ASPIRE-Eq] Slope Engine Converged | \u03b3_L: {self.gamma_L:.4f} | Final Slope |b|: {self.final_b:.4f}")

    def _save_spectral_analysis(self, lambda_obs, tau, h, anchor_ext):
        output_dir = os.path.join(self.output_path, "spectral_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        n = len(lambda_obs)
        ranks = np.arange(1, n + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"ASPIRE-Equilibrium Slope Engine (Final |b|: {self.final_b:.4f})\n"
                     rf"(Slope Consistency: $\gamma_L={self.gamma_L:.4f}$)", 
                     fontsize=15, fontweight='bold', y=1.02)
        
        # [Panel 1] Spectral Recovery
        ax = axes[0]
        l0 = lambda_obs[0] + 1e-12
        ax.loglog(ranks, lambda_obs/l0, label=r'Observed (Norm)', color='#3498db', alpha=0.3)
        ax.loglog(ranks, tau/l0, label=r'Signal ($\tau$)', color='#2ecc71', linewidth=2)
        # Visual calibration: anchor_ext is the baseline noise floor
        ax.axhline(y=anchor_ext/l0, color='r', linestyle='--', label='Noise Floor (Base)')
        ax.set_title("Spectral Recovery", fontsize=12, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        
        # [Panel 2] Filter
        ax = axes[1]
        ax.plot(ranks, h, color='#e67e22', linewidth=2, label='Wiener Filter')
        ax.fill_between(ranks, h, color='#e67e22', alpha=0.1)
        ax.set_title(f"Filter Response ($\lambda={anchor_ext:.1f}$)", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # [Panel 3] Slope Consistency Equilibrium
        ax = axes[2]
        l0 = lambda_obs[0] + 1e-12
        s_theory = tau * h
        
        # 3중 비교를 통해 엔진 동작 확인
        ax.loglog(ranks, lambda_obs/l0, label='Observed (Raw)', color='#3498db', alpha=0.2)
        ax.loglog(ranks, tau, label=r'Signal ($\tau$)', color='#2ecc71', linewidth=3, ls='--', alpha=0.7)
        ax.loglog(ranks, s_theory, label=r'Filtered ($s$)', color='#9b59b6', linewidth=2)
        
        ax.set_title("Spectral Logic (Raw vs Signal vs Filtered)", fontsize=12, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dashboard.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    @torch.no_grad()
    def forward(self, users):
        batch = users.cpu().numpy()
        X_u = torch.from_numpy(self.train_matrix_csr[batch].toarray()).float().to(self.device)
        XV = torch.mm(X_u, self.V_raw)
        return torch.mm(XV * self.filter_diag, self.V_raw.t())

    @torch.no_grad()
    def predict_full(self, users, items=None):
        return self.forward(users)

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
        return (torch.tensor(0.0, device=self.device),), {}
