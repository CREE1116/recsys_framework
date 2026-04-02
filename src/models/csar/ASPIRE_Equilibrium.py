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
        [Full-Spectrum Equilibrium Engine]
        Uses the ENTIRE spectrum with 0.5 Damping (Momentum).
        Balances responsiveness and stability for global scale restoration.
        """
        gamma_L = 0.5  
        eps = 1e-12
        momentum = 0.5 
        final_b = 0.0

        for i in range(self.max_iter):
            prev_gamma_L = gamma_L
            
            # 1. Reconstruct Distorted Signal 
            tau_k = lambda_obs ** gamma_L
            
            # 2. Apply Wiener Filter
            h_k = tau_k / (tau_k + anchor_ext + eps)
            
            # 3. Get Filtered Signal Corrected by Filter
            s_k = tau_k * h_k
            
            # 4. Global Band Detection (Full Spectrum)
            # We measure the entire signal to find the global scale-invariant point.
            valid_mask = np.ones_like(tau_k, dtype=bool)
            n_valid = len(tau_k)
            
            if n_valid < 2:
                print(f"[ASPIRE-Eq] Iter {i+1:2d} | Insufficient signal. Stopping.")
                break
            
            # Global reference point (last element)
            s_min_anchor = tau_k[-1] + eps
            
            # 5. Measure GLOBAL slope (b) on the FILTERED signal s_k
            b = self._get_plateau_slope_mle(s_k, valid_mask, s_min=s_min_anchor)

            # 6. Damped Update (No Momentum)
            target_gamma_L = 1.0 / (1.0 + b + eps)
            gamma_L = (momentum * prev_gamma_L) + (1.0 - momentum) * target_gamma_L
            gamma_L = np.clip(gamma_L, 0.01, 1.0)
            
            # 7. Log Convergence
            print(f"[ASPIRE-Eq] Iter {i+1:2d} | γ_L: {prev_gamma_L:.4f} | b: {b:.4f} | n: {n_valid}")
            
            final_b = b
            if abs(gamma_L - prev_gamma_L) < self.tol:
                print(f"[ASPIRE-Eq] Engine Converged at Iter {i+1}")
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
        
        # Set Noise Floor (Directly from config)
        anchor_ext = float(self.config['model'].get('lambda_base', 100.0))
        
        # Execute Slope-Consistency Engine (Raw Scale)
        self.gamma_L, self.final_b = self._infer_gamma_slope_consistency(lambda_obs, anchor_ext)
        
        # Build Final Weight Matrix W = (V * h) @ V.T (EASE-style single-pass)
        tau_final = lambda_obs ** self.gamma_L
        h_np = tau_final / (tau_final + anchor_ext + 1e-12)
        
        V_np = v.cpu().numpy()[:, sort_idx]
        # Pre-calculating W to reduce 2 MMs to 1 during inference
        W_np = (V_np * h_np) @ V_np.T
        self.register_buffer("W", torch.from_numpy(W_np).float().to(self.device))
        
        # Keep original V only if needed for saving analysis
        self._save_spectral_analysis(lambda_obs, tau_final, h_np, anchor_ext)
        print(f"[ASPIRE-Eq] Accelerated Build Complete | \u03b3_L: {self.gamma_L:.4f} | W: {W_np.shape}")

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
        # Choice of acceleration based on device compatibility
        if self.device.type == 'mps':
            # MPS: Standard Dense MM is stable and fast for Mac unified memory
            X_u = torch.from_numpy(self.train_matrix_csr[batch].toarray()).float().to(self.device)
            return torch.mm(X_u, self.W)
        elif self.device.type == 'cuda':
            # CUDA: Sparse-Dense MM is memory efficient
            from src.utils.gpu_accel import to_torch_sparse_csr
            X_u_sparse = to_torch_sparse_csr(self.train_matrix_csr[batch], device=self.device)
            return torch.sparse.mm(X_u_sparse, self.W)
        else:
            # CPU fallback
            X_u = torch.from_numpy(self.train_matrix_csr[batch].toarray()).float()
            return torch.mm(X_u, self.W.cpu())

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
