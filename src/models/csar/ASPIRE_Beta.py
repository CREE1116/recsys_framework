import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.utils.gpu_accel import EVDCacheManager

class ASPIRE_Beta(BaseModel):
    """
    ASPIRE_Beta: Beta Fixed-point Equilibrium Engine
    [Algebraic Logic]
    - Uses Beta (b) as the direct target for fixed-point iteration.
    - Beta = b(Beta) where gamma = 1 / (1 + beta).
    - Identity-centric root-finding for superior convergence.
    """
    def __init__(self, config, data_loader):
        super(ASPIRE_Beta, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_config = config.get('model', {})
        self.k         = model_config.get('k', None)
        self.max_iter  = model_config.get('max_iter', 20)
        self.tol       = 1e-4

        # Buffers
        self.register_buffer("V_raw",       torch.empty(0, 0))
        self.register_buffer("filter_diag", torch.empty(0))

        self.gamma_L = 1.0 
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
        [Beta Fixed-point Engine]
        gamma = 1 / (1 + beta)
        beta_new = Hill_Estimator(s_k)
        """
        beta = 1.0     # Initial beta (corresponds to gamma=0.5)
        gamma_L = 1.0 / (1.0 + beta)
        eps = 1e-12
        final_b = 0.0

        for i in range(self.max_iter):
            prev_beta = beta
            gamma_L = 1.0 / (1.0 + beta + eps)
            
            # 1. Transform / Reconstruct
            tau_k = lambda_obs ** gamma_L
            s_k = tau_k * (tau_k / (tau_k + anchor_ext + eps))
            
            # 2. Measure Slope using Hill Estimator
            # We measure the entire signal as per user request
            valid_mask = np.ones_like(s_k, dtype=bool)
            b = self._get_plateau_slope_mle(s_k, valid_mask, s_min=s_k[-1]+eps)
            
            # 3. Fixed-point Update: beta = b
            beta = b
            
            self._log(f"[ASPIRE-Beta] Iter {i+1:2d} | \u03b2: {prev_beta:.4f} -> {beta:.4f} | \u03b3_L: {gamma_L:.4f}")
            
            final_b = b
            if abs(beta - prev_beta) < self.tol:
                self._log(f"[ASPIRE-Beta] Engine Converged at Iter {i+1}")
                break
                
        return 1.0 / (1.0 + beta + eps), final_b

    def _get_plateau_slope_mle(self, s_k, valid_mask, s_min=None):
        s_valid = s_k[valid_mask]
        if len(s_valid) < 2: return 0.0
        if s_min is None: s_min = s_valid[-1] + 1e-12
        zeta = np.mean(np.log((s_valid + 1e-12) / s_min))
        # Zipf slope b = zeta
        return np.abs(zeta)

    @torch.no_grad()
    def _build(self, X_sparse, dataset_name):
        manager = EVDCacheManager(device=self.device.type)
        _, s, v, _ = manager.get_evd(X_sparse, k=self.k, dataset_name=dataset_name)

        lambda_obs = s.cpu().numpy() ** 2
        sort_idx = np.argsort(lambda_obs)[::-1]
        lambda_obs = lambda_obs[sort_idx]
        
        anchor_ext = float(self.config['model'].get('lambda_base', 100.0))
        
        # Beta Engine Execution
        self.gamma_L, self.final_b = self._infer_gamma_slope_consistency(lambda_obs, anchor_ext)
        
        # Pre-calculate W = (V * h) @ V.T for Inference speed
        tau_final = lambda_obs ** self.gamma_L
        h_np = tau_final / (tau_final + anchor_ext + 1e-12)
        
        V_np = v.cpu().numpy()[:, sort_idx]
        W_np = (V_np * h_np) @ V_np.T
        self.register_buffer("W", torch.from_numpy(W_np).float().to(self.device))

        self._save_spectral_analysis(lambda_obs, tau_final, h_np, anchor_ext)
        print(f"[ASPIRE-Beta] Accelerated Build: \u03b3_L={self.gamma_L:.4f} | W: {W_np.shape}")

    def _save_spectral_analysis(self, lambda_obs, tau, h, anchor_ext):
        output_dir = os.path.join(self.output_path, f"spectral_analysis_{self.__class__.__name__}")
        os.makedirs(output_dir, exist_ok=True)
        n = len(lambda_obs)
        ranks = np.arange(1, n + 1)
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        l0 = lambda_obs[0] + 1e-12
        axes.loglog(ranks, lambda_obs/l0, label='Observed', color='#3498db', alpha=0.3)
        axes.loglog(ranks, tau/l0, label=rf'Signal ($\gamma_L={self.gamma_L:.4f}$)', color='#2ecc71', linewidth=2)
        axes.set_title(f"{self.__class__.__name__} Analysis", fontsize=15)
        axes.grid(True, which="both", ls="-", alpha=0.2)
        axes.legend()
        plt.savefig(os.path.join(output_dir, "analysis.png"), dpi=150)
        plt.close(fig)

    @torch.no_grad()
    def forward(self, users):
        batch = users.cpu().numpy()
        if self.device.type == 'mps':
            X_u = torch.from_numpy(self.train_matrix_csr[batch].toarray()).float().to(self.device)
            return torch.mm(X_u, self.W)
        elif self.device.type == 'cuda':
            from src.utils.gpu_accel import to_torch_sparse_csr
            X_u_sparse = to_torch_sparse_csr(self.train_matrix_csr[batch], device=self.device)
            return torch.sparse.mm(X_u_sparse, self.W)
        else:
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
