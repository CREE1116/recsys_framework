import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.utils.gpu_accel import EVDCacheManager

class ASPIRE_Residual(BaseModel):
    """
    ASPIRE_Residual: Gradient-based Equilibrium Engine
    [Optimization Logic]
    - Defines a loss function: Loss = sum( (s_k - tau_k) / tau_k )^2.
    - Uses Gradient Descent to find gamma that minimizes this scale error.
    - No damping needed as it follows the energy gradient.
    """
    def __init__(self, config, data_loader):
        super(ASPIRE_Residual, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_config = config.get('model', {})
        self.k         = model_config.get('k', None)
        self.max_iter  = model_config.get('max_iter', 20)
        self.lr        = model_config.get('lr', 0.1)
        self.tol       = 1e-5

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
        [Residual Gradient Descent Engine]
        Finds gamma that minimizes the "Flattening Error".
        """
        gamma_L = 0.5  
        eps = 1e-12
        log_lambda = np.log(lambda_obs + eps)

        for i in range(self.max_iter):
            prev_gamma_L = gamma_L
            
            # 1. Transform / Reconstruct
            tau_k = np.exp(gamma_L * log_lambda)
            h_k = tau_k / (tau_k + anchor_ext + eps)
            
            # 2. Define Gradient (Simplification of dLoss/dGamma)
            # Loss ~ sum( (h_k - 1)^2 )
            # Grad ~ sum( 2 * (h_k - 1) * d(h_k)/d(gamma) )
            # d(h_k)/d(gamma) ~ h_k * (1 - h_k) * log_lambda
            grad = np.mean(2.0 * (h_k - 1.0) * h_k * (1.0 - h_k) * log_lambda)
            
            # 3. Step
            gamma_L = gamma_L - self.lr * grad
            gamma_L = np.clip(gamma_L, 0.01, 1.0)
            
            self._log(f"[ASPIRE-Residual] Iter {i+1:2d} | \u03b3_L: {prev_gamma_L:.4f} -> {gamma_L:.4f} | Grad: {grad:.6f}")
            
            if abs(gamma_L - prev_gamma_L) < self.tol:
                self._log(f"[ASPIRE-Residual] Engine Converged at Iter {i+1}")
                break
                
        return gamma_L, 0.0 # final_b is not the primary measure here

    @torch.no_grad()
    def _build(self, X_sparse, dataset_name):
        manager = EVDCacheManager(device=self.device.type)
        _, s, v, _ = manager.get_evd(X_sparse, k=self.k, dataset_name=dataset_name)

        lambda_obs = s.cpu().numpy() ** 2
        sort_idx = np.argsort(lambda_obs)[::-1]
        lambda_obs = lambda_obs[sort_idx]
        
        anchor_ext = float(self.config['model'].get('lambda_base', 100.0))
        
        self.gamma_L, self.final_b = self._infer_gamma_slope_consistency(lambda_obs, anchor_ext)
        
        tau_final = lambda_obs ** self.gamma_L
        h_np = tau_final / (tau_final + anchor_ext + 1e-12)
        
        V_np = v.cpu().numpy()[:, sort_idx]
        W_np = (V_np * h_np) @ V_np.T
        self.register_buffer("W", torch.from_numpy(W_np).float().to(self.device))

        self._save_spectral_analysisUnpacked(lambda_obs, tau_final, h_np, anchor_ext)
        print(f"[ASPIRE-Residual] Accelerated Build: \u03b3_L={self.gamma_L:.4f} | W: {W_np.shape}")

    def _save_spectral_analysisUnpacked(self, lambda_obs, tau, h, anchor_ext):
        output_dir = os.path.join(self.output_path, f"spectral_analysis_{self.__class__.__name__}")
        os.makedirs(output_dir, exist_ok=True)
        n = len(lambda_obs)
        ranks = np.arange(1, n + 1)
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        l0 = lambda_obs[0] + 1e-12
        axes.loglog(ranks, lambda_obs/l0, label='Observed', color='#3498db', alpha=0.3)
        axes.loglog(ranks, tau/l0, label=rf'Signal (GD Energy Optimization)', color='#e67e22', linewidth=2)
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
