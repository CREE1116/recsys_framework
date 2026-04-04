import torch
import numpy as np
import time
from src.models.base_model import BaseModel
from src.utils.gpu_accel import get_device, _build_gram

class VD_IPS_EASE(BaseModel):
    """
    VD-IPS-EASE: Variance-Decomposed Inverse Propensity Score EASE.
    
    [Logic]
    1. Items with high frequency (n) but low User-level Variance (Var) are systemic bias.
    2. Bernoulli Variance: Var_i = p_i * (1 - p_i) where p_i = n_i / N.
    3. Systemic Propensity: p_sys_i = n_i / (Var_i + eps).
    4. IPS Normalization: G_star = D_sys^-alpha/2 G D_sys^-alpha/2.
    5. Solve EASE on G_star.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.reg_lambda = float(model_config.get('reg_lambda', 500.0))
        self.ips_power = float(model_config.get('ips_power', 0.5))
        self.eps = float(model_config.get('eps', 1e-8))
        self.device = get_device()

        self.register_buffer("W", torch.empty(self.n_items, self.n_items))

        # Build training matrix (sparse for EASE-family in this framework)
        from scipy.sparse import csr_matrix
        df = data_loader.train_df
        self.train_matrix_csr = csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df['user_id'].values, df['item_id'].values)),
            shape=(self.n_users, self.n_items)
        )
        
        self._build(self.train_matrix_csr)

    @torch.no_grad()
    def _estimate_p_sys(self, R_sparse):
        """
        Estimate systemic propensity p_sys_i = n_i / (Var_u(R_ui) + eps)
        Var_u(R_ui) = p_obs * (1 - p_obs) where p_obs is normalized n_i.
        """
        N = self.n_users
        n = torch.from_numpy(np.asarray(R_sparse.sum(axis=0)).flatten()).to(self.device).float()
        
        # p = n_i / N (Probability of observing interaction)
        p_obs = n / N
        # Bernoulli variance: p * (1 - p)
        var_u = p_obs * (1.0 - p_obs)
        
        # Systemic propensity: High frequency, low variance = Systemic Bias
        p_sys = n / (var_u + self.eps)
        
        # Normalize: average should be 1.0
        p_sys = p_sys / (p_sys.mean() + self.eps)
        
        return p_sys, n, var_u

    @torch.no_grad()
    def _build(self, R_sparse):
        t0 = time.time()
        m = self.n_items
        self._log(f"Building VD-IPS-EASE (lambda={self.reg_lambda}, ips_power={self.ips_power}) on {self.device}")
        
        # 1. Estimate Systemic Propensity p_sys
        p_sys, n_vec, var_u = self._estimate_p_sys(R_sparse)
        self.p_sys_stats = (float(p_sys.min()), float(p_sys.max()))
        self.n_vec = n_vec
        self.var_u = var_u

        # 2. Build Gram Matrix G = R^T R
        G = _build_gram(R_sparse, self.device)
        
        # 3. IPS Normalization Level (Total power = ips_power)
        each_side_power = self.ips_power / 2.0
        D_inv = 1.0 / (torch.pow(p_sys, each_side_power) + self.eps)
        
        # G_star = D_inv * G * D_inv
        G_star = G * D_inv.unsqueeze(1) * D_inv.unsqueeze(0)
        del G
        
        # 4. Solve EASE Closed-form: P = (G_star + lambda I)^-1
        I = torch.eye(m, device=self.device, dtype=torch.float32)
        A = G_star + self.reg_lambda * I
        
        try:
            P = torch.linalg.inv(A)
        except RuntimeError:
            P = torch.from_numpy(np.linalg.inv(A.cpu().numpy())).to(self.device).float()
        del A, G_star

        # 5. Final Weights W = P / (-diag(P))
        diag_P = torch.diag(P)
        W = P / (-diag_P.unsqueeze(0) + self.eps)
        W.fill_diagonal_(0.0)
        
        self.W.copy_(W)
        self._log(f"VD-IPS-EASE Build completed in {time.time()-t0:.2f}s")
        self._log(f"  p_sys range: [{self.p_sys_stats[0]:.3f}, {self.p_sys_stats[1]:.3f}]")

    @torch.no_grad()
    def forward(self, users):
        if users.dim() > 1 and users.shape[1] == self.n_items:
            return torch.matmul(users, self.W)
        batch_ids = users.cpu().numpy()
        X_u = torch.from_numpy(self.train_matrix_csr[batch_ids].toarray()).float().to(self.device).detach()
        return torch.matmul(X_u, self.W)

    @torch.no_grad()
    def predict_full(self, users, items=None):
        scores = self.forward(users)
        if items is not None:
            return scores.gather(1, items)
        return scores

    @torch.no_grad()
    def predict_for_pairs(self, users, items):
        scores = self.forward(users)
        return scores.gather(1, items.unsqueeze(1)).squeeze(1)

    def get_final_item_embeddings(self):
        return self.W

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), {}

    def get_user_scores(self, user_indices):
        user_data = self.train_matrix_csr[user_indices].toarray()
        user_torch = torch.from_numpy(user_data).float().to(self.device).detach()
        return torch.matmul(user_torch, self.W)

    def diagnostics(self):
        from scipy.stats import spearmanr
        n_np = self.n_vec.cpu().numpy()
        v_np = self.var_u.cpu().numpy()
        p_np = self.var_u.cpu().numpy() # Just for reference
        
        # We want to see how p_sys aligns with n vs var
        p_sys_np = (n_np / (v_np + self.eps))
        
        corr_n, _ = spearmanr(n_np, p_sys_np)
        corr_var, _ = spearmanr(v_np, p_sys_np)
        
        return {
            "lam": self.get_device,
            "p_sys_min": self.p_sys_stats[0],
            "p_sys_max": self.p_sys_stats[1],
            "corr_n_psys": float(corr_n),
            "corr_var_psys": float(corr_var)
        }
