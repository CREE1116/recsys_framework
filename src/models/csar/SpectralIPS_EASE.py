import torch
import numpy as np
import time
from src.models.base_model import BaseModel
from src.utils.gpu_accel import get_device, _build_gram

class SpectralIPS_EASE(BaseModel):
    """
    Spectral IPS-EASE: Low-Rank Popularity Debiasing.
    
    [Logic]
    1. Build Raw Gram Matrix G = R^T R.
    2. Extract Item Degree vector n (raw popularity).
    3. Project n onto the subspace spanned by the top-k eigenvectors of G:
       p_sys = V_k @ (V_k^T @ n)
    4. Normalize G using p_sys as the IPS propensity:
       G_star = D_sys^-1/2 G D_sys^-1/2  where D_sys = diag(p_sys)
    5. Solve EASE on G_star.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.k = int(model_config.get('k', 128))
        self.reg_lambda = float(model_config.get('reg_lambda', 500.0))
        self.device = get_device()
        self.eps = 1e-12

        self.register_buffer("W", torch.empty(self.n_items, self.n_items))

        # Build training matrix (standard for EASE-family in this framework)
        from scipy.sparse import csr_matrix
        df = data_loader.train_df
        self.train_matrix_csr = csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df['user_id'].values, df['item_id'].values)),
            shape=(self.n_users, self.n_items)
        )
        
        self._build(self.train_matrix_csr)

    @torch.no_grad()
    def _build(self, R_sparse):
        t0 = time.time()
        m = self.n_items
        self._log(f"Building Spectral IPS-EASE (k={self.k}, lambda={self.reg_lambda}) on {self.device}")
        
        # 1. Build Gram Matrix G = R^T R
        G = _build_gram(R_sparse, self.device)
        
        # 2. Eigen-decomposition of G
        try:
            # Using torch.linalg.eigh for symmetric matrix G
            evals, evecs = torch.linalg.eigh(G.cpu())
            evals_np = evals.numpy()
            
            # --- Hybrid k via MP-Law + Energy Ratio ---
            n_users, n_items = self.n_users, self.n_items
            gamma = n_items / n_users
            lambda_mp = n_users * (1 + np.sqrt(gamma))**2
            
            # 1. MP-Law Count
            k_mp = int(np.sum(evals_np > lambda_mp))
            
            # 2. Energy Ratio Count (90% threshold for representation)
            total_energy = np.sum(evals_np)
            energy_cumsum = np.cumsum(evals_np[::-1]) # Descending order cumsum
            k_energy = np.searchsorted(energy_cumsum, 0.9 * total_energy) + 1
            
            # 3. Hybrid Selection (with lower bound for coverage)
            k_eff = max(k_mp, k_energy, 100)
            k_eff = min(k_eff, m)
            
            self._log(f"  [Spectral-Auto] MP-k: {k_mp} | Energy-k (90%): {k_energy} | Selected k: {k_eff}")
            
            V_k = evecs[:, -k_eff:].to(self.device).float()
            self.top_evals = evals_np[-k_eff:]
        except RuntimeError:
            self._log("Linalg eigh failed, using CPU SVD fallback.")
            from scipy.sparse.linalg import svds
            U, S, Vh = svds(G.cpu().double().numpy(), k=self.k)
            V_k = torch.from_numpy(Vh.T).to(self.device).float()
            self.top_evals = S**2

        # 3. Low-Rank Popularity Filtering
        # n_vec: raw degrees
        n_vec = torch.from_numpy(np.asarray(R_sparse.sum(axis=0)).flatten()).to(self.device).float()
        
        # Project n onto V_k: p_sys = V_k V_k^T n
        p_sys_raw = V_k @ (V_k.T @ n_vec)
        
        # --- Diagnostics for p_sys alignment ---
        from scipy.stats import spearmanr
        n_np = n_vec.cpu().numpy()
        p_np = p_sys_raw.cpu().numpy()
        corr, _ = spearmanr(n_np, p_np)
        
        self._log(f"  [Diagnostics] p_sys < 1: {np.sum(p_np < 1)} / {m}")
        self._log(f"  [Diagnostics] p_sys < 0: {np.sum(p_np < 0)} / {m}")
        self._log(f"  [Diagnostics] n vs p_sys Spearman: {corr:.3f}")
        
        p_sys = torch.clamp(p_sys_raw, min=1.0)
        
        # --- Geometric Smoothing ---
        # Instead of just p_sys, use sqrt(n_vec * p_sys) to stabilize Coverage
        d_ideal = torch.sqrt(n_vec * p_sys + self.eps)
        d_ideal = torch.clamp(d_ideal, min=1.0)
        
        self.p_sys_stats = (float(d_ideal.min()), float(d_ideal.max()))

        # 4. IPS Normalization: G_star = D^-1/2 G D^-1/2
        # Use ips_power to control debiasing strength (default total power 0.5 -> 0.25 each side)
        ips_power = float(self.config.get('model', {}).get('ips_power', 0.5))
        each_side_power = ips_power / 2.0
        
        D_inv = 1.0 / (torch.pow(d_ideal, each_side_power) + self.eps)
        G_star = G * D_inv.unsqueeze(1) * D_inv.unsqueeze(0)
        del G
        
        # 5. EASE Closed-form Solve: P = (G_star + lambda I)^-1
        I = torch.eye(m, device=self.device, dtype=torch.float32)
        A = G_star + self.reg_lambda * I
        
        try:
            P = torch.linalg.inv(A)
        except RuntimeError:
            P = torch.from_numpy(np.linalg.inv(A.cpu().numpy())).to(self.device).float()
        del A, G_star

        # 6. Final Weights W = P / (-diag(P)) with zeros on diagonal
        diag_P = torch.diag(P)
        W = P / (-diag_P.unsqueeze(0) + self.eps)
        W.fill_diagonal_(0.0)
        
        self.W.copy_(W)
        self._log(f"Spectral IPS-EASE Build completed in {time.time()-t0:.2f}s")
        self._log(f"  p_sys range: [{self.p_sys_stats[0]:.2f}, {self.p_sys_stats[1]:.2f}]")

    @torch.no_grad()
    def forward(self, users):
        """Standard interaction-fetching forward pass."""
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
        return {
            "k": self.k,
            "lambda": self.reg_lambda,
            "p_sys_min": self.p_sys_stats[0],
            "p_sys_max": self.p_sys_stats[1],
            "top_eval_ratio": float(self.top_evals[-1] / (self.top_evals[-2] if len(self.top_evals)>1 else 1.0))
        }
