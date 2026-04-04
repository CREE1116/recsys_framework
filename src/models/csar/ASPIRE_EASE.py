import torch
import numpy as np
import time
from src.models.base_model import BaseModel
from src.utils.gpu_accel import get_device, _build_gram

class ASPIRE_EASE(BaseModel):
    """
    ASPIRE-EASE: The Ultimate Spectral-Equilibrium Model.
    
    [Stage 1] Spectral Orthogonalization (Find Ideal Beta)
    [Stage 2] Model Equilibrium (Fixed-Point Solvers on W)
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.reg_lambda = float(model_config.get('reg_lambda', 10.0))
        self.max_iter_equilibrium = int(model_config.get('max_iter_equilibrium', 5))
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
    def _compute_spectral_balancing_d(self, G, max_iter=50, tol=1e-6):
        """Estimate D using the self-consistent spectral balancing operator: d_new = (G^2) @ (1/d)"""
        m = G.shape[0]
        # Initialize with degrees
        d = G.sum(dim=1) + self.eps
        
        # Precompute G * G for efficiency in the loop
        G2 = G * G
        
        for it in range(max_iter):
            prev_d = d.clone()
            
            inv_d = 1.0 / (d + self.eps)
            # The heart of spectral balancing: d_i = sum_j (G_ij^2 / d_j)
            d_new = torch.mv(G2, inv_d)
            
            # Prevent scale drift (maintain total energy)
            d_new = d_new * (d.mean() / (d_new.mean() + self.eps))
            
            diff = torch.norm(d_new - prev_d) / (torch.norm(prev_d) + self.eps)
            
            d = d_new
            if diff < tol:
                self._log(f"  [ASPIRE-EASE] D converged at iter {it+1}")
                break
                
        return d

    @torch.no_grad()
    def _build(self, R_sparse):
        t0 = time.time()
        m = self.n_items
        self._log(f"Building ASPIRE-EASE (Spectral Balancing, lambda={self.reg_lambda})")
        
        # 1. Build Raw Gram Matrix G
        G = _build_gram(R_sparse, self.device)
        
        # 2. Compute Ideal Correction Weights D
        # Use item subsampling for fast D estimation on large datasets
        sub_limit = 1500
        if m > sub_limit:
            diag = G.diagonal()
            _, idx = torch.topk(diag, k=sub_limit)
            G_sub = G[idx][:, idx]
            d_sub = self._compute_spectral_balancing_d(G_sub)
            
            # Map d_sub back to full scale using degree proportions
            d_full = G.sum(dim=1) + self.eps
            # Heuristic: D_ideal ~ D_raw * (d_sub / d_raw_sub) ?
            # Smoother: Use d_sub as a pivot to adjust global alpha.
            # But simpler: apply spectral balancing to full G since it's O(M^2)
            d_ideal = self._compute_spectral_balancing_d(G)
        else:
            d_ideal = self._compute_spectral_balancing_d(G)
        
        # 3. Final Build using G_ideal = D^-1/2 G D^-1/2
        inv_sqrt_d = 1.0 / (torch.sqrt(d_ideal) + self.eps)
        G_final = G * inv_sqrt_d.unsqueeze(1) * inv_sqrt_d.unsqueeze(0)
        del G
        
        I = torch.eye(m, device=self.device, dtype=torch.float32)
        A = G_final + self.reg_lambda * I
        try:
            P = torch.linalg.inv(A)
        except:
            P = torch.from_numpy(np.linalg.inv(A.cpu().numpy())).to(self.device).float()
        
        W = I - (self.reg_lambda * P)
        W.fill_diagonal_(0.0)
        
        self.W.copy_(W)
        self._log(f"ASPIRE-EASE Build completed in {time.time()-t0:.2f}s")

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
        return {
            "lambda": self.reg_lambda,
            "W_mean": float(self.W.mean()),
            "W_diag_sum": float(torch.diag(self.W).sum())
        }
