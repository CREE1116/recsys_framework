import torch
import numpy as np
import time
from src.models.base_model import BaseModel
from src.utils.gpu_accel import get_device, _build_gram

class AutoEASE_MNAR(BaseModel):
    """
    AutoEASE-MNAR: Universal Popularity Correction via the 10% Alignment Rule.
    
    [Logic]
    1. Build Raw Gram Matrix G.
    2. Identify the Popularity Axis (p1) - top eigenvector of G.
    3. Find alpha where the influence of p1 in the filtered space G_tilde
       drops to exactly 10% of its original relative energy.
    4. Solve EASE closed-form on G_tilde.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.reg_lambda = float(model_config.get('reg_lambda', 50.0))
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
    def _estimate_alpha_alignment(self, G):
        """
        Estimate alpha using the 10% Alignment Rule.
        Finds alpha such that p1 mapping energy drops to 1/10th of original.
        """
        m_full = G.shape[0]
        sub_limit = 1500
        
        # 1. Subsample items for fast spectral analysis (if needed)
        if m_full > sub_limit:
            diag = G.diagonal()
            _, idx = torch.topk(diag, k=sub_limit)
            G_sub = G[idx][:, idx]
            m = sub_limit
        else:
            G_sub = G
            m = m_full

        # 2. Extract Popularity Axis (p1) from Raw Sub-Gram
        # Top eigenvector represents the dominant popularity bias
        try:
            # CPU fallback for linalg.eigh (MPS compatibility)
            evals, evecs = torch.linalg.eigh(G_sub.cpu())
            p1 = evecs[:, -1].to(self.device) # Max eigenvector
        except:
            # Fallback: Degree-based approximation of p1
            p1 = (G_sub.sum(dim=1) + self.eps)
            p1 = p1 / (p1.norm() + self.eps)

        # 3. Initial Alignment Ratio
        # R(a) = (p1^T @ G_tilde @ p1) / Trace(G_tilde)
        # Trace of raw G_sub is sum of item frequencies.
        initial_p1_energy = (p1 @ G_sub @ p1).item()
        initial_trace = G_sub.trace().item()
        initial_ratio = initial_p1_energy / initial_trace
        
        target_ratio = initial_ratio * 0.10 # The 10% Rule discovered in Exp 3
        self._log(f"[AutoEASE] Initial Pop-Alignment: {initial_ratio:.4f} | Target: {target_ratio:.4f}")

        # 4. Binary Search for Alpha in [0.0, 1.5]
        # This is extremely fast (O(M^2) * 8 iterations) as we reuse G_sub and p1
        low, high = 0.0, 1.5
        d = G_sub.sum(dim=1) + self.eps
        best_alpha = 0.0
        
        for _ in range(8):
            alpha = (low + high) / 2.0
            
            # Normalize Sub-Gram
            inv_d_pow = 1.0 / (torch.pow(d, alpha / 2.0) + self.eps)
            # Alignment check: (p1^T @ (D^-a/2 G D^-a/2) @ p1) / Trace(...)
            # = ((p1/d^a/2)^T @ G @ (p1/d^a/2)) / Sum(diag(G)/d^a)
            p1_normed = p1 * inv_d_pow
            p1_energy = (p1_normed @ G_sub @ p1_normed).item()
            curr_trace = (G_sub.diagonal() * (inv_d_pow**2)).sum().item()
            
            curr_ratio = p1_energy / (curr_trace + self.eps)
            
            if curr_ratio > target_ratio:
                low = alpha
            else:
                high = alpha
            best_alpha = alpha

        return best_alpha

    @torch.no_grad()
    def _build(self, R_sparse):
        t0 = time.time()
        m = self.n_items
        self._log(f"Building AutoEASE-MNAR (10% Alignment Rule, lambda={self.reg_lambda}) on {self.device}")
        
        # 1. Build Gram Matrix G = R^T R
        G = _build_gram(R_sparse, self.device)
        
        # 2. Accurate Alpha Estimation via 10% Alignment Rule
        self.alpha = self._estimate_alpha_alignment(G)
        self._log(f"  Final Estimated Alpha: {self.alpha:.4f}")
        
        # 3. Final Normalization with Estimated Alpha
        d = G.sum(dim=1) + self.eps
        inv_d_pow = 1.0 / (torch.pow(d, self.alpha / 2.0) + self.eps)
        G_tilde = G * inv_d_pow.unsqueeze(1) * inv_d_pow.unsqueeze(0)
        del G
        
        # 4. Solver (Unconstrained Wiener Filter)
        I = torch.eye(m, device=self.device, dtype=torch.float32)
        A = G_tilde + self.reg_lambda * I
        
        try:
            P = torch.linalg.inv(A)
        except RuntimeError:
            self._log("Linalg inv failed, using CPU fallback.")
            P = torch.from_numpy(np.linalg.inv(A.cpu().numpy())).to(self.device).float()
        del A, G_tilde
        
        W = I - (self.reg_lambda * P)
        del P
        
        # Self-loop removal
        W.fill_diagonal_(0.0)
        
        self.W.copy_(W)
        self._log(f"AutoEASE-MNAR Build completed in {time.time()-t0:.2f}s (Final Alpha: {self.alpha:.4f})")

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
        scores = torch.matmul(user_torch, self.W)
        return scores
        
    def diagnostics(self):
        return {
            "lambda": self.reg_lambda,
            "alpha": self.alpha,
            "W_mean": float(self.W.mean()),
            "W_diag_sum": float(torch.diag(self.W).sum())
        }
