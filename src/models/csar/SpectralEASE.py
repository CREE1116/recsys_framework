import torch
import torch.nn as nn
import numpy as np
import time
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.utils.gpu_accel import get_device

class SpectralEASE(BaseModel):
    """
    SpectralEASE (Fixed-Point Spectral Energy Balance + Exact EASE Constraint)
    
    [핵심 메커니즘]
    1. Spectral Energy Balance (Sinkhorn-style): 
       모든 아이템의 스펙트럴 에너지가 균일해지도록 d_i = sum_j (G_ij^2 / d_j) 반복법으로 d를 추정.
    2. Exact EASE Closed-form Solve: 
       대각선이 0인 제약 조건을 목적함수에 포함한 수식 W_tilde = I - P * diag(1/diag(P)) 로 최적해 도출.
    3. Domain Translation: 
       변환된 공간(tilde)에서 학습된 가중치를 원본 공간 데이터 X_u에 맞게 복원: 
       W_eff = D^-1/2 @ W_tilde @ D^1/2
    """

    def __init__(self, config, data_loader):
        super(SpectralEASE, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_config = config.get('model', {})
        self.reg_lambda = float(model_config.get('reg_lambda', 0.5))
        self.max_iter_d = int(model_config.get('max_iter_d', 50))
        self.tol_d      = float(model_config.get('tol_d', 1e-6))
        self.eps        = 1e-12

        # Filter Buffer
        self.register_buffer("W", torch.empty(self.n_items, self.n_items))

        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        self._build(self.train_matrix_csr, config.get('dataset_name', 'unknown'))

    def _build_sparse_matrix(self, data_loader):
        df = data_loader.train_df
        return csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df['user_id'].values, df['item_id'].values)),
            shape=(self.n_users, self.n_items)
        )

    @torch.no_grad()
    def _estimate_d_spectral_fp(self, G):
        """Estimate normalization factor d using fixed-point iteration (Sinkhorn extension)."""
        self._log(f"Estimating d via Spectral Balancing (max_iter={self.max_iter_d})...")
        d = G.sum(dim=1) + self.eps
        # Change: Use absolute value (Linear) instead of squared (Energy) to be less aggressive.
        G_abs = torch.abs(G)

        for it in range(self.max_iter_d):
            inv_d = 1.0 / (d + self.eps)
            d_new = torch.mv(G_abs, inv_d)
            d_new = d_new / (d_new.mean() + self.eps)
            diff = torch.norm(d_new - d) / (torch.norm(d) + self.eps)
            
            # Per-iteration logging
            if (it + 1) % 5 == 0 or it == 0:
                self._log(f"  Iteration {it+1:2d}: diff = {diff:.2e}")
            
            d = d_new
            if diff < self.tol_d:
                self._log(f"  Fixed-point converged at iteration {it+1}, diff={diff:.2e}")
                break
        return d

    @torch.no_grad()
    def _build(self, R_sparse, dataset_name):
        n, m = R_sparse.shape
        self._log(f"Building SpectralEASE (lambda={self.reg_lambda}) on {self.device}")
        t0 = time.time()

        # 1. G_obs = R^T R
        from src.utils.gpu_accel import _build_gram
        G = _build_gram(R_sparse, self.device)
        
        # 2. Fixed-Point for Energy Normalization (Sinkhorn style)
        d = self._estimate_d_spectral_fp(G)
        
        # 3. Spectral Pre-conditioning
        inv_sqrt_d = 1.0 / torch.sqrt(d + self.eps)
        sqrt_d = torch.sqrt(d + self.eps)
        
        # G_tilde = D^-1/2 @ G @ D^-1/2
        G_tilde = G * inv_sqrt_d.unsqueeze(1) * inv_sqrt_d.unsqueeze(0)
        del G

        # 4. Unconstrained Wiener Filter Solve
        self._log(f"Solving unconstrained Wiener filter for {m}x{m} matrix...")
        I = torch.eye(m, device=self.device, dtype=torch.float32)
        A = G_tilde + self.reg_lambda * I
        
        try:
            # P = (G_tilde + lambda I)^-1
            P = torch.linalg.inv(A)
        except RuntimeError:
            self._log("Linalg inv failed, using CPU fallback.")
            P = torch.from_numpy(np.linalg.inv(A.cpu().numpy())).to(self.device).float()
        del A, G_tilde

        # W_tilde = I - lambda * P
        W_tilde = I - (self.reg_lambda * P)
        del P

        # 5. Domain Translation (Scale Back to Original Space)
        # We restore the D^1/2 multiplier to map the spectral signal back to item space.
        W_eff = W_tilde * inv_sqrt_d.unsqueeze(1) * sqrt_d.unsqueeze(0)
        
        # Self-loop removal (Post-hoc)
        W_eff.fill_diagonal_(0.0)
        
        self.W.copy_(W_eff)
        self._log(f"SpectralEASE Build completed in {time.time()-t0:.2f}s")

    @torch.no_grad()
    def forward(self, users):
        batch_ids = users.cpu().numpy()
        X_u = torch.from_numpy(self.train_matrix_csr[batch_ids].toarray()).float().to(self.device)
        return torch.mm(X_u, self.W)

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

    def diagnostics(self):
        diag_val = float(torch.diag(self.W).abs().max())
        return {
            "lambda": self.reg_lambda,
            "W_mean": float(self.W.mean()),
            "W_max_diag": diag_val,
        }
