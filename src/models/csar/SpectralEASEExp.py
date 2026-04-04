import torch
import torch.nn as nn
import numpy as np
import time
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.utils.gpu_accel import get_device

class SpectralEASEExp(BaseModel):
    """
    SpectralEASE Experimental (Extends base with propensity_type and alpha flags for Exp 5)
    
    [핵심 메커니즘]
    1. Spectral Energy Balance (Sinkhorn-style): 
       모든 아이템의 스펙트럴 에너지가 균일해지도록 d_i = sum_j (G_ij / d_j) 반복법으로 d를 추정.
    2. Exact EASE Closed-form Solve: 
       대각선이 0인 제약 조건을 목적함수에 포함한 수식 W_tilde = I - P * diag(1/diag(P)) 로 최적해 도출.
    3. Domain Translation: 
       변환된 공간(tilde)에서 학습된 가중치를 원본 공간 데이터 X_u에 맞게 복원: 
       W_eff = D^-1/2 @ W_tilde @ D^1/2
    """

    def __init__(self, config, data_loader):
        super(SpectralEASEExp, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_config = config.get('model', {})
        self.reg_lambda = float(model_config.get('reg_lambda', 0.5))
        # num_hops: Low-pass filter의 강도를 결정. 
        # 값이 커질수록 더 초저주파(초거시적 시스템 편향)만 남김.
        self.num_hops   = int(model_config.get('num_hops', 3))
        self.tol_d      = float(model_config.get('tol_d', 1e-6))
        # use_lagrange: True (Always), False (Never), or "adaptive" (Item-specific)
        self.use_lagrange = model_config.get('use_lagrange', True)
        # propensity_type: "sinkhorn" (Balanced) or "frequency" (Degree-based)
        self.propensity_type = model_config.get('propensity_type', 'sinkhorn')
        # alpha: Propensity exponent (1.0 = Standard ASPIRE/Balanced, 0.5 = Square-root smoothing)
        self.alpha      = float(model_config.get('alpha', 1.0))
        self.eps        = 1e-8

        # Filter Parameter (Instead of buffer, to follow user's snippet preference)
        self.W = nn.Parameter(torch.empty(self.n_items, self.n_items))

        self.to(self.device)
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
        """
        Estimate d using Symmetric Sinkhorn Balancing (Row/Col scaling).
        This reaches a deeper equilibrium than simple fixed-point iterations.
        """
        self._log(f"Estimating d via Sinkhorn Balancing (Max Iters={self.num_hops}, tol={self.tol_d})...")
        n = G.shape[0]
        # Initialize factors: Warm-start using Row Sums (Degrees)
        d_init = torch.sum(G, dim=1)
        d_row = torch.sqrt(d_init + self.eps)
        d_col = torch.sqrt(d_init + self.eps)
        
        for it in range(self.num_hops):
            # Row scaling: d_row_new = sqrt( (G @ d_col) / d_row )
            row_sums = torch.mv(G, d_col) / (d_row + self.eps)
            d_row_new = torch.sqrt(row_sums + self.eps)
            
            # Column scaling: d_col_new = sqrt( (G @ d_row_new) / d_col )
            col_sums = torch.mv(G, d_row_new) / (d_col + self.eps)
            d_col_new = torch.sqrt(col_sums + self.eps)
            
            diff = max(torch.norm(d_row_new - d_row), torch.norm(d_col_new - d_col)) / (torch.norm(d_row) + self.eps)
            d_row, d_col = d_row_new, d_col_new
            
            if diff < self.tol_d:
                self._log(f"  Sinkhorn converged at iter {it+1}, diff={diff:.2e}")
                break
                
        # Final balancing factor d = d_row * d_col
        # This ensures the spectral energy is evenly distributed as per ASPIRE theory.
        d = d_row * d_col
        return d

    @torch.no_grad()
    def _build(self, R_sparse, dataset_name):
        n, m = R_sparse.shape
        self._log(f"Building SpectralEASE-Wiener (lambda={self.reg_lambda}) on {self.device}")
        t0 = time.time()

        # 1. Build Gram Matrix G = R^T R
        from src.utils.gpu_accel import _build_gram
        G = _build_gram(R_sparse, self.device)
        
        # 2. Estimate Propensity Factor D
        if self.propensity_type == "frequency":
            # Simple item popularity (Degree of the Gram matrix)
            # In implicit R, this is precisely the number of interactions per item.
            d = torch.from_numpy(np.array(R_sparse.sum(axis=0)).flatten()).to(self.device).float()
            self._log(f"Using frequency-based propensity (alpha={self.alpha})")
        else:
            # Full Spectral Balancing (Sinkhorn)
            d = self._estimate_d_spectral_fp(G)
            self._log(f"Using Sinkhorn-based spectral propensity (alpha={self.alpha})")
        
        # Diagnostic: Compare d with simple RowSum
        d_rowsum = G.sum(dim=1)
        corr = torch.corrcoef(torch.stack([d, d_rowsum]))[0, 1]
        self._log(f"Estimated d vs RowSum correlation: {corr:.4f}")
        
        # 3. Spectral Pre-conditioning (Exponent-based D Correction: D^-alpha/2 @ G @ D^-alpha/2)
        # alpha=1.0 is full balancing, alpha=0.5 is popularity-preserving smoothing.
        inv_sqrt_d = 1.0 / torch.sqrt(torch.pow(d + self.eps, self.alpha))
        G_tilde = G * inv_sqrt_d.unsqueeze(1) * inv_sqrt_d.unsqueeze(0)
        del G

        # 4. Adaptive Lagrange / Wiener Optimization
        # Calculate diag_ratio BEFORE adding reg_lambda to preserve pure correlation density
        diag_val = G_tilde.diagonal()
        # Item-wise density ratio: high for sparse items, low for dense (popular) items
        diag_ratio = diag_val / (diag_val + G_tilde.mean(dim=1) + self.eps)
        
        self._log(f"Solving SpectralEASE (lambda={self.reg_lambda}, lagrange={self.use_lagrange})...")
        G_tilde.diagonal().add_(self.reg_lambda)
        
        try:
            P = torch.linalg.inv(G_tilde)
        except RuntimeError:
            self._log("Linalg inv failed, using CPU fallback.")
            P = torch.from_numpy(np.linalg.inv(G_tilde.cpu().numpy())).to(self.device).float()
        del G_tilde

        I = torch.eye(m, device=self.device, dtype=torch.float32)
        diag_P = P.diagonal()

        if self.use_lagrange == "adaptive":
            self._log("  Applying Adaptive Lagrange Constraint...")
            # Hybrid: Sparse -> Lagrange (Identity penalty), Dense -> Wiener (Global penalty)
            # Row-wise interpolation: W = I - P * (diag_ratio / diag_P + (1 - diag_ratio) * lambda)
            # Use unsqueeze(0) for column-wise multiplication (since P is M x M and scalars are per-item)
            scaling = (diag_ratio.unsqueeze(0) / torch.clamp(diag_P.unsqueeze(0), min=self.eps)) + \
                      (1.0 - diag_ratio).unsqueeze(0) * self.reg_lambda
            W_tilde = I - (P * scaling)
        elif self.use_lagrange is True:
            self._log("  Applying Strict Lagrange Constraint (Zero-Diagonal)...")
            W_tilde = I - (P / torch.clamp(diag_P.unsqueeze(0), min=self.eps))
        else:
            self._log("  Solving Unconstrained Wiener (No Diagonal Constraint)...")
            W_tilde = I - (self.reg_lambda * P)
            
        del P

        # 5. Domain Translation
        # Intentionally skipped: Using W_tilde directly penalizes items
        # scaled down by p_sys_inv, maintaining the debiasing effect during inference.
        W_eff = W_tilde
        
        if self.W.numel() == 0:
            self.W = nn.Parameter(W_eff)
        else:
            self.W.data.copy_(W_eff)
            
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
