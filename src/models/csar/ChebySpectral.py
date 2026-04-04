import torch
import torch.nn as nn
import numpy as np
import time
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.utils.gpu_accel import get_device

class ChebySpectral(BaseModel):
    """
    ChebySpectral: Inversion-free Spectral Filter using Chebyshev Polynomials.
    
    1. Spectral Energy Balance (Sinkhorn-style d estimation).
    2. Approximate the Wiener filter f(x) = x / (x + lambda) using Chebyshev series.
    3. No zero-diagonal constraint (Constraint-free).
    """

    def __init__(self, config, data_loader):
        super(ChebySpectral, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_config = config.get('model', {})
        self.cheb_order = int(model_config.get('cheb_order', 20))
        # use_lagrange: True (Always), False (Never), or "adaptive" (Item-specific)
        self.use_lagrange = model_config.get('use_lagrange', True)
        self.num_hops   = int(model_config.get('num_hops', 3))
        self.tol_d      = float(model_config.get('tol_d', 1e-6))
        self.eps        = 1e-8

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
        """
        Estimate d using Symmetric Sinkhorn Balancing (Row/Col scaling).
        This reaches a deeper equilibrium than simple fixed-point iterations.
        """
        self._log(f"Estimating d via Sinkhorn Balancing (Max Iters={self.num_hops}, tol={self.tol_d})...")
        n = G.shape[0]
        # Initialize factors
        d_row = torch.ones(n, device=self.device)
        d_col = torch.ones(n, device=self.device)
        
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
    def _power_iteration(self, A, n_iter=10):
        """Estimate the maximum eigenvalue of A using power iteration."""
        m = A.shape[0]
        v = torch.randn(m, 1, device=self.device)
        v = v / torch.norm(v)
        
        for _ in range(n_iter):
            v_next = torch.mm(A, v)
            v = v_next / torch.norm(v_next)
            
        # Rayleigh quotient: lambda_max = (v^T A v) / (v^T v)
        # Since v is normalized, v^T A v = v^T v_next
        lambda_max = torch.mm(v.t(), torch.mm(A, v)).item()
        return max(lambda_max, 1e-6)

    def _get_cheb_coefficients(self, f, K):
        """Compute K+1 Chebyshev coefficients for function f on [-1, 1]."""
        # Node indices j=0..K
        j = torch.arange(K + 1, device=self.device).float()
        # Chebyshev nodes (zeros of T_{K+1})
        nodes = torch.cos(np.pi * (j + 0.5) / (K + 1))
        # Function values at nodes
        vals = f(nodes)
        
        coeffs = torch.zeros(K + 1, device=self.device)
        # c_k calculation for f(x) = 0.5*c0 + sum_{k=1}^K c_k T_k(x)
        for k in range(K + 1):
            weights = torch.cos(k * np.pi * (j + 0.5) / (K + 1))
            coeffs[k] = (2.0 / (K + 1)) * torch.sum(vals * weights)
            
        return coeffs

    @torch.no_grad()
    def _build(self, R_sparse, dataset_name):
        n, m = R_sparse.shape
        self._log(f"Building ChebySpectral (lambda={self.reg_lambda}, order={self.cheb_order})")
        t0 = time.time()

        # 1. G_obs = R^T R
        from src.utils.gpu_accel import _build_gram
        G = _build_gram(R_sparse, self.device)
        
        # 2. Fixed-Point for Spectral Balancing
        d = self._estimate_d_spectral_fp(G)
        
        # 3. Spectral Pre-conditioning (Full Balancing: D^-1/2 on each side)
        # G_tilde = D^-1/2 @ G @ D^-1/2
        inv_sqrt_d = 1.0 / torch.sqrt(d + self.eps)
        
        G_tilde = G * inv_sqrt_d.unsqueeze(1) * inv_sqrt_d.unsqueeze(0)
        del G
        
        # 4. Adaptive Bias density estimation
        # Calculate diag_ratio BEFORE normalization/filtering
        diag_val = G_tilde.diagonal()
        diag_ratio = diag_val / (diag_val + G_tilde.mean(dim=1) + self.eps)

        # 4. Accurate Eigenvalue Scaling for Stability
        # Estimate lambda_max to map range [0, lambda_max] -> [-1, 1]
        self._log("Estimating lambda_max via Power Iteration...")
        l_max = self._power_iteration(G_tilde, n_iter=10)
        # Add a small safety margin (1.05x) to ensure we are strictly within [-1, 1]
        l_max *= 1.05
        self._log(f"  Estimated lambda_max (with margin) = {l_max:.4f}")

        # Map x in [0, l_max] to t = (2*x/l_max) - 1
        # f(x) = x / (x + lambda)
        # g(t) = f(x(t)) where x = (t+1)*l_max / 2
        def g(t):
            x = (t + 1.0) * l_max / 2.0
            return x / (x + self.reg_lambda + self.eps)

        coeffs = self._get_cheb_coefficients(g, self.cheb_order)
        
        # Clenshaw Recurrence for Matrix Polynomial p(G_hat)
        # G_hat = (2/l_max) * G_tilde - I
        I = torch.eye(m, device=self.device)
        G_hat = (2.0 / l_max) * G_tilde - I
        del G_tilde

        self._log(f"Applying Clenshaw recurrence (order {self.cheb_order})...")
        
        # Standard Clenshaw for f(x) = 0.5*c0 + sum c_k T_k(x)
        # Note: We use 0.5*c0 convention.
        # b_{K+2} = 0, b_{K+1} = 0
        # b_k = c_k*I + 2*G_hat*b_{k+1} - b_{k+2}
        b2 = torch.zeros((m, m), device=self.device)
        b1 = torch.zeros((m, m), device=self.device)
        
        for k in range(self.cheb_order, 0, -1):
            # tmp = c_k*I + 2*G_hat*b_1 - b_2
            tmp = 2.0 * torch.mm(G_hat, b1) - b2
            tmp.diagonal().add_(coeffs[k])
            b2 = b1
            b1 = tmp
            
        # Final result: W_wiener = 0.5 * c0 * I + G_hat * b1 - b2
        W_wiener = torch.mm(G_hat, b1) - b2
        W_wiener.diagonal().add_(0.5 * coeffs[0])
        
        if self.use_lagrange == "adaptive":
            self._log("  Applying Adaptive Lagrange Constraint...")
            # P = (G + lambda I)^-1 = (I - W_wiener) / lambda
            P = (torch.eye(m, device=self.device) - W_wiener) / (self.reg_lambda + self.eps)
            diag_P = P.diagonal()
            # Scaling: W = I - P * (diag_ratio / diag_P + (1 - diag_ratio) * lambda)
            scaling = (diag_ratio.unsqueeze(0) / torch.clamp(diag_P.unsqueeze(0), min=self.eps)) + \
                      (1.0 - diag_ratio).unsqueeze(0) * self.reg_lambda
            W = torch.eye(m, device=self.device) - (P * scaling)
        elif self.use_lagrange is True:
            self._log("  Applying Strict Lagrange Constraint (Zero-Diagonal)...")
            P = (torch.eye(m, device=self.device) - W_wiener) / (self.reg_lambda + self.eps)
            diag_P = P.diagonal()
            W = torch.eye(m, device=self.device) - (P / torch.clamp(diag_P.unsqueeze(0), min=self.eps))
        else:
            W = W_wiener

        self.W.copy_(W)
        diag_W = torch.diag(self.W)
        self._log(f"ChebySpectral Build completed. W_diag_abs_max: {diag_W.abs().max().item():.2e}")
        self._log(f"Build completed in {time.time()-t0:.2f}s")

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
        diag_W = torch.diag(self.W)
        return {
            "lambda": self.reg_lambda,
            "order": self.cheb_order,
            "use_lagrange": self.use_lagrange,
            "W_mean": float(self.W.mean()),
            "W_diag_abs_max": float(diag_W.abs().max()),
            "W_diag_mean": float(diag_W.mean()),
        }
