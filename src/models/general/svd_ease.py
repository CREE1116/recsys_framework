import torch
import numpy as np
import scipy.sparse as sp
from .ease import EASE
from ...utils.gpu_accel import SVDCacheManager

class SVDEASE(EASE):
    """
    SVD-EASE: Scalable EASE using SVD and Woodbury Identity.
    
    1. Approximation of Gram matrix: G ≈ V * Sigma^2 * V^T
    2. Precision matrix P = (G + λI)^-1 via Woodbury identity:
       P = (1/λ) * (I - V * (λ*Sigma^-2 + I)^-1 * V^T)
    3. EASE solution: B = I - P * diag(P)^-1
    """
    def __init__(self, config, data_loader):
        super(SVDEASE, self).__init__(config, data_loader)
        
        self.k = self.config['model'].get('k', 256)
        self.k = self.k[0] if isinstance(self.k, list) else self.k
        
        # SVD Manager for caching results
        self.svd_manager = SVDCacheManager(device=self.device.type)
        
        print(f"[SVD-EASE] Initialized with k={self.k}, λ={self.reg_lambda}")

    def fit(self, data_loader):
        print(f"Fitting SVD-EASE model (k={self.k}, λ={self.reg_lambda})...")
        
        # 1. Build Interaction Matrix
        rows = data_loader.train_df['user_id'].values
        cols = data_loader.train_df['item_id'].values
        data = np.ones(len(rows))
        
        X_sparse = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=float
        )
        self.train_matrix_csr = X_sparse
        
        # 2. Perform SVD
        # Note: We can use the normalized matrix like GF-CF if needed, 
        # but standard EASE uses the raw interaction matrix.
        dataset_name = self.config.get('dataset_name', 'unknown_svdease')
        u, s, v, _ = self.svd_manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)
        
        # s: (k,), v: (n_items, k)
        s = s.to(self.device).float()
        v = v.to(self.device).float()
        
        # 3. Woodbury Identity Implementation
        # P = (1/λ) * (I - V * (λ * Sigma^-2 + I)^-1 * V^T)
        # Let D = λ * s^-2 + I  (diagonal matrix k x k)
        # P = (1/λ) * (I - V * D^-1 * V^T)
        
        # D = λ / s^2 + 1
        d_diag = self.reg_lambda / (s ** 2 + 1e-10) + 1.0
        d_inv_diag = 1.0 / d_diag
        
        # Instead of explicitly forming P (n_items x n_items), which can be memory intensive,
        # we compute diag(P) and then use the factored form for predictions.
        
        # diag(P) = (1/λ) * (1 - diag(V * D^-1 * V^T))
        # row i of (V @ diag(d_inv)) @ V^T is: sum_j (V_ij * d_inv_j * V_ij) = sum_j (V_ij^2 * d_inv_j)
        v_sq = v ** 2
        v_d_inv_v_diag = (v_sq @ d_inv_diag.unsqueeze(1)).squeeze()
        
        p_diag = (1.0 / self.reg_lambda) * (1.0 - v_d_inv_v_diag)
        
        # 4. Final weight matrix B = I - P * diag(P)^-1
        # B = I - (1/λ) * (I - V * D^-1 * V^T) * diag(P)^-1
        
        # Let W = diag(P)^-1
        # B = I - (1/λ) * (W - V @ D^-1 @ V^T @ W)
        
        # We store v and d_inv_diag, and diag(P) for efficient forward pass
        self.register_buffer('v', v)
        self.register_buffer('d_inv_diag', d_inv_diag)
        self.register_buffer('p_diag_inv', 1.0 / (p_diag + 1e-10))
        
        print("SVD-EASE model fitted.")

    def forward(self, users):
        """
        Forward pass using Woodbury factors:
        Score = X_u @ B
        B = I - (1/λ) * (P_diag_inv - V @ D^-1 @ V^T @ P_diag_inv)
        Score = X_u - (1/λ) * (X_u @ P_diag_inv - (X_u @ V) @ D^-1 @ V^T @ P_diag_inv)
        """
        device = self.device
        batch_users_np = users.cpu().numpy()
        x_u = torch.from_numpy(self.train_matrix_csr[batch_users_np].toarray()).float().to(device)
        
        # X_u @ diag(P_diag_inv)
        x_p_inv = x_u * self.p_diag_inv.unsqueeze(0)
        
        # (X_u @ V) @ D^-1 @ V^T @ diag(P_diag_inv)
        xv = torch.matmul(x_u, self.v)
        xvd = xv * self.d_inv_diag.unsqueeze(0)
        final_term = torch.matmul(xvd, self.v.t()) * self.p_diag_inv.unsqueeze(0)
        
        scores = x_u - (1.0 / self.reg_lambda) * (x_p_inv - final_term)
        
        return scores

    def get_final_item_embeddings(self):
        """Return V as item factors"""
        return self.v

    def get_embeddings(self):
        """ILD 계산 등을 위해 임베딩 반환 (None fallback 방지)"""
        return None, self.v
