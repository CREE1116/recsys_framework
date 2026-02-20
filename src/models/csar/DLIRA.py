import torch
import torch.nn as nn
import numpy as np
import os
import json
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from src.models.base_model import BaseModel

class DLIRA(BaseModel):
    """
    Directed LIRA (D-LIRA)
    A closed-form system that unifies similarity and directionality.
    K = (Xp.T @ Xp + lambda*I)^-1 @ (Xp.T @ Xf)
    Approximated via SVD for scalability.
    """
    def __init__(self, config, data_loader):
        super(DLIRA, self).__init__(config, data_loader)
        self.n_items = data_loader.n_items
        self.n_users = data_loader.n_users
        
        model_cfg = config.get('model', {})
        self.reg_lambda = model_cfg.get('reg_lambda', 500.0)
        self.eps = float(model_cfg.get('eps', 1e-8))
        self.K_svd = model_cfg.get('K_svd', 64)  # SVD rank
        self.visualize = model_cfg.get('visualize', True)

        # 1. Past(t) and Future(t+1) Matrix Construction (Sparse)
        print("[D-LIRA] Building shifted matrices (Past/Future)...")
        X_p_sparse, X_f_sparse = self._build_shifted_matrices_sparse(data_loader)
        device = self.device
        
        # 2. Closed-form Solution via SVD
        print(f"[D-LIRA] Computing Directed Kernel Approximation (k={self.K_svd})...")
        
        # Instead of full X_p (Pairs x Items), we work with Gram matrix G = X_p.T @ X_p if possible
        # but for SVD of X_p: U S V^T. 
        # Actually svds works on X_p_sparse directly and is efficient.
        try:
            # Note: k must be < min(shape)
            k = min(self.K_svd, X_p_sparse.shape[1] - 1)
            # scipy.sparse.linalg.svds is memory efficient
            _, s_k_np, Vt_k_np = svds(X_p_sparse.astype(float), k=k)
            
            # Sort descending
            idx = np.argsort(s_k_np)[::-1]
            s_k_np = s_k_np[idx]
            Vt_k_np = Vt_k_np[idx, :]
            
            V_k = torch.from_numpy(Vt_k_np.T).float().to(device)  # (n_items, k)
            s_k = torch.from_numpy(s_k_np).float().to(device)      # (k,)
            
            # (G + lambda*I)^{-1} ≈ V_k @ diag((s_k^2 + lambda)^{-1}) @ V_k.T
            diag_inv = 1.0 / (s_k**2 + self.reg_lambda) # (k,)
            
            # C = X_p.T @ X_f (Items x Items)
            # Compute this sparsely first
            C_sparse = X_p_sparse.T @ X_f_sparse
            C = torch.from_numpy(C_sparse.toarray()).float().to(device)
            
            # K ≈ V_k @ diag(diag_inv) @ (V_k.T @ C)
            Vt_k_C = torch.mm(V_k.t(), C)
            K_approx = torch.mm(V_k, diag_inv.unsqueeze(1) * Vt_k_C)
            
        except Exception as e:
            print(f"[D-LIRA] SVD failed: {e}. Falling back to small lambda identity.")
            K_approx = torch.eye(self.n_items, device=device)

        # 3. Asymmetric Normalization
        # d_r: row sums (source), d_c: col sums (target)
        d_r = torch.pow(K_approx.abs().sum(dim=1) + self.eps, -0.5)
        d_c = torch.pow(K_approx.abs().sum(dim=0) + self.eps, -0.5)
        K_final = d_r.view(-1, 1) * K_approx * d_c.view(1, -1)

        self.register_buffer('K_final', K_final)
        
        # 4. Training Matrix for inference
        self.train_matrix_csr = self._build_train_matrix(data_loader)

    def _build_shifted_matrices_sparse(self, data_loader):
        train_df = data_loader.train_df.copy()
        if 'timestamp' not in train_df.columns:
            train_df['timestamp'] = train_df.index
            
        df = train_df.sort_values(['user_id', 'timestamp'])
        df['next_item'] = df.groupby('user_id')['item_id'].shift(-1)
        df = df.dropna(subset=['next_item'])
        
        item_p = df['item_id'].values.astype(int)
        item_f = df['next_item'].values.astype(int)
        
        n_pairs = len(df)
        rows = np.arange(n_pairs)
        
        X_p = csr_matrix((np.ones(n_pairs), (rows, item_p)), shape=(n_pairs, self.n_items))
        X_f = csr_matrix((np.ones(n_pairs), (rows, item_f)), shape=(n_pairs, self.n_items))
        
        return X_p, X_f

    def _build_train_matrix(self, data_loader):
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        return csr_matrix((np.ones(len(train_df)), (rows, cols)), shape=(self.n_users, self.n_items))

    def forward(self, users, mask_observed=True):
        device = self.device
        batch_users_np = users.cpu().numpy()
        X_u_sparse = self.train_matrix_csr[batch_users_np]
        X_u = torch.from_numpy(X_u_sparse.toarray()).float().to(device)
        
        scores = torch.mm(X_u, self.K_final)
        
        if mask_observed:
            rows, cols = X_u.nonzero(as_tuple=True)
            scores[rows, cols] = -1e9
            
        return scores

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), {}

    def predict_for_pairs(self, user_ids, item_ids):
        device = self.device
        batch_users_np = user_ids.cpu().numpy()
        X_u_sparse = self.train_matrix_csr[batch_users_np]
        X_u = torch.from_numpy(X_u_sparse.toarray()).float().to(device)
        
        # (B, I) @ (I, B_items) -> diagonal
        # Efficiently: sum(X_u * K_final[:, items], dim=1)
        relevant_K = self.K_final[:, item_ids] # (I, B)
        # This is memory intensive if B is large. Do it row-wise or dot.
        scores = (X_u * relevant_K.t()).sum(dim=1)
        return scores

    def get_final_item_embeddings(self):
        return self.K_final
