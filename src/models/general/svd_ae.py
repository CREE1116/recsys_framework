import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
import time

class SVD_AE(BaseModel):
    """
    SVD-AE (Simple Autoencoders for Collaborative Filtering) - IJCAI 2024
    Reference: Hong et al., "SVD-AE: Simple Autoencoders for Collaborative Filtering"
    
    Logic:
    1. Train standard EASE to get weight matrix B.
    2. Compute full reconstruction R = X @ B.
    3. Perform truncated SVD on R: R ≈ U_k @ Sigma_k @ V_k.T
    4. Use denoised R_k for prediction.
    """
    def __init__(self, config, data_loader):
        super(SVD_AE, self).__init__(config, data_loader)
        
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.k = config['model'].get('k', 100) # Rank for SVD truncation
        
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        self.weight_matrix = None # Initial EASE weights
        self.train_matrix_csr = None
        
        # We store the low-rank components of the denoised reconstruction matrix R
        self.U_k = None
        self.Sigma_k = None
        self.V_k = None

    def fit(self, data_loader):
        print(f"\n[SVD-AE] Fitting with lambda={self.reg_lambda}, k={self.k}...")
        start_time = time.time()
        
        # 1. Build Interaction Matrix X
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df), dtype=np.float32)
        
        self.train_matrix_csr = sp.csr_matrix(
            (values, (rows, cols)), 
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        
        # Dense X for computation (N x M)
        X = torch.from_numpy(self.train_matrix_csr.toarray()).float().to(self.device)
        
        # 2. Train Standard EASE
        print("[SVD-AE] Computing EASE weights...")
        G = X.t() @ X
        diag_indices = torch.arange(self.n_items, device=self.device)
        G[diag_indices, diag_indices] += self.reg_lambda
        
        try:
             P = torch.linalg.inv(G)
        except (RuntimeError, NotImplementedError):
             print("[SVD-AE] MPS inv failed, fallback to CPU...")
             P = torch.linalg.inv(G.cpu()).to(self.device)
             
        # B = P @ (X^T X)
        # Recompute G_raw without lambda
        G_raw = G.clone()
        G_raw[diag_indices, diag_indices] -= self.reg_lambda
        
        B = P @ G_raw
        B.fill_diagonal_(0)
        self.weight_matrix = B
        
        # 3. Compute Reconstruction R = X @ B
        print("[SVD-AE] Computing Reconstruction R...")
        R = X @ B
        
        # 4. SVD Truncation (Denoising)
        N, M = R.shape
        max_k = min(N, M)
        if self.k > max_k:
             print(f"[SVD-AE] Warning: Requested k={self.k} > min(N, M)={max_k}. Capping to {max_k}.")
             self.k = max_k
             
        print(f"[SVD-AE] Computing SVD on R (rank={self.k})...")
        # Use torch.svd_lowrank or randomized SVD for speed if checking large matrices
        # Standard svd might be slow for full N x M
        try:
             U, S, V = torch.svd_lowrank(R, q=self.k, niter=2)
             # U: (N x k), S: (k,), V: (M x k)
        except (RuntimeError, NotImplementedError):
             print("[SVD-AE] MPS svd failed, fallback to CPU...")
             U, S, V = torch.svd_lowrank(R.cpu(), q=self.k, niter=2)
             U, S, V = U.to(self.device), S.to(self.device), V.to(self.device)
             
        self.U_k = U
        self.Sigma_k = S
        self.V_k = V
        
        elapsed = time.time() - start_time
        print(f"[SVD-AE] Training complete. Time: {elapsed:.2f}s")

    def forward(self, user_ids, item_ids=None):
        if self.U_k is None:
             raise RuntimeError("[SVD-AE] Model not fitted.")
             
        # SVD-AE predictions are directly from the denoised matrix R_k
        # This is tricky: R is (N x M). 
        # For training users, we just look up the row in R_k.
        # But what about *new* users during inference if any? EASE supports inductive.
        # SVD-AE as formulated is transductive (requires user in R).
        # We assume user_ids are indices into the training matrix rows (valid for this framework's standard eval).
        
        if isinstance(user_ids, torch.Tensor):
            u_indices = user_ids.to(self.device)
        else:
            u_indices = torch.tensor(user_ids, device=self.device)
            
        # R_k[u] = U_k[u] @ Sigma_k @ V_k.T
        # U_k is (N x k). slice rows u.
        
        batch_U = self.U_k[u_indices] # (Batch x k)
        scores = (batch_U * self.Sigma_k) @ self.V_k.t() # (Batch x k) @ (k x M) -> (Batch x M)
        
        if item_ids is not None:
             if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids, device=self.device)
             batch_indices = torch.arange(len(user_ids), device=self.device)
             return scores[batch_indices, item_ids]
             
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)

    def calc_loss(self, batch_data):
        return torch.tensor(0.0, device=self.device), None

    def get_final_item_embeddings(self):
        # SVD-AE doesn't technically have "item embeddings" in standard sense, 
        # but V_k could be considered latent factors.
        return self.V_k
