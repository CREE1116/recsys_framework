import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
import time

class Infinity_AE(BaseModel):
    """
    Infinity-AE (Infinite-Width Shallow Autoencoder) - NeurIPS 2022
    Reference: Sachdeva et al., "∞-AE: Infinite-Width Shallow Autoencoders"
    
    Logic:
    1. Compute User-User Gram matrix G = X @ X.T
    2. Compute Squared Kernel K = G @ G
    3. Solve Alpha = (K + lambda*I)^-1 @ K
    4. Compute Item-Item weights B = X.T @ Alpha @ X
    5. Enforce diag(B) = 0
    """
    def __init__(self, config, data_loader):
        super(Infinity_AE, self).__init__(config, data_loader)
        
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        self.weight_matrix = None
        self.train_matrix_csr = None

    def fit(self, data_loader):
        self._log(f"Fitting with lambda={self.reg_lambda}...")
        start_time = time.time()
        
        # 1. Build Dense Interaction Matrix X (N x M)
        # Note: Infinity-AE requires N x N kernel operations, which is O(N^3).
        # For large N (e.g. > 20k), this will be very slow/expensive.
        
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df), dtype=np.float32)
        
        X_csr = sp.csr_matrix(
            (values, (rows, cols)), 
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        self.train_matrix_csr = X_csr
        
        # Convert to Dense Tensor
        # Caution: X can be large. 
        X = torch.from_numpy(X_csr.toarray()).float().to(self.device)

        # [User Request] Normalize by degree
        deg = torch.sum(X, dim=1, keepdim=True)
        X = X / torch.sqrt(deg + 1e-8)
        
        # 2. Compute Kernel K = (X @ X.T)^2
        self._log("Computing User-User Gram Matrix...")
        G = X @ X.t()  # (N x N)
        
        self._log("Computing Squared Kernel...")
        K = G @ G # (N x N)
        del G  # 메모리 해제
        
        # 3. Ridge Regression: Alpha = (K + lambda*I)^-1 @ K
        self._log("Solving Kernel Regression...")
        diag_indices = torch.arange(self.n_users, device=self.device)
        K_reg = K.clone()
        K_reg[diag_indices, diag_indices] += self.reg_lambda
        
        try:
             Alpha = torch.linalg.solve(K_reg, K)
        except RuntimeError:
             # 대규모 행렬에서 MPS 메모리 부족 시 CPU fallback
             self._log("GPU solve failed, fallback to CPU...")
             Alpha = torch.linalg.solve(K_reg.cpu(), K.cpu()).to(self.device)
        del K, K_reg  # 메모리 해제
        
        # 4. Compute B = X.T @ Alpha @ X
        self._log("Computing Weight Matrix B...")
        temp = Alpha @ X
        del Alpha
        B = X.t() @ temp
        del temp
        
        # 5. Enforce Diagonal Constraint
        B.fill_diagonal_(0)
        
        self.weight_matrix = B
        
        elapsed = time.time() - start_time
        self._log(f"Training complete via Kernel Method. Time: {elapsed:.2f}s")

    def forward(self, user_ids, item_ids=None):
        if self.weight_matrix is None:
             raise RuntimeError("[Infinity-AE] Model not fitted.")
             
        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = np.array(user_ids)
            
        # Get user history
        user_input_sparse = self.train_matrix_csr[u_ids_np]
        user_input = torch.from_numpy(user_input_sparse.toarray()).float().to(self.device)
        
        # Scores = X @ B
        scores = user_input @ self.weight_matrix
        
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
        return self.weight_matrix
