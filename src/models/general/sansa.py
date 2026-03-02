import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
import time

class SANSA(BaseModel):
    """
    SANSA (Scalable Approximate NonSymmetric Autoencoder) - RecSys 2023
    "Scalable Linear Shallow Autoencoders for Collaborative Filtering"
    
    Refined Implementation:
    1. Gram Matrix: A = X^T X + lambda I
    2. Cholesky Decomposition: A = L L^T (Dense, CPU fallback)
    3. Invert Factor: L_inv = L^{-1}
    4. Sparsify L_inv -> K (Top-k or Threshold)
    5. Approximate Weights: B = I - lambda * K^T K
    6. Prediction: Scores = X B = X - lambda * (X K^T) K
    """
    def __init__(self, config, data_loader):
        super(SANSA, self).__init__(config, data_loader)
        
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.density = config['model'].get('density', 0.05) # Target density for K
        self.target_k = config['model'].get('target_k', 0)  # Optional: Top-k per row for K
        
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        # We store K (Sparse) and potentially B?
        # Storing B (M x M) might be dense if K^T K is not sparse enough.
        # Paper suggests computing X B on the fly using K.
        # B = I - lambda K^T K
        # X B = X - lambda X K^T K
        # We store K.
        
        self.K_indices = None
        self.K_values = None
        self.K_shape = None
        
        self.train_matrix_csr = None

    def fit(self, data_loader):
        print(f"\n{'='*60}")
        self._log(f"Fitting with lambda={self.reg_lambda}, density={self.density}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 1. Build Interaction Matrix
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
        
        # 2. Compute Gram Matrix A = X^T X + lambda I
        self._log("Computing Gram Matrix...")
        G = (X_csr.T @ X_csr).toarray().astype(np.float32)
        diag_indices = np.arange(self.n_items)
        G[diag_indices, diag_indices] += self.reg_lambda
        
        # 3. Cholesky → L^-1 for sparsification
        self._log("Cholesky Decomposition...")
        from src.utils.gpu_accel import get_device
        device = get_device()
        
        try:
            # GPU-accelerated Cholesky
            import torch
            G_t = torch.from_numpy(G).float().to(device)
            L_t = torch.linalg.cholesky(G_t)
            del G_t, G
            
            # Solve L @ L_inv = I on GPU
            self._log("Inverting Cholesky Factor...")
            I_t = torch.eye(self.n_items, device=device, dtype=torch.float32)
            L_inv_t = torch.linalg.solve_triangular(L_t, I_t, upper=False)
            del L_t, I_t
            L_inv_np = L_inv_t.cpu().numpy()
            del L_inv_t
        except (RuntimeError, NotImplementedError):
            self._log("GPU failed, CPU fallback...")
            from scipy.linalg import cho_factor, solve_triangular as sp_solve_tri
            cho, low = cho_factor(G)
            del G
            L = np.linalg.cholesky(cho if not low else cho.T)
            del cho
            L_inv_np = sp_solve_tri(L, np.eye(self.n_items, dtype=np.float32), lower=True)
            del L
        
        # Global thresholding or Row-wise top-k?
        # Paper implies we want K to be sparse.
        # Let's use global density for simplicity or top-k per row if specified.
        
        K_sparse = np.zeros_like(L_inv_np)

        if self.target_k > 0:
            # Vectorized top-k per row (numpy argpartition across full matrix)
            # L_inv_np: (M, M)
            abs_L = np.abs(L_inv_np)
            # argpartition returns indices of k largest in row (unordered)
            top_k_idx = np.argpartition(abs_L, -self.target_k, axis=1)[:, -self.target_k:]
            # Construct row indices
            row_idx = np.arange(self.n_items)[:, None] * np.ones(self.target_k, dtype=int)
            K_sparse[row_idx, top_k_idx] = L_inv_np[row_idx, top_k_idx]
            self._log(f"Sparsified K: top-{self.target_k} per row (vectorized)")
        else:
            # Global threshold
            threshold = np.percentile(np.abs(L_inv_np), 100 * (1 - self.density))
            mask = np.abs(L_inv_np) >= threshold
            K_sparse[mask] = L_inv_np[mask]
             
        # Convert K to sparse tensor
        K_coo = sp.coo_matrix(K_sparse)
        
        self.K_indices = torch.LongTensor([K_coo.row, K_coo.col])
        self.K_values = torch.FloatTensor(K_coo.data)
        self.K_shape = (self.n_items, self.n_items)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        self._log("Training complete!")
        print(f"  - Time: {elapsed:.2f}s")
        print(f"  - K nnz: {K_coo.nnz:,} (Density: {K_coo.nnz / (self.n_items**2):.4f})")
        print(f"{'='*60}\n")
        
    def forward(self, user_ids, item_ids=None):
        if self.K_indices is None:
             raise RuntimeError("[SANSA] Model not fitted.")
             
        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = np.array(user_ids)
            
        user_input_sparse = self.train_matrix_csr[u_ids_np]
        user_input = torch.from_numpy(
            user_input_sparse.toarray()
        ).float().to(self.device)
        
        # Reconstruct K on device
        K = torch.sparse_coo_tensor(
            self.K_indices, self.K_values, self.K_shape
        ).to(self.device)
        
        # Prediction: Scores = X - lambda * X @ K^T @ K
        # 1. A = X @ K^T
        # Sparse MM if possible
        try:
             # K is sparse, K.t() is sparse. user_input is dense.
             # Dense @ Sparse -> Dense supported?
             # user_input (B x M), K.t() (M x M)
             # torch.mm(dense, sparse) is NOT supported usually.
             
             # But torch.sparse.mm(sparse, dense) is supported.
             # We can use: (K @ X^T)^T = X @ K^T
             
             # Actually, let's densify K for computation if M is not too huge?
             # Or use sparse_mm with care.
             
             # MPS Fallback reasoning:
             # Sparsity is high, so densifying K might lose memory benefit but be faster on MPS?
             # K_dense = K.to_dense()
             # temp = user_input @ K_dense.t()
             # term2 = temp @ K_dense
             
             # Let's try to keep it sparse if possible.
             # X (Batch, M)
             # K (M, M)
             
             # X @ K^T
             # = (K @ X^T)^T
             
             K_dense = K.to_dense() # Fallback for compatibility/ease
             term1 = user_input @ K_dense.t()
             term2 = term1 @ K_dense
             
             scores = user_input - self.reg_lambda * term2
             
        except (RuntimeError, NotImplementedError):
             # CPU Fallback
             X_cpu = user_input.cpu()
             K_cpu = K.cpu().to_dense()
             
             term1 = X_cpu @ K_cpu.t()
             term2 = term1 @ K_cpu
             
             scores_cpu = X_cpu - self.reg_lambda * term2
             scores = scores_cpu.to(self.device)
             
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
        # SANSA approximates weights, returns B = I - lambda K^T K
        # Construct B on demand
        K = torch.sparse_coo_tensor(
            self.K_indices, self.K_values, self.K_shape
        ).to(self.device)
        K_dense = K.to_dense()
        I = torch.eye(self.n_items, device=self.device)
        B = I - self.reg_lambda * (K_dense.t() @ K_dense)
        return B
