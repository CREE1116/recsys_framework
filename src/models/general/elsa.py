import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
import time

class ELSA(BaseModel):
    """
    ELSA (Scalable Linear Shallow Autoencoder)
    Reference: Vančura et al., RecSys 2022
    
    Key idea:
    1. Compute EASE: B = (G + λI)^{-1} G, diag(B)=0
    2. Decompose: B ≈ L @ S + R (low-rank + sparse)
    """
    def __init__(self, config, data_loader):
        super(ELSA, self).__init__(config, data_loader)
        
        # Hyperparameters
        self.rank = config['model'].get('rank', 256)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.sparse_k = config['model'].get('sparse_k', 50)  # Top-k per row
        
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        # Low-rank components (will be computed in fit())
        self.register_buffer('L', torch.empty(0, 0))  # (M × d)
        self.register_buffer('S', torch.empty(0, 0))  # (d × M)
        self.register_buffer('R', torch.empty(0, 0))  # (M × M) sparse
        
        self.train_matrix_csr = None
        
        self._log(f"Config: rank={self.rank}, λ={self.reg_lambda}, sparse_k={self.sparse_k}")

    def _build_sparse_matrix(self, data_loader):
        """Build user-item interaction matrix"""
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df), dtype=np.float32)
        
        return sp.csr_matrix(
            (values, (rows, cols)), 
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )

    def fit(self, data_loader):
        """
        ELSA training: Closed-form + SVD decomposition
        """
        self._log(f"\n{'='*60}")
        self._log("Starting training...")
        self._log(f"{'='*60}")
        
        start_time = time.time()
        
        # 1. Build interaction matrix
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        X = self.train_matrix_csr
        
        self._log(f"Matrix: {X.shape}, nnz={X.nnz:,}")
        
        # GPU-accelerated EASE solution
        from src.utils.gpu_accel import gpu_gram_solve
        P = gpu_gram_solve(X, self.reg_lambda, device=self.device, return_tensor=True)
        
        diag = torch.diagonal(P).clone()
        B_t = -P / diag.view(1, -1)
        B_t.diagonal().fill_(0)
        del P
        
        # 3. Low-rank decomposition via SVD
        self._log(f"Computing SVD (rank={self.rank}) on {self.device}...")
        
        try:
            U, Sigma, Vh = torch.linalg.svd(B_t, full_matrices=False)
            U_d = U[:, :self.rank]
            Sigma_d = Sigma[:self.rank]
            Vh_d = Vh[:self.rank, :]
            del U, Sigma, Vh
        except RuntimeError:
            self._log("GPU SVD failed, fallback to CPU...")
            B_np = B_t.cpu().numpy()
            U, Sigma, Vh = np.linalg.svd(B_np, full_matrices=False)
            U_d = torch.from_numpy(U[:, :self.rank]).to(self.device)
            Sigma_d = torch.from_numpy(Sigma[:self.rank]).to(self.device)
            Vh_d = torch.from_numpy(Vh[:self.rank, :]).to(self.device)
            del B_np, U, Sigma, Vh

        # L @ S decomposition
        sqrt_sigma = torch.sqrt(Sigma_d)
        L = U_d * sqrt_sigma.view(1, -1)
        S = sqrt_sigma.view(-1, 1) * Vh_d
        
        # 4. Sparse residual
        self._log("Filtering sparse residual...")
        R = B_t - (L @ S)
        
        # Sparsify: keep top-k per row (using torch.topk)
        mask = torch.zeros_like(R, dtype=torch.bool)
        _, topk_indices = torch.topk(torch.abs(R), self.sparse_k, dim=1)
        mask.scatter_(1, topk_indices, True)
        R = R * mask
        
        # 5. Store components
        self.register_buffer('L', L)
        self.register_buffer('S', S)
        self.register_buffer('R', R.to_sparse_coo().coalesce())
        
        # 6. Statistics
        elapsed = time.time() - start_time
        R_coalesced = self.R.coalesce()
        R_nnz = R_coalesced._nnz()

        self._log(f"\n{'='*60}")
        self._log("Training complete!")
        self._log(f"  - Time: {elapsed:.2f}s")
        self._log(f"  - Low-rank: {L.shape}")
        self._log(f"  - Sparse nnz: {R_nnz:,}")
        mem_approx = (L.numel() + S.numel() + R_nnz) * 4 / 1e6
        self._log(f"  - Memory (approx): {mem_approx:.1f} MB")
        self._log(f"{'='*60}\n")

    def forward(self, user_ids, item_ids=None):
        """
        Forward pass: scores = (X @ L) @ S + X @ R
        """
        if self.L is None:
            raise RuntimeError("[ELSA] Model not fitted. Call fit() first.")
        
        # Get user histories
        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = np.array(user_ids)
        
        user_input_sparse = self.train_matrix_csr[u_ids_np]
        user_input = torch.from_numpy(
            user_input_sparse.toarray()
        ).float().to(self.device)
        
        # Low-rank component: (X @ L) @ S
        latent = user_input @ self.L  # (batch × d)
        scores_lr = latent @ self.S   # (batch × M)
        
        # Sparse component: X @ R
        # Use coalesced R directly from buffer
        if self.device.type == 'mps':
            # MPS does not support torch.sparse.mm; fall back to dense matmul
            scores_sp = torch.mm(user_input, self.R.to_dense().t())
        else:
            scores_sp = torch.sparse.mm(user_input, self.R.t())
        
        # Combine
        scores = scores_lr + scores_sp
        
        if item_ids is not None:
            if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids, device=self.device)
            batch_indices = torch.arange(len(user_ids), device=self.device)
            return scores[batch_indices, item_ids]
        
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        """Predict scores for (user, item) pairs"""
        return self.forward(user_ids, item_ids)

    def calc_loss(self, batch_data):
        """No training loop needed - closed-form solution"""
        return torch.tensor(0.0, device=self.device), None

    def get_final_item_embeddings(self):
        """Return low-rank component L (or B approximation)"""
        if self.L is None:
            raise RuntimeError("[ELSA] Model not fitted.")
        return self.L

    def get_train_matrix(self, data_loader):
        """Return training matrix"""
        if self.train_matrix_csr is None:
            self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        return self.train_matrix_csr
