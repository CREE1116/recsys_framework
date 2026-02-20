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
        self.L = None  # (M × d)
        self.S = None  # (d × M)
        self.R = None  # (M × M) sparse
        
        self.train_matrix_csr = None
        
        print(f"[ELSA] Config: rank={self.rank}, λ={self.reg_lambda}, sparse_k={self.sparse_k}")

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
        print(f"\n{'='*60}")
        print(f"[ELSA] Starting training...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 1. Build interaction matrix
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        X = self.train_matrix_csr
        
        print(f"[ELSA] Matrix: {X.shape}, nnz={X.nnz:,}")
        
        # 2. Compute EASE solution
        print(f"[ELSA] Computing EASE solution...")
        
        # Option A: For small datasets (M < 10K)
        if self.n_items < 10000:
            G = X.T.dot(X).toarray()  # (M × M)
            G_tensor = torch.from_numpy(G).float().to(self.device)
            
            # B = (G + λI)^{-1} @ G
            # MPS workaround: Inverse on CPU
            G_cpu = G_tensor.cpu()
            I_cpu = torch.eye(self.n_items)
            P_cpu = torch.inverse(G_cpu + self.reg_lambda * I_cpu)
            P = P_cpu.to(self.device)
            
            B = P @ G_tensor
            B.fill_diagonal_(0)  # EASE constraint
            
            B_np = B.cpu().numpy()
            
        else:
            # Option B: For large datasets - use iterative solver
            raise NotImplementedError(
                "[ELSA] Large-scale version not implemented. "
                "Use column-wise optimization or approximate methods."
            )
        
        # 3. Low-rank decomposition via SVD
        print(f"[ELSA] Computing SVD (rank={self.rank})...")
        
        U, Σ, Vt = np.linalg.svd(B_np, full_matrices=False)
        
        # Truncate to rank d
        Σ_d = Σ[:self.rank]
        U_d = U[:, :self.rank]
        Vt_d = Vt[:self.rank, :]
        
        # L @ S decomposition
        L = U_d @ np.diag(np.sqrt(Σ_d))  # (M × d)
        S = np.diag(np.sqrt(Σ_d)) @ Vt_d  # (d × M)
        
        # 4. Sparse residual
        print(f"[ELSA] Computing sparse residual...")
        
        B_lowrank = L @ S
        R = B_np - B_lowrank
        
        # Sparsify: keep top-k per row
        R_sparse = np.zeros_like(R)
        for i in range(self.n_items):
            topk_idx = np.argpartition(np.abs(R[i]), -self.sparse_k)[-self.sparse_k:]
            R_sparse[i, topk_idx] = R[i, topk_idx]
        
        # 5. Store components
        self.L = torch.from_numpy(L).float()
        self.S = torch.from_numpy(S).float()
        
        # Convert R to sparse tensor
        R_coo = sp.coo_matrix(R_sparse)
        indices = torch.LongTensor([R_coo.row, R_coo.col])
        values = torch.FloatTensor(R_coo.data)
        self.R = torch.sparse_coo_tensor(
            indices, values, 
            (self.n_items, self.n_items)
        )
        
        # 6. Statistics
        elapsed = time.time() - start_time
        
        # Reconstruction error
        B_approx = B_lowrank + R_sparse
        recon_error = np.linalg.norm(B_np - B_approx) / np.linalg.norm(B_np)
        
        print(f"\n{'='*60}")
        print(f"[ELSA] Training complete!")
        print(f"  - Time: {elapsed:.2f}s")
        print(f"  - Low-rank: {L.shape}")
        print(f"  - Sparse nnz: {R_coo.nnz:,}")
        print(f"  - Reconstruction error: {recon_error:.4f}")
        print(f"  - Memory (approx): {(L.size + S.size + R_coo.nnz)*4/1e6:.1f} MB")
        print(f"{'='*60}\n")

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
        
        # Move components to device
        L = self.L.to(self.device)
        S = self.S.to(self.device)
        R = self.R.to(self.device)
        
        # Low-rank component: (X @ L) @ S
        latent = user_input @ L  # (batch × d)
        scores_lr = latent @ S   # (batch × M)
        
        # Sparse component: X @ R
        # MPS workaround: Sparse MM on CPU
        try:
            scores_sp = torch.sparse.mm(user_input, R.t().coalesce())
        except (RuntimeError, NotImplementedError):
            # Fallback to CPU if MPS fails
            user_input_cpu = user_input.cpu()
            R_cpu = R.cpu()
            scores_sp_cpu = torch.sparse.mm(user_input_cpu, R_cpu.t().coalesce())
            scores_sp = scores_sp_cpu.to(self.device)
        
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
