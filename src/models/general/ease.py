import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel

class EASE(BaseModel):
    """
    EASE (Embarrassingly Shallow Autoencoders for Sparse Data)
    - Closed-form solution: B = (X^T X + lambda I)^-1 X^T X
    - No gradient descent training needed.
    """
    def __init__(self, config, data_loader):
        super(EASE, self).__init__(config, data_loader)
        self.reg_lambda = config['model']['reg_lambda']
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        # [MEMORY FIX] Do NOT allocate large dense zero matrix on CPU at init.
        # This prevents the "DefaultCPUAllocator: not enough memory" error for large datasets.
        self.register_buffer('weight_matrix', torch.empty(0, 0))
        self.is_sparse = False

        # 학습에 사용된 희소 행렬을 저장 (Inference 시 사용)
        self.train_matrix_csr = None
        # CUDA sparse version of train_matrix — populated in fit() for GPU inference
        self._X_cuda = None


    def fit(self, data_loader):
        self._log(f"Fitting (λ={self.reg_lambda}) with advanced GPU solver...")
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        
        X = sp.csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix_csr = X
        
        # 1. Solve (X^T X + λI)^-1 via improved GPU solver
        from src.utils.gpu_accel import gpu_gram_solve
        dataset_name = self.config.get('dataset_name', 'unknown')
        # This now uses torch.linalg.inv -> np.linalg.inv native fallback internally
        P = gpu_gram_solve(X, self.reg_lambda, device=self.device, dataset_name=dataset_name, return_tensor=True)
        
        # 2. Post-process B = I - P / diag(P) (Exact EASE Constraint)
        # Numerical stability epsilon
        eps = 1e-12
        diag_P = torch.diagonal(P)
        I = torch.eye(self.n_items, device=self.device, dtype=torch.float32)
        B = I - (P / torch.clamp(diag_P.unsqueeze(0), min=eps))
        del P, I
        
        # 6. Sparsification if requested (still on GPU)
        threshold = self.config['model'].get('threshold', 0.0)
        if threshold > 0:
            mask = torch.abs(B) >= threshold
            B = B * mask 
            self.is_sparse = True
            self.weight_matrix = B.to_sparse().coalesce()
            del mask, B
        else:
            self.weight_matrix = B
        
        self._log(f"Fitted on {self.device}. B: {self.n_items}x{self.n_items} (Sparse={self.is_sparse})")

        # [CUDA OPT] Pre-upload train matrix as CUDA sparse CSR tensor.
        # Avoids repeated .toarray() → numpy → GPU transfer on every forward() call.
        self._X_cuda = None
        if self.device.type == 'cuda':
            try:
                from src.utils.gpu_accel import SVDCacheManager
                self._X_cuda = SVDCacheManager._to_cuda_sparse(X)
                self._log("Train matrix uploaded to CUDA sparse (forward acceleration).")
            except Exception as e:
                self._log(f"CUDA sparse upload failed ({e}), using CPU fallback.")

    def forward(self, user_ids, item_ids=None):
        if self.train_matrix_csr is None:
             raise RuntimeError("EASE model has not been fitted yet. Call fit() first.")

        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = np.asarray(user_ids)

        # --- User vector retrieval ---
        if self._X_cuda is not None:
            # [CUDA Sparse Path] Row-selection via sparse E @ X:
            # E is (B x N_users) selection matrix; E @ X_cuda → (B, N_items) dense, no .toarray()
            B = len(u_ids_np)
            row_idx = torch.arange(B, device=self.device)
            col_idx = torch.from_numpy(u_ids_np.astype(np.int64)).to(self.device)
            vals    = torch.ones(B, device=self.device)
            E = torch.sparse_coo_tensor(
                torch.stack([row_idx, col_idx]), vals,
                (B, self.n_users), device=self.device
            ).to_sparse_csr()
            user_input = torch.mm(E, self._X_cuda)  # (B, N_items) dense
        else:
            # [CPU Fallback] .toarray() + async non-blocking GPU transfer
            user_np = self.train_matrix_csr[u_ids_np].toarray()
            t = torch.from_numpy(user_np).float()
            if self.device.type == 'cuda':
                user_input = t.pin_memory().to(self.device, non_blocking=True)
            else:
                user_input = t.to(self.device)

        if self.is_sparse:
            # weight_matrix: Sparse(I, I), torch.sparse.mm only supports Sparse @ Dense
            scores = torch.sparse.mm(self.weight_matrix, user_input.t()).t()
        else:
            scores = user_input @ self.weight_matrix

        return scores

    def calc_loss(self, batch_data):
        # EASE optimizes a closed-form solution, no gradient descent training.
        return (torch.tensor(0.0, device=self.device),), None

    def predict_for_pairs(self, user_ids, item_ids):
        # Not typically efficient for EASE, but implemented for compatibility
        scores = self.forward(user_ids) # [B, N_items]
        # Gather specific item scores
        batch_indices = torch.arange(len(user_ids), device=user_ids.device)
        return scores[batch_indices, item_ids]

    def get_final_item_embeddings(self):
        # EASE doesn't have "embeddings", it has Item-Item weights.
        return self.weight_matrix
