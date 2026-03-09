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

    def fit(self, data_loader):
        self._log(f"Fitting (λ={self.reg_lambda})...")
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        
        X = sp.csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix_csr = X
        
        # 1. Solve (X^T X + λI)^-1 via GPU Cholesky/Eigen
        # Now returns a PyTorch tensor directly on the target device
        from src.utils.gpu_accel import gpu_gram_solve
        P = gpu_gram_solve(X, self.reg_lambda, device=self.device, return_tensor=True)
        
        # 2. Post-process B = -P / diag(P) on GPU
        # shape P: (M, M)
        diag = torch.diagonal(P).clone() # shape (M,)
        
        # B = -P / diag_j
        # Broadcase diag to (1, M) for division
        B = -P / diag.view(1, -1)
        del P
        
        # Set diagonal to ZERO
        B.fill_diagonal_(0)
        
        # 3. Sparsification if requested (still on GPU)
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

    def forward(self, user_ids, item_ids=None):
        if self.train_matrix_csr is None:
             raise RuntimeError("EASE model has not been fitted yet. Call fit() first.")

        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = user_ids
            
        user_input_sparse = self.train_matrix_csr[u_ids_np]
        user_input = torch.from_numpy(user_input_sparse.toarray()).float().to(self.device)
        
        if self.is_sparse:
            # Sparse-Dense Multiplication: (W @ X^T)^T
            # MPS does not support torch.sparse.mm, so fall back to dense matmul
            if self.device.type == 'mps':
                scores = user_input @ self.weight_matrix.to_dense()
            else:
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
