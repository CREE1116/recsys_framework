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
        
        # B matrix를 저장할 버퍼 등록 (학습되지 않는 파라미터)
        self.register_buffer('weight_matrix', torch.zeros(self.n_items, self.n_items))
        
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
        
        # GPU-accelerated Cholesky solve
        from src.utils.gpu_accel import gpu_gram_solve
        P = gpu_gram_solve(X, self.reg_lambda)
        
        # B = -P / diag(P), with zero diagonal
        diag = np.diag(P).copy()
        B = -P / diag[None, :]
        del P
        np.fill_diagonal(B, 0)
        
        self.weight_matrix.copy_(torch.from_numpy(B).float())
        del B
        
        self._log(f"Fitted. B: {self.n_items}x{self.n_items}")

    def forward(self, user_ids, item_ids=None):
        """
        user_ids: (batch_size) - List of user IDs to predict for
        """
        if self.train_matrix_csr is None:
             raise RuntimeError("EASE model has not been fitted yet. Call fit() first.")

        # 1. Get user Interaction Vectors from Train Matrix (using sparse slicing)
        # user_ids is likely a Tensor, convert to numpy for slicing
        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = user_ids
            
        # Efficiently slice CSR matrix for the batch of users
        # shape: (batch_size, n_items)
        user_input_sparse = self.train_matrix_csr[u_ids_np]
        
        # Convert to Dense Tensor for multiplication
        # shape: (batch_size, n_items)
        user_input = torch.from_numpy(user_input_sparse.toarray()).float().to(self.device)
        
        # 2. Compute Scores: S = X @ B
        # (batch_size, n_items) @ (n_items, n_items) -> (batch_size, n_items)
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
