import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from .ease import EASE

class NormEASE(EASE):
    """
    NormEASE: EASE with LIRA-style symmetric normalization.
    B_norm = D^-0.5 * B * D^-0.5 where D = diag(sum_j |B_ij|)
    """
    def __init__(self, config, data_loader):
        super(NormEASE, self).__init__(config, data_loader)
        self.normalize = config['model'].get('normalize', True)

    def fit(self, data_loader):
        self._log(f"Fitting NormEASE on {self.device} (λ={self.reg_lambda})...")
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        
        X = sp.csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix_csr = X
        
        # 1. Solve on GPU
        from src.utils.gpu_accel import gpu_gram_solve
        P = gpu_gram_solve(X, self.reg_lambda, device=self.device, return_tensor=True)
        
        # 2. B = -P / diag(P) with zero diagonal
        diag = torch.diagonal(P).clone()
        B = -P / diag.view(1, -1).clamp(min=1e-12)
        del P
        B.fill_diagonal_(0)
        
        # 3. Symmetric Normalization on GPU
        if self.normalize:
            # d = sum_j |B_ij|
            d = torch.sum(torch.abs(B), dim=1) # (M,)
            d_inv_sqrt = torch.pow(d.clamp(min=1e-12), -0.5)
            
            # B = d_inv_sqrt * B * d_inv_sqrt
            B = d_inv_sqrt.view(-1, 1) * B * d_inv_sqrt.view(1, -1)
            self._log("Applied Symmetric Normalization on GPU.")
            
        self.weight_matrix = B
        self._log("Fitted.")
