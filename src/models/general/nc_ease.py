import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from .ease import EASE

class NCEASE(EASE):
    """
    NC-EASE: No-Constraint EASE.
    Essentially Ridge Regression / Wiener Filter without the diag(B)=0 constraint.
    B = (X^T X + λI)^-1 (X^T X) = I - λ(X^T X + λI)^-1
    """
    def __init__(self, config, data_loader):
        super(NCEASE, self).__init__(config, data_loader)
        self._log(f"Initialized (λ={self.reg_lambda})")

    def fit(self, data_loader):
        self._log(f"Fitting NCEASE on {self.device} (λ={self.reg_lambda})...")
        
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        
        X = sp.csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix_csr = X
        
        # 1. Solve (X^T X + λI)^-1 via GPU
        from src.utils.gpu_accel import gpu_gram_solve
        P = gpu_gram_solve(X, self.reg_lambda, device=self.device, return_tensor=True)
        
        # 2. B = I - λP on GPU
        # B = - λP + I
        B = -self.reg_lambda * P
        B.diagonal().add_(1.0)
        del P
        
        # 3. Store
        self.weight_matrix = B
        self._log("Fitted.")
