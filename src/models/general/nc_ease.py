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
        print(f"[NC-EASE] Initialized with λ={self.reg_lambda}")

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
        
        # B = I - λP
        B = np.eye(self.n_items, dtype=np.float32) - self.reg_lambda * P
        del P
        
        self.weight_matrix.copy_(torch.from_numpy(B).float())
        del B
        
        self._log("Fitted.")
