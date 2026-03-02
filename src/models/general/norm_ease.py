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
        print("Fitting NormEASE model...")
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        
        X = sp.csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix_csr = X
        
        # GPU-accelerated Cholesky solve
        from src.utils.gpu_accel import gpu_gram_solve
        P = gpu_gram_solve(X, self.reg_lambda)
        
        diag = np.diag(P).copy()
        B = -P / diag[None, :]
        del P
        np.fill_diagonal(B, 0)
        
        # Symmetric Normalization
        if self.normalize:
            d = np.abs(B).sum(axis=1)
            d[d == 0] = 1.0
            d_inv_sqrt = np.power(d, -0.5)
            B = d_inv_sqrt[:, None] * B * d_inv_sqrt[None, :]
            print("[NormEASE] Applied Symmetric Normalization.")
            
        self.weight_matrix.copy_(torch.from_numpy(B).float())
        del B
        
        print("NormEASE model fitted.")
