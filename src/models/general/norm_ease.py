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
        # 1. Standard EASE fitting logic
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        
        X = sp.csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix_csr = X
        
        G = X.transpose().dot(X).toarray()
        diag_indices = np.diag_indices(self.n_items)
        G[diag_indices] += self.reg_lambda
        
        P = np.linalg.inv(G)
        
        diag = np.diag(P)
        B = -P / diag[None, :]
        np.fill_diagonal(B, 0)
        
        # 2. Symmetric Normalization (LIRA-style)
        if self.normalize:
            # Row-wise absolute sum for D
            d = np.abs(B).sum(axis=1)
            d[d == 0] = 1.0
            d_inv_sqrt = np.power(d, -0.5)
            
            # B_norm = D^-0.5 * B * D^-0.5
            B = d_inv_sqrt[:, None] * B * d_inv_sqrt[None, :]
            print("[NormEASE] Applied Symmetric Normalization.")
            
        # 3. Store as Tensor
        self.weight_matrix.copy_(torch.from_numpy(B).float())
        
        print("NormEASE model fitted.")
