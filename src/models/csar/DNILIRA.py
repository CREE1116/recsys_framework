import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from src.models.base_model import BaseModel

class DNILIRA(BaseModel):
    """
    Dual-Normalized Interaction LIRA (DNILIRA)
    
    Refined normalization for both symmetric and asymmetric components:
    1. S_norm: Symmetric normalization of LIRA kernel using its absolute row sums.
    2. A_norm: Asymmetric flow normalization using In-degree + Out-degree.
    """
    def __init__(self, config, data_loader):
        super(DNILIRA, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_cfg = config.get('model', {})
        self.reg_lambda    = model_cfg.get('reg_lambda', 500.0)
        self.beta          = model_cfg.get('beta', 0.2)
        self.eps           = float(model_cfg.get('eps', 1e-8))

        # 1. Build Data structures
        train_matrix_csr = self._build_sparse_matrix(data_loader)
        sequences = self._extract_sequences(data_loader)
        device = self.device
        
        # 2. Symmetric Part (Full Matrix Wiener Filter + Symmetric Norm)
        print("[DNILIRA] Building Full Symmetric Skeleton (No SVD) ...")
        X = torch.from_numpy(train_matrix_csr.toarray()).float().to(device)
        G = torch.mm(X.t(), X) # [n_items, n_items]
        
        I = torch.eye(self.n_items, device=device)
        # Full Wiener Filter: S = G @ inv(G + lambda*I)
        S = torch.mm(G, torch.linalg.inv(G + self.reg_lambda * I))
        
        # Refined Normalization for S: d_s = S.abs().sum(dim=1)
        d_s_inv_sqrt = torch.pow(S.abs().sum(dim=1) + self.eps, -0.5)
        S_norm = d_s_inv_sqrt.view(-1, 1) * S * d_s_inv_sqrt.view(1, -1)
        
        # 3. Asymmetric Part (Transition Flow + Refined Norm)
        print("[DNILIRA] Building Asymmetric Flow ...")
        n = self.n_items
        T_counts = torch.zeros(n, n, device=device)
        for seq in sequences:
            for t in range(len(seq) - 1):
                i, j = seq[t], seq[t + 1]
                T_counts[i, j] += 1.0
        
        flow = T_counts - T_counts.t()
        # Refined Normalization for A: d_a = In-degree + Out-degree
        # T_counts.sum(dim=1) is out-degree, T_counts.sum(dim=0) is in-degree
        d_a_inv_sqrt = torch.pow(T_counts.sum(dim=1) + T_counts.sum(dim=0) + self.eps, -0.5)
        A_norm = d_a_inv_sqrt.view(-1, 1) * flow * d_a_inv_sqrt.view(1, -1)
        
        # 4. K_unified = S_norm + beta * A_norm
        # beta is a mixing ratio
        K_unified = S_norm + self.beta * A_norm
        self.register_buffer('K_unified', K_unified)
        
        self.train_matrix_csr = train_matrix_csr

    def _build_sparse_matrix(self, data_loader):
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        return csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items))

    def _extract_sequences(self, data_loader):
        train_df = data_loader.train_df.copy()
        if 'timestamp' not in train_df.columns:
            train_df['timestamp'] = train_df.index
        train_df = train_df.sort_values(by=['user_id', 'timestamp'])
        return train_df.groupby('user_id')['item_id'].apply(list).tolist()

    def forward(self, users, mask_observed=True):
        device = self.device
        batch_users = users.cpu().numpy()
        X_u = torch.from_numpy(self.train_matrix_csr[batch_users].toarray()).float().to(device)
        
        scores = torch.mm(X_u, self.K_unified)
        
        if mask_observed:
            rows, cols = X_u.nonzero(as_tuple=True)
            scores[rows, cols] = -1e9
            
        return scores

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), {}

    def predict_for_pairs(self, user_ids, item_ids):
        device = self.device
        X_u = torch.from_numpy(self.train_matrix_csr[user_ids.cpu().numpy()].toarray()).float().to(device)
        scores = []
        for b in range(len(user_ids)):
            scores.append(torch.dot(X_u[b], self.K_unified[:, item_ids[b]]))
        return torch.stack(scores)

    def get_final_item_embeddings(self):
        return self.K_unified
