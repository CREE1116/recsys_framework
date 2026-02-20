import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel

class SMFLIRA(BaseModel):
    """
    Spectral Momentum Filter LIRA (SMF-LIRA)
    
    Breakthrough: Separates 'Latent Interest Flow' and 'Item-level Flow' 
    using spectral power without SVD decomposition.
    """
    def __init__(self, config, data_loader):
        super(SMFLIRA, self).__init__(config, data_loader)
        self.n_items = data_loader.n_items
        self.n_users = data_loader.n_users
        
        model_cfg = config.get('model', {})
        self.reg_lambda = model_cfg.get('reg_lambda', 500.0)
        self.beta = model_cfg.get('beta', 0.5)   # 관심사 모멘텀 가중치
        self.gamma = model_cfg.get('gamma', 0.1) # 아이템 디테일 가중치
        self.eps = float(model_cfg.get('eps', 1e-8))

        # 1. 데이터 빌드
        print("[SMF-LIRA] Preparing sparse matrices from train_df...")
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        X = torch.from_numpy(self.train_matrix_csr.toarray()).float().to(self.device)
        
        sequences = self._extract_sequences(data_loader)
        T_counts = self._build_transition_matrix(sequences)
        n_items = self.n_items

        print("[SMF-LIRA] Computing Spectral Terrain (S) ...")
        # 2. Symmetric Terrain (Wiener Filter 기반 지형 구축)
        G = torch.mm(X.t(), X)
        I = torch.eye(n_items, device=self.device)
        # S = G(G + lambda*I)^-1
        S = torch.mm(G, torch.linalg.inv(G + self.reg_lambda * I))

        print("[SMF-LIRA] Extracting Multi-scale Momentum ...")
        # 3. Asymmetric Raw Flow
        A = T_counts - T_counts.t()

        # 4. Spectral Filtering (SVD 없는 관심사 분리)
        # S_low: S를 거듭제곱하여 고유값이 큰 성분(거시적 관심사)만 증폭
        S_low = torch.mm(S, S) 
        
        # Interest-level Flow (관심사 대륙 간의 거시적 해류)
        # S_low 사이에서 흐르는 A만 추출
        Phi_latent = torch.mm(torch.mm(S_low, A), S_low)
        
        # Item-level Flow (개별 아이템 간의 미세한 골목길)
        # 전체 흐름에서 관심사 흐름을 제외한 나머지 디테일
        Phi_item = A - Phi_latent

        # 5. Momentum Fusion & Dual Normalization
        # K_raw = S + (관심사 모멘텀) + (아이템 모멘텀)
        K_raw = S + self.beta * Phi_latent + self.gamma * Phi_item
        
        # Dual-Norm
        d_s = torch.pow(K_raw.abs().sum(dim=1) + self.eps, -0.5)
        d_a = torch.pow(K_raw.abs().sum(dim=0) + self.eps, -0.5)
        self.K_final = d_s.view(-1, 1) * K_raw * d_a.view(1, -1)
        
        self.register_buffer('K_buffer', self.K_final)

    def forward(self, users, mask_observed=True):
        device = self.device
        batch_users_np = users.cpu().numpy()
        X_u = torch.from_numpy(self.train_matrix_csr[batch_users_np].toarray()).float().to(device)
        
        scores = torch.mm(X_u, self.K_buffer)
        
        if mask_observed:
            rows, cols = X_u.nonzero(as_tuple=True)
            scores[rows, cols] = -1e9
        return scores

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

    def _build_transition_matrix(self, sequences):
        n = self.n_items
        T = torch.zeros(n, n, device=self.device)
        for seq in sequences:
            if len(seq) < 2: continue
            for i in range(len(seq) - 1):
                T[seq[i], seq[i+1]] += 1.0
        return T

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), {}

    def predict_for_pairs(self, user_ids, item_ids):
        device = self.device
        batch_users_np = user_ids.cpu().numpy()
        X_u = torch.from_numpy(self.train_matrix_csr[batch_users_np].toarray()).float().to(device)
        
        # Vectorized scoring for specific user-item pairs
        relevant_K = self.K_buffer[:, item_ids] # [n_items, batch_size]
        scores = (X_u * relevant_K.t()).sum(dim=1)
        return scores

    def get_final_item_embeddings(self):
        return self.K_buffer
