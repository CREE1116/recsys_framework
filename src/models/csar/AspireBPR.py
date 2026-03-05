import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from ..general.mf import MF
from src.utils.gpu_accel import SVDCacheManager
from src.models.csar.ASPIRELayer import estimate_alignment_slope

class AspireBPR(MF):
    """
    AspireBPR: BPR-MF with Adaptive Spectral Regularization
    ASPIRE의 필터링 이론을 MF의 임베딩 규제항(Spectral Penalty)으로 이식한 모델입니다.
    """
    def __init__(self, config, data_loader):
        super(AspireBPR, self).__init__(config, data_loader)
        
        # 1. Hyperparameters
        self.embedding_dim = config['model']['embedding_dim']
        self.spectral_lambda = config['model'].get('spectral_lambda', 0.1)
        self.beta_config = config['model'].get('beta', 'auto_compromise')
        self.k_svd = self.embedding_dim # d 차원 임베딩은 d 차원 스펙트럼 성분과 대응
        
        # 2. Build Spectral Components
        self._build_spectral_reg()

    def _build_spectral_reg(self):
        self._log(f"Building Spectral Regularization components (k={self.k_svd})...")
        
        dataset_name = self.config.get('dataset_name')
        item_popularity = getattr(self.data_loader, 'item_popularity', None)
        
        # 1. Load SVD Components (Prefer cache by name, avoid global matrix reconstruction)
        svd_mgr = SVDCacheManager()
        try:
            # X_sparse=None: dataset_name 기반 캐시 강제 로드 시도
            _, s, v, _ = svd_mgr.get_svd(X_sparse=None, k=self.k_svd, dataset_name=dataset_name)
        except RuntimeError as e:
            self._log(f"SVD Cache not found ({e}). Building sparse matrix as fallback...")
            # Fallback: MF처럼 깔끔하게 처리하되, SVD가 꼭 필요할 때만 임시 생성
            from scipy.sparse import csr_matrix
            train_df = self.data_loader.train_df
            rows, cols = train_df['user_id'].values, train_df['item_id'].values
            vals = np.ones(len(train_df))
            X_tmp = csr_matrix((vals, (rows, cols)), shape=(self.n_users, self.n_items))
            _, s, v, _ = svd_mgr.get_svd(X_tmp, k=self.k_svd, dataset_name=dataset_name)
        
        # Move to device
        device = self.user_embedding.weight.device
        self.register_buffer('singular_values', s.to(device)) # (k,)
        self.register_buffer('singular_vectors', v.to(device)) # (I, k)
        
        # Auto Beta: β = max(0, 2a - 1) — MCAR 기준선(a=0.5)으로부터의 이탈량
        if isinstance(self.beta_config, str):
            a = estimate_alignment_slope(
                singular_values=self.singular_values, 
                item_popularity=item_popularity,
                dataset_name=dataset_name,
                alpha=self.spectral_lambda
            )
            self.alignment_slope = a
            self.beta = max(0.0, 2.0 * a - 1.0)
            self._log(f"SWLS a={a:.3f} -> β={self.beta:.3f}")
        else:
            self.beta = float(self.beta_config)
            self.alignment_slope = 0.5

        # Precompute Spectral Weights: sigma^{2(1-beta)}
        # This weight penalizes the projection onto popular directions (high sigma)
        # based on the ASPIRE shrinkage principle.
        exponent = 2.0 * (1.0 - self.beta)
        spectral_weights = torch.pow(torch.maximum(self.singular_values, torch.tensor(1e-9, device=device)), exponent)
        self.register_buffer('spectral_weights', spectral_weights)

    def get_spectral_reg_loss(self):
        """
        Spectral Regularization Loss:
        Items' embeddings translated to SVD space and penalized by spectral_weights.
        Loss = Sum_k [ w_k * ||V_k^T E||^2 ]
        """
        E = self.item_embedding.weight # (Items, D)
        V = self.singular_vectors # (Items, K)
        
        # Projection of embeddings onto singular vectors
        # E_spectral: (K, D)
        E_spectral = torch.mm(V.t(), E)
        
        # Weighted squared norm
        # spectral_weights: (K,)
        sq_norms = torch.sum(E_spectral ** 2, dim=1) # (K,)
        term = self.spectral_weights * sq_norms
        
        return torch.sum(term)

    def calc_loss(self, batch_data):
        # 1. Main Loss (BPR)
        users = batch_data['user_id']
        pos_items = batch_data['pos_item_id']
        neg_items = batch_data['neg_item_id']

        u_emb = self.user_embedding(users)
        p_emb = self.item_embedding(pos_items)
        n_emb = self.item_embedding(neg_items)

        pos_scores = torch.sum(u_emb * p_emb, dim=-1)
        neg_scores = torch.sum(u_emb * n_emb, dim=-1)
        
        loss_main = self.loss_fn(pos_scores, neg_scores)

        # 2. Spectral Regularization (Global on whole Item Embedding Matrix)
        # This replaces Item-side L2 regularization with a spectrum-aware version.
        loss_spectral = self.get_spectral_reg_loss() / self.n_items
        
        # 3. Standard L2 Regularization (User-side only)
        # We only apply standard L2 to users; items are governed by spectral loss.
        # Use get_l2_reg_loss for mathematical consistency (includes factor 2 and batch scaling)
        loss_l2_user = self.get_l2_reg_loss(u_emb)

        params_to_log = {
            'loss_main': loss_main.item(),
            'loss_spectral': loss_spectral.item(),
            'loss_l2_user': loss_l2_user.item(),
            'beta': self.beta,
            'a': self.alignment_slope
        }

        return (loss_main, self.spectral_lambda * loss_spectral, loss_l2_user), params_to_log

    def __str__(self):
        return f"AspireBPR(dim={self.embedding_dim}, spectral_lambda={self.spectral_lambda}, beta={self.beta:.3f})"
