import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp

from ..base_model import BaseModel
from ...utils.gpu_accel import SVDCacheManager

class PureSVD(BaseModel):
    """
    PureSVD: Singular Value Decomposition for Collaborative Filtering.
    A true 'One-Shot' Matrix Factorization model using analytic decomposition (SVD).
    Uses SVDCacheManager for caching and MPS acceleration.
    """
    def __init__(self, config, data_loader):
        super(PureSVD, self).__init__(config, data_loader)
        
        self.embedding_dim = self.config['model']['embedding_dim']
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        
        self.svd_manager = SVDCacheManager(device=self.device.type)
        
        self.register_buffer('user_factors', torch.empty(0))
        self.register_buffer('item_factors', torch.empty(0))
        self.register_buffer('sigma', torch.empty(0))
        self.register_buffer('user_factors_scaled', torch.empty(0))
        
        self._log(f"Initialized (embedding_dim={self.embedding_dim})")

        # Cache manager 등록
        self.register_cache_manager('svd', self.svd_manager)

    def fit(self, data_loader):
        """Perform SVD on the user-item interaction matrix via SVDCacheManager."""
        self._log("Building interaction matrix...")
        rows = data_loader.train_df['user_id'].values
        cols = data_loader.train_df['item_id'].values
        data = np.ones(len(rows))
        
        user_item_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        
        # Check rank limits safely without modifying config class-wide state unnecessarily
        min_dim = min(user_item_matrix.shape)
        k = min(self.embedding_dim, min_dim - 1)
        if self.embedding_dim >= min_dim:
            self._log(f"Warning: requested embedding_dim={self.embedding_dim} >= min_dim={min_dim}. Using k={k}.")

        # SVDCacheManager: 캐싱 + MPS 가속 자동 지원
        dataset_name = self.config.get('dataset_name', 'unknown')
        u, s, v, energy = self.svd_manager.get_svd(
            user_item_matrix, k=k, dataset_name=dataset_name
        )
        
        # 안전한 버퍼 덮어쓰기 (재등록 방지)
        self.user_factors = u.to(self.device).float()
        self.sigma = s.to(self.device).float()
        self.item_factors = v.to(self.device).float()
        
        # Pre-compute U * Sigma for efficiency
        self.user_factors_scaled = self.user_factors * self.sigma.unsqueeze(0)
        
        self._log(f"Fitted. Energy captured: {energy:.4f}")


    def forward(self, users):
        """
        Predict scores for given users over all items.
        """
        # users: tensor of user indices
        batch_users_scaled = self.user_factors_scaled[users] # (batch, k)
        
        # item_factors is (n_items, k)
        scores = batch_users_scaled @ self.item_factors.T
        
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        """
        Predict for specific user-item pairs.
        """
        u_embeds = self.user_factors_scaled[user_ids] # (batch, k)
        i_embeds = self.item_factors[item_ids]        # (batch, k)
        
        scores = torch.sum(u_embeds * i_embeds, dim=1)
        return scores

    def get_final_item_embeddings(self):
        """
        Return item embeddings for visualization.
        Usually PureSVD represents items as V^T (or V * Sigma).
        We'll return V^T (stored as item_factors).
        """
        return self.item_factors

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
