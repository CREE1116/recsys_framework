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
        
        self.user_factors = None
        self.item_factors = None
        self.sigma = None
        
        self._log(f"Initialized (embedding_dim={self.embedding_dim})")

    def fit(self, data_loader):
        """Perform SVD on the user-item interaction matrix via SVDCacheManager."""
        self._log("Building interaction matrix...")
        rows = data_loader.train_df['user_id'].values
        cols = data_loader.train_df['item_id'].values
        data = np.ones(len(rows))
        
        user_item_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=float
        )
        
        # Check rank limits
        min_dim = min(user_item_matrix.shape)
        if self.embedding_dim >= min_dim:
            self._log(f"Warning: embedding_dim={self.embedding_dim} >= min_dim={min_dim}. Capping to {min_dim - 1}.")
            self.embedding_dim = min_dim - 1

        # SVDCacheManager: 캐싱 + MPS 가속 자동 지원
        dataset_name = self.config.get('dataset_name', 'unknown')
        u, s, v, energy = self.svd_manager.get_svd(
            user_item_matrix, k=self.embedding_dim, dataset_name=dataset_name
        )
        
        self.user_factors = u.to(self.device).float()    # (n_users, k)
        self.sigma = s.to(self.device).float()            # (k,)
        self.item_factors = v.to(self.device).float()     # (n_items, k)
        
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
        # scores = batch_users_scaled @ item_factors.T
        scores = torch.matmul(batch_users_scaled, self.item_factors.transpose(0, 1))
        
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
        For SVD, this is V^T * Sigma? Or just V^T?
        Usually PureSVD represents items as V^T (or V * Sigma).
        We'll returns V^T (stored as item_factors).
        """
        if self.item_factors is None:
            return torch.zeros((self.n_items, self.embedding_dim), device=self.device)
        return self.item_factors

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
