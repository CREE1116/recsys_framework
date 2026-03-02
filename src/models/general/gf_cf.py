import torch
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
from ...utils.gpu_accel import SVDCacheManager


class GF_CF(BaseModel):
    """
    GF-CF: Graph Filter based Collaborative Filtering (CIKM 2021)
    "How Powerful is Graph Convolution for Recommendation?"
    
    Refined Implementation:
    1. Build Generalized Normalized Bipartite Graph Matrix.
    2. Perform SVD via SVDCacheManager (supports MPS & Caching).
    3. Apply Graph Spectral Filter with Alpha scaling.
    """
    def __init__(self, config, data_loader):
        super(GF_CF, self).__init__(config, data_loader)
        
        # Hyperparameters
        k = self.config['model'].get('k', 256)
        self.k = k[0] if isinstance(k, list) else k
        self.filter_type = self.config['model'].get('filter_type', 'ideal')
        
        # alpha: scaling singular values (Sigma^alpha)
        # alpha=1.0 is standard linear filtering. alpha=0.0 is ideal low-pass.
        self.alpha = self.config['model'].get('alpha', 1.0)
        
        # norm_weight: control normalization strength (D^-w)
        # w=0.5 is standard symmetric normalization. w=0.0 is PureSVD (no norm).
        self.norm_weight = self.config['model'].get('norm_weight', 0.5)
        
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        
        # SVD Manager initialization
        self.svd_manager = SVDCacheManager(device=self.device.type)
        
        self.user_factors = None
        self.item_factors = None
        self.sigma = None
        
        print(f"GF-CF initialized: k={self.k}, filter={self.filter_type}, alpha={self.alpha}, norm_weight={self.norm_weight}")

    def _normalize_matrix(self, R, weight):
        """Generalized Symmetric Normalization: D_u^-w * R * D_i^-w"""
        if weight == 0:
            return R
            
        user_sums = np.array(R.sum(axis=1)).flatten()
        item_sums = np.array(R.sum(axis=0)).flatten()
        
        user_inv_w = np.power(user_sums, -weight, where=user_sums > 0)
        item_inv_w = np.power(item_sums, -weight, where=item_sums > 0)
        
        # D_u^-w @ R
        R_norm = sp.diags(user_inv_w) @ R
        # R_norm @ D_i^-w
        R_norm = R_norm @ sp.diags(item_inv_w)
        
        return R_norm

    def fit(self, data_loader):
        """SVD on normalized interaction matrix + apply graph filter."""
        print(f"Building and Normalizing interaction matrix (w={self.norm_weight}) for GF-CF...")
        rows = data_loader.train_df['user_id'].values
        cols = data_loader.train_df['item_id'].values
        data = np.ones(len(rows))
        
        R = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=float
        )
        
        # Apply Generalized Normalization
        R_norm = self._normalize_matrix(R, self.norm_weight)
        
        # Perform SVD using SVDManager
        # NOTE: dataset_name must include norm_weight to avoid cache collision if normalization changes
        dataset_name = self.config.get('dataset_name', 'unknown_gfcf')
        cache_suffix = f"_w{self.norm_weight:.2f}"
        
        u, s, v, _ = self.svd_manager.get_svd(R_norm, k=self.k, dataset_name=f"{dataset_name}{cache_suffix}")
        
        # Move factors to device
        self.user_factors = u.to(self.device).float()
        self.item_factors = v.to(self.device).float() # (n_items, k)
        self.sigma = s.to(self.device).float()
        
        # Apply alpha scaling to singular values
        # Score = U * Sig^alpha * V^T
        # For GF-CF (CIKM'21), the ideal linear filter is alpha=1.0 on a normalized matrix.
        scaled_sigma = torch.pow(self.sigma, self.alpha)
        
        # Final User Factors = U * Sigma^alpha
        self.user_factors_scaled = self.user_factors * scaled_sigma.unsqueeze(0)
        
        print(f"GF-CF fitted. alpha={self.alpha}, user_factors: {self.user_factors_scaled.shape}, item_factors: {self.item_factors.shape}")

    def forward(self, users):
        """Predict scores for given users over all items."""
        # Score = User_scaled @ Item_factors^T
        batch_users_scaled = self.user_factors_scaled[users]  # (batch, k)
        return torch.matmul(batch_users_scaled, self.item_factors.T)  # (batch, n_items)

    def predict_for_pairs(self, user_ids, item_ids):
        """Predict for specific user-item pairs."""
        u_embeds = self.user_factors_scaled[user_ids]
        i_embeds = self.item_factors[item_ids]
        return torch.sum(u_embeds * i_embeds, dim=1)

    def get_final_item_embeddings(self):
        if self.item_factors is None:
            return torch.zeros((self.n_items, self.k), device=self.device)
        return self.item_factors

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
