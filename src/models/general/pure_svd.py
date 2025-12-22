import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from ..base_model import BaseModel

class PureSVD(BaseModel):
    """
    PureSVD: Singular Value Decomposition for Collaborative Filtering.
    A true 'One-Shot' Matrix Factorization model using analytic decomposition (SVD).
    
    Prediction score for user u, item i:
    Score = U_u \cdot \Sigma \cdot V_i^T
    
    This is effectively computing the reconstructed matrix R_hat = U * Sigma * Vt.
    """
    def __init__(self, config, data_loader):
        super(PureSVD, self).__init__(config, data_loader)
        
        self.embedding_dim = self.config['model']['embedding_dim']
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        
        self.user_factors = None # U
        self.item_factors = None # Vt (transposed in typical storage, but svds returns Vt usually)
        self.sigma = None        # Sigma (diagonal)
        
        print(f"PureSVD model initialized with embedding_dim={self.embedding_dim}.")

    def fit(self, data_loader):
        """
        Perform SVD on the user-item interaction matrix.
        """
        print("Building interaction matrix for PureSVD...")
        # Construct CSR matrix (Users x Items)
        rows = data_loader.train_df['user_id'].values
        cols = data_loader.train_df['item_id'].values
        data = np.ones(len(rows))
        
        user_item_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=float
        )
        
        print(f"Performing SVD (scipy.sparse.linalg.svds) with k={self.embedding_dim}...")
        # svds returns: u, s, vt
        # u: (n_users, k)
        # s: (k,)
        # vt: (k, n_items)
        u, s, vt = svds(user_item_matrix, k=self.embedding_dim)
        
        # Sort singular values in descending order (svds returns ascending usually)
        # But for reconstruction, order doesn't matter as long as indices match.
        # We'll just store them.
        
        self.user_factors = torch.FloatTensor(u.copy()).to(self.device)
        self.sigma = torch.FloatTensor(np.diag(s).copy()).to(self.device)
        self.item_factors = torch.FloatTensor(vt.T.copy()).to(self.device) # Store as (n_items, k) for easier lookup
        
        # Pre-compute U * Sigma for efficiency?
        # Score = (U * Sigma) @ Vt
        # Let's verify dimensions:
        # (n_users, k) * (k, k) -> (n_users, k)
        # (n_users, k) @ (n_items, k).T -> (n_users, n_items)
        
        self.user_factors_scaled = torch.matmul(self.user_factors, self.sigma)
        
        print("PureSVD model fitted successfully.")

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
