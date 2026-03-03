import torch
import torch.nn as nn
import numpy as np
import time
from scipy.sparse import csr_matrix

from ..base_model import BaseModel
from .LIRALayer import SpectralPowerLIRALayer

class SpectralPowerLIRA(BaseModel):
    """
    SpectralPowerLIRA: Applies power transformation to eigenvalues.
    Rather than element-wise sharpening, this shapes the energy spectrum.
    S_approx = V * diag(s^p / (s^p + lambda)) * V^T
    """
    def __init__(self, config, data_loader):
        super(SpectralPowerLIRA, self).__init__(config, data_loader)
        
        # Hyperparameters
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        self.k = config['model'].get('k', 200)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.power = config['model'].get('power', 1.0)
        
        self.visualize = config['model'].get('visualize', False)
        self.dataset_name = config.get('dataset_name', 'unknown')
        
        # LIRA Layer
        self.model = SpectralPowerLIRALayer(
            k=self.k, 
            reg_lambda=self.reg_lambda,
            power=self.power
        ).to(self.device)
        
        self._log(f"Initialized (k={self.k}, λ={self.reg_lambda}, p={self.power})")

    def _build_sparse_matrix(self, data_loader):
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        return csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items))

    def fit(self, data_loader):
        """Build SpectralPowerLIRA model"""
        self._log(f"\n{'='*60}")
        self._log(f"Training (k={self.k}, λ={self.reg_lambda}, Power={self.power})")
        self._log(f"{'='*60}")
        
        start_time = time.time()
        
        # 1. Build interaction matrix
        X_sparse = self._build_sparse_matrix(data_loader)
        
        # 2. Build Layer
        self.model.build(X_sparse, dataset_name=self.dataset_name)
        
        elapsed = time.time() - start_time
        self._log(f"\n{'='*60}")
        self._log(f"Training completed in {elapsed:.2f}s")
        self._log(f"{'='*60}\n")
        
        if self.visualize:
            self.model.visualize_matrices(X_sparse, save_dir=f"checkpoints/{self.dataset_name}/spectral_power_lira_k{self.k}")

    def get_train_matrix(self, data_loader):
        return self._build_sparse_matrix(data_loader)

    def forward(self, users, items=None):
        return self.predict_full(users, items)

    def predict_full(self, users, items=None):
        train_matrix = self.get_train_matrix(self.data_loader)
        user_history_dense = torch.from_numpy(train_matrix[users.cpu().numpy()].toarray()).float().to(self.device)
        
        # In SpectralPowerLIRA, the layer expects (X_batch, user_ids, mask)
        scores = self.model(user_history_dense, user_ids=users)
        
        if items is not None:
             return scores.gather(1, items)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        return self.predict_full(user_ids, item_ids)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device, requires_grad=True),), None

    def get_final_item_embeddings(self):
        """
        Return item similarity matrix S_approx = V @ diag(filter) @ V.T
        Note: Unlike sparse versions, this calculates dense S on the fly.
        """
        if not hasattr(self.model, 'V_raw') or getattr(self.model, 'V_raw', None) is None:
            raise RuntimeError("[SpectralPowerLIRA] Model not fitted.")
            
        V = self.model.V_raw
        filter_diag = self.model.filter_diag
        
        # S = V @ diag(F) @ V.T
        S = torch.mm(V * filter_diag, V.t())
        return S
