import torch
import torch.nn as nn
import numpy as np
import os
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.models.csar.LIRALayer import TaylorLIRALayer

class TaylorLIRA(BaseModel):
    """
    TaylorLIRA - O(NNZ) matrix inversion using Neumann Series.
    Svd-free. Sparse preserving with element-wise power & EPS thresholding.
    """
    def __init__(self, config, data_loader):
        super(TaylorLIRA, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.reg_lambda = model_config.get('reg_lambda', 500.0)
        self.power = model_config.get('power', 2.0)
        self.threshold = model_config.get('threshold', 1e-6)
        self.K = model_config.get('K', 2)  # Taylor expansion order
        
        # TaylorLIRA Layer
        self.lira_layer = TaylorLIRALayer(
            reg_lambda=self.reg_lambda,
            power=self.power,
            threshold=self.threshold,
            K=self.K
        )
        self.lira_layer.to(self.device)
        
        print(f"[TaylorLIRA] Initialized with λ={self.reg_lambda}, p={self.power}, threshold={self.threshold}, K={self.K}")
        
        # Build Sparse Matrix from DataLoader
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        
        # Build Low-Rank Model immediately
        dataset_name = config.get('dataset_name', 'unknown')
        self.lira_layer.build(self.train_matrix_csr, dataset_name=dataset_name)
        self.lira_layer.to(self.device)

    def _build_sparse_matrix(self, data_loader):
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        return csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items))

    def get_train_matrix(self, data_loader):
        return self.train_matrix_csr

    def forward(self, users, items=None):
        return self.predict_full(users, items)
    
    def predict_full(self, users, items=None):
        # Get user history
        train_matrix = self.get_train_matrix(self.data_loader)
        
        # Batch conversion to dense is required for projection, 
        # but TaylorLIRALayer sparse.mm expects dense X_batch.
        batch_users = users.cpu().numpy()
        user_history_dense = torch.from_numpy(train_matrix[batch_users].toarray()).float().to(self.device)
        
        # Forward pass: X_batch @ S_sparse
        scores = self.lira_layer(user_history_dense, user_ids=users)
        
        if items is not None:
             return scores.gather(1, items)
        return scores

    def fit(self, data_loader):
        # Already built in __init__ for EASE-family logic flow
        print(f"\n[TaylorLIRA] Training completed during initialization.")
        print("="*60 + "\n")

    def predict_for_pairs(self, user_ids, item_ids):
        scores = self.predict_full(user_ids)
        if scores.dim() == 1:
            return scores[item_ids]
        return scores.gather(1, item_ids)

    def get_embeddings(self):
        return None, None

    def get_final_item_embeddings(self):
        return None
    
    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device, requires_grad=True),), None
