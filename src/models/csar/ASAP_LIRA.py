
import torch
import torch.nn as nn
import numpy as np
import os
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.models.csar.LIRALayer import ASAPLIRALayer
from src.utils.gpu_accel import SVDCacheManager

class ASAP_LIRA(BaseModel):
    """
    ASAP-LIRA: Adaptive Spectral Adaptive Pipeline LIRA
    """
    def __init__(self, config, data_loader):
        super(ASAP_LIRA, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.eta = config['model'].get('eta', 1.0)
        self.z = config['model'].get('z', 1.0)
        self.visualize = config['model'].get('visualize', False)
        
        # ASAPLIRA Layer
        self.lira_layer = ASAPLIRALayer( 
            reg_lambda=self.reg_lambda,
            eta=self.eta,
            z=self.z
        )
        self.lira_layer.to(self.device)
        
        # Initialize
        self._log(f"Initialized (λ={self.reg_lambda}, eta={self.eta}, z={self.z})")
        
        # Build Sparse Matrix from DataLoader
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        
        self.lira_layer.build(self.train_matrix_csr)
        self.lira_layer.to(self.device)

    def _build_sparse_matrix(self, data_loader):
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        return csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items))

    def get_train_matrix(self, data_loader):
        return self.train_matrix_csr

    def fit(self, data_loader):
        self._log(f"\n{'='*60}")
        self._log(f"Training (λ={self.reg_lambda}, η={self.eta}, z={self.z})")
        self._log("="*60)
        
        if not hasattr(self.lira_layer, 'S_sparse') or self.lira_layer.S_sparse is None:
             self.lira_layer.build(self.get_train_matrix(data_loader))
             
        # Path for analysis
        analysis_dir = SVDCacheManager.get_analysis_dir(self.config)
        
        if self.visualize:
            self._log(f"Saving visualizations to {analysis_dir}...")
            self.lira_layer.visualize_matrices(
                X_sparse=self.train_matrix_csr, 
                save_dir=analysis_dir
            )
        
        self._log("="*60 + "\n")

    def forward(self, users, items=None):
        return self.predict_full(users, items)

    def predict_full(self, users, items=None):
        train_matrix = self.get_train_matrix(self.data_loader)
        user_history_dense = torch.from_numpy(train_matrix[users.cpu().numpy()].toarray()).float().to(self.device)
        scores = self.lira_layer(user_history_dense, user_ids=users, mask_observed=False)
        
        if items is not None:
             return scores.gather(1, items)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        return self.predict_full(user_ids, item_ids)

    def get_embeddings(self):
        return None, self.lira_layer.S_sparse.to_dense()

    def get_final_item_embeddings(self):
        return self.lira_layer.S_sparse.to_dense()
    
    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device, requires_grad=True),), None
