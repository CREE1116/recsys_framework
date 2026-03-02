
import torch
import torch.nn as nn
import numpy as np
import os
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.models.csar.LIRALayer import LIRALayer
from src.utils.gpu_accel import SVDCacheManager

class LIRA(BaseModel):
    """
    Linear Interest-covariance Ridge Analysis
    """
    def __init__(self, config, data_loader):
        super(LIRA, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.visualize = config['model'].get('visualize', False)
        
        # LIRA Layer
        self.lira_layer = LIRALayer( 
            reg_lambda=self.reg_lambda
        )
        self.lira_layer.to(self.device)
        
        # Initialize
        print(f"[LIRA] Initialized with λ={self.reg_lambda}, Visualize={self.visualize}")
        
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
        print(f"\n[LIRA] Training (Lambda={self.reg_lambda})")
        print("="*60)
        
        # Build is already done in __init__ for this model structure
        # But if fit is called explicitly, we can rebuild or just log.
        # Given the current structure, __init__ calls build. 
        # Let's ensure S is built.
        
        if not hasattr(self.lira_layer, 'S') or self.lira_layer.S.numel() == 0:
             self.lira_layer.build(self.get_train_matrix(data_loader))
             
        # Path for analysis
        analysis_dir = SVDCacheManager.get_analysis_dir(self.config)
        
        if self.visualize:
            # New Visualization
            print(f"[LIRA] Saving visualizations to {analysis_dir}...")
            self.lira_layer.visualize_matrices(
                X_sparse=self.train_matrix_csr, 
                save_dir=analysis_dir
            )
        
        print("="*60 + "\n")

    def forward(self, users, items=None):
        # Compatible with AbstractModel
        return self.predict_full(users, items)


    def predict_full(self, users, items=None):
        # Get user history
        train_matrix = self.get_train_matrix(self.data_loader)
        # Check device
        # train_matrix is CSR. We need to batch convert to dense for LIRA layer if N is large.
        user_history_dense = torch.from_numpy(train_matrix[users.cpu().numpy()].toarray()).float().to(self.device)
        scores = self.lira_layer(user_history_dense, user_ids=users)
        
        if items is not None:
             return scores.gather(1, items)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        # Simple implementation using full prediction
        return self.predict_full(user_ids, item_ids)

    def get_embeddings(self):
        return None, self.lira_layer.S

    def get_final_item_embeddings(self):
        return self.lira_layer.S
    
    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device, requires_grad=True),), None
