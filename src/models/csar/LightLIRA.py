import torch
import torch.nn as nn
import numpy as np
import os
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.models.csar.LIRALayer import LightLIRALayer
from src.utils.gpu_accel import SVDCacheManager

class LightLIRA(BaseModel):
    """
    LightLIRA - Scalable LIRA using SVD-based Spectral Filtering.
    - O(nk) inference complexity.
    - No massive N*N matrix inversion.
    """
    def __init__(self, config, data_loader):
        super(LightLIRA, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.k = config['model'].get('k', 100000)
        self.visualize = config['model'].get('visualize', False) # Keep visualize config for fit method
        
        # LightLIRA Layer
        self.lira_layer = LightLIRALayer(
            k=self.k,
            reg_lambda=self.reg_lambda
        )
        self.lira_layer.to(self.device)
        

        
        self._log(f"Initialized (k={self.k}, λ={self.reg_lambda})")
        
        # Build Sparse Matrix from DataLoader
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        
        # Build Low-Rank Model immediately
        dataset_name = config.get('dataset_name', 'unknown')
        self.lira_layer.build(self.train_matrix_csr, dataset_name=dataset_name)
        self.lira_layer.to(self.device)

        # Cache manager 등록
        self.register_cache_manager('svd', SVDCacheManager(device=self.device.type))

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
        
        # Batch conversion to dense is required for projection
        batch_users = users.cpu().numpy()
        user_history_dense = torch.from_numpy(train_matrix[batch_users].toarray()).float().to(self.device)
        
        # Forward pass: (X @ V) * filter @ V.T
        scores = self.lira_layer(user_history_dense, user_ids=users)
        
        if items is not None:
             return scores.gather(1, items)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        # Optimized for pair prediction
        scores = self.predict_full(user_ids)
        return scores.gather(1, item_ids)

    def fit(self, data_loader):
        self._log(f"\n{'='*60}")
        self._log(f"Training (λ={self.reg_lambda})")
        self._log("="*60)
        
        # Always perform visualization (Lightweight vs Heavyweight controlled by visualize flag)
        self._log("Precomputing Gram Matrix...")
        try:
            # Use SVDCacheManager to resolve analysis path
            from src.utils.gpu_accel import SVDCacheManager
            analysis_dir = SVDCacheManager.get_analysis_dir(self.config)
            os.makedirs(analysis_dir, exist_ok=True)
            self._log(f"Analysis directory created/verified: {os.path.abspath(analysis_dir)}")
            
            # Use 'visualize' config to control heavyweight heatmaps
            self.lira_layer.visualize_matrices(
                X_sparse=self.train_matrix_csr, 
                save_dir=analysis_dir,
                lightweight=not self.visualize
            )
            self._log(f"Analysis results saved to {analysis_dir}")
        except Exception as e:
            self._log(f"Visualization skipped: {e}")
            import traceback
            traceback.print_exc()
        self._log("="*60 + "\n")

    def get_embeddings(self):
        # Return V_k as Item Embeddings (approximated)
        return None, self.lira_layer.V_k

    def get_final_item_embeddings(self):
        return self.lira_layer.V_k
    
    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device, requires_grad=True),), None
