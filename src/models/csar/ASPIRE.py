import torch
import torch.nn as nn
import numpy as np
import os
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.models.csar.ASPIRELayer import ASPIRELayer, MNARGammaCacheManager
from src.utils.gpu_accel import SVDCacheManager

class ASPIRE(BaseModel):
    """
    ASPIRE - Scalable LIRA using SVD with Popularity Decay.
    h(sigma_k) = sigma_k^(2-beta) / (sigma_k^(2-beta) + alpha)
    """
    def __init__(self, config, data_loader):
        super(ASPIRE, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.alpha = model_config.get('alpha', 500.0)
        self.beta = model_config.get('beta', 0.5)
        self.target_energy = model_config.get('target_energy', 0.9)
        self.k = model_config.get('k', 128)
        self.visualize = model_config.get('visualize', True)
        self.estimator_type = model_config.get('estimator_type', 'huber')
        
        # ASPIRE Layer
        self.lira_layer = ASPIRELayer(
            k=self.k,
            alpha=self.alpha,
            beta=self.beta,
            target_energy=self.target_energy,
            estimator_type=self.estimator_type
        )
        self.lira_layer.to(self.device)
        
        self._log(f"Initialized (k={self.k}, α={self.alpha}, β={self.beta}, method={self.estimator_type})")
        
        # Build Sparse Matrix from DataLoader
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        
        # Build Low-Rank Model immediately
        dataset_name = config.get('dataset_name', 'unknown')
        self.lira_layer.build(self.train_matrix_csr, dataset_name=dataset_name)
        self.lira_layer.to(self.device)

        # Cache managers 등록 (Trainer가 자동 관리)
        self.register_cache_manager('svd', SVDCacheManager(device=self.device.type))
        self.register_cache_manager('mnar_gamma', MNARGammaCacheManager())

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
        self._log(f"Training (k={self.k}, α={self.alpha}, β={self.beta})")
        self._log("="*60)
        
        # Always perform visualization (Lightweight vs Heavyweight controlled by visualize flag)
        self._log(f"Analyzing model (Visualize Heavyweight: {self.visualize})...")
        try:
            # Create analysis directory
            analysis_dir = os.path.join(self.output_path, 'analysis')
            os.makedirs(analysis_dir, exist_ok=True)
            self._log(f"Analysis directory created/verified: {os.path.abspath(analysis_dir)}")
            
            # Use 'visualize' config to control heavyweight heatmaps
            if hasattr(self.lira_layer, 'visualize_matrices'):
                self.lira_layer.visualize_matrices(
                    X_sparse=self.train_matrix_csr, 
                    save_dir=analysis_dir,
                    lightweight=not self.visualize
                )
                self._log(f"Analysis results saved to {analysis_dir}")
        except Exception as e:
            self._log(f"Visualization skipped: {e}")
        self._log("="*60 + "\n")

    def get_embeddings(self):
        # Return V_k as Item Embeddings (approximated)
        return None, self.lira_layer.V_k

    def get_final_item_embeddings(self):
        return self.lira_layer.V_k
    
    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None
