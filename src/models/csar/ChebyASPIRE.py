import torch
import torch.nn as nn
import numpy as np
import time
from scipy.sparse import csr_matrix
from ..base_model import BaseModel
from .ASPIRELayer import ChebyASPIRELayer, MNARGammaCacheManager

class ChebyASPIRE(BaseModel):
    """
    ChebyASPIRE: SVD-free ASPIRE via Chebyshev polynomial approximation.
    f(S) = S^{3/4} / (S^{3/4} + alpha * I)
    """
    def __init__(self, config, data_loader):
        super(ChebyASPIRE, self).__init__(config, data_loader)
        
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.alpha = model_config.get('alpha', 500.0)
        self.degree = model_config.get('degree', 20)
        self.beta = model_config.get('beta', 0.5)
        self.lambda_max_estimate = model_config.get('lambda_max_estimate', 'auto')
        self.threshold = model_config.get('threshold', 1e-4)
        self.visualize = model_config.get('visualize', True)
        
        # ChebyASPIRE Layer
        self.lira_layer = ChebyASPIRELayer(
            alpha=self.alpha,
            degree=self.degree,
            beta=self.beta,
            lambda_max_estimate=self.lambda_max_estimate,
            threshold=self.threshold
        )
        self.lira_layer.to(self.device)
        
        # Build during fit (not here)
        self.train_matrix_csr = None
        
        self._log(f"Initialized (degree={self.degree}, alpha={self.alpha}, beta={self.beta}, threshold={self.threshold})")

        # Cache manager 등록
        self.register_cache_manager('mnar_gamma', MNARGammaCacheManager())

    def _build_sparse_matrix(self, data_loader):
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        return csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items))

    def get_train_matrix(self, data_loader):
        if self.train_matrix_csr is None:
            self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        return self.train_matrix_csr

    def fit(self, data_loader):
        """Build ChebyASPIRE model"""
        self._log(f"\n{'='*60}")
        self._log(f"Training ChebyASPIRE (degree={self.degree}, alpha={self.alpha}, beta={self.beta})")
        self._log(f"{'='*60}")
        
        start_time = time.time()
        
        # 1. Build interaction matrix
        X_sparse = self.get_train_matrix(data_loader)
        
        # 2. Build Layer
        dataset_name = self.config.get('dataset_name', 'unknown')
        self.lira_layer.build(X_sparse, dataset_name=dataset_name)
        self.lira_layer.to(self.device)
        
        self._log(f"Analyzing model (Visualize Heavyweight: {self.visualize})...")
        try:
            import os
            analysis_dir = os.path.join(self.output_path, 'analysis')
            os.makedirs(analysis_dir, exist_ok=True)
            self._log(f"Analysis directory created/verified: {os.path.abspath(analysis_dir)}")
            if hasattr(self.lira_layer, 'visualize_matrices'):
                self.lira_layer.visualize_matrices(
                    X_sparse=self.train_matrix_csr, 
                    save_dir=analysis_dir,
                    lightweight=not self.visualize
                )
                self._log(f"Analysis results saved to {analysis_dir}")
        except Exception as e:
            self._log(f"Visualization skipped: {e}")
        
        elapsed = time.time() - start_time
        self._log(f"\n{'='*60}")
        self._log(f"Training completed in {elapsed:.2f}s")
        self._log(f"{'='*60}\n")

    def forward(self, users, items=None):
        return self.predict_full(users, items)

    def predict_full(self, users, items=None):
        # Use the already built CSR matrix
        X_csr = self.train_matrix_csr 
        user_history_dense = torch.from_numpy(X_csr[users.cpu().numpy()].toarray()).float().to(self.device)
        scores = self.lira_layer(user_history_dense, user_ids=users)
        
        if items is not None:
             return scores.gather(1, items)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        return self.predict_full(user_ids, item_ids)

    def get_final_item_embeddings(self):
        # f(S) implicitly handled; return Identity for now as item-item similarity is not stored
        return torch.eye(self.n_items, device=self.device)
    
    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None

    def __str__(self):
        return f"ChebyASPIRE(degree={self.degree}, alpha={self.alpha}, threshold={self.threshold})"
