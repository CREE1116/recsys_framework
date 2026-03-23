import torch
import torch.nn as nn
import numpy as np
import time
from scipy.sparse import csr_matrix
from ..base_model import BaseModel
from .ASPIRELayer import ChebyASPIRELayer

class ChebyASPIRE(BaseModel):
    """
    ChebyASPIRE: SVD-free ASPIRE via Chebyshev polynomial approximation.
    h(sigma_tilde; gamma, tau) = sigma_tilde^gamma / (sigma_tilde^gamma + tau^gamma)
    tau: cut-off threshold [0,1] independent of gamma.
    """
    def __init__(self, config, data_loader):
        super(ChebyASPIRE, self).__init__(config, data_loader)
        
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.alpha = model_config.get('alpha', 0.1)
        self.gamma = model_config.get('gamma', 1.0)
        self.degree = model_config.get('degree', 20)
        self.lambda_max_estimate = model_config.get('lambda_max_estimate', 'auto')
        self.filter_mode = model_config.get('filter_mode', 'standard')
        self.visualize = model_config.get('visualize', True)
        self.precompute_scores = model_config.get('precompute_scores', True)
        
        # ChebyASPIRE Layer
        # Remove explicitly passed args from kwargs to avoid 'multiple values for keyword argument' error
        layer_kwargs = model_config.copy()
        for key in ['alpha', 'gamma', 'degree', 'lambda_max_estimate', 'filter_mode']:
            layer_kwargs.pop(key, None)

        self.lira_layer = ChebyASPIRELayer(
            degree=self.degree,
            gamma=self.gamma,
            lambda_max_estimate=self.lambda_max_estimate,
            **layer_kwargs
        )
        self.lira_layer.to(self.device)
        
        # Build during fit
        self.train_matrix_csr = None
        
        self._log(f"Initialized (degree={self.degree}, mode=gamma_only, gamma={self.gamma:.2f})")

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
        diag = self.diagnostics()
        if diag.get('filter_mode') == 'gamma_only':
            mode_str = "(gamma_only)"
        else:
            mode_str = f"α={diag['alpha']:.4f} (standard)"
            
        self._log(f"Training ChebyASPIRE | degree={self.degree} {mode_str} gamma={diag['gamma']:.2f}")
        self._log(f"{'='*60}")
        
        start_time = time.time()
        
        # 1. Build interaction matrix
        X_sparse = self.get_train_matrix(data_loader)
        
        # 2. Build Layer
        dataset_name = self.config.get('dataset_name', 'unknown')
        self.lira_layer.build(X_sparse, dataset_name=dataset_name)
        self.lira_layer.to(self.device)
        
        # 3. Precompute Full Scores (Optimization for evaluation speed)
        # Only if item_weights (dense matrix) was not built (i.e., for large datasets)
        if self.precompute_scores and self.lira_layer.item_weights.numel() == 0:
            self.lira_layer.precompute(X_sparse, dataset_name=dataset_name, 
                                      matrix_id=self.lira_layer.matrix_id, device=self.device)
        
        if self.visualize:
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
        # 1. Use precomputed cache if available
        if self.lira_layer.scores_cache is not None:
            scores = self.lira_layer.scores_cache[users.cpu().numpy()].to(self.device)
        else:
            # Fallback to iterative forward
            X_csr = self.train_matrix_csr 
            user_history_dense = torch.from_numpy(X_csr[users.cpu().numpy()].toarray()).float().to(self.device)
            scores = self.lira_layer(user_history_dense)
        
        if items is not None:
             return scores.gather(1, items)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        return self.predict_full(user_ids, item_ids)

    def get_final_item_embeddings(self):
        # Implicitly handled; return Identity
        return torch.eye(self.n_items, device=self.device)
    
    def diagnostics(self):
        layer = self.lira_layer
        return {
            "gamma": float(self.gamma),
            "alpha": float(layer.alpha) if layer.alpha is not None else 0.0,
            "alpha_abs": float(layer.alpha_abs) if layer.alpha_abs is not None else 0.0,
            "degree": int(self.degree),
            "filter_mode": self.filter_mode
        }

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None

    def __str__(self):
        return f"ChebyASPIRE(degree={self.degree}, gamma={self.gamma:.2f}, mode={self.filter_mode})"
