import torch
import torch.nn as nn
import numpy as np
import os
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.models.csar.ASPIRELayer import ASPIRELayer
from src.utils.gpu_accel import SVDCacheManager, EVDCacheManager

class ASPIRE(BaseModel):
    """
    ASPIRE - Spectral Filtering for MNAR-robust Recommendation.
    h(sigma_tilde_k; gamma, tau) = sigma_tilde_k^gamma / (sigma_tilde_k^gamma + tau^gamma)
    gamma (gamma): filter shape | tau (tau): cut-off threshold [0, 1]
    """
    def __init__(self, config, data_loader):
        super(ASPIRE, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.tau = model_config.get('tau', model_config.get('alpha', 0.3))  # backward compat
        self.gamma = model_config.get('gamma', 1.0)
        self.target_energy = model_config.get('target_energy', 0.9)
        self.k = model_config.get('k', None)
        self.visualize = model_config.get('visualize', True)
        
        # ASPIRE Layer
        self.lira_layer = ASPIRELayer(
            k=self.k,
            tau=self.tau,
            gamma=self.gamma,
            target_energy=self.target_energy
        )
        self.lira_layer.to(self.device)
        
        self._log(f"Initialized (k={self.k}, t={self.tau:.3f}, g={self.gamma:.2f})")
        
        # Build Sparse Matrix from DataLoader
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        
        # Build Low-Rank Model immediately
        dataset_name = config.get('dataset_name', 'unknown')
        self.lira_layer.build(self.train_matrix_csr, dataset_name=dataset_name)
        self.lira_layer.to(self.device)

        # Cache managers 등록
        self.register_cache_manager('evd', EVDCacheManager(device=self.device.type))
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
        train_matrix = self.get_train_matrix(self.data_loader)
        batch_users = users.cpu().numpy()
        user_history_dense = torch.from_numpy(train_matrix[batch_users].toarray()).float().to(self.device)
        
        scores = self.lira_layer(user_history_dense)
        
        if items is not None:
             return scores.gather(1, items)
        return scores

    def predict_for_pairs(self, users, items):
        """배치 단위 (user, item) 쌍에 대한 점수 예측 (평가용)"""
        train_matrix = self.get_train_matrix(self.data_loader)
        batch_users = users.cpu().numpy()
        user_history_dense = torch.from_numpy(train_matrix[batch_users].toarray()).float().to(self.device)
        
        scores = self.lira_layer(user_history_dense)
        return scores.gather(1, items.unsqueeze(1)).squeeze(1)

    def diagnostics(self):
        """모델 상태 및 스펙트럼 진단 정보 반환"""
        layer = self.lira_layer
        return {
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "tau_gamma": float(layer.tau_gamma),
            "rho": float(layer.rho),
            "n_components": int(layer.k),
        }

    def fit(self, data_loader):
        diag = self.diagnostics()
        self._log(f"\n{'='*60}")
        self._log(f"Training ASPIRE | g={diag['gamma']:.2f} t={diag['tau']:.4f} (t^g={diag['tau_gamma']:.4f}) rho={diag['rho']:.4f}")
        self._log("="*60)
        
        if self.visualize:
            try:
                analysis_dir = os.path.join(self.output_path, 'analysis')
                os.makedirs(analysis_dir, exist_ok=True)
                self.lira_layer.visualize_matrices(
                    X_sparse=self.train_matrix_csr, 
                    save_dir=analysis_dir,
                    lightweight=False
                )
                self._log(f"Analysis saved to {analysis_dir}")
            except Exception as e:
                self._log(f"Visualization skipped: {e}")
        self._log("="*60 + "\n")

    def get_final_item_embeddings(self):
        return self.lira_layer.V_k
    
    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None

class ASPIRE_Norm(ASPIRE):
    """ASPIRE with Symmetric Normalization"""
    def __init__(self, config, data_loader):
        if 'model' not in config: config['model'] = {}
        config['model']['symmetric_norm'] = True
        super(ASPIRE_Norm, self).__init__(config, data_loader)
