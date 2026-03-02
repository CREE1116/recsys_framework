import torch
import numpy as np

from ..base_model import BaseModel
from .LIRALayer import CGLIRALayer


class CGLIRA(BaseModel):
    def __init__(self, config, data_loader):
        super(CGLIRA, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.reg_lambda = model_config.get('reg_lambda', 500.0)
        self.max_iter = model_config.get('max_iter', 30)
        self.tol = float(model_config.get('tol', 1e-6))
        
        # CGLIRA Layer
        self.lira_layer = CGLIRALayer(
            reg_lambda=self.reg_lambda,
            max_iter=self.max_iter,
            tol=self.tol
        )
        self.lira_layer.to(self.device)
        
        print(f"[CGLIRA] Initialized with λ={self.reg_lambda}, max_iter={self.max_iter}, tol={self.tol}")
        
        # Build Sparse Matrix from DataLoader
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        
        # Build Layer immediately
        dataset_name = config.get('dataset_name', 'unknown')
        self.lira_layer.build(self.train_matrix_csr, dataset_name=dataset_name)
        self.lira_layer.to(self.device)

    def _build_sparse_matrix(self, data_loader):
        from scipy.sparse import csr_matrix
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        return csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items))

    def get_train_matrix(self, data_loader):
        return self.train_matrix_csr

    def fit(self, train_loader, val_loader=None):
        # Explicit solving is done lazily at inference time using CG
        # The sparse matrix components were prepared in build()
        pass

    def predict_full(self, users, items=None):
        # Get user history
        train_matrix = self.get_train_matrix(self.data_loader)
        
        # Batch conversion to dense is required for projection, 
        # but CGLIRALayer expects dense X_batch
        batch_users = users.cpu().numpy()
        user_history_dense = torch.from_numpy(train_matrix[batch_users].toarray()).float().to(self.device)
        
        # Forward pass iteratively solves (S + lambda I) Z = S X_batch^T
        scores = self.lira_layer(user_history_dense, user_ids=users)
        
        if items is not None:
             return scores.gather(1, items)
        return scores
        
    def predict_for_pairs(self, user_ids, item_ids):
        scores = self.predict_full(user_ids)
        if scores.dim() == 1:
            return scores[item_ids]
        return scores.gather(1, item_ids)

    def forward(self, users, items=None):
        return self.predict_full(users, items)

    def get_embeddings(self):
        return None, None

    def get_final_item_embeddings(self):
        return None

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device, requires_grad=True),), None
