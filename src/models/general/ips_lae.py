import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
from src.utils.gpu_accel import gpu_gram_solve, gpu_cholesky_solve, GramMatrixCacheManager

class IPS_LAE(BaseModel):
    """
    Onishi et al. (2025) Propensity-Weighted Linear Autoencoder.
    SIGIR-AP 2025.
    
    Supports:
    - Backbone: ease, edlae, rdlae
    - Propensity: logsigmoid, powerlaw
    """
    def __init__(self, config, data_loader):
        super(IPS_LAE, self).__init__(config, data_loader)
        model_config = config.get('model', {})
        self.reg_lambda = model_config.get('reg_lambda', 500.0)
        self.wtype      = model_config.get('wtype', 'powerlaw')   # logsigmoid | powerlaw
        self.wbeta      = model_config.get('wbeta', 0.4)
        self.backbone   = model_config.get('backbone', 'ease')    # ease | edlae | rdlae
        self.drop_p     = model_config.get('drop_p', 0.5)
        self.alpha      = model_config.get('alpha', 0.5)

        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        self.register_buffer('weight_matrix', torch.empty(0, 0))
        self.train_matrix_csr = None

    def _compute_inv_propensity(self, X):
        if self.wtype == 'logsigmoid':
            freqs = np.ravel(X.sum(axis=0))
            log_freqs = np.log(freqs + 1)
            alpha_logit = -self.wbeta * (np.min(log_freqs) + np.max(log_freqs)) / 2
            logits = alpha_logit + self.wbeta * log_freqs
            p = 1 / (1 + np.exp(-logits))
        elif self.wtype == 'powerlaw':
            pop = np.ravel(X.sum(axis=0))
            norm_pop = pop / (np.max(pop) + 1e-12)
            p = np.power(norm_pop, self.wbeta)
        else:
            raise ValueError(f"Unknown wtype: {self.wtype}")
        return 1 / (p + 1e-12)

    def fit(self, data_loader):
        self._log(f"Fitting IPS_LAE ({self.backbone}, wtype={self.wtype}, wbeta={self.wbeta:.2f})...")
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        
        X = sp.csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix_csr = X
        dataset_name = self.config.get('dataset_name', 'unknown')

        # 1. Get G = X^T X on device (No caching)
        self._log(f"Computing Gram matrix G = X^T X on {self.device}...")
        if self.device.type in ('cuda', 'mps'):
            # Optimized: bfloat16 for memory efficiency (ADA GPU)
            from src.utils.gpu_accel import _build_gram
            G = _build_gram(X, self.device)
        else:
            G = (X.T @ X).toarray().astype(np.float32)
            G = torch.from_numpy(G).to(self.device)

        # 2. Backbone specific fit (on device)
        if self.backbone == 'ease':
            G_reg = G.clone()
            G_reg.diagonal().add_(self.reg_lambda)
            P = gpu_cholesky_solve(G_reg, rhs_np=None, device=self.device, return_tensor=True)
            del G_reg
            B = P / (-torch.diag(P) + 1e-12)
            del P
        elif self.backbone == 'edlae':
            diag_G = torch.diag(G)
            gamma = diag_G * self.drop_p / (1 - self.drop_p) + self.reg_lambda
            G_reg = G.clone()
            G_reg.diagonal().add_(gamma)
            P = gpu_cholesky_solve(G_reg, rhs_np=None, device=self.device, return_tensor=True)
            del G_reg
            B = P / (-torch.diag(P) + 1e-12)
            del P
        elif self.backbone == 'rdlae':
            diag_G = torch.diag(G)
            gamma = diag_G * self.drop_p / (1 - self.drop_p) + self.reg_lambda
            G_reg = G.clone()
            G_reg.diagonal().add_(gamma)
            P = gpu_cholesky_solve(G_reg, rhs_np=None, device=self.device, return_tensor=True)
            del G_reg
            diag_P = torch.diag(P)
            cond = (1 - gamma * diag_P) > self.alpha
            lag = ((1 - self.alpha) / (diag_P + 1e-12) - gamma) * cond.float()
            # B = P * -(gamma + lag) [Column-wise scaling]
            B = P * -(gamma + lag).view(1, -1)
            del P
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

        # 3. Propensity weighting
        w_np = self._compute_inv_propensity(X)
        w = torch.from_numpy(w_np).to(self.device).float()
        
        # B = B * w [Column-wise scaling]
        B = B * w.view(1, -1)
        B.fill_diagonal_(0)

        self.weight_matrix = B
        self._log(f"Fitted IPS_LAE on {self.device}.")

    def forward(self, user_ids, item_ids=None):
        if self.train_matrix_csr is None:
             raise RuntimeError("Model has not been fitted yet.")

        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = user_ids
            
        user_input_sparse = self.train_matrix_csr[u_ids_np]
        user_input = torch.from_numpy(user_input_sparse.toarray()).float().to(self.device)
        
        scores = user_input @ self.weight_matrix
        return scores

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None

    def predict_for_pairs(self, user_ids, item_ids):
        scores = self.forward(user_ids)
        batch_indices = torch.arange(len(user_ids), device=user_ids.device)
        return scores[batch_indices, item_ids]

    def get_final_item_embeddings(self):
        return self.weight_matrix
