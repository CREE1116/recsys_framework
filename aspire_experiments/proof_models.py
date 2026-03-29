import torch
import torch.nn as nn
import numpy as np
from src.models.base_model import BaseModel
from src.utils.gpu_accel import EVDCacheManager, SVDCacheManager
from scipy.sparse import csr_matrix

class AspireFilter_Test:
    @staticmethod
    def apply_filter(vals: torch.Tensor, gamma: float = 1.0, alpha: float = 1.0, 
                    mode: str = 'gamma_only', is_gram: bool = False, skip_top_k: int = 0) -> tuple[torch.Tensor, float, float]:
        """
        테스트용 필터: gamma와 alpha(lambda)를 모두 지원
        """
        s = torch.clamp(vals.float(), min=1e-12)
        exp = float(gamma) if not is_gram else float(gamma) / 2.0
        s_gamma = torch.pow(s, exp)
        s_max_gamma = s_gamma.max().item()
        if skip_top_k > 0:
            # For skipped components, use the 'natural' exponent 2 (Standard Tikhonov/EASE scale)
            # This ensures they are not 'restored' but kept as Goliath components.
            s_gamma = s_gamma.clone()
            s_gamma[:skip_top_k] = torch.pow(s[:skip_top_k], 2.0) 

        if mode == 'gamma_only':
            # h(s_max) = 0.5 (alpha is ignored)
            effective_lambda = s_max_gamma
            alpha_val = 1.0
        else:
            # standard: uses the provided alpha (lambda)
            effective_lambda = float(alpha)
            alpha_val = float(alpha)
            
        h = s_gamma / (s_gamma + effective_lambda + 1e-10)
        return h.float(), alpha_val, float(effective_lambda)

class ASPIRELayer_Test(nn.Module):
    def __init__(self, k=200, gamma=1.0, alpha=1.0, filter_mode="gamma_only", skip_top_k=0, **kwargs):
        super().__init__()
        self.k = k
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.filter_mode = filter_mode
        self.skip_top_k = int(skip_top_k)
        self.target_energy = kwargs.get("target_energy", 0.9)

        self.register_buffer("singular_values", torch.empty(0))
        self.register_buffer("V_raw",           torch.empty(0, 0))
        self.register_buffer("filter_diag",     torch.empty(0))

    @torch.no_grad()
    def build(self, X_sparse, dataset_name=None, device=None):
        dev = device if device is not None else torch.device("cpu")
        
        # --- Optimization: Use EVDCacheManager for k=None (Full) or SVDCacheManager for truncated ---
        if self.k is None:
            print(f"  [ASPIRE] No k provided. Requesting Full EVD...")
            manager = EVDCacheManager(device=str(dev))
            _, s, v, _ = manager.get_evd(X_sparse, k=None, dataset_name=dataset_name)
            self.k = len(s)
        else:
            manager = SVDCacheManager(device=dev)
            _, s, v, _ = manager.get_svd(X_sparse, k=self.k, dataset_name=dataset_name)
            self.k = min(int(self.k), len(s), 10000)
        
        # Use raw singular values (DO NOT NORMALIZE to preserved top signal power)
        self.register_buffer("singular_values", s[:self.k].to(dev))
        self.register_buffer("V_raw", v[:, :self.k].to(dev))

        # Apply filter on raw singular values.
        # h = s^gamma / (s^gamma + alpha)
        h, _, _ = AspireFilter_Test.apply_filter(
            self.singular_values, gamma=self.gamma, alpha=self.alpha, 
            mode=self.filter_mode, is_gram=False, skip_top_k=self.skip_top_k
        )
        self.register_buffer("filter_diag", h)

    def forward(self, X_batch):
        XV = torch.mm(X_batch, self.V_raw)
        return torch.mm(XV * self.filter_diag, self.V_raw.t())

class ASPIRE_Test(BaseModel):
    """실험용 ASPIRE 테스트 모델 (Gamma/Alpha 조절 가능)"""
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.alpha = model_config.get('alpha', 1.0)
        self.gamma = model_config.get('gamma', 1.0)
        self.filter_mode = model_config.get('filter_mode', 'gamma_only')
        self.skip_top_k = model_config.get('skip_top_k', 0)
        self.k = model_config.get('k', None)
        self.target_energy = model_config.get('target_energy', 1.0)
        
        self.lira_layer = ASPIRELayer_Test(
            k=self.k,
            gamma=self.gamma,
            alpha=self.alpha,
            filter_mode=self.filter_mode,
            skip_top_k=self.skip_top_k,
            target_energy=self.target_energy
        )
        self.lira_layer.to(self.device)
        
        # Build Matrix
        train_df = data_loader.train_df
        R = csr_matrix((np.ones(len(train_df)), (train_df['user_id'].values, train_df['item_id'].values)), 
                       shape=(self.n_users, self.n_items))
        self.train_matrix_csr = R
        
        dataset_name = config.get('dataset_name', 'unknown')
        self.lira_layer.build(self.train_matrix_csr, dataset_name=dataset_name, device=self.device)

    def forward(self, users, items=None):
        batch_users = users.cpu().numpy()
        user_history_dense = torch.from_numpy(self.train_matrix_csr[batch_users].toarray()).float().to(self.device)
        scores = self.lira_layer(user_history_dense)
        if items is not None:
             return scores.gather(1, items)
        return scores

    def fit(self, data_loader):
        pass # Built in __init__ for simplicity in tests

    def predict_for_pairs(self, users, items):
        return self.forward(users, items.unsqueeze(1)).squeeze(1)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None

    def get_final_item_embeddings(self):
        return self.lira_layer.V_raw

class EASE_Test(BaseModel):
    """실험용 EASE 테스트 모델 (Diagonal Removal 포함)"""
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.alpha = model_config.get('alpha', 100.0)
        
        # Build
        train_df = data_loader.train_df
        R = csr_matrix((np.ones(len(train_df)), (train_df['user_id'].values, train_df['item_id'].values)), 
                       shape=(self.n_users, self.n_items))
        self.train_matrix_csr = R
        
        print(f"[EASE_Test] Building Gram matrix and inverting (alpha={self.alpha})...")
        G = (R.T @ R).toarray().astype(np.float32)
        G += np.eye(self.n_items) * self.alpha
        
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0)
        
        self.B = torch.from_numpy(B).to(self.device).float()
        print(f"[EASE_Test] Built.")

    def forward(self, users, items=None):
        batch_users = users.cpu().numpy()
        user_history_dense = torch.from_numpy(self.train_matrix_csr[batch_users].toarray()).float().to(self.device)
        scores = torch.mm(user_history_dense, self.B)
        if items is not None:
             return scores.gather(1, items)
        return scores

    def fit(self, data_loader):
        pass

    def predict_for_pairs(self, users, items):
        return self.forward(users, items.unsqueeze(1)).squeeze(1)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), None

    def get_final_item_embeddings(self):
        return self.B



