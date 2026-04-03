import torch
import torch.nn as nn
import numpy as np
import time
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.utils.gpu_accel import gpu_gram_solve, get_device

class EASE_MNAR(BaseModel):
    """
    EASE-MNAR (Popularity Normalized Spectral Filter)
    
    [핵심 메커니즘]
    1. Gram Matrix Build: G = R^T R
    2. Popularity Proxy (Degree): d_i = sum_j G_ij (아이템별 등장 빈도의 공분산 합)
    3. MNAR Correction: G_tilde = D^(-alpha/2) @ G @ D^(-alpha/2)
       - alpha=1.0: 완전 정규화
       - alpha=0.0: 일반 EASE
    4. Wiener Filter: W = (G_tilde + lambda I)^-1 G_tilde
    5. Diagonal Removal: W_ii = 0 (자기 추천 방지)
    """

    def __init__(self, config, data_loader):
        super(EASE_MNAR, self).__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items

        model_config = config.get('model', {})
        self.reg_lambda = float(model_config.get('reg_lambda', 0.5))
        self.alpha      = float(model_config.get('alpha', 0.5))
        self.eps        = 1e-12

        # Filter Buffer
        self.register_buffer("W", torch.empty(self.n_items, self.n_items))

        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        self._build(self.train_matrix_csr, config.get('dataset_name', 'unknown'))

    def _build_sparse_matrix(self, data_loader):
        df = data_loader.train_df
        return csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df['user_id'].values, df['item_id'].values)),
            shape=(self.n_users, self.n_items)
        )

    @torch.no_grad()
    def _build(self, R_sparse, dataset_name):
        n, m = R_sparse.shape
        self._log(f"Building EASE-MNAR (lambda={self.reg_lambda}, alpha={self.alpha}) on {self.device}")
        t0 = time.time()

        # 1. G_obs = R^T R
        # Use bfloat16 batch multiplication to prevent OCP/thermal issues if large
        from src.utils.gpu_accel import _build_gram
        G = _build_gram(R_sparse, self.device)
        
        # 2. Degree-based Normalization
        # D (degree / popularity proxy)
        d = G.sum(dim=1)
        d_inv_sqrt = 1.0 / (torch.pow(d + self.eps, self.alpha / 2.0))
        
        # G_tilde = D^-1/2 @ G @ D^-1/2
        # Broadcasting: (M, 1) * (M, M) * (1, M)
        G_tilde = G * d_inv_sqrt.unsqueeze(1)
        G_tilde = G_tilde * d_inv_sqrt.unsqueeze(0)
        del G

        # 3. Unconstrained Wiener Filter Solve
        self._log(f"Solving unconstrained Wiener filter for {m}x{m} matrix...")
        
        # Add lambda to diagonal
        I = torch.eye(m, device=self.device, dtype=torch.float32)
        A = G_tilde + self.reg_lambda * I
        
        try:
            # P = (G_tilde + lambda I)^-1
            P = torch.linalg.inv(A)
        except RuntimeError:
            self._log("Linalg inv failed, using CPU fallback.")
            P = torch.from_numpy(np.linalg.inv(A.cpu().numpy())).to(self.device).float()
        
        del A, G_tilde

        # W = I - lambda * P
        W = I - (self.reg_lambda * P)
        del P
        
        # Self-loop removal (Post-hoc)
        W.fill_diagonal_(0.0)
        
        self.W.copy_(W)
        self._log(f"EASE-MNAR Build completed in {time.time()-t0:.2f}s")

    @torch.no_grad()
    def forward(self, users):
        batch_ids = users.cpu().numpy()
        X_u = torch.from_numpy(self.train_matrix_csr[batch_ids].toarray()).float().to(self.device)
        return torch.mm(X_u, self.W)

    @torch.no_grad()
    def predict_full(self, users, items=None):
        scores = self.forward(users)
        if items is not None:
            return scores.gather(1, items)
        return scores

    @torch.no_grad()
    def predict_for_pairs(self, users, items):
        scores = self.forward(users)
        return scores.gather(1, items.unsqueeze(1)).squeeze(1)

    def get_final_item_embeddings(self):
        return self.W

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), {}

    def diagnostics(self):
        return {
            "lambda": self.reg_lambda,
            "alpha": self.alpha,
            "W_mean": float(self.W.mean()),
            "W_diag_sum": float(torch.diag(self.W).sum())
        }
