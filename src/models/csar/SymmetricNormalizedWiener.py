import torch
import numpy as np
import time
from scipy.sparse import csr_matrix
from src.models.base_model import BaseModel
from src.utils.gpu_accel import get_device, _build_gram

class SymmetricNormalizedWiener(BaseModel):
    """
    Symmetric Normalized Wiener Filter (Ablation Model).
    --------------------------------------------------
    1. Build Gram Matrix G = R^T R.
    2. Extract Item Degree vector d (raw popularity).
    3. Normalize G: G_tilde = D^-1/2 G D^-1/2  where D = diag(d).
    4. Solve Unconstrained Wiener Filter on G_tilde:
       W = I - lambda * (G_tilde + lambda I)^-1.
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        model_config = config.get('model', {})
        self.reg_lambda = float(model_config.get('reg_lambda', 500.0))
        self.device = get_device()
        self.eps = 1e-12

        # Filter Buffer
        self.register_buffer("W", torch.empty(self.n_items, self.n_items))

        # Build training matrix
        df = data_loader.train_df
        self.train_matrix_csr = csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df['user_id'].values, df['item_id'].values)),
            shape=(self.n_users, self.n_items)
        )
        
        self._build(self.train_matrix_csr)

    @torch.no_grad()
    def _build(self, R_sparse):
        m = self.n_items
        self._log(f"Building SymmetricNormalizedWiener (lambda={self.reg_lambda}) on {self.device}")
        t0 = time.time()

        # 1. G_obs = R^T R
        G = _build_gram(R_sparse, self.device)
        
        # 2. Extract item degrees (row/column sums of G)
        d = G.sum(dim=1) + self.eps
        
        # 3. Symmetric Normalization: G_tilde = D^-1/2 G D^-1/2
        inv_sqrt_d = 1.0 / torch.sqrt(d)
        G_tilde = G * inv_sqrt_d.unsqueeze(1) * inv_sqrt_d.unsqueeze(0)
        del G

        # 4. Unconstrained Wiener Filter Solve
        self._log(f"Solving unconstrained Wiener filter for {m}x{m} matrix...")
        I = torch.eye(m, device=self.device, dtype=torch.float32)
        A = G_tilde + self.reg_lambda * I
        
        try:
            P = torch.linalg.inv(A)
        except RuntimeError:
            self._log("Linalg inv failed, using CPU fallback.")
            P = torch.from_numpy(np.linalg.inv(A.cpu().numpy())).to(self.device).float()
        del A, G_tilde

        # 4. Constrained Symmetric EASE with Lagrange (Zero-Diagonal Constraint)
        # Formula: W_tilde = I - P * diag(1/diag(P))
        # This ensures diag(W_tilde) = 0 explicitly, improving ranking quality.
        self._log(f"Solving Constrained Symmetric EASE (lambda={self.reg_lambda})...")
        diag_P = P.diagonal()
        # Ensure numerical stability for division
        W_tilde = I - (P / torch.clamp(diag_P.unsqueeze(0), min=self.eps))
        del P

        # 5. Direct weight copy (No reconstruction, matches ASPIRE-Wiener setup)
        self.W.copy_(W_tilde)
        self._log(f"SymmetricNormalizedWiener Build completed in {time.time()-t0:.2f}s")

    @torch.no_grad()
    def forward(self, users):
        if users.dim() > 1 and users.shape[1] == self.n_items:
            return torch.matmul(users, self.W)
        batch_ids = users.cpu().numpy()
        X_u = torch.from_numpy(self.train_matrix_csr[batch_ids].toarray()).float().to(self.device).detach()
        return torch.matmul(X_u, self.W)

    @torch.no_grad()
    def predict_full(self, users, items=None):
        scores = self.forward(users)
        if items is not None:
            return scores.gather(1, items)
        return scores

    @torch.no_grad()
    def get_user_scores(self, user_indices):
        user_data = self.train_matrix_csr[user_indices].toarray()
        user_torch = torch.from_numpy(user_data).float().to(self.device).detach()
        return torch.matmul(user_torch, self.W)

    def calc_loss(self, batch_data):
        return (torch.tensor(0.0, device=self.device),), {}

    def get_final_item_embeddings(self):
        return self.W

    @torch.no_grad()
    def predict_for_pairs(self, users, items):
        scores = self.forward(users)
        return scores.gather(1, items.unsqueeze(1)).squeeze(1)
