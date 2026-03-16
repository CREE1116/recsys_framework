import torch
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel

class RLAE(BaseModel):
    """
    RLAE (Relaxed Linear AutoEncoder) - SIGIR 2023
    "It's Enough: Relaxing Diagonal Constraints"
    
    Closed-form solution with diagonal inequality constraint.
    """
    def __init__(self, config, data_loader):
        super(RLAE, self).__init__(config, data_loader)
        self.reg_lambda = config['model'].get('reg_lambda', 500.0)
        self.b = config['model'].get('b', 0.5)  # Diagonal bound
        
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        self.register_buffer('weight_matrix', torch.empty(0, 0))
        self.train_matrix_csr = None
        
        self._log(f"Initialized: λ={self.reg_lambda}, b={self.b}")

    def _build_sparse_matrix(self, data_loader):
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df), dtype=np.float32)
        
        return sp.csr_matrix(
            (values, (rows, cols)), 
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )

    def fit(self, data_loader):
        self._log(f"\n{'='*60}")
        self._log(f"Training RLAE on {self.device}")
        self._log(f"{'='*60}")
        
        import time
        start_time = time.time()
        
        # 1. Build interaction matrix
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        X = self.train_matrix_csr
        
        # 2. Solve (X^TX + λI)^-1 on GPU
        from src.utils.gpu_accel import gpu_gram_solve
        dataset_name = self.config.get('dataset_name', 'unknown')
        P = gpu_gram_solve(X, self.reg_lambda, device=self.device, dataset_name=dataset_name, return_tensor=True)
        
        # 3. Compute diagonal terms on GPU
        P_diag = torch.diagonal(P).clone() # (M,)
        
        # 4. Compute μ for diagonal constraint
        # constraint_vals = 1 - P_diag * λ
        constraint_vals = 1.0 - P_diag * self.reg_lambda
        active_mask = constraint_vals > self.b
        n_active = active_mask.sum().item()
        
        mu = torch.zeros(self.n_items, device=self.device, dtype=torch.float32)
        if n_active > 0:
             mu[active_mask] = (1.0 - self.b) / P_diag[active_mask].clamp(min=1e-12) - self.reg_lambda
        
        # 5. B = I - P @ diag(λ + μ)
        # Use element-wise multiplication for diag term
        diag_term = self.reg_lambda + mu # (M,)
        B = - (P * diag_term.view(1, -1))
        B.diagonal().add_(1.0)
        
        del P
        self.weight_matrix = B
        
        elapsed = time.time() - start_time
        self._log(f"Training complete in {elapsed:.2f}s!")
        self._log(f"  - Active constraints: {n_active}/{self.n_items}")
        self._log(f"{'='*60}\n")

    def forward(self, user_ids, item_ids=None):
        if self.train_matrix_csr is None:
            raise RuntimeError("[RLAE] Model not fitted.")
        
        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = np.array(user_ids)
        
        user_input_sparse = self.train_matrix_csr[u_ids_np]
        user_input = torch.from_numpy(
            user_input_sparse.toarray()
        ).float().to(self.device)
        
        scores = user_input @ self.weight_matrix
        
        if item_ids is not None:
            if not isinstance(item_ids, torch.Tensor):
                item_ids = torch.tensor(item_ids, device=self.device)
            batch_indices = torch.arange(len(user_ids), device=self.device)
            return scores[batch_indices, item_ids]
        
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        return self.forward(user_ids, item_ids)

    def calc_loss(self, batch_data):
        return torch.tensor(0.0, device=self.device), None

    def get_final_item_embeddings(self):
        return self.weight_matrix

    def get_train_matrix(self, data_loader):
        if self.train_matrix_csr is None:
            self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        return self.train_matrix_csr