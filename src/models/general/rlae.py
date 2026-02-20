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
        
        self.register_buffer('weight_matrix', torch.zeros(self.n_items, self.n_items))
        self.train_matrix_csr = None
        
        print(f"[RLAE] Initialized: λ={self.reg_lambda}, b={self.b}")

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
        print(f"\n{'='*60}")
        print(f"[RLAE] Training with b={self.b}, λ={self.reg_lambda}")
        print(f"{'='*60}")
        
        import time
        start_time = time.time()
        
        # 1. Build interaction matrix
        self.train_matrix_csr = self._build_sparse_matrix(data_loader)
        X = self.train_matrix_csr
        
        # 2. Compute Gram matrix
        G = X.T.dot(X).toarray()  # (M × M)
        G_tensor = torch.from_numpy(G).float().to(self.device)
        
        # 3. Compute P = (G + λI)^{-1}
        I = torch.eye(self.n_items, device=self.device)
        try:
            P = torch.inverse(G_tensor + self.reg_lambda * I)
        except (RuntimeError, NotImplementedError):
            print("[RLAE] MPS inverse failed, fallback to CPU...")
            P = torch.inverse(G_tensor.cpu() + self.reg_lambda * I.cpu()).to(self.device)
        
        # 4. Compute μ for each diagonal element
        μ = torch.zeros(self.n_items, device=self.device)
        P_diag = torch.diag(P)
        
        n_active = 0
        # [Vectorized] Constraint check
        constraint_vals = 1 - P_diag * self.reg_lambda
        # Active if constraint_val > b
        active_mask = constraint_vals > self.b
        
        # μ = (1 - b)/P_ii - λ if active, else 0
        μ[active_mask] = (1 - self.b) / P_diag[active_mask] - self.reg_lambda
        n_active = active_mask.sum().item()
        
        # 5. Compute B = I - P @ diag(λ + μ)
        diag_term = self.reg_lambda + μ
        B = I - P @ torch.diag(diag_term)
        
        # 6. Store
        self.weight_matrix.copy_(B)
        
        elapsed = time.time() - start_time
        
        # Statistics
        diag_vals = torch.diag(B).cpu().numpy()
        
        print(f"\n{'='*60}")
        print(f"[RLAE] Training complete!")
        print(f"  - Time: {elapsed:.2f}s")
        print(f"  - Active constraints: {n_active}/{self.n_items}")
        print(f"  - Diagonal stats:")
        print(f"    * Min: {diag_vals.min():.4f}")
        print(f"    * Max: {diag_vals.max():.4f}")
        print(f"    * Mean: {diag_vals.mean():.4f}")
        print(f"  - Constraint: diag(B) ≤ {self.b}")
        print(f"  - Max violation: {max(0, diag_vals.max() - self.b):.6f}")
        print(f"{'='*60}\n")

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