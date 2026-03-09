import torch
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel

class CoOccurrence(BaseModel):
    """
    Simple Count-based Item-Item Similarity Model.
    S = X^T X (Co-occurrence Matrix)
    
    Supports normalization:
    - 'cosine': S_ij = C_ij / sqrt(C_ii * C_jj)
    - 'jaccard': S_ij = C_ij / (C_ii + C_jj - C_ij)
    - 'none': S_ij = C_ij
    """
    def __init__(self, config, data_loader):
        super(CoOccurrence, self).__init__(config, data_loader)
        self.metric = config['model'].get('similarity_metric', 'cosine')
        self.normalize = config['model'].get('normalize', True)
        
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        # Similarity matrix buffer
        self.register_buffer('similarity_matrix', torch.zeros(self.n_items, self.n_items))
        self.train_matrix_csr = None

    def fit(self, data_loader):
        self._log(f"Fitting CoOccurrence model on {self.device} (metric={self.metric})...")
        
        # 1. Construct interaction matrix
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df), dtype=np.float32)
        
        X_sp = sp.csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix_csr = X_sp
        
        # 2. Compute Co-occurrence G = X^T X on GPU
        dev = self.device
        
        # Convert to torch sparse COO for GPU matmul
        X_coo = X_sp.tocoo()
        indices = torch.from_numpy(np.vstack((X_coo.row, X_coo.col))).long()
        values = torch.from_numpy(X_coo.data).float()
        X_t = torch.sparse_coo_tensor(indices, values, X_sp.shape, device=dev).coalesce()
        
        self._log(f"Computing Gram matrix G = X.T @ X on {dev}...")
        # G = X.T @ X
        if dev.type == 'mps':
            # MPS does not support torch.sparse.mm; fall back to dense matmul
            X_dense = X_t.to_dense()
            G = torch.mm(X_dense.t(), X_dense)
            del X_dense
        else:
            G = torch.sparse.mm(X_t.t(), X_t.to_dense())  # Returns dense
        del X_t, X_coo
        
        # 3. Normalize on GPU
        diag = torch.diagonal(G).clone()
        
        if self.metric == 'cosine':
            sqrt_diag = torch.sqrt(diag).clamp(min=1e-12)
            # S = G / (sqrt_diag_i * sqrt_diag_j)
            S = G / (sqrt_diag.view(-1, 1) * sqrt_diag.view(1, -1))
        elif self.metric == 'jaccard':
            # S = G / (G_ii + G_jj - G_ij)
            D = diag.view(-1, 1) + diag.view(1, -1)
            denominator = (D - G).clamp(min=1e-12)
            S = G / denominator
        else:
            S = G
            if self.normalize:
                m = S.max()
                if m > 0: S /= m

        # 4. Zero out diagonal
        S.fill_diagonal_(0.0)
        
        # 5. Store
        self.similarity_matrix = S
        self._log("CoOccurrence model fitted on GPU.")

    def forward(self, user_ids, item_ids=None):
        if self.train_matrix_csr is None:
             raise RuntimeError("Model not fitted yet.")
             
        # Slice user history
        if isinstance(user_ids, torch.Tensor):
            u_ids_np = user_ids.cpu().numpy()
        else:
            u_ids_np = user_ids
            
        user_input_sparse = self.train_matrix_csr[u_ids_np]
        user_input = torch.from_numpy(user_input_sparse.toarray()).float().to(self.device)
        
        # Predict: User_Vector @ Sim_Matrix
        # [B, I] @ [I, I] -> [B, I]
        scores = user_input @ self.similarity_matrix
        
        return scores
        
    def predict_for_pairs(self, user_ids, item_ids):
        # Predict scores for specific (user, item) pairs
        # user_ids: [N], item_ids: [N]
        scores = self.forward(user_ids) # [N, n_items]
        # Gather specific item scores
        batch_indices = torch.arange(len(user_ids), device=self.device)
        return scores[batch_indices, item_ids]
        
    def calc_loss(self, batch_data):
        # CoOccurrence is count-based, no gradient training.
        return (torch.tensor(0.0, device=self.device),), None

    def get_final_item_embeddings(self):
        return self.similarity_matrix
