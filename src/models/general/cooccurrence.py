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
        self._log(f"Fitting CoOccurrence model (metric={self.metric})...")
        
        # 1. Construct Sparse Interaction Matrix X
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(train_df))
        
        X = sp.csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items), dtype=np.float32)
        self.train_matrix_csr = X
        
        # 2. Compute Co-occurrence G = X^T X
        # shape: (n_items, n_items)
        G = X.transpose().dot(X).toarray()
        
        # Diagonal elements (item popularity / self-occurrence)
        diag = np.diag(G).copy()
        
        # 3. Normalize
        if self.metric == 'cosine':
            # Cosine: G_ij / sqrt(G_ii * G_jj)
            # sqrt_diag = G_ii^0.5
            sqrt_diag = np.sqrt(diag)
            # Avoid division by zero
            sqrt_diag[sqrt_diag == 0] = 1.0
            
            # Outer product for denominator: D_ij = sqrt(G_ii) * sqrt(G_jj)
            denominator = np.outer(sqrt_diag, sqrt_diag)
            S = G / denominator
            
        elif self.metric == 'jaccard':
            # Jaccard: G_ij / (G_ii + G_jj - G_ij)
            # D_ij = G_ii + G_jj
            D = np.add.outer(diag, diag)
            denominator = D - G
            # Avoid division by zero
            denominator[denominator == 0] = 1.0
            S = G / denominator
            
        else: # 'none' or 'raw'
            S = G
            if self.normalize:
               # Simple max normalization if requested but no metric specific
               m = S.max()
               if m > 0: S /= m

        # 4. Zero out diagonal (usually we don't recommend the item itself based on itself)
        np.fill_diagonal(S, 0.0)
        
        # 5. Store
        self.similarity_matrix.copy_(torch.from_numpy(S).float())
        self._log("CoOccurrence model fitted.")

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
        return (torch.tensor(0.0, device=self.device, requires_grad=True),), None

    def get_final_item_embeddings(self):
        return self.similarity_matrix
