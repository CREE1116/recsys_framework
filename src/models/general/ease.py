import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from ..base_model import BaseModel

class EASE(BaseModel):
    """
    EASE (Embarrassingly Shallow Autoencoders for Sparse Data)
    - Closed-form solution: B = (X^T X + lambda I)^-1 X^T X
    - No gradient descent training needed.
    """
    def __init__(self, config, data_loader):
        super(EASE, self).__init__(config, data_loader)
        
        self.reg_lambda = self.config['model'].get('reg_lambda', 500.0)
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        
        # Learned Weight Matrix (Items x Items)
        # We store it as a buffer or parameter, but calculate it in 'fit'
        self.register_buffer('weight_matrix', torch.zeros(self.n_items, self.n_items))
        
    def fit(self, data_loader):
        """
        Closed-form update for EASE.
        Requires constructing the full interaction matrix X.
        """
        print(f"Fitting EASE with lambda={self.reg_lambda}...")
        
        # 1. Build Sparse Interaction Matrix X
        # data_loader.train_df contains [user_id, item_id]
        train_df = data_loader.train_df
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        values = np.ones(len(rows), dtype=np.float32)
        
        X = sp.csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items))
        
        # 2. Calculate Gram Matrix G = X^T X
        # Result is (n_items x n_items) - Dense if n_items is small, but can be large
        # For MovieLens/Amazon, n_items can be 3k~10k, so (10k x 10k) matrix is ~400MB float32 (Manageable).
        # We assume n_items fits in memory.
        G = X.transpose().dot(X).toarray()
        
        # 3. Add Lambda to Diagonal
        diag_indices = np.diag_indices(self.n_items)
        G[diag_indices] += self.reg_lambda
        
        # 4. Invert P = G^-1
        print("Inverting Gram matrix...")
        P = np.linalg.inv(G)
        
        # 5. Calculate B = I - P * diag(P)^-1
        # B_ij = - P_ij / P_jj if i != j else 0
        B = P / (-np.diag(P))
        B[diag_indices] = 0.0
        
        # 6. Store B
        self.weight_matrix.copy_(torch.from_numpy(B).float())
        print("EASE fitting complete.")

    def forward(self, users):
        # EASE predicts for ONE user based on their history vector x_u
        # Score_u = x_u * B
        
        # 1. Construct input vector x_u for the batch of users
        # This is expensive to do on-the-fly if history is long.
        # But EASE is fast.
        
        # We need the user's history from training data
        batch_size = users.size(0)
        x_u = torch.zeros(batch_size, self.n_items, device=users.device)
        
        users_list = users.cpu().numpy()
        for i, u_id in enumerate(users_list):
            hist_items = list(self.data_loader.user_history.get(u_id, []))
            x_u[i, hist_items] = 1.0
            
        # 2. Multiply with B
        scores = torch.matmul(x_u, self.weight_matrix.to(users.device))
        
        return scores

    def calc_loss(self, batch_data):
        # EASE is not trained via SGD
        return torch.tensor(0.0), {}

    def predict_for_pairs(self, user_ids, item_ids):
        # Not typically efficient for EASE, but implemented for compatibility
        scores = self.forward(user_ids) # [B, N_items]
        # Gather specific item scores
        batch_indices = torch.arange(len(user_ids), device=user_ids.device)
        return scores[batch_indices, item_ids]

    def get_final_item_embeddings(self):
        # EASE doesn't have "embeddings", it has Item-Item weights.
        # We can return the weight matrix itself effectively acting as embeddings context
        return self.weight_matrix
