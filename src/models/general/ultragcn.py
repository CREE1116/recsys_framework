import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from ..base_model import BaseModel
from tqdm import tqdm

class UltraGCN(BaseModel):
    """
    UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation
    Model: Standard Matrix Factorization (User/Item Embeddings)
    Training: 
        L = L_main + lambda * L_C + gamma * L_I
        L_main: BCE/BPR Loss
        L_C: User-Item Constraint (Weighted MSE align)
        L_I: Item-Item Constraint (Weighted MSE align)
    """
    def __init__(self, config, data_loader):
        super(UltraGCN, self).__init__(config, data_loader)
        
        self.embedding_dim = config['model'].get('embedding_dim', config['model'].get('embedding_size', 64))
        
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        
        # Hyperparameters
        self.w1 = float(config['model'].get('w1', 1e-4)) # L_C weight
        self.w2 = float(config['model'].get('w2', 1e-4)) # L_I weight
        self.w3 = float(config['model'].get('w3', 1e-4)) # L2 reg
        self.w4 = float(config['model'].get('w4', 1e-4)) # Negative log-likelihood weight (if using BCE)
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Init weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        self.ii_neighbor_mat = None # Item-Item Neighbor Matrix (Sparse)
        self.ii_constraint_k = config['model'].get('ii_neighbor_num', 10)
        
        self._log(f"Initialized (w1={self.w1}, w2={self.w2}, w3={self.w3}, w4={self.w4})")

    def fit(self, data_loader):
        if self.ii_neighbor_mat is not None: return
        
        self._log("Constructing Item-Item Similarity Graph for Constraint Loss...")
        
        try:
            # 1. User-Item Graph
            train_df = data_loader.train_df
            rows = train_df['user_id'].values
            cols = train_df['item_id'].values
            values = np.ones(len(train_df), dtype=np.float32)
            
            R = sp.csr_matrix((values, (rows, cols)), shape=(self.n_users, self.n_items))
            
            # 2. Item-Item Co-occurrence: G = R^T @ R
            # Warning: For very large datasets, this might be slow/heavy.
            # But standard UltraGCN does this.
            # Use dot product.
            G = R.transpose().dot(R)
            
            # 3. Get Top-K Neighbors for each item
            # We want to keep G sparse.
            # G is (N, N).
            
            # Diagonal zero
            G.setdiag(0)
            G.eliminate_zeros()
            
            # Normalize (Cosine) - Optional but good for weights
            # To save memory, we might just use raw counts or simplified weights.
            # UltraGCN implementation often uses pre-computed "Item-Item" file.
            # Here we compute on the fly.
            
            # Convert to Lil for efficient row slicing or just use CSR
            # To get Top-K efficiently:
            # For each row, pick Top-K.
            
            ii_rows = []
            ii_cols = []
            ii_vals = []
            
            # If N_items is small (<10k), dense is fine.
            # If N_items is large, iterate.
            if self.n_items <= 10000:
                G_dense = G.toarray().astype(np.float32)
                # Move to device first so topk runs on GPU/MPS
                G_tensor = torch.from_numpy(G_dense).to(self.device)
                vals, indices = torch.topk(G_tensor, self.ii_constraint_k, dim=1)
                del G_tensor

                self.ii_neighbors = indices.long()   # [N_items, K], already on self.device
                self.ii_weights = vals.float()        # [N_items, K], already on self.device
                
            else:
                # Large scale: Iterate rows of sparse matrix
                # This could be slow in Python.
                # Simplified: Just sample uniformly from co-occurring items?
                # No, UltraGCN needs specific neighbors.
                
                self._log("Large item set detected. Computing Top-K row by row...")
                neighbors = np.zeros((self.n_items, self.ii_constraint_k), dtype=np.int64)
                weights = np.zeros((self.n_items, self.ii_constraint_k), dtype=np.float32)
                
                for i in tqdm(range(self.n_items)):
                    row = G.getrow(i)
                    if row.nnz > 0:
                        # get indices and data
                        idx = row.indices
                        dat = row.data.astype(np.float32)
                        
                        if len(dat) > self.ii_constraint_k:
                            # partial sort
                            top_k_idx = np.argpartition(dat, -self.ii_constraint_k)[-self.ii_constraint_k:]
                            neighbors[i] = idx[top_k_idx]
                            weights[i] = dat[top_k_idx]
                        else:
                            # Pad with first neighbor or self?
                            # Just fill what we have
                            length = len(dat)
                            neighbors[i, :length] = idx
                            weights[i, :length] = dat
                            # Pad rest with i (self loop with 0 weight)
                            neighbors[i, length:] = i
                
                self.ii_neighbors = torch.from_numpy(neighbors).to(self.device).long()
                self.ii_weights = torch.from_numpy(weights).to(self.device).float()
                
            # Normalize weights locally
            row_sum = self.ii_weights.sum(dim=1, keepdim=True) + 1e-9
            self.ii_weights /= row_sum
            
            self._log("Graph construction complete.")
            
        except Exception as e:
            self._log(f"Error building graph: {e}. Disabling Item-Item Constraint.")
            self.ii_neighbors = None

    def forward(self, user_ids, item_ids=None):
        u_emb = self.user_embedding(user_ids)
        if item_ids is not None:
            i_emb = self.item_embedding(item_ids)
            return torch.sum(u_emb * i_emb, dim=1)
        else:
            # All items
            return u_emb @ self.item_embedding.weight.t()

    def predict_for_pairs(self, user_ids, item_ids):
        # UltraGCN forward already supports item_ids
        return self.forward(user_ids, item_ids)

    def calc_loss(self, batch_data):
        user_ids = batch_data['user_id'].squeeze()
        pos_items = batch_data['pos_item_id'].squeeze()
        neg_items = batch_data['neg_item_id'].squeeze() 
        
        u_emb = self.user_embedding(user_ids)
        pos_emb = self.item_embedding(pos_items)
        
        # 1. Main Recommendation Loss (Log Sigmoid)
        pos_score = torch.sum(u_emb * pos_emb, dim=1)
        pos_loss = -F.logsigmoid(pos_score).mean()
        
        neg_loss = 0
        if neg_items.dim() == 2:
            neg_emb = self.item_embedding(neg_items) # [B, Neg, D]
            u_emb_expanded = u_emb.unsqueeze(1)      # [B, 1, D]
            neg_score = torch.sum(u_emb_expanded * neg_emb, dim=2) # [B, Neg]
            neg_loss = -F.logsigmoid(-neg_score).mean()
        else:
            neg_emb = self.item_embedding(neg_items)
            neg_score = torch.sum(u_emb * neg_emb, dim=1)
            neg_loss = -F.logsigmoid(-neg_score).mean()
            
        main_loss = pos_loss + neg_loss
        
        # 2. Constraint Loss L_C (User-Item)
        l_c = F.mse_loss(u_emb, pos_emb, reduction='mean')
        
        # 3. Item-Item Constraint Loss L_I
        l_i = torch.tensor(0.0, device=self.device)
        if self.ii_neighbors is not None:
            rand_idx = torch.randint(0, self.ii_constraint_k, (len(pos_items),), device=self.device)
            neighbor_ids = self.ii_neighbors[pos_items, rand_idx]
            neighbor_weights = self.ii_weights[pos_items, rand_idx]
            neighbor_emb = self.item_embedding(neighbor_ids)
            dist_sq = torch.sum((pos_emb - neighbor_emb)**2, dim=1)
            l_i = (neighbor_weights * dist_sq).mean()
            
        # 4. L2 Regularization (Use framework standard helper)
        l2_reg = self.get_l2_reg_loss(u_emb, pos_emb, neg_emb if neg_items.dim() != 2 else neg_emb.view(-1, self.embedding_dim))
        
        return (main_loss, self.w1 * l_c, self.w2 * l_i, l2_reg), {
            'loss_main': main_loss.item(), 
            'loss_lc': l_c.item(), 
            'loss_li': l_i.item(), 
            'loss_l2': l2_reg.item()
        }
        
    def get_final_item_embeddings(self):
        return self.item_embedding.weight
