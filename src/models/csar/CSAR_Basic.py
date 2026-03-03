import torch
import torch.nn as nn
from src.loss import BPRLoss, SampledSoftmaxLoss
from ..base_model import BaseModel
from .csar_layers import CSAR_basic
import numpy as np

class CSAR_Basic(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR_Basic, self).__init__(config, data_loader)
        
        self.embedding_dim = self.config['model']['embedding_dim']
        if isinstance(self.embedding_dim, list):
            self.embedding_dim = self.embedding_dim[0]
            
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
            
        self.num_interests = self.config['model']['num_interests']
        if isinstance(self.num_interests, list):
            self.num_interests = self.num_interests[0]
        
        self.reg_lambda = self.config['model'].get('reg_lambda', 500.0)
        if isinstance(self.reg_lambda, (list, np.ndarray)):
            self.reg_lambda = self.reg_lambda[0]
        self.align_weight = self.config['model'].get('align_weight', 0.0)
        self.ema_momentum = self.config['model'].get('ema_momentum', 1.0) # Default 1.0 = no EMA
        
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # CSAR Basic Layer (Normalize=True hardcoded)
        self.model_layer = CSAR_basic(self.num_interests, self.embedding_dim, reg_lambda=self.reg_lambda, normalize=True)
        
        self._init_weights()
        
        # Ridge mode related (Always active)
        self.X = self._prepare_interaction_matrix()
        # Initialize G as Identity for stability from epoch 0
        self.register_buffer('_cached_G', torch.eye(self.num_interests))
        self.register_buffer('_cached_d_inv_sqrt', torch.ones(self.num_interests))
        self.register_buffer('g_stability', torch.tensor(0.0))

        # Loss
        self.ols_eps = self.config['model'].get('ols_eps', 1e-4)
        if isinstance(self.ols_eps, (list, np.ndarray)):
            self.ols_eps = self.ols_eps[0]
        self.loss_type = self.config['train'].get('loss_type', 'bpr')
        if self.loss_type == 'sampled_softmax':
            self.loss_fn = SampledSoftmaxLoss(temperature=0.1)
        else:
            self.loss_fn = BPRLoss()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _prepare_interaction_matrix(self):
        """Build symmetrically normalized interaction matrix as a Sparse Tensor (stays Sparse)"""
        train_df = self.data_loader.train_df
        rows = torch.tensor(train_df['user_id'].values, dtype=torch.long)
        cols = torch.tensor(train_df['item_id'].values, dtype=torch.long)
        vals = torch.ones(len(train_df), dtype=torch.float32)
        X = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (self.n_users, self.n_items))
        
        X = X.coalesce()
        
        # Symmetrically normalize X: D_u^-0.5 * X * D_i^-0.5
        user_deg = torch.sparse.sum(X, dim=1).to_dense()
        item_deg = torch.sparse.sum(X, dim=0).to_dense()

        def inv_sqrt(d):
            res = torch.zeros_like(d)
            mask = d > 0
            res[mask] = torch.pow(d[mask], -0.5)
            return res

        d_u_inv = inv_sqrt(user_deg)
        d_i_inv = inv_sqrt(item_deg)

        indices = X.indices()
        v = X.values()
        v_norm = d_u_inv[indices[0]] * v * d_i_inv[indices[1]]
        
        X_norm = torch.sparse_coo_tensor(indices, v_norm, (self.n_users, self.n_items))
        return X_norm.cpu().coalesce() # Keep on CPU for sparse operations support

    def on_epoch_start(self, epoch):
        """Update G via Scalable Unified Ridge LS (v5.0) + Update Stability"""
        with torch.no_grad():
            user_emb = self.user_embedding.weight
            item_emb = self.item_embedding.weight
            
            # Compute current memberships
            M_u = self.model_layer.get_membership(user_emb)
            M_i = self.model_layer.get_membership(item_emb)
            
            # Compute new G via Unified Ridge LS (CPU fallback handles Sparse X)
            new_G, _ = self.model_layer.get_gram_matrix(
                mode='unified', M_u=M_u, M_i=M_i, X=self.X
            )
            new_G = new_G.to(self.device)
            
            # Update Stability Score (v5.5 Dynamic Alignment)
            # diff = ||G_new - G_old||_F
            diff = torch.norm(new_G - self._cached_G, p='fro')
            # stability = exp(-diff). Higher = more stable.
            # We use a slow EMA to smooth the stability score
            current_stability = torch.exp(-diff).to(self.device)
            if epoch == 0:
                self.g_stability = current_stability
            else:
                alpha = 0.9 # Stability EMA
                self.g_stability = alpha * self.g_stability + (1 - alpha) * current_stability
            
            # Apply EMA for G
            m = self.ema_momentum
            self._cached_G = m * self._cached_G + (1 - m) * new_G
            
        self._log(f"G (Unified-Ridge) updated (EMA m={self.ema_momentum}, stability={self.g_stability.item():.4f}) at epoch {epoch}")

    def _get_target_g(self):
        """Returns the data-driven Ridge matrix G (teacher target)"""
        return self._cached_G, self._cached_d_inv_sqrt

    def forward(self, users):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding.weight
        
        # Inference uses learned PP^T propagation
        user_mem = self.model_layer.get_membership(user_emb)
        item_mem = self.model_layer.get_membership(item_emb)
        
        G_learned = self.model_layer.get_propagation_matrix()
        
        # Score = (User_Mem @ G_learned) @ Item_Mem.T
        user_prop = torch.matmul(user_mem, G_learned)
        scores = torch.matmul(user_prop, item_mem.t())
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Inference uses learned PP^T propagation
        user_mem = self.model_layer.get_membership(user_emb)
        item_mem = self.model_layer.get_membership(item_emb)
        
        G_learned = self.model_layer.get_propagation_matrix()
        
        # Score = (User_Mem @ G_learned) * Item_Mem -> sum over K
        user_prop = torch.matmul(user_mem, G_learned)
        scores = (user_prop * item_mem).sum(dim=-1)
        return scores

    def calc_loss(self, batch_data):
        users = batch_data['user_id']
        pos_items = batch_data['pos_item_id']
        neg_items = batch_data['neg_item_id']
        
        pos_scores = self.predict_for_pairs(users, pos_items)
        neg_scores = self.predict_for_pairs(users, neg_items)
        
        bpr_loss = self.loss_fn(pos_scores, neg_scores)
        
        # Dynamic Alignment Loss (v5.5)
        # weight scales with G stability
        align_loss = torch.tensor(0.0, device=self.device)
        dynamic_align_weight = self.align_weight * self.g_stability.item()
        
        if dynamic_align_weight > 0:
            G_target, _ = self._get_target_g()
            align_loss = self.model_layer.get_alignment_loss(G_target)
        
        # Return tuple: (Loss-to-optimize, Additional-loss-to-log)
        return (bpr_loss, dynamic_align_weight * align_loss), {
            "scale": self.model_layer.scale.item(),
            "g_stability": self.g_stability.item(),
            "dynamic_align_weight": dynamic_align_weight
        }

    def get_final_item_embeddings(self):
        return self.item_embedding.weight.detach()

    def get_interest_keys(self):
        return self.model_layer.interest_keys.detach()
