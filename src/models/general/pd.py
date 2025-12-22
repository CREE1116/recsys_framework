import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel

class PD(BaseModel):
    """
    Personalized Diversification (개량 버전)
    - BPR + user-specific diversity
    - score = relevance + alpha_u * diversity_score
    """
    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)
        
        self.embedding_dim = config['model']['embedding_dim']
        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_dim, padding_idx=self.n_items)
        
        # User-specific diversity preference (learnable)
        self.user_diversity_pref = nn.Embedding(self.n_users, 1)
        nn.init.constant_(self.user_diversity_pref.weight, 0.5)
        
        self._init_history_tensor()
        self._init_weights()
    
    def _init_history_tensor(self):
        """Precompute padded user history"""
        user_history = self.data_loader.user_history
        if not user_history:
            # 빈 history인 경우 더미 텐서
            self.register_buffer('history_tensor', torch.full((self.n_users, 1), self.n_items, dtype=torch.long))
            return
        max_len = max(len(items) for items in user_history.values()) if user_history else 1
        max_len = max(max_len, 1)  # 최소 1
        history_tensor = torch.full((self.n_users, max_len), self.n_items, dtype=torch.long)
        for u in range(self.n_users):
            items = user_history.get(u, set())
            if items:
                l = len(items)
                history_tensor[u, :l] = torch.tensor(list(items), dtype=torch.long)
        self.register_buffer('history_tensor', history_tensor)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        with torch.no_grad():
            self.item_embedding.weight[self.n_items] = 0
    
    def forward(self, users):
        """
        Compute score = relevance + alpha * diversity
        """
        u_emb = self.user_embedding(users)                  # [B, D]
        i_emb_all = self.item_embedding.weight[:self.n_items] # [N, D]
        
        # Relevance
        relevance = torch.matmul(u_emb, i_emb_all.T)       # [B, N]
        
        # Diversity
        diversity_score = self.compute_diversity(users)   # [B, N]
        
        # Normalize diversity to zero mean, unit std per user
        diversity_score = (diversity_score - diversity_score.mean(dim=1, keepdim=True)) / \
                          (diversity_score.std(dim=1, keepdim=True) + 1e-8)
        
        # Alpha
        alpha = torch.sigmoid(self.user_diversity_pref(users))  # [B,1]
        
        score = relevance + alpha * diversity_score           # [B, N]
        return score
    
    def compute_diversity(self, users):
        """Negative similarity to user history"""
        hist_idx = self.history_tensor[users]               # [B, L]
        hist_embs = self.item_embedding(hist_idx)           # [B, L, D]
        mask = (hist_idx != self.n_items).float().unsqueeze(2) # [B,L,1]
        mean_hist_emb = (hist_embs * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)  # [B,D]
        sim = torch.matmul(mean_hist_emb, self.item_embedding.weight[:self.n_items].T)        # [B,N]
        return -sim
    
    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)
        device = users.device
        
        # Full score
        scores = self.forward(users)                        # [B,N]
        
        batch_idx = torch.arange(users.size(0), device=device)
        pos_score = scores[batch_idx, pos_items]
        neg_score = scores[batch_idx, neg_items]
        
        # BPR loss
        bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
        
        # Alpha regularization (prevent collapse)
        alpha = torch.sigmoid(self.user_diversity_pref(users)).squeeze()
        div_reg = (alpha - 0.5).pow(2).mean()
        
        # Optional: diversity variance regularization (stabilize scale)
        diversity_score = self.compute_diversity(users)
        diversity_score = (diversity_score - diversity_score.mean(dim=1, keepdim=True)) / \
                          (diversity_score.std(dim=1, keepdim=True) + 1e-8)
        div_var_reg = (diversity_score.var(dim=1).mean() - 1.0).pow(2)
        
        
        return (bpr_loss , 0.01 * div_reg , 0.01 * div_var_reg,), {
            'bpr': bpr_loss.item(),
            'div_reg': div_reg.item(),
            'div_var_reg': div_var_reg.item(),
            'avg_alpha': alpha.mean().item()
        }
    
    def predict_for_pairs(self, user_ids, item_ids):
        """
        Predict for specific user-item pairs
        """
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        
        # Alpha
        alpha = torch.sigmoid(self.user_diversity_pref(user_ids)).squeeze()
        
        # Diversity score: negative similarity to user history
        hist_idx = self.history_tensor[user_ids]
        hist_embs = self.item_embedding(hist_idx)
        mask = (hist_idx != self.n_items).float().unsqueeze(2)
        mean_hist_emb = (hist_embs * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        sim = (i_emb * mean_hist_emb).sum(dim=1)
        diversity_score = -sim
        
        # Normalize
        diversity_score = (diversity_score - diversity_score.mean()) / (diversity_score.std() + 1e-8)
        
        return (u_emb * i_emb).sum(dim=1) + alpha * diversity_score
    
    def get_final_item_embeddings(self):
        return self.item_embedding.weight[:self.n_items].detach()
