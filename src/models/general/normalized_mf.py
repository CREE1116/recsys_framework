import torch
import torch.nn as nn
from ..base_model import BaseModel
from src.loss import NormalizedSampledSoftmaxLoss

class NormalizedMF(BaseModel):
    """
    Normalized Matrix Factorization (NormalizedMF)
    
    Standard Matrix Factorization architecture trained with 
    Normalized Sampled Softmax Loss (Z-Score + Temperature + Correction).
    """
    def __init__(self, config, data_loader):
        super(NormalizedMF, self).__init__(config, data_loader)
        
        self.embedding_dim = config['model']['embedding_dim']
        self.temperature = config['model'].get('temperature', 0.1)
        self.use_zscore = config['model'].get('use_zscore', False)
        
        # Check for explicit negative sampling
        self.num_negatives = config['train'].get('num_negatives', 0)
        self.is_explicit = self.num_negatives > 0
        
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        self.loss_fn = NormalizedSampledSoftmaxLoss(
            self.data_loader.n_items, 
            temperature=self.temperature
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding.weight
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores
    
    def get_final_item_embeddings(self):
        return self.item_embedding.weight.detach()

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        
        user_emb = self.user_embedding(users) # [B, D]
        pos_item_emb = self.item_embedding(pos_items) # [B, D]
        
        if self.is_explicit:
            # Explicit High-Hard Negatives (e.g. Popularity Sampling)
            neg_items = batch_data['neg_item_id'] # [B, num_negatives]
            
            # Pos Scores: [B, 1]
            pos_scores = (user_emb * pos_item_emb).sum(dim=-1, keepdim=True)
            
            # Neg Scores: [B, N]
            # Flatten for embedding lookup
            B, N = neg_items.size()
            flat_neg_items = neg_items.view(-1)
            flat_neg_emb = self.item_embedding(flat_neg_items) # [B*N, D]
            neg_emb = flat_neg_emb.view(B, N, -1) # [B, N, D]
            
            # Dot Product with broadcasting
            # user_emb: [B, 1, D]
            neg_scores = (user_emb.unsqueeze(1) * neg_emb).sum(dim=-1) # [B, N]
            
            # Concat for Loss: [Pos, Negs] -> [B, 1+N]
            scores = torch.cat([pos_scores, neg_scores], dim=1)
            
            loss = self.loss_fn(scores, is_explicit=True)
            
        else:
            # Implicit In-Batch Negatives
            # Calculate In-Batch Scores [B, B]
            scores = torch.matmul(user_emb, pos_item_emb.t())
            
            # Calculate Loss (Implicit In-Batch Negatives)
            loss = self.loss_fn(scores, is_explicit=False)
        
        return (loss,), None

    def __str__(self):
        return f"NormalizedMF(embedding_dim={self.embedding_dim}, temp={self.temperature}, zscore={self.use_zscore})"
