import torch
import torch.nn as nn
from ..base_model import BaseModel

class MF_Listwise(BaseModel):
    def __init__(self, config, data_loader):
        super(MF_Listwise, self).__init__(config, data_loader)
        
        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_negatives = self.config['train'].get('num_negatives', 30)
        self.topk = self.config['model'].get('topk', 10)
        self.zscore = self.config['model'].get('zscore', False)

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)

        self._init_weights()
        
        # Explicit Listwise Loss
        from src.loss import NDCGWeightedListwiseBPR
        self.is_explicit = self.num_negatives > 0
        self.loss_fn = NDCGWeightedListwiseBPR(k=self.topk, use_zscore=self.zscore, is_explicit=self.is_explicit)

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        return (user_emb * item_emb).sum(dim=1)

    def predict_for_pairs(self, user_ids, item_ids):
        # same as forward but matching signature
        return self.forward(user_ids, item_ids)
        
    def get_final_item_embeddings(self):
         return self.item_embedding.weight.detach()

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)

        user_emb = self.user_embedding(users)
        
        # Explicit Negative Sampling (Listwise on K negatives)
        if self.is_explicit:
            neg_items = batch_data['neg_item_id'] # [B, num_negatives]
            if neg_items.dim() == 1:
                neg_items = neg_items.unsqueeze(1)
            
            # Helper to get Item Embeddings
            def get_item_embeddings(item_ids):
                # item_ids: [B, N] -> Flatten -> [B*N]
                B_size, N_size = item_ids.size()
                flat_ids = item_ids.view(-1)
                flat_embs = self.item_embedding(flat_ids) # [B*N, K]
                return flat_embs.view(B_size, N_size, -1) # [B, N, K]

            # Pos Scores
            pos_item_emb = self.item_embedding(pos_items) # [B, K]
            pos_scores = (user_emb * pos_item_emb).sum(dim=1).unsqueeze(1) # [B, 1]
            
            # Neg Scores
            neg_item_embs = get_item_embeddings(neg_items) # [B, N, K]
            # user_emb: [B, K] -> [B, 1, K]
            neg_scores = (user_emb.unsqueeze(1) * neg_item_embs).sum(dim=-1) # [B, N]
            
            # Stack: [Pos, Neg1, Neg2...] -> [B, 1+N]
            scores = torch.cat([pos_scores, neg_scores], dim=1)
            
        else:
            # Fallback to single negative or in-batch if num_negatives=0 (unlikely for this model)
            # Just mimicking standard MF BPR logic for structure but we enforce explicit listwise
            raise ValueError("MF_Listwise requires num_negatives > 0 in config (Explicit Sampling)")

        loss = self.loss_fn(scores)
        
        return (loss,), {}

    def __str__(self):
        return f"MF_Listwise(dim={self.embedding_dim}, Negs={self.num_negatives})"
