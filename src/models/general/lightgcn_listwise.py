import torch
import torch.nn as nn
from .lightgcn import LightGCN

class LightGCN_Listwise(LightGCN):
    def __init__(self, config, data_loader):
        # Override to ensure our loss_fn is initialized in our way
        super(LightGCN_Listwise, self).__init__(config, data_loader)
        
        self.num_negatives = self.config['train'].get('num_negatives', 30)
        self.topk = self.config['model'].get('topk', 10)
        self.zscore = self.config['model'].get('zscore', False)

        # Explicit Listwise Loss
        from src.loss import NDCGWeightedListwiseBPR
        self.is_explicit = self.num_negatives > 0
        self.loss_fn = NDCGWeightedListwiseBPR(k=self.topk, use_zscore=self.zscore)

    def calc_loss(self, batch_data):
        user_embeds, item_embeds = self.get_embeddings()

        users = batch_data['user_id'].squeeze()
        pos_items = batch_data['pos_item_id'].squeeze()
        
        user_vec = user_embeds[users]
        
        # Explicit Negative Sampling (Listwise on K negatives)
        if self.is_explicit:
            neg_items = batch_data['neg_item_id'] # [B, num_negatives]
            if neg_items.dim() == 1:
                neg_items = neg_items.unsqueeze(1)
            
            # Helper to get Item Embeddings from Graph Propagated Embeddings
            def get_item_vectors(item_ids):
                # item_ids: [B, N] -> Flatten -> [B*N]
                B_size, N_size = item_ids.size()
                flat_ids = item_ids.view(-1)
                flat_vecs = item_embeds[flat_ids] # [B*N, K]
                return flat_vecs.view(B_size, N_size, -1) # [B, N, K]

            # Pos Scores
            pos_vec = item_embeds[pos_items] # [B, K]
            pos_scores = (user_vec * pos_vec).sum(dim=1).unsqueeze(1) # [B, 1]
            
            # Neg Scores
            neg_vecs = get_item_vectors(neg_items) # [B, N, K]
            # user_vec: [B, K] -> [B, 1, K]
            neg_scores = (user_vec.unsqueeze(1) * neg_vecs).sum(dim=-1) # [B, N]
            
            # Stack: [Pos, Neg1, Neg2...] -> [B, 1+N]
            scores = torch.cat([pos_scores, neg_scores], dim=1)
            
        else:
            raise ValueError("LightGCN_Listwise requires num_negatives > 0 in config (Explicit Sampling)")

        loss = self.loss_fn(scores)
        
        return (loss,), None

    def __str__(self):
        return f"LightGCN_Listwise(n_layers={self.n_layers}, Negs={self.num_negatives})"
