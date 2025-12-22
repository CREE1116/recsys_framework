import torch
import torch.nn as nn
import torch.nn.functional as F
from .CSAR_Listwise import CSAR_Listwise
from src.loss import NDCGWeightedListwiseBPR

class CSAR_R_Listwise(CSAR_Listwise):
    """
    CSAR_R_Listwise: CSAR_Listwise + Residual Connection
    Uses NDCGWeightedListwiseBPR with precomputed residual scores.
    """
    def __init__(self, config, data_loader):
        super(CSAR_R_Listwise, self).__init__(config, data_loader)

        # CSAR_Listwise Loss Params
        self.topk = self.config['model'].get('topk', 10)
        self.zscore = self.config['model'].get('zscore', False)

        # NDCGWeightedListwiseBPR Loss
        # NDCGWeightedListwiseBPR Loss
        self.num_negatives = self.config['train'].get('num_negatives', 1)
        self.is_explicit = self.num_negatives > 0
        self.loss_fn = NDCGWeightedListwiseBPR(k=self.topk, use_zscore=self.zscore, is_explicit=self.is_explicit)

    def forward(self, users):
        user_embs = self.user_embedding(users)
        all_item_embs = self.item_embedding.weight
        
        # Embedding Dropout (Training only)
        if self.training and self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)
        
        # 1. Interest Score
        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(all_item_embs)
        interest_scores = torch.einsum('bk,nk->bn', user_interests, item_interests)
        
        # 2. Residual Score
        res_scores = torch.matmul(user_embs, all_item_embs.t())
        
        return interest_scores + res_scores

    def predict_for_pairs(self, user_ids, item_ids):
        user_embs = self.user_embedding(user_ids)
        item_embs = self.item_embedding(item_ids)
        
        # Embedding Dropout (Training only)
        if self.training and self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)
            item_embs = F.dropout(item_embs, p=self.emb_dropout, training=True)

        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(item_embs)

        interest_scores = (user_interests * item_interests).sum(dim=-1)
        res_scores = (user_embs * item_embs).sum(dim=-1)
        
        return interest_scores + res_scores

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        
        user_embs = self.user_embedding(users)
        pos_item_embs = self.item_embedding(pos_items)
        
        # Embedding Dropout (Training only)
        if self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)
            pos_item_embs = F.dropout(pos_item_embs, p=self.emb_dropout, training=True)
        
        user_intensities = self.attention_layer(user_embs)
        batch_pos_item_intensities = self.attention_layer(pos_item_embs)
        
        if self.is_explicit:
            neg_items = batch_data['neg_item_id'] # [B, num_negatives]
                
            # Helper: Get Interest & Residual Scores for Items
            def get_scores(item_ids):
                # item_ids: [B, N] -> Flatten
                B_size, N_size = item_ids.size()
                flat_ids = item_ids.view(-1)
                
                # 1. Interest
                flat_item_interests = self.attention_layer(self.item_embedding(flat_ids))
                # 2. Residual
                flat_item_embs = self.item_embedding(flat_ids)
                
                # Reshape back to [B, N, K]
                return flat_item_interests.view(B_size, N_size, -1), flat_item_embs.view(B_size, N_size, -1)

            user_interests = user_intensities.unsqueeze(1) # [B, 1, K]
            user_embs_exp = user_embs.unsqueeze(1)         # [B, 1, K]

            # Pos Scores
            pos_item_interests = self.attention_layer(pos_item_embs).unsqueeze(1)
            pos_item_embs_exp = pos_item_embs.unsqueeze(1)
            
            pos_int_score = (user_interests * pos_item_interests).sum(dim=-1)
            pos_res_score = (user_embs_exp * pos_item_embs_exp).sum(dim=-1)
            pos_total = pos_int_score + pos_res_score # [B, 1]
            
            # Neg Scores
            neg_int_vecs, neg_res_vecs = get_scores(neg_items)
            neg_int_score = (user_interests * neg_int_vecs).sum(dim=-1)
            neg_res_score = (user_embs_exp * neg_res_vecs).sum(dim=-1)
            neg_total = neg_int_score + neg_res_score # [B, N]
            
            total_scores = torch.cat([pos_total, neg_total], dim=1)

        else:
            # 1. Interest Scores (B, B)
            interest_scores = torch.matmul(user_intensities, batch_pos_item_intensities.t())
            
            # 2. Residual Scores (B, B)
            batch_item_embs = self.item_embedding(pos_items)
            res_scores = torch.matmul(user_embs, batch_item_embs.t())
            
            total_scores = interest_scores + res_scores
        
        # Pass Precomputed Scores to Loss
        loss = self.loss_fn(total_scores)

        # Orthogonal Loss
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")

        params_to_log = {
            'scale': self.attention_layer.scale.item() if hasattr(self.attention_layer.scale, 'item') else 0.0
        }

        return (loss, self.lamda * orth_loss), params_to_log

    def __str__(self):
        return f"CSAR_R_Listwise(K={self.num_interests}, TopK={self.topk}, ZScore={self.zscore})"
