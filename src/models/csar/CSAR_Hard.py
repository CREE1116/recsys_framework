import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer


class CSAR_Hard(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR_Hard, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.scale = self.config['model'].get('scale', True)
        self.init_method = self.config['model'].get('init_method', 'xavier')
        self.normalize = self.config['model'].get('normalize', False)
        
        # Inclusion Regularization Weights (ProtoMF names preferred)
        self.sim_proto_weight = float(self.config['model'].get('sim_proto_weight', self.config['model'].get('balance_weight', 0.1)))
        self.sim_batch_weight = float(self.config['model'].get('sim_batch_weight', self.config['model'].get('entropy_weight', 0.1)))
        
        self.num_negatives = self.config['train'].get('num_negatives', 1)
        self.is_explicit = self.num_negatives > 0

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        self.attention_layer = CoSupportAttentionLayer(
            self.num_interests, self.embedding_dim, 
            scale=self.scale,
            normalize=self.normalize,
            init_method=self.init_method
        )

        self._init_weights()
        
        # CSAR_Hard uses NormalizedSampledSoftmaxLoss (same as CSAR_Sampled)
        temperature = config['model'].get('temperature', 1.0)
        from src.loss import NormalizedSampledSoftmaxLoss
        self.loss_fn = NormalizedSampledSoftmaxLoss(self.data_loader.n_items, temperature=temperature)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users):
        user_embs = self.user_embedding(users)
        all_item_embs = self.item_embedding.weight
        
        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(all_item_embs)

        scores = torch.einsum('bk,nk->bn', user_interests, item_interests)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        user_embs = self.user_embedding(user_ids)
        item_embs = self.item_embedding(item_ids)
        
        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(item_embs)

        scores = (user_interests * item_interests).sum(dim=-1)
        return scores

    def get_final_item_embeddings(self):
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()

    def prototype_similarity(self, embeddings, prototypes):
        """
        ProtoMF-style cosine similarity: [0, 1] range (shifted_and_div).
        """
        emb_norm = F.normalize(embeddings, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        cos_sim = torch.matmul(emb_norm, proto_norm.t())
        return (1 + cos_sim) / 2

    def _inclusion_regularizer(self, sim_mtx):
        """
        Inclusion criteria (max-sim based), exactly matching ProtoMF.
        sim_mtx: [B, K] similarities in [0, 1] range.
        """
        # [B]
        max_over_proto = sim_mtx.max(dim=1).values
        # [K]
        max_over_batch = sim_mtx.max(dim=0).values

        batch_inclusion_loss = -max_over_proto.mean()
        proto_inclusion_loss = -max_over_batch.mean()
        return batch_inclusion_loss, proto_inclusion_loss

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        
        user_embs = self.user_embedding(users)
        pos_item_embs = self.item_embedding(pos_items)
        
        user_intensities = self.attention_layer(user_embs) 
        pos_item_intensities = self.attention_layer(pos_item_embs)
        
        # Loss Calculation: InfoNCE style
        if self.is_explicit:
            neg_items = batch_data['neg_item_id'] # [B, num_negatives]
            
            def get_item_interests(item_ids):
                B_size, N_size = item_ids.size()
                flat_ids = item_ids.view(-1)
                flat_interests = self.attention_layer(self.item_embedding(flat_ids))
                return flat_interests.view(B_size, N_size, -1)

            user_interests = user_intensities.unsqueeze(1)
            
            # Pos Scores
            pos_item_interests = self.attention_layer(self.item_embedding(pos_items)).unsqueeze(1)
            pos_scores = (user_interests * pos_item_interests).sum(dim=-1)
            
            # Neg Scores
            neg_item_interests = get_item_interests(neg_items)
            neg_scores = (user_interests * neg_item_interests).sum(dim=-1)
            
            scores = torch.cat([pos_scores, neg_scores], dim=1)
            
        else:
            # In-Batch Scores
            batch_pos_item_intensities = self.attention_layer(self.item_embedding(pos_items))
            scores = torch.matmul(user_intensities, batch_pos_item_intensities.t())
        
        loss = self.loss_fn(scores, is_explicit=self.is_explicit)

        # --- Inclusion Regularization (Max-Sim, ProtoMF-style) ---
        # Regularization uses Cosine Similarity (separate from scoring intensities)
        u_sim = self.prototype_similarity(user_embs, self.attention_layer.interest_keys)
        pos_i_sim = self.prototype_similarity(pos_item_embs, self.attention_layer.interest_keys)

        # User side
        batch_inc_u, proto_inc_u = self._inclusion_regularizer(u_sim)
        # Item side
        batch_inc_i, proto_inc_i = self._inclusion_regularizer(pos_i_sim)
        
        # Total Inclusion Reg
        inclusion_loss = self.sim_batch_weight * (batch_inc_u + batch_inc_i) + \
                         self.sim_proto_weight * (proto_inc_u + proto_inc_i)

        # --- Score (Activation) Regularization ---
        # Prevent raw intensities (from Softplus) from exploding
        score_reg_weight = float(self.config['model'].get('score_reg_weight', 0.01))
        score_reg_loss = score_reg_weight * ((user_intensities ** 2).mean() + (pos_item_intensities ** 2).mean())

        reg_loss = inclusion_loss + score_reg_loss

        params_to_log = {
            'scale': self.attention_layer.scale.item(),
            'Batch_Inc': (batch_inc_u + batch_inc_i).item() / 2,
            'Proto_Inc': (proto_inc_u + proto_inc_i).item() / 2,
            'Score_Reg': score_reg_loss.item()
        }

        return (loss, reg_loss), params_to_log

    def __str__(self):
        return f"CSAR_Hard(K={self.num_interests}, H_w={self.entropy_weight}, Bal_w={self.balance_weight})"
