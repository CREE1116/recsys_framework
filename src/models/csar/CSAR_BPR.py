import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.loss import BPRLoss, DynamicMarginBPRLoss

from ..base_model import BaseModel


from .csar_layers import CoSupportAttentionLayer


class CSAR_BPR(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR_BPR, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda = self.config['model']['orth_loss_weight']
        self.scale = self.config['model'].get('scale', True)
        self.Dummy = self.config['model'].get('dummy', False)
        self.init_method = self.config['model'].get('init_method', 'xavier')
        self.emb_dropout = self.config['model'].get('emb_dropout', 0.0)
        self.orth_loss_type = self.config['model'].get('orth_loss_type', 'l1')  # L1이 더 균등한 페널티

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # CoSupportAttentionLayer 사용
        self.attention_layer = CoSupportAttentionLayer(
            self.num_interests, self.embedding_dim, 
            scale=self.scale,
            init_method=self.init_method
        )

        self._init_weights()
        self.loss_fn = BPRLoss()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        # attention_layer의 가중치는 해당 클래스 내부에서 초기화됨

    def forward(self, users):
        # 사용자 임베딩과 아이템 임베딩 가져오기
        user_embs = self.user_embedding(users)
        all_item_embs = self.item_embedding.weight
        
        # Embedding Dropout (Training only)
        if self.training and self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)

        # co-support attention layer를 통해 관심사 가중치 계산
        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(all_item_embs)

        # 최종 점수 계산
        scores = torch.einsum('bk,nk->bn', user_interests, item_interests)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        """
        사용자-아이템 쌍 점수 계산
        user_ids: [B] 또는 [B, 1]
        item_ids: [B], [B, 1], 또는 [B, N] (다중 negative)
        """
        user_embs = self.user_embedding(user_ids)  # [B, D] 또는 [B, 1, D]
        item_embs = self.item_embedding(item_ids)  # [B, D], [B, 1, D], 또는 [B, N, D]
        
        # Embedding Dropout (Training only)
        if self.training and self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)
            item_embs = F.dropout(item_embs, p=self.emb_dropout, training=True)

        # 차원 처리: [B, 1, D] → [B, D]
        if user_embs.dim() == 3 and user_embs.shape[1] == 1:
            user_embs = user_embs.squeeze(1)
        
        # item_embs 차원 확인
        original_shape = item_embs.shape
        if item_embs.dim() == 3:
            # [B, N, D] → flatten → attention → reshape
            B, N, D = item_embs.shape
            item_embs_flat = item_embs.view(B * N, D)
            item_interests = self.attention_layer(item_embs_flat)  # [B*N, K]
            item_interests = item_interests.view(B, N, -1)  # [B, N, K]
            
            user_interests = self.attention_layer(user_embs)  # [B, K]
            # Broadcasting: [B, 1, K] * [B, N, K] → [B, N, K]
            scores = (user_interests.unsqueeze(1) * item_interests).sum(dim=-1)  # [B, N]
        else:
            # [B, D] 또는 squeeze된 경우
            if item_embs.dim() == 3 and item_embs.shape[1] == 1:
                item_embs = item_embs.squeeze(1)
            user_interests = self.attention_layer(user_embs)
            item_interests = self.attention_layer(item_embs)
            scores = (user_interests * item_interests).sum(dim=-1)  # [B]
        
        return scores

    def _get_user_interests(self, user_ids):
        """ 사용자의 K-dim 관심사 벡터를 계산합니다. """
        user_embs = self.user_embedding(user_ids)
        if self.training and self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)
        return self.attention_layer(user_embs)

    def _get_item_interests(self, item_ids):
        """ 특정 아이템들의 K-dim 관심사 벡터를 계산합니다. """
        item_embs = self.item_embedding(item_ids)
        if self.training and self.emb_dropout > 0:
            item_embs = F.dropout(item_embs, p=self.emb_dropout, training=True)
        return self.attention_layer(item_embs)

    def _get_all_item_interests(self):
        """ *모든* 아이템의 K-dim 관심사 벡터를 계산합니다. """
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs)

    def calc_loss(self, batch_data):
        users = batch_data['user_id']          # [B, 1]
        pos_items = batch_data['pos_item_id']  # [B, 1]
        neg_items = batch_data['neg_item_id']  # [B, N]

        pos_scores = self.predict_for_pairs(users, pos_items)  # [B]
        neg_scores = self.predict_for_pairs(users, neg_items)  # [B, N]
        
        # BPR Loss: 각 negative와 비교 후 평균
        loss = self.loss_fn(pos_scores, neg_scores)

        # Orthogonal Loss
        orth_loss = self.attention_layer.get_orth_loss(loss_type=self.orth_loss_type)

        # [추가] L2 규제
        u_emb = self.user_embedding(users)
        p_emb = self.item_embedding(pos_items)
        n_emb = self.item_embedding(neg_items)
        l2_loss = self.get_l2_reg_loss(u_emb, p_emb, n_emb)

        params_to_log = {
            'scale': self.attention_layer.scale.item(),
            'loss_main': loss.item(),
            'loss_orth': orth_loss.item(),
            'loss_l2': l2_loss.item()
        }

        return (loss, self.lamda * orth_loss, l2_loss), params_to_log
    
    def get_final_item_embeddings(self):
        """CSAR_BPR의 최종 아이템 임베딩 (Topic-space)을 반환합니다."""
        print("Getting final item embeddings from CSAR_BPR...")
        return self._get_all_item_interests().detach()


    def __str__(self):
        return f"CSAR_BPR(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"
