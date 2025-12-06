import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.loss import orthogonal_loss,BPRLoss, DynamicMarginBPRLoss # 이 임포트는 더 이상 필요 없지만, 혹시 다른 곳에서 사용될까봐 유지

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
        self.soft_relu = self.config['model'].get('soft_relu', False)
        self.dynamic_bpr = self.config['model'].get('dynamic_bpr', False)

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # CoSupportAttentionLayer 사용
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim, scale=self.scale, Dummy=self.Dummy, soft_relu=self.soft_relu)

        self._init_weights()
        self.loss_fn = BPRLoss() if not self.dynamic_bpr else DynamicMarginBPRLoss()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        # attention_layer의 가중치는 해당 클래스 내부에서 초기화됨

    def forward(self, users):
        # 사용자 임베딩과 아이템 임베딩 가져오기
        user_embs = self.user_embedding(users)
        all_item_embs = self.item_embedding.weight

        # co-support attention layer를 통해 관심사 가중치 계산
        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(all_item_embs)

        # 최종 점수 계산
        scores = torch.einsum('bk,nk->bn', user_interests, item_interests)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        # 사용자-아이템 쌍 점수 계산 (포인트와이즈용)
        user_embs = self.user_embedding(user_ids)
        item_embs = self.item_embedding(item_ids)

        # co-support attention layer를 통해 관심사 가중치 계산
        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(item_embs)

        scores = (user_interests * item_interests).sum(dim=-1)
        return scores

    def _get_user_interests(self, user_ids):
        """ 사용자의 K-dim 관심사 벡터를 계산합니다. """
        user_embs = self.user_embedding(user_ids)
        return self.attention_layer(user_embs)

    def _get_item_interests(self, item_ids):
        """ 특정 아이템들의 K-dim 관심사 벡터를 계산합니다. """
        item_embs = self.item_embedding(item_ids)
        return self.attention_layer(item_embs)

    def _get_all_item_interests(self):
        """ *모든* 아이템의 K-dim 관심사 벡터를 계산합니다. """
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs)

    def calc_loss(self, batch_data):
        # [수정] DataLoader가 [B, 1] 형태로 반환하므로 차원 축소
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1) # [수정] 차원 축소

        # --- 1. BPR Loss 계산 ---
        # 헬퍼 메서드를 사용해 각 관심사 벡터를 *한 번만* 계산
        user_interests = self._get_user_interests(users)         # [B, K]
        pos_item_interests = self._get_item_interests(pos_items) # [B, K]
        neg_item_interests = self._get_item_interests(neg_items) # [B, K]

        # Pairwise 점수 계산 (predict_for_pairs와 동일한 로직)
        pos_scores = (user_interests * pos_item_interests).sum(dim=-1) # [B]
        neg_scores = (user_interests * neg_item_interests).sum(dim=-1) # [B]
        
        # BPR 손실
        loss = self.loss_fn(pos_scores, neg_scores)

        # attention_layer에서 직교 손실 계산
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")
        params_to_log = {'scale': self.attention_layer.scale.item()}

        return (loss, self.lamda * orth_loss), params_to_log
    
    def get_final_item_embeddings(self):
        """CSAR_BPR의 최종 아이템 임베딩 (Topic-space)을 반환합니다."""
        print("Getting final item embeddings from CSAR_BPR...")
        return self._get_all_item_interests().detach()


    def __str__(self):
        return f"CSAR_BPR(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"
