import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer

class CSAR_Listwise(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR_Listwise, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda = self.config['model']['orth_loss_weight']
        self.scale = self.config['model'].get('scale', True)
        self.Dummy = self.config['model'].get('dummy', False)
        # Standard Deviation Scaling Factor (Beta)
        self.std_power = self.config['model'].get('std_power', 0.2)
        
        # Loss Parameters
        self.loss_temp = self.config['model'].get('loss_temp', 1.0)
        self.num_negatives = self.config['train'].get('num_negatives', 1)
        self.topk = self.config['model'].get('topk', 10)
        self.zscore = self.config['model'].get('zscore', False)
        self.emb_dropout = self.config['model'].get('emb_dropout', 0.0)
        
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim, scale=self.scale)

        self._init_weights()
        
        # NDCGWeightedListwiseBPR Loss (Explicit 모드 고정 - 데이터로더에서 네거티브 샘플링)
        from src.loss import NDCGWeightedListwiseBPR
        self.loss_fn = NDCGWeightedListwiseBPR(k=self.topk, use_zscore=self.zscore, is_explicit=True)

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
        # 사용자-아이템 쌍 점수 계산 (포인트와이즈용)
        user_embs = self.user_embedding(user_ids)
        item_embs = self.item_embedding(item_ids)
        
        # Embedding Dropout (Training only)
        if self.training and self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)
            item_embs = F.dropout(item_embs, p=self.emb_dropout, training=True)

        # co-support attention layer를 통해 관심사 가중치 계산
        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(item_embs)

        scores = (user_interests * item_interests).sum(dim=-1)
        return scores

    def get_final_item_embeddings(self):
        """CSAR의 최종 아이템 임베딩 (Topic-space)을 반환합니다."""
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id']  # [B, num_negatives]
        
        user_embs = self.user_embedding(users)
        pos_item_embs = self.item_embedding(pos_items)
        
        # Embedding Dropout (Training only)
        if self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)
            pos_item_embs = F.dropout(pos_item_embs, p=self.emb_dropout, training=True)
        
        # User Interest Vectors [B, K]
        user_interests = self.attention_layer(user_embs)
        
        # Pos Item Interests [B, K]
        pos_item_interests = self.attention_layer(self.item_embedding(pos_items))
        
        # Neg Item Interests [B, N, K]
        B, N = neg_items.size()
        flat_neg_ids = neg_items.view(-1)
        neg_item_interests = self.attention_layer(self.item_embedding(flat_neg_ids)).view(B, N, -1)
        
        # Pos Scores [B, 1]
        pos_scores = (user_interests * pos_item_interests).sum(dim=-1, keepdim=True)
        
        # Neg Scores [B, N]
        # user_interests: [B, K] -> [B, 1, K]
        neg_scores = (user_interests.unsqueeze(1) * neg_item_interests).sum(dim=-1)
        
        # Stack: [Pos, Neg1, Neg2...] -> [B, 1+N]
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        
        loss = self.loss_fn(scores)

        # Orthogonal Loss
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")

        params_to_log = {
            'scale': self.attention_layer.scale.item() if hasattr(self.attention_layer.scale, 'item') else 0.0
        }

        return (loss, self.lamda * orth_loss), params_to_log

    def __str__(self):
        return f"CSAR_Listwise(K={self.num_interests}, TopK={self.topk}, ZScore={self.zscore})"
