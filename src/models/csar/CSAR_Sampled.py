import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.loss import orthogonal_loss # 이 임포트는 더 이상 필요 없지만, 혹시 다른 곳에서 사용될까봐 유지

from ..base_model import BaseModel


from .csar_layers import CoSupportAttentionLayer


class CSAR_Sampled(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR_Sampled, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda = self.config['model']['orth_loss_weight']
        self.scale = self.config['model'].get('scale', True)
        self.Dummy = self.config['model'].get('dummy', False)
        self.soft_relu = self.config['model'].get('soft_relu', False)
        # Standard Deviation Scaling Factor (Beta)
        # 0.0 = No scaling (Centering only), 1.0 = Full Z-score, 0.5 = Half scaling
        self.std_power = self.config['model'].get('std_power', 0.2)

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim, scale=self.scale, Dummy=self.Dummy, soft_relu=self.soft_relu)

        self._init_weights()
        
        # CSAR_Sampled uses NormalizedSampledSoftmaxLoss (Sampled Softmax)
        temperature = config['model'].get('temperature', 1.0)
        from src.loss import NormalizedSampledSoftmaxLoss
        self.loss_fn = NormalizedSampledSoftmaxLoss(self.data_loader.n_items, temperature=temperature)

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

    def get_final_item_embeddings(self):
        """CSAR의 최종 아이템 임베딩 (Topic-space)을 반환합니다."""
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        # CSAR_Sampled (InfoNCE) uses In-Batch Negatives implicitly via CSARLossPower
        
        user_intensities = self.attention_layer(self.user_embedding(users)) 
        pos_item_intensities = self.attention_layer(self.item_embedding(pos_items))
        
        # Loss Calculation: CSARLossPower (InfoNCE)
        loss = self.loss_fn(user_intensities, pos_item_intensities)

        # Orthogonal Loss
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")

        params_to_log = {'scale': self.attention_layer.scale.item()}

        return (loss, self.lamda * orth_loss), params_to_log

    def __str__(self):
        return f"CSAR_Sampled(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"
