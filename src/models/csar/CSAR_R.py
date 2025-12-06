import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import orthogonal_loss, PolyLoss

from ..base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer


class CSAR_R(BaseModel):
    """
    CSAR_R (Residual Connection) + Cross Entropy Loss (Pointwise)
    """
    def __init__(self, config, data_loader):
        super(CSAR_R, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda = self.config['model']['orth_loss_weight']
        self.cross_entropy_temp = self.config['model'].get('cross_entropy_temp', 1.0)
        self.poly_loss = self.config['model'].get('poly_loss', False)
        
        self.soft_relu = self.config['model'].get('soft_relu', False)
        self.scale = self.config['model'].get('scale', True)
        self.Dummy = self.config['model'].get('dummy', False)

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim, scale=self.scale, Dummy=self.Dummy, soft_relu=self.soft_relu)

        self.loss_fn = PolyLoss() if self.poly_loss else nn.CrossEntropyLoss()
        self._init_weights()

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

        # 최종 점수 계산 (Residual Connection 포함)
        # 1. Interest-based Score
        interest_scores = torch.einsum('bk,nk->bn', user_interests, item_interests)
        
        # 2. Residual Score (Direct Embedding Dot Product)
        res_scores = torch.matmul(user_embs, all_item_embs.T)

        return interest_scores + res_scores

    def predict_for_pairs(self, user_ids, item_ids):
        # 사용자-아이템 쌍 점수 계산 (포인트와이즈용)
        user_embs = self.user_embedding(user_ids)
        item_embs = self.item_embedding(item_ids)

        # co-support attention layer를 통해 관심사 가중치 계산
        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(item_embs)

        # 1. Interest-based Score
        interest_scores = (user_interests * item_interests).sum(dim=-1)
        
        # 2. Residual Score
        res_scores = (user_embs * item_embs).sum(dim=-1)
        
        return interest_scores + res_scores

    def get_final_item_embeddings(self):
        """CSAR_R의 최종 아이템 임베딩 (Topic-space)을 반환합니다."""
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()

    def calc_loss(self, batch_data):
        users = batch_data['user_id']
        items = batch_data['item_id'] # [B] (이것이 Positive 아이템 인덱스)
        # ratings = batch_data['rating'] # [B] (사용 안 함)

        # 1. 전체 점수 (Input 1)
        # self.predict(users) 호출 -> [B, N_items]
        preds = self.forward(users) 
        
        # 2. Positive 아이템 인덱스 (Input 2)
        loss = self.loss_fn(preds/self.cross_entropy_temp, items) 

        # attention_layer에서 직교 손실 계산
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")
        
        params_to_log = {}
        if isinstance(self.attention_layer.scale, nn.Parameter):
            params_to_log['scale'] = self.attention_layer.scale.item()

        return (loss, self.lamda * orth_loss), params_to_log

    def __str__(self):
        return f"CSAR_R(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"