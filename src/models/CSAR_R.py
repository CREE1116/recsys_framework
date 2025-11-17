import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import orthogonal_loss

from .base_model import BaseModel


class CSAR_R(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR_R, self).__init__(config, data_loader)

        self.num_interests = self.config['model']['num_interests']
        self.embedding_dim = self.config['model']['embedding_dim']
        self.lamda = self.config['model']['orth_loss_weight']

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        self.global_interest_keys = nn.Parameter(torch.randn(self.num_interests, self.embedding_dim))
        self.scale = nn.Parameter(torch.tensor(self.num_interests ** -0.5))

        self._init_weights()
        self.orth_loss_fn = orthogonal_loss("l2")

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.global_interest_keys)

    def forward(self, users):
        # 사용자 관심사 가중치 계산
        user_embs = self.user_embedding(users)
        user_attention_logits = torch.einsum('bd,kd->bk', user_embs, self.global_interest_keys) * self.scale
        user_interests = F.softplus(user_attention_logits)

        # 전체 아이템 관심사 가중치 계산
        all_item_embs = self.item_embedding.weight
        item_attention_logits = torch.einsum('nd,kd->nk', all_item_embs, self.global_interest_keys) * self.scale
        item_interest_probs = F.softplus(item_attention_logits)

        # 최종 점수 계산
        scores = torch.einsum('bk,nk->bn', user_interests, item_interest_probs)
        res_scores = torch.matmul(user_embs, all_item_embs.T) # [B, D] @ [D, N] -> [B, N]
        return scores + res_scores

    def predict_for_pairs(self, user_ids, item_ids):
        # 사용자-아이템 쌍 점수 계산 (포인트와이즈용)
        user_embs = self.user_embedding(user_ids)
        item_embs = self.item_embedding(item_ids)
        user_attention_logits = torch.einsum('bd,kd->bk', user_embs, self.global_interest_keys) * self.scale
        user_interests = F.softplus(user_attention_logits)

        item_attention_logits = torch.einsum('bd,kd->bk', item_embs, self.global_interest_keys) * self.scale
        item_interest_probs = F.softplus(item_attention_logits)

        scores = (user_interests * item_interest_probs).sum(dim=-1)  # [B] shape의 스칼라 점수
        res_scores = (user_embs * item_embs).sum(dim=-1)
        return scores + res_scores

    def predict(self, users):
        """
        주어진 사용자들에 대한 모든 아이템 추천 점수를 반환합니다.
        BaseModel의 추상 메서드 구현.
        """
        return self.forward(users)

    def calc_loss(self, batch_data):
        users = batch_data['user_id']
        items = batch_data['item_id'] # [B] (이것이 Positive 아이템 인덱스)
        # ratings = batch_data['rating'] # [B] (사용 안 함)

        # [수정됨] 1. 전체 점수 (Input 1)
        # self.predict(users) 호출 -> [B, N_items]
        preds = self.predict(users) 
        
        # [수정됨] 2. Positive 아이템 인덱스 (Input 2)
        # 'ratings.float()' 대신 'items' (LongTensor)를 사용
        # (CrossEntropy는 target으로 LongTensor를 기대함)
        loss = F.cross_entropy(preds, items) 

        orth_loss = self.orth_loss_fn(self.global_interest_keys)

        total_loss = loss + self.lamda * orth_loss
        params_to_log = {'scale': self.scale.item()}

        return (total_loss, self.lamda * orth_loss), params_to_log

    def __str__(self):
        return f"CSAR(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"