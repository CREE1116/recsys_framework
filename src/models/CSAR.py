import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.loss import orthogonal_loss # 이 임포트는 더 이상 필요 없지만, 혹시 다른 곳에서 사용될까봐 유지

from .base_model import BaseModel


class CoSupportAttentionLayer(nn.Module):
    """
    d-차원의 임베딩을 K-차원의 비음수 관심사 가중치 벡터로 변환하는 레이어.
    """
    def __init__(self, num_interests, embedding_dim):
        super(CoSupportAttentionLayer, self).__init__() 
        self.num_interests = num_interests
        self.embedding_dim = embedding_dim
        # K-Anchor (관심사 키)
        self.interest_keys = nn.Parameter(torch.empty(num_interests, embedding_dim))
        # 학습 가능한 스케일 (온도 파라미터의 역수)
        # self.scale = nn.Parameter(torch.tensor(num_interests ** -0.5))
        # self.scale = torch.tensor(num_interests ** -0.5)
        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform 초기화를 사용하여 관심사 키를 초기화합니다."""
        nn.init.xavier_uniform_(self.interest_keys)

    def forward(self, embedding_tensor):
        """
        입력 텐서를 K-차원 관심사 가중치로 변환합니다.
        
        Args:
            embedding_tensor (torch.Tensor): [..., d] shape의 임베딩 텐서.
                                             (e.g., [B, d] 또는 [N, d])
        
        Returns:
            torch.Tensor: [..., K] shape의 비음수 관심사 가중치 텐서.
        """
        attention_logits = torch.einsum('...d,kd->...k', embedding_tensor, self.interest_keys)
        interest_weights = F.softplus(attention_logits)
        
        return interest_weights
    
    @staticmethod
    def l1_orthogonal_loss(keys):
        """
        L1 norm을 사용하여 관심사 키의 직교성을 강제합니다.

        Args:
            keys (torch.Tensor): 전역 관심사 키 텐서. [num_interests, D]

        Returns:
            torch.Tensor: 직교성 손실값.
        """
        K = keys.size(0)
        keys_normalized = F.normalize(keys, p=2, dim=1)
        cosine_similarity = torch.matmul(keys_normalized, keys_normalized.t())
        identity_matrix = torch.eye(K, device=keys.device)
        
        # 비대각선 요소의 개수
        num_off_diagonal_elements = K * K - K
        
        # 제곱 대신 절댓값을 사용하여 수치적 안정성 확보
        loss = torch.abs(cosine_similarity - identity_matrix).sum()
        return loss / num_off_diagonal_elements 
    
    @staticmethod
    def l2_orthogonal_loss(keys):
        """
        L2 norm을 사용하여 관심사 키의 직교성을 강제합니다.

        Args:
            keys (torch.Tensor): 전역 관심사 키 텐서. [num_interests, D]

        Returns:
            torch.Tensor: 직교성 손실값.
        """
        K = keys.size(0)
        keys_normalized = F.normalize(keys, p=2, dim=1)
        cosine_similarity = torch.matmul(keys_normalized, keys_normalized.t())
        identity_matrix = torch.eye(K, device=keys.device)
        off_diag = cosine_similarity - identity_matrix
        
        # 1. 제곱의 합 (L2 Loss)
        l2_sum_loss = (off_diag ** 2).sum()
        
        # 2. 비대각선 요소의 개수로 나누어 정규화
        num_off_diagonal_elements = K * K - K
        normalized_loss = l2_sum_loss / num_off_diagonal_elements 
        return normalized_loss 

    def get_orth_loss(self, loss_type="l2"):
        """
        이 레이어가 소유한 관심사 키에 대한 직교 손실을 반환합니다.
        
        Args:
            loss_type (str): "l1" 또는 "l2". 기본값은 "l2".
        
        Returns:
            torch.Tensor: 직교성 손실값.
        """
        if loss_type == "l1":
            return self.l1_orthogonal_loss(self.interest_keys)
        else:
            return self.l2_orthogonal_loss(self.interest_keys)


class CSAR(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda = self.config['model']['orth_loss_weight']
        self.ce_temp = self.config['model'].get('cross_entropy_temp', 0.2)

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # CoSupportAttentionLayer 사용
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim)

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
        users = batch_data['user_id']
        items = batch_data['item_id']

        preds = self.forward(users) 
        loss = F.cross_entropy(preds/self.ce_temp, items, reduction='mean') 

        # attention_layer에서 직교 손실 계산
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")

        total_loss = loss + self.lamda * orth_loss
        # params_to_log = {'scale': self.attention_layer.scale.item()}

        return (total_loss, self.lamda * orth_loss), None

    def __str__(self):
        return f"CSAR(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"
