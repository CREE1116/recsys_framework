import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.loss import orthogonal_loss,BPRLoss # 이 임포트는 더 이상 필요 없지만, 혹시 다른 곳에서 사용될까봐 유지

from .base_model import BaseModel

    
class CSAR_Deep(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR_Deep, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.lamda = self.config['model']['orth_loss_weight']
        self.k_dim_list = self.config['model']['k_list']
        

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # HierarchicalCSAR 사용
        self.attention_layer = HierarchicalCSAR(self.embedding_dim, self.k_dim_list)

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

    def get_final_item_embeddings(self):
        """CSAR_Deep의 최종 아이템 임베딩 (Topic-space)을 반환합니다."""
        return self._get_all_item_interests().detach()

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
        orth_loss = self.attention_layer.get_total_orth_loss(loss_type="l1")

        params_to_log = self.attention_layer.get_all_scales()

        return (loss, self.lamda * orth_loss), params_to_log

    def __str__(self):
        return f"CSAR_BPR(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"

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
        self.scale = nn.Parameter(torch.tensor(self.num_interests**-0.5))
        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform 초기화를 사용하여 관심사 키를 초기화합니다."""
        nn.init.xavier_uniform_(self.interest_keys)

    def forward(self, embedding_tensor):
        attention_logits = torch.einsum('...d,kd->...k', embedding_tensor, self.interest_keys) * self.scale
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

class HierarchicalCSAR(nn.Module):
    def __init__(self, input_dim, k_list):
        super(HierarchicalCSAR, self).__init__()
        self.input_dim = input_dim
        self.k_list = k_list
        
        # CSAR 레이어들만
        self.csar_layers = nn.ModuleList([
            CoSupportAttentionLayer(
                num_interests=k_list[i],
                embedding_dim=k_list[i-1] if i > 0 else input_dim
            )
            for i in range(len(k_list))
        ])

    def forward(self, x):
        current = x
        for i, csar_layer in enumerate(self.csar_layers):
            if i > 0: 
                current = F.layer_norm(current, (current.size(-1),))
            current = csar_layer(current)
        return current

    def get_total_orth_loss(self, loss_type="l2"):
        total_loss = 0.0
        for layer in self.csar_layers:
            total_loss += layer.get_orth_loss(loss_type=loss_type)
        return total_loss
    def get_all_scales(self):
        return {f'layer_{i}_scale': layer.scale.item() for i, layer in enumerate(self.csar_layers)}
    
    def __str__(self):
        return f"HierarchicalCSAR(input_dim={self.input_dim}, structure={self.k_list})"