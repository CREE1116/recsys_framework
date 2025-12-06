from .csar_layers import CoSupportAttentionLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.loss import orthogonal_loss, BPRLoss, DynamicMarginBPRLoss,CSARLoss,CSARLossPower

from .base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer


class CSAR_BPR_CE(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR_BPR_CE, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda = self.config['model']['orth_loss_weight']
        self.scale = self.config['model'].get('scale', True)
        self.Dummy = self.config['model'].get('dummy', False)
        self.soft_relu = self.config['model'].get('soft_relu', False)
        # Loss Weighting (User Feedback)
        # Default 1.0 (Equal weight, or handled in calc_loss)
        self.loss_weight = self.config['model'].get('loss_weight', 1.0)
        
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # CoSupportAttentionLayer 사용
        self.attention_layer = CoSupportAttentionLayer(
            self.num_interests, self.embedding_dim, 
            scale=self.scale, Dummy=self.Dummy, soft_relu=self.soft_relu
        )

        self._init_weights()
        
        # CSARLoss 초기화 (Fixed Temperature InfoNCE + Consistent Sampled Softmax Correction)
        # Final Score Matrix Norm 적용됨 -> Temperature 설정 (Default 1.0)
        # [수정] Temperature는 Model Parameter로 이동
        temperature = config['model'].get('temperature', 1.0)
        self.loss_fn = CSARLossPower(self.data_loader.n_items, temperature=temperature)

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

    def _get_user_interests(self, user_ids):
        """사용자의 K-dim 관심사 벡터를 계산합니다."""
        user_embs = self.user_embedding(user_ids)
        return self.attention_layer(user_embs)

    def _get_item_interests(self, item_ids):
        """특정 아이템들의 K-dim 관심사 벡터를 계산합니다."""
        item_embs = self.item_embedding(item_ids)
        return self.attention_layer(item_embs)

    def _get_all_item_interests(self):
        """*모든* 아이템의 K-dim 관심사 벡터를 계산합니다."""
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs)

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        # BPR을 위해 neg_items 필요
        neg_items = batch_data['neg_item_id'].squeeze(-1)
        
        # 1. Forward Pass (Intensity 추출) for CSARLoss (InfoNCE)
        user_intensities = self.attention_layer(self.user_embedding(users)) 
        pos_item_intensities = self.attention_layer(self.item_embedding(pos_items))
        
        # 2. BPR Score Calculation
        # BPR은 (User, Pos) - (User, Neg) 차이를 최적화
        # 여기서도 Intensity 기반 Score 사용: (User_Int * Item_Int).sum
        neg_item_intensities = self.attention_layer(self.item_embedding(neg_items))
        
        pos_scores = (user_intensities * pos_item_intensities).sum(dim=-1)
        neg_scores = (user_intensities * neg_item_intensities).sum(dim=-1)
        
        # 3. Loss Combination
        # Loss A: CSARLoss (InfoNCE) - Global/Batch Negative Effect
        loss_csar = self.loss_fn(user_intensities, pos_item_intensities)
        
        # Loss B: BPR Loss - Pairwise Ranking (Hard Negative Sampling Effect depending on sampler)
        # BPRLoss는 별도 선언 필요 (현재 self.bpr_loss_fn이 없으므로 직접 계산 or init에서 선언)
        # 간단히 Softplus(-diff) 사용
        loss_bpr = F.softplus(neg_scores - pos_scores).mean()

        # Orthogonal Loss
        orth_loss = self.attention_layer.get_orth_loss()
    
        return (loss_csar, self.loss_weight * loss_bpr, self.lamda * orth_loss), {'csar_loss': loss_csar.item(), 'bpr_loss': loss_bpr.item()}
    
    def get_final_item_embeddings(self):
        """CSAR_BPR의 최종 아이템 임베딩 (Topic-space)을 반환합니다."""
        print("Getting final item embeddings from CSAR_BPR...")
        return self._get_all_item_interests().detach()

    def __str__(self):
        return f"CSAR_BPR_CE(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"
