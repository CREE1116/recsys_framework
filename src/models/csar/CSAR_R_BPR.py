import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import BPRLoss, DynamicMarginBPRLoss
from ..base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer

class CSAR_R_BPR(BaseModel):
    """
    CSAR_R (Residual Connection) + BPR Loss
    """
    def __init__(self, config, data_loader):
        super(CSAR_R_BPR, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda = self.config['model']['orth_loss_weight']
        self.scale = self.config['model'].get('scale', True)
        self.Dummy = self.config['model'].get('dummy', False)
        self.dynamic_bpr = self.config['model'].get('dynamic_bpr', False)
        self.emb_dropout = self.config['model'].get('emb_dropout', 0.0)

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # CoSupportAttentionLayer 사용
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim, scale=self.scale)

        self._init_weights()
        self.loss_fn = BPRLoss() if not self.dynamic_bpr else DynamicMarginBPRLoss()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        # attention_layer의 가중치는 해당 클래스 내부에서 초기화됨

    def forward(self, users):
        # Inference용 (Top-K 추천 등)
        # 사용자 임베딩과 아이템 임베딩 가져오기
        user_embs = self.user_embedding(users)
        all_item_embs = self.item_embedding.weight
        
        # Embedding Dropout (Training only)
        if self.training and self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)

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
        # Evaluation용 (Pairwise Score)
        user_embs = self.user_embedding(user_ids)
        item_embs = self.item_embedding(item_ids)
        
        # Embedding Dropout (Training only)
        if self.training and self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)
            item_embs = F.dropout(item_embs, p=self.emb_dropout, training=True)

        # co-support attention layer를 통해 관심사 가중치 계산
        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(item_embs)

        # 1. Interest-based Score
        interest_scores = (user_interests * item_interests).sum(dim=-1)
        
        # 2. Residual Score
        res_scores = (user_embs * item_embs).sum(dim=-1)
        
        return interest_scores + res_scores

    def _get_user_interests(self, user_ids):
        """ 사용자의 K-dim 관심사 벡터를 계산합니다. """
        user_embs = self.user_embedding(user_ids)
        return self.attention_layer(user_embs)

    def _get_item_interests(self, item_ids):
        """ 특정 아이템들의 K-dim 관심사 벡터를 계산합니다. """
        item_embs = self.item_embedding(item_ids)
        return self.attention_layer(item_embs)
    
    def get_final_item_embeddings(self):
        """
        CSAR_R의 경우 최종 아이템 임베딩을 정의하기가 모호할 수 있음 (Residual 때문).
        하지만 Topic-space 분석을 위해 Interest Vector를 반환하거나, 
        혹은 검색을 위해 Concatenation 등을 고려할 수 있음.
        여기서는 기존 CSAR과 동일하게 Interest Vector를 반환.
        """
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()

    def calc_loss(self, batch_data):
        # DataLoader가 [B, 1] 형태로 반환하므로 차원 축소
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)

        # --- 1. BPR Loss 계산 ---
        # 임베딩 조회
        user_embs = self.user_embedding(users)
        pos_item_embs = self.item_embedding(pos_items)
        neg_item_embs = self.item_embedding(neg_items)
        
        # Embedding Dropout (Training only)
        if self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)
            pos_item_embs = F.dropout(pos_item_embs, p=self.emb_dropout, training=True)
            neg_item_embs = F.dropout(neg_item_embs, p=self.emb_dropout, training=True)

        # 관심사 벡터 계산
        user_interests = self.attention_layer(user_embs)
        pos_item_interests = self.attention_layer(pos_item_embs)
        neg_item_interests = self.attention_layer(neg_item_embs)

        # Positive Score (Interest + Residual)
        pos_interest_scores = (user_interests * pos_item_interests).sum(dim=-1)
        pos_res_scores = (user_embs * pos_item_embs).sum(dim=-1)
        pos_scores = pos_interest_scores + pos_res_scores

        # Negative Score (Interest + Residual)
        neg_interest_scores = (user_interests * neg_item_interests).sum(dim=-1)
        neg_res_scores = (user_embs * neg_item_embs).sum(dim=-1)
        neg_scores = neg_interest_scores + neg_res_scores
        
        # BPR 손실
        loss = self.loss_fn(pos_scores, neg_scores)

        # attention_layer에서 직교 손실 계산
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")
        
        # Scale 파라미터 로깅 (Parameter인 경우에만)
        params_to_log = {}
        if isinstance(self.attention_layer.scale, nn.Parameter):
            params_to_log['scale'] = self.attention_layer.scale.item()

        return (loss, self.lamda * orth_loss), params_to_log
    
    def __str__(self):
        return f"CSAR_R_BPR(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"
