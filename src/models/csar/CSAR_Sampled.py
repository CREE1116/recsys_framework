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
        self.init_method = self.config['model'].get('init_method', 'xavier')
        # self.Dummy removed
        self.normalize = self.config['model'].get('normalize', False)
        # self.emb_dropout removed
        self.score_reg_weight = self.config['model'].get('score_reg_weight', 0.0)
        
        self.num_negatives = self.config['train'].get('num_negatives', 1)
        self.is_explicit = self.num_negatives > 0

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        self.attention_layer = CoSupportAttentionLayer(
            self.num_interests, self.embedding_dim, 
            scale=self.scale,
            normalize=self.normalize,
            init_method=self.init_method
        )

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
        
        # Embedding Dropout removed

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
        # Embedding Dropout removed


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
        
        user_embs = self.user_embedding(users)
        pos_item_embs = self.item_embedding(pos_items)
        
        # Embedding Dropout (Training only)
        # Embedding Dropout removed

        
        user_intensities = self.attention_layer(user_embs) 
        pos_item_intensities = self.attention_layer(pos_item_embs)
        
        
        # Loss Calculation: CSARLossPower (InfoNCE)
        # Explicit Negative Sampling (InfoNCE on K+1 classes)
        if self.is_explicit:
            neg_items = batch_data['neg_item_id'] # [B, num_negatives]
            
            # Helper to get Item Interests
            def get_item_interests(item_ids):
                # item_ids: [B, N] -> Flatten -> [B*N]
                B_size, N_size = item_ids.size()
                flat_ids = item_ids.view(-1)
                flat_interests = self.attention_layer(self.item_embedding(flat_ids)) # [B*N, K]
                return flat_interests.view(B_size, N_size, -1) # [B, N, K]

            user_interests = user_intensities.unsqueeze(1) # [B, 1, K]
            
            # Pos Scores
            pos_item_interests = self.attention_layer(self.item_embedding(pos_items)).unsqueeze(1) # [B, 1, K]
            pos_scores = (user_interests * pos_item_interests).sum(dim=-1) # [B, 1]
            
            # Neg Scores
            neg_item_interests = get_item_interests(neg_items) # [B, N, K]
            neg_scores = (user_interests * neg_item_interests).sum(dim=-1) # [B, N]
            
            # Stack: [Pos, Neg1, Neg2...] -> [B, 1+N]
            scores = torch.cat([pos_scores, neg_scores], dim=1)
            
        else:
            # In-Batch Scores (B, B)
            batch_pos_item_intensities = self.attention_layer(self.item_embedding(pos_items))
            scores = torch.matmul(user_intensities, batch_pos_item_intensities.t())
        
        loss = self.loss_fn(scores, is_explicit=self.is_explicit)

        # Orthogonal Loss
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")
        
        # Score (Activation) Regularization
        # 관심사 강도(Intensity) 자체가 너무 커지지 않도록 제어 (Exploding 방지)
        reg_loss_user = (user_intensities ** 2).mean()
        reg_loss_item = (pos_item_intensities ** 2).mean()
        score_reg_loss = reg_loss_user + reg_loss_item

        params_to_log = {'scale': self.attention_layer.scale.item()}

        return (loss, self.lamda * orth_loss, self.score_reg_weight * score_reg_loss), params_to_log

    def __str__(self):
        return f"CSAR_Sampled(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"
