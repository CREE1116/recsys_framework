import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from .csar_layers import DualViewCoSupportAttentionLayer


class CSAR_DualView(BaseModel):
    """
    Dual-View CSAR: Like View(좋아하는 관심사)와 Dislike View(싫어하는 관심사)를 사용.
    Score = Like - Dislike
    """
    def __init__(self, config, data_loader):
        super(CSAR_DualView, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda = self.config['model']['orth_loss_weight']
        self.exclusive_lamda = self.config['model']['exclusive_loss_weight']
        self.scale = self.config['model'].get('scale', True)
        self.normalize = self.config['model'].get('normalize', False)
        self.emb_dropout = self.config['model'].get('emb_dropout', 0.0)
        self.l2_reg_weight = self.config['model'].get('l2_reg_weight', 0.0)
        
        self.num_negatives = self.config['train'].get('num_negatives', 1)

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        self.attention_layer = DualViewCoSupportAttentionLayer(
            self.num_interests, self.embedding_dim, 
            scale=self.scale, normalize=self.normalize
        )

        self._init_weights()
        
        # Sampled Softmax Loss
        temperature = config['model'].get('temperature', 0.1)
        use_zscore = config['model'].get('use_zscore', False)
        from src.loss import NormalizedSampledSoftmaxLoss
        self.loss_fn = NormalizedSampledSoftmaxLoss(
            self.data_loader.n_items, temperature=temperature, use_zscore=use_zscore
        )

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users):
        """Full evaluation: 모든 아이템에 대한 점수 계산"""
        user_embs = self.user_embedding(users)
        all_item_embs = self.item_embedding.weight
        
        if self.training and self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)

        # Dual View: like/dislike
        u_like, u_dislike = self.attention_layer(user_embs)
        i_like, i_dislike = self.attention_layer(all_item_embs)

        # Score = Like - Dislike
        like_score = torch.einsum('bk,nk->bn', u_like, i_like)
        dislike_score = torch.einsum('bk,nk->bn', u_dislike, i_dislike)
        return like_score - dislike_score

    def predict_for_pairs(self, user_ids, item_ids):
        """Pairwise evaluation: 특정 유저-아이템 쌍 점수 계산"""
        user_embs = self.user_embedding(user_ids)
        item_embs = self.item_embedding(item_ids)
        
        if self.training and self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)
            item_embs = F.dropout(item_embs, p=self.emb_dropout, training=True)

        # Dual View: like/dislike
        u_like, u_dislike = self.attention_layer(user_embs)
        i_like, i_dislike = self.attention_layer(item_embs)

        # Element-wise score
        like_score = (u_like * i_like).sum(dim=-1)
        dislike_score = (u_dislike * i_dislike).sum(dim=-1)
        return like_score - dislike_score

    def get_final_item_embeddings(self):
        """아이템의 Like View 임베딩 반환 (다양성 메트릭용)"""
        all_item_embs = self.item_embedding.weight
        like_view, _ = self.attention_layer(all_item_embs)
        return like_view.detach()

    def _exclusive_loss(self, like, dislike):
        # 1. L2 정규화 (Normalize)
        # 엡실론(eps)을 더해줘서 0으로 나누는 에러 방지
        like_norm = F.normalize(like.detach(), p=2, dim=-1, eps=1e-8)
        dislike_norm = F.normalize(dislike, p=2, dim=-1, eps=1e-8)
        
        # 2. 내적 (Dot Product) -> 이것이 곧 코사인 유사도
        # Softplus를 통과했으므로 두 벡터는 모두 양수입니다. (1사분면)
        # 따라서 결과값은 0(직교) ~ 1(일치) 사이로 나옵니다.
        cosine_sim = (like_norm * dislike_norm).sum(dim=-1)
        
        # 3. [핵심] 제곱(Square) 패널티 적용
        # 이유: 큰 겹침(0.9)은 아주 강하게 때리고, 작은 겹침(0.1)은 봐주기 위함.
        # 0.9 -> 0.81 (강한 로스)
        # 0.1 -> 0.01 (미미한 로스 -> 무시하고 Main Loss에 집중하게 해줌)
        loss = (cosine_sim ** 2).mean()
        
        return loss

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        gt_items = batch_data['pos_item_id'].squeeze(-1)
        ng_items = batch_data['neg_item_id'] # [B, num_negatives]
        
        B, N = ng_items.size()
        
        # === Embeddings ===
        # 1. 임베딩 Lookup
        user_embs = self.user_embedding(users)
        gt_embs = self.item_embedding(gt_items)
        ng_embs = self.item_embedding(ng_items.view(-1))

        # === [추가됨] Explicit L2 Regularization ===
        # 설명: 배치에 등장한 유저/아이템 임베딩과 Attention Layer(Key) 파라미터에 대해
        #      강력한 L2 규제를 적용하여 Magnitude Bias를 억제합니다.
        reg_loss = 0
        
        # 1) 배치 내 활성화된 임베딩 규제
        for emb in [user_embs, gt_embs, ng_embs]:
            reg_loss += torch.norm(emb, p=2) ** 2
            
        # 2) Attention Layer 내부 파라미터(Key, Linear Weight 등) 규제
        for param in self.attention_layer.parameters():
            reg_loss += torch.norm(param, p=2) ** 2
            
        # 3) 가중치 적용 (보통 1e-3 ~ 1e-4 수준으로 강하게 설정 추천)
        # 배치 사이즈로 나누어 평균을 맞추는 경우도 있지만, 
        # 여기서는 "강한 규제"를 위해 합(Sum)을 그대로 사용하거나 0.5만 곱해줍니다.
        reg_loss = self.l2_reg_weight * reg_loss * 0.5
        
        # -------------------------------------------------------

        if self.emb_dropout > 0:
            user_embs = F.dropout(user_embs, p=self.emb_dropout, training=True)
            gt_embs = F.dropout(gt_embs, p=self.emb_dropout, training=True)
            ng_embs = F.dropout(ng_embs, p=self.emb_dropout, training=True)
        
        # === Dual View Interests ===
        u_like, u_dislike = self.attention_layer(user_embs)
        gt_like, gt_dislike = self.attention_layer(gt_embs)
        ng_like, ng_dislike = self.attention_layer(ng_embs)

        exclusive_loss = self._exclusive_loss(u_like, u_dislike) + \
                         (self._exclusive_loss(gt_like, gt_dislike) + \
                          self._exclusive_loss(ng_like, ng_dislike)) / 2
        
        # Reshape negatives
        ng_like = ng_like.view(B, N, -1)
        ng_dislike = ng_dislike.view(B, N, -1)
        
        # === Score Calculation ===
        # Ground Truth score: like - dislike
        gt_score = (u_like * gt_like).sum(dim=-1, keepdim=True) - \
                   (u_dislike * gt_dislike).sum(dim=-1, keepdim=True)
        
        # Negative scores
        u_like_exp = u_like.unsqueeze(1)
        u_dislike_exp = u_dislike.unsqueeze(1)
        
        ng_like_score = (u_like_exp * ng_like).sum(dim=-1)
        ng_dislike_score = (u_dislike_exp * ng_dislike).sum(dim=-1)
        ng_score = ng_like_score - ng_dislike_score
        
        # Concat
        scores = torch.cat([gt_score, ng_score], dim=1)
        
        # === Loss ===
        main_loss = self.loss_fn(scores, is_explicit=True)
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")
        
        params_to_log = {
            'pos_scale': self.attention_layer.pos_scale.item(),
            'neg_scale': self.attention_layer.neg_scale.item()
        }
        
        # Return 튜플에 reg_loss 추가
        return (main_loss, self.lamda * orth_loss, self.exclusive_lamda * exclusive_loss, reg_loss), params_to_log
        
    def __str__(self):
        return f"CSAR_DualView(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"
