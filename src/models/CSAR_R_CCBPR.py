import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import BPRLoss, DynamicMarginBPRLoss
from .base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer

class CurricularConsistencyBPRLoss(nn.Module):
    """
    Entropy-regularized Self-Paced Learning based Loss.
    
    Mathematical Justification:
    Instead of assuming Gaussian noise (which leads to 1/sigma^2 weights and instability),
    we adopt a Self-Paced Learning framework with Entropic Regularization.
    
    Objective: min_{w, theta} L(theta, w) = sum(w_i * L_BPR_i) - 1/beta * H(w)
    where H(w) is the entropy of weights.
    The optimal weight w*_i derived from this objective is w*_i = exp(-beta * Uncertainty_i).
    """
    def __init__(self, beta=1.0, consistency_weight=0.1):
        super().__init__()
        self.beta = beta  # Inverse Temperature parameter (controls curricular sharpness)
        self.consistency_weight = consistency_weight

    def forward(self, mf_pos, mf_neg, csar_pos, csar_neg, total_pos, total_neg):
        # 1. Measure Epistemic Uncertainty (Disagreement)
        # Gradient is detached to prevent the model from manipulating uncertainty directly.
        with torch.no_grad():
            diff_mf = mf_pos - mf_neg
            diff_csar = csar_pos - csar_neg
            disagreement = torch.abs(diff_mf - diff_csar)

        # 2. Compute Curriculum Weights (Derived from MaxEnt Principle)
        # w* = exp(-beta * cost)
        # Prevents weight explosion/collapse issues of Inverse-Variance weighting (1/sigma^2).
        curriculum_weight = torch.exp(-self.beta * disagreement)

        # 3. Weighted Main Task Loss
        # Focus learning on 'reliable' (low disagreement) samples first.
        bpr_loss = F.softplus(-(total_pos - total_neg))
        weighted_bpr_loss = (curriculum_weight * bpr_loss).mean()

        # 4. Consistency Regularization
        # Forces the two views (MF & CSAR) to converge, reducing uncertainty over time.
        # This acts as the 'driver' for the curriculum, increasing reliable samples.
        consistency_loss = F.mse_loss(mf_pos - mf_neg, csar_pos - csar_neg)

        # Total Loss
        total_loss = weighted_bpr_loss + self.consistency_weight * consistency_loss
        
        # Logging purposes
        return total_loss, weighted_bpr_loss, disagreement, consistency_loss

class RobustCurriculumBPRLoss(nn.Module):
    def __init__(self, beta=1.0, consistency_weight=0.1):
        super().__init__()
        self.beta = beta 
        self.consistency_weight = consistency_weight

    def forward(self, mf_pos, mf_neg, csar_pos, csar_neg, total_pos, total_neg):
        # 1. 불확실성(Disagreement) 측정 (Gradient Detach 필수)
        with torch.no_grad():
            diff_mf = mf_pos - mf_neg
            diff_csar = csar_pos - csar_neg
            disagreement = torch.abs(diff_mf - diff_csar)

        # 2. Welsch Weighting (Robustness 핵심)
        # 수학적 정의: w(r) = exp(-beta * r^2)
        # 제곱(**2)을 해주면 Welsch 분포의 정의와 완벽하게 일치합니다.
        # 이상치일수록 가중치가 0으로 더 부드럽고 빠르게 수렴합니다.
        curriculum_weight = torch.exp(-self.beta * torch.square(disagreement))

        # 3. Main Task (BPR)
        bpr_loss = F.softplus(-(total_pos - total_neg))
        weighted_bpr = (curriculum_weight * bpr_loss).mean()

        # 4. Consistency (일관성 유도)
        consistency_loss = F.mse_loss(mf_pos - mf_neg, csar_pos - csar_neg)

        return weighted_bpr + self.consistency_weight * consistency_loss, weighted_bpr,disagreement, consistency_loss

class CSAR_R_CCBPR(BaseModel):
    def __init__(self, config, data_loader):
        super(CSAR_R_CCBPR, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda = self.config['model']['orth_loss_weight']
        self.soft_relu = self.config['model'].get('soft_relu', False)
        self.scale = self.config['model'].get('scale', True)
        self.Dummy = self.config['model'].get('dummy', False)

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # CoSupportAttentionLayer 사용
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim, scale=self.scale, Dummy=self.Dummy, soft_relu=self.soft_relu)

        self._init_weights()
        self.loss_fn = RobustCurriculumBPRLoss()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        # attention_layer의 가중치는 해당 클래스 내부에서 초기화됨

    def forward(self, users):
        # Inference용 (Top-K 추천 등)
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
        # Evaluation용 (Pairwise Score)
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
        
        # BPR 손실 (UncertaintyBPRLoss)
        loss, bpr,disagreement, consistency_loss = self.loss_fn(
            mf_pos=pos_res_scores, 
            mf_neg=neg_res_scores, 
            csar_pos=pos_interest_scores, 
            csar_neg=neg_interest_scores, 
            total_pos=pos_scores, 
            total_neg=neg_scores
        )

        # attention_layer에서 직교 손실 계산
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")
        
        # Scale 파라미터 로깅 (Parameter인 경우에만)
        params_to_log = {'bpr': bpr.item(), 'disagreement': disagreement.mean().item(), 'consistency_loss': consistency_loss.item()}
        if isinstance(self.attention_layer.scale, nn.Parameter):
            params_to_log['scale'] = self.attention_layer.scale.item()

        return (loss, self.lamda * orth_loss), params_to_log
    
    def __str__(self):
        return f"CSAR_R_BPR(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"
