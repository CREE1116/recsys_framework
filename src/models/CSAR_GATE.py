
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import BPRLoss
from .base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer # CSAR 레이어는 모듈화되어 있다고 가정
class CSAR_GATE(BaseModel):
    """
    CSAR with Learnable Hard-Thresholding.
    
    [핵심 철학]
    - "Deep Learning but behaves like Item-KNN"
    - 대중적인 관심사(Cluster)는 문턱을 낮춰 관대하게 받아들이고,
    - 니치한 관심사(Cluster)는 문턱을 높여 확실한 신호만 통과시킨다.
    - 이 '문턱(Threshold)'을 사람이 정하지 않고 모델이 학습한다.
    """
    def __init__(self, config, data_loader):
        super(CSAR_Learnable_Gated, self).__init__(config, data_loader)

        # 1. 하이퍼파라미터 로드
        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda_orth = self.config['model'].get('orth_loss_weight', 0.1)
        self.scale = self.config['model'].get('scale', True)
        
        # 2. 임베딩 레이어
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # 3. CSAR 핵심 레이어 (Latent Interest Extraction)
        self.attention_layer = CoSupportAttentionLayer(
            self.num_interests, 
            self.embedding_dim, 
            scale=self.scale
        )
        # 4. [핵심] 학습 가능한 임계값 (Per-Channel Learnable Thresholds)
        # 초기값 0.1: 너무 크면 초반 학습이 죽고, 너무 작으면 노이즈가 낌.
        # (1, K) 형태로 선언하여 각 관심사(Cluster)별로 다른 기준을 가짐.
        self.threshold_bias = nn.Parameter(torch.full((1, self.num_interests), 0.1))

        # 5. 기타 초기화
        self.loss_fn = BPRLoss()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        # attention_layer와 threshold_bias는 자체 초기화됨

    def _apply_learnable_gating(self, interests):
        """
        [Gating Logic] Shifted ReLU Mechanism
        수식: y = ReLU(x - threshold)
        
        - 미분 가능: Threshold(bias)에 대해서도 그라디언트가 흐름.
        - 적응형: 데이터 분포에 맞춰 '얼마나 쳐낼지' 스스로 학습함.
        """
        # 임계값은 항상 양수여야 하므로 Softplus로 안전장치
        real_threshold = F.softplus(self.threshold_bias) 
        
        # 노이즈 제거 (Threshold보다 작은 값은 0이 됨 -> Gradient 차단)
        # Threshold보다 큰 값은 (x - th)만큼 남아서 통과 -> Gradient 흐름
        gated_interests = F.relu(interests - real_threshold)
        
        return gated_interests

    def forward(self, users):
        """
        전체 아이템에 대한 추천 점수 계산 (Inference / Validation 용)
        """
        user_embs = self.user_embedding(users)
        all_item_embs = self.item_embedding.weight

        # 1. Raw Interests 추출
        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(all_item_embs)

        # 2. [Gating] 유저의 모호한 관심사 제거
        # (아이템은 그대로 두어 정보량 보존, 유저만 엄격하게 필터링)
        user_interests = self._apply_learnable_gating(user_interests)

        # 3. 점수 계산
        scores = torch.einsum('bk,nk->bn', user_interests, item_interests)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        """
        특정 User-Item 쌍에 대한 점수 계산 (Train / Test 용)
        """
        user_embs = self.user_embedding(user_ids)
        item_embs = self.item_embedding(item_ids)

        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(item_embs)

        # [Gating]
        user_interests = self._apply_learnable_gating(user_interests)

        scores = (user_interests * item_interests).sum(dim=-1)
        return scores

    def calc_loss(self, batch_data):
        """
        학습 루프
        """
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)

        # -------------------------------------------------------
        # 1. Interest Extraction & Gating
        # -------------------------------------------------------
        # 임베딩 추출
        u_emb = self.user_embedding(users)
        p_emb = self.item_embedding(pos_items)
        n_emb = self.item_embedding(neg_items)

        # 관심사 변환
        u_int = self.attention_layer(u_emb)
        p_int = self.attention_layer(p_emb)
        n_int = self.attention_layer(n_emb)

        # [핵심] Gating 적용 (User Side)
        # 여기서 노이즈가 0이 되면서, BPR이 엉뚱한 정보를 학습하는걸 원천 차단함
        u_int_gated = self._apply_learnable_gating(u_int)

        # -------------------------------------------------------
        # 2. BPR Loss
        # -------------------------------------------------------
        pos_scores = (u_int_gated * p_int).sum(dim=-1)
        neg_scores = (u_int_gated * n_int).sum(dim=-1)
        
        bpr_loss = self.loss_fn(pos_scores, neg_scores)

        # -------------------------------------------------------
        # 3. Auxiliary Losses
        # -------------------------------------------------------
        # (1) Orthogonal Loss: 관심사 키들이 서로 겹치지 않게 (다양성 확보)
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l2")
        
        # (2) Threshold Regularization (Optional)
        # 임계값이 너무 커져서(Dead Neuron) 학습이 멈추는 것을 방지하기 위한 약한 제약
        # 혹은 너무 작아져서(All Pass) Gating이 무의미해지는 것을 방지
        # 여기서는 0으로 수렴하지 않게 아주 약하게만 잡음
        threshold_reg = torch.norm(self.threshold_bias, p=2) * 1e-5

        total_loss = bpr_loss + (self.lamda_orth * orth_loss) + threshold_reg

        # 로깅 (학습된 Threshold의 평균값을 보면 모델이 얼마나 엄격해졌는지 알 수 있음)
        with torch.no_grad():
            mean_threshold = F.softplus(self.threshold_bias).mean().item()

        log_dict = {
            'bpr': bpr_loss.item(),
            'orth': orth_loss.item(),
            'th_mean': mean_threshold
        }

        return total_loss, log_dict

    def get_final_item_embeddings(self):
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()

    def __str__(self):
        return f"CSAR_Learnable_Gated(K={self.num_interests}, D={self.embedding_dim})"