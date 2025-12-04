import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import BPRLoss
from .base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer

class CSAR_R_KD(BaseModel):
    """
    CSAR_R (Residual Connection) + BPR Loss
    """
    def __init__(self, config, data_loader):
        super(CSAR_R_KD, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda = self.config['model']['orth_loss_weight']
        self.scale = self.config['model'].get('scale', True)
        self.Dummy = self.config['model'].get('dummy', False)

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # CoSupportAttentionLayer 사용
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim, scale=self.scale, Dummy=self.Dummy)

        self._init_weights()
        self.loss_fn = BPRLoss()

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

        # --- 1. 임베딩 및 관심사 추출 ---
        user_embs = self.user_embedding(users)
        pos_item_embs = self.item_embedding(pos_items)
        neg_item_embs = self.item_embedding(neg_items)

        # 관심사 벡터 (CSAR 파트)
        user_interests = self.attention_layer(user_embs)
        pos_item_interests = self.attention_layer(pos_item_embs)
        neg_item_interests = self.attention_layer(neg_item_embs)

        # --- 2. 점수 계산 (분리해서 계산!) ---
        
        # A. CSAR Part (Interest)
        pos_interest_scores = (user_interests * pos_item_interests).sum(dim=-1)
        neg_interest_scores = (user_interests * neg_item_interests).sum(dim=-1)

        # B. MF Part (Residual/Prior)
        pos_res_scores = (user_embs * pos_item_embs).sum(dim=-1)
        neg_res_scores = (user_embs * neg_item_embs).sum(dim=-1)

        # C. Combined (Final Prediction)
        pos_scores = pos_interest_scores + pos_res_scores
        neg_scores = neg_interest_scores + neg_res_scores
        
        # --- 3. Loss 구성 (베이지안 3단 합체) ---

        # Loss 1: Main BPR (최종 성능 책임)
        # (Total Score가 순서를 잘 맞추는가?)
        loss_main = self.loss_fn(pos_scores, neg_scores)

        # Loss 2: Aux BPR (뼈대 강화)
        # (MF 혼자서도 순서를 잘 맞추는가? -> Prior가 튼튼해짐)
        loss_aux = self.loss_fn(pos_res_scores, neg_res_scores)

        # Loss 3: Distillation (지식 증류 - Sigmoid & BCE)
        # (CSAR이 MF의 안정성을 닮아가라. 단, Teacher인 MF는 detach 필수!)
        # Positive에 대해서만 증류해도 되고, Negative까지 해도 됨. 여기선 둘 다.
        
        # Teacher(MF)를 확률(0~1)로 변환 (Sigmoid)
        with torch.no_grad():
            teacher_pos_probs = torch.sigmoid(pos_res_scores)
            teacher_neg_probs = torch.sigmoid(neg_res_scores)
        
        # Student(CSAR)가 Teacher의 확률 분포를 BCE로 따라함
        distill_pos = F.binary_cross_entropy_with_logits(pos_interest_scores, teacher_pos_probs)
        distill_neg = F.binary_cross_entropy_with_logits(neg_interest_scores, teacher_neg_probs)
        loss_distill = (distill_pos + distill_neg) / 2

        # --- 4. 최종 합산 ---
        # orth_loss: 관심사 직교 규제
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")
        
        # 가중치 설정 (하이퍼파라미터 튜닝 포인트)
        # alpha: MF 보조 로스 가중치 (보통 0.5 ~ 1.0)
        # beta: 증류 로스 가중치 (희소 데이터일수록 높임, 0.1 ~ 1.0)
        alpha = 0.5
        beta = 0.5

        # 로깅용
        params_to_log = {}
        if isinstance(self.attention_layer.scale, nn.Parameter):
            params_to_log['scale'] = self.attention_layer.scale.item()
        
        return (loss_main , beta * loss_distill, self.lamda * orth_loss), params_to_log
    
    def __str__(self):
        return f"CSAR_R_BPR(num_interests={self.num_interests}, embedding_dim={self.embedding_dim})"
