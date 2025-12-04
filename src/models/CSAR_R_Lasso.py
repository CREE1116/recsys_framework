
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import BPRLoss
from .base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer

class CSAR_R_Lasso(BaseModel):
    """
    CSAR Hybrid with L1 Sparsity (Lasso)
    
    [철학]
    "No Free Lunch (공짜 점심은 없다)"
    - MF(Residual)는 기본 제공되지만, CSAR(Interest)를 쓰려면 대가를 치러야 한다.
    - 이 '대가(L1 Penalty)' 때문에 모델은 정말 필요한 관심사만 남기고 스스로 0으로 컷오프한다.
    """
    def __init__(self, config, data_loader):
        super(CSAR_R_Lasso, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.num_interests = self.config['model']['num_interests']
        self.lamda_orth = self.config['model'].get('orth_loss_weight', 0.1)
        self.scale = self.config['model'].get('scale', True)
        
        # [핵심] Sparsity 가중치 (Lasso 강도)
        # 이 값이 클수록 더 엄격하게 자릅니다. (보통 1e-3 ~ 1e-4 추천)
        # 튜닝이 필요 없다고 하셨지만, '얼마나 엄격할지'는 정해줘야 합니다. 
        # 한 번만 정하면 모델이 알아서 맞춥니다.
        self.l1_lambda = self.config['model'].get('l1_lambda', 1e-4)

        # Embeddings & Layers
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim, scale=self.scale)

        self.loss_fn = BPRLoss()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_final_item_embeddings(self):
        """
        CSAR_R의 경우 최종 아이템 임베딩을 정의하기가 모호할 수 있음 (Residual 때문).
        하지만 Topic-space 분석을 위해 Interest Vector를 반환하거나, 
        혹은 검색을 위해 Concatenation 등을 고려할 수 있음.
        여기서는 기존 CSAR과 동일하게 Interest Vector를 반환.
        """
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()

    def forward(self, users):
        u_emb = self.user_embedding(users)
        all_i_emb = self.item_embedding.weight

        # 1. MF Score (Base)
        mf_score = torch.matmul(u_emb, all_i_emb.t())

        # 2. CSAR Score (Detail)
        u_int = self.attention_layer(u_emb)
        i_int = self.attention_layer(all_i_emb)
        csar_score = torch.einsum('bk,nk->bn', u_int, i_int)
        
        return mf_score + csar_score

    def predict_for_pairs(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        # MF
        mf_score = (u_emb * i_emb).sum(dim=-1)

        # CSAR (Activations for L1)
        u_int = self.attention_layer(u_emb) # [B, K]
        i_int = self.attention_layer(i_emb)
        csar_score = (u_int * i_int).sum(dim=-1)
        
        return mf_score + csar_score

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)

        # 점수 계산 및 관심사 벡터 추출
        pos_scores = self.predict_for_pairs(users, pos_items)
        neg_scores = self.predict_for_pairs(users, neg_items)
        u_int_pos = self.attention_layer(self.user_embedding(users))
        # 1. BPR Loss
        bpr_loss = self.loss_fn(pos_scores, neg_scores)

        # 2. [핵심] L1 Sparsity Loss (Auto-Cutoff)
        # 유저가 활성화한 관심사들의 '합'을 벌점으로 부과합니다.
        # 모델은 BPR을 유지하면서 이 합을 0으로 줄이려 노력합니다 -> 노이즈 제거됨
        l1_loss = u_int_pos.abs().mean()

        # 3. Orthogonal Loss
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l2" )
        params_to_log = {
            'l1_sparsity': l1_loss.item(), # 이 값이 줄어들수록 모델이 엄격해진 것
            'scale': self.attention_layer.scale.item()
        }

        return (bpr_loss, self.lamda_orth * orth_loss, self.l1_lambda * l1_loss), params_to_log