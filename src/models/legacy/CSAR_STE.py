import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import BPRLoss
from .base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer

class CSAR_STE(BaseModel):
    """
    CSAR Select Hybrid with Straight-Through Estimator (STE).
    
    [Philosophy]
    "No Heuristics, Just Math."
    - Forward Pass: Strict Top-K selection (Noise-free).
    - Backward Pass: Soft Gradient Flow (Learning-friendly).
    
    이 방식은 모델이 '어떻게 해야 Top-K 안에 들 수 있는지'를
    미분 가능한 방식으로 학습하게 만듭니다.
    """
    def __init__(self, config, data_loader):
        super(CSAR_STE, self).__init__(config, data_loader)

        self.embedding_dim = self.config['model']['embedding_dim']
        self.active_k = self.config['model']['active_interests']
        self.lamda_orth = self.config['model'].get('orth_loss_weight', 0.1)
        
        # Shared Embeddings
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        # CSAR Layer
        self.attention_layer = CoSupportAttentionLayer(self.embedding_dim, self.embedding_dim)

        self.loss_fn = BPRLoss()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _get_ste_interests(self, embedding):
        interests = self.attention_layer(embedding)  # [B, D]

        # === 안정적 Top-K 선택 ===
        vals, idx = torch.topk(interests, k=self.active_k, dim=1)

        mask_hard = torch.zeros_like(interests)
        mask_hard.scatter_(1, idx, 1.0)

        # === Softmask (Gradient Flow) ===
        interests_soft = torch.softmax(interests / 0.2, dim=1)

        # === STE ===
        return (interests * mask_hard - interests_soft).detach() + interests_soft

    
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

        mf_score = torch.matmul(u_emb, all_i_emb.t())

        # STE 적용된 Sparse Interests
        u_int = self._get_ste_interests(u_emb) 
        i_int = self.attention_layer(all_i_emb) # 아이템은 정보 보존 (Dense)
        
        csar_score = torch.einsum('bk,nk->bn', u_int, i_int)
        
        return mf_score + csar_score

    def predict_for_pairs(self, user_ids, item_ids):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        mf_score = (u_emb * i_emb).sum(dim=-1)

        u_int = self._get_ste_interests(u_emb)
        i_int = self.attention_layer(i_emb)
        csar_score = (u_int * i_int).sum(dim=-1)

        return mf_score + csar_score

    def calc_loss(self, batch_data):
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].squeeze(-1)

        pos_scores = self.predict_for_pairs(users, pos_items)
        neg_scores = self.predict_for_pairs(users, neg_items)
        
        bpr_loss = self.loss_fn(pos_scores, neg_scores)
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l2" )
        params_to_log = {
            'scale': self.attention_layer.scale.item()
        }

        return (bpr_loss ,self.lamda_orth * orth_loss), params_to_log