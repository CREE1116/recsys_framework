import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import BPRLoss
from .base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer, AdaptiveContrastiveLoss

class CSAR_R_contrastive(BaseModel):
    """
    CSAR_R (Residual) + Entropy-Adaptive Contrastive Learning
    """
    def __init__(self, config, data_loader):
        super(CSAR_R_contrastive, self).__init__(config, data_loader)

        self.num_interests = self.config['model']['num_interests']
        self.embedding_dim = self.config['model']['embedding_dim']
        
        # 가중치
        self.lamda_orth = self.config['model'].get('orth_loss_weight', 0.1) 
        self.lamda_cl = self.config['model'].get('contrastive_loss_weight', 0.1)
        self.scale = self.config['model'].get('scale', True)
        
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim, scale=self.scale)

        self._init_weights()
        self.bpr_loss_fn = BPRLoss()
        
        # 대조 학습 모듈
        self.cl_criterion = AdaptiveContrastiveLoss(
            noise_sigma=config['model'].get('noise_sigma', 0.1),
            base_tau=config['model'].get('base_tau', 0.1),
            alpha=config['model'].get('alpha', 0.5)
        )

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users):
        # Inference용
        user_embs = self.user_embedding(users)
        all_item_embs = self.item_embedding.weight
        
        user_interests = self.attention_layer(user_embs)
        all_item_interests = self.attention_layer(all_item_embs)
        
        # 1. Interest Score
        interest_scores = torch.einsum("bk,nk->bn", user_interests, all_item_interests)
        
        # 2. Residual Score
        res_scores = torch.matmul(user_embs, all_item_embs.T)
        
        return interest_scores + res_scores

    def predict_for_pairs(self, user_ids, item_ids):
        # Evaluation용
        user_embs = self.user_embedding(user_ids)
        item_embs = self.item_embedding(item_ids)
        
        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(item_embs)
        
        # 1. Interest Score
        interest_scores = (user_interests * item_interests).sum(dim=-1)
        
        # 2. Residual Score
        res_scores = (user_embs * item_embs).sum(dim=-1)
        
        return interest_scores + res_scores

    def get_final_item_embeddings(self):
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()

    def calc_loss(self, batch_data):
        # 데이터 로딩
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id'].view(-1) # Neg Flatten

        # 임베딩 조회
        user_embs = self.user_embedding(users)
        pos_item_embs = self.item_embedding(pos_items)
        neg_item_embs = self.item_embedding(neg_items)

        # ------------------------------------------------------------
        # 1. Main Task (BPR) - Interest + Residual
        # ------------------------------------------------------------
        u_ints = self.attention_layer(user_embs)
        p_ints = self.attention_layer(pos_item_embs)
        n_ints = self.attention_layer(neg_item_embs)

        # 점수 계산 (Residual 포함)
        pos_scores = (u_ints * p_ints).sum(dim=-1) + (user_embs * pos_item_embs).sum(dim=-1)
        neg_scores = (u_ints * n_ints).sum(dim=-1) + (user_embs * neg_item_embs).sum(dim=-1)
        
        loss_bpr = self.bpr_loss_fn(pos_scores, neg_scores)

        # ------------------------------------------------------------
        # 2. Auxiliary Task (Contrastive)
        # ------------------------------------------------------------
        loss_cl_u = self.cl_criterion(u_ints, self.attention_layer.interest_keys)
        
        # 아이템은 중복 제거 후 계산 (효율성)
        batch_items = torch.unique(torch.cat([pos_items, neg_items]))
        batch_item_embs = self.item_embedding(batch_items)
        i_ints_unique = self.attention_layer(batch_item_embs)
        
        loss_cl_i = self.cl_criterion(i_ints_unique, self.attention_layer.interest_keys)

        # ------------------------------------------------------------
        # 3. Regularization & Logging
        # ------------------------------------------------------------
        to_log = {
            'loss/bpr_loss': loss_bpr.item(),
            'loss/contrastive_loss_user': loss_cl_u.item(),
            'loss/contrastive_loss_item': loss_cl_i.item(),
        }
        
        # Orthogonal Loss (Optional, Contrastive가 어느정도 역할을 하지만 명시적으로 줄 수도 있음)
        # 여기서는 config에 orth_loss_weight가 있으면 추가
        total_loss = loss_bpr + self.lamda_cl * loss_cl_u + self.lamda_cl * loss_cl_i
        
        if self.lamda_orth > 0:
            orth_loss = self.attention_layer.get_orth_loss(loss_type="l1")
            total_loss += self.lamda_orth * orth_loss
            to_log['loss/orth_loss'] = orth_loss.item()

        return (total_loss, self.lamda_cl * loss_cl_u), to_log
