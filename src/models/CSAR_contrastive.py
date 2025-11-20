import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.loss import BPRLoss
from .base_model import BaseModel
from .CSAR_BPR import CoSupportAttentionLayer 
from src.loss import EntropyAdaptiveInfoNCE

class CSAR_contrastive(BaseModel):
    """
    CSAR + Entropy-Adaptive Contrastive Learning
    SCI 투고용 핵심 논리: "Sparse한 유저일수록 관심사 엔트로피가 높으므로, 적응형 온도를 통해 학습 난이도를 조절한다."
    """
    def __init__(self, config, data_loader):
        super(CSAR_contrastive, self).__init__(config, data_loader)

        self.num_interests = self.config['model']['num_interests']
        self.embedding_dim = self.config['model']['embedding_dim']
        
        # 가중치 (논문 실험 시 Sensitivity Analysis 필수)
        self.lamda_orth = self.config['model'].get('orth_loss_weight', 0.1) 
        self.lamda_cl = self.config['model'].get('contrastive_loss_weight', 0.1)
        
        # CL Hyperparams
        self.noise_sigma = self.config['model'].get('noise_sigma', 0.1)
        self.base_tau = self.config['model'].get('base_tau', 0.1)
        self.alpha = self.config['model'].get('alpha', 0.5)

        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)
        
        self.attention_layer = CoSupportAttentionLayer(self.num_interests, self.embedding_dim)

        self._init_weights()
        self.bpr_loss_fn = BPRLoss()
        
        self.contrastive_module = EntropyAdaptiveInfoNCE(
            base_tau=self.config['model'].get('base_tau', 0.1),
            alpha=self.config['model'].get('alpha', 0.5)
        )

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    # --- Utility: Noise Generator ---
    def _add_noise(self, x):
        """학습 중에만 가우시안 노이즈 추가"""
        if self.training:
            noise = torch.randn_like(x) * self.noise_sigma
            return x + noise
        return x

    def forward(self, users):
        # Inference용 (수정 없음)
        user_embs = self.user_embedding(users)
        all_item_embs = self.item_embedding.weight
        user_interests = self.attention_layer(user_embs)
        all_item_interests = self.attention_layer(all_item_embs)
        scores = torch.einsum("bk,nk->bn", user_interests, all_item_interests)
        return scores

    def predict_for_pairs(self, user_ids, item_ids):
        # Evaluation용 (수정 없음)
        user_embs = self.user_embedding(user_ids)
        item_embs = self.item_embedding(item_ids)
        user_interests = self.attention_layer(user_embs)
        item_interests = self.attention_layer(item_embs)
        scores = (user_interests * item_interests).sum(dim=-1)
        return scores

    def get_final_item_embeddings(self):
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()

    def calc_loss(self, batch_data):
        # 1. Data Loading & BPR Logic (기존과 동일)
        users = batch_data['user_id'].squeeze(-1)
        pos_items = batch_data['pos_item_id'].squeeze(-1)
        neg_items = batch_data['neg_item_id']
        if neg_items.dim() == 2: neg_items = neg_items.view(-1)

        user_embs = self.user_embedding(users)
        pos_item_embs = self.item_embedding(pos_items)
        neg_item_embs = self.item_embedding(neg_items)

        user_interests = self.attention_layer(user_embs)
        pos_interests = self.attention_layer(pos_item_embs)
        neg_interests = self.attention_layer(neg_item_embs)

        # BPR Calc
        pos_scores = (user_interests * pos_interests).sum(dim=-1)
        if neg_interests.dim() == 2:
            neg_scores = (user_interests * neg_interests).sum(dim=-1)
        else:
            neg_scores = (user_interests.unsqueeze(1) * neg_interests).sum(dim=-1)
        
        bpr_loss = self.bpr_loss_fn(pos_scores, neg_scores)

        # --- 2. Contrastive Logic (이제 모듈을 써서 깔끔하게!) ---
        
        # [User Side]
        # 1) Tau 진단 (모듈에 위임)
        tau_u = self.contrastive_module.get_adaptive_tau(
            user_interests, self.attention_layer.interest_keys
        )
        # 2) Noise & Loss (모듈에 위임)
        cl_loss_u = self.contrastive_module(
            self._add_noise(user_interests), 
            self._add_noise(user_interests), 
            tau_u
        )

        # [Item Side] (In-batch unique)
        batch_items = torch.cat([pos_items, neg_items], dim=0)
        unique_items = torch.unique(batch_items)
        item_interests_unique = self.attention_layer(self.item_embedding(unique_items))
        
        # 1) Tau 진단
        tau_i = self.contrastive_module.get_adaptive_tau(
            item_interests_unique, self.attention_layer.interest_keys
        )
        # 2) Noise & Loss
        cl_loss_i = self.contrastive_module(
            self._add_noise(item_interests_unique), 
            self._add_noise(item_interests_unique), 
            tau_i
        )

        # --- 3. Orthogonal Loss ---
        orth_loss = self.attention_layer.get_orth_loss("l2")

        to_log = {
            'loss/bpr_loss': bpr_loss.item(),
            'loss/contrastive_loss_user': cl_loss_u.item(),
            'loss/contrastive_loss_item': cl_loss_i.item(),
            'loss/orth_loss': orth_loss.item()
        }
        return (bpr_loss,self.lamda_cl * cl_loss_u, self.lamda_cl * cl_loss_i,self.lamda_orth * orth_loss), to_log