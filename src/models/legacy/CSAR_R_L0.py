import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss import BPRLoss
from .base_model import BaseModel
from .csar_layers import CoSupportAttentionLayer

class L0Gate(nn.Module):
    """
    Hard-Concrete based L0 gate (유저별).
    - alpha: [n_users, K] learnable logits
    - sampling: hard 0/1에 가깝게 샘플링하되 soft relaxation(s)을 반환
    - expected_active: approx sigmoid(alpha) 를 L0 페널티로 사용
    """
    def __init__(self, n_users, K, droprate=0.5, temperature=2./3., limit_low=-0.1, limit_high=1.1, eps=1e-6):
        super().__init__()
        self.n_users = n_users
        self.K = K
        self.droprate = float(droprate)
        self.temperature = float(temperature)
        self.limit_low = float(limit_low)
        self.limit_high = float(limit_high)
        self.eps = eps

        # 유저별 alpha 파라미터 (logit)
        # 초기값은 0에 가깝게 해서 초기엔 gate가 절반 확률로 활성화되도록 함
        self.alpha = nn.Parameter(torch.zeros(n_users, K))

        # precompute for expected prob offset if needed (kept simple here)
        # (고급 튜닝: droprate을 직접 맞추려면 alpha-offset 계산을 넣어야 함)
    
    def _sample_u(self, shape, device):
        return torch.rand(shape, device=device).clamp(self.eps, 1.0 - self.eps)

    def sample_gate(self, user_ids, training=True):
        """
        user_ids: [B] LongTensor
        returns:
          gate_hard: [B, K] (0/1)
          gate_soft: [B, K] (continuous relaxation in [0,1])
          prob_active: [B, K] (sigmoid(alpha) approx expected prob) - for penalty
        """
        device = self.alpha.device
        a = self.alpha[user_ids]  # [B, K]

        # expected activation prob (간단 근사) - 이걸 L0 penalty로 사용
        prob_active = torch.sigmoid(a)  # [B, K]

        if training:
            # Concrete sample
            u = self._sample_u(a.shape, device=device)
            logistic = torch.log(u) - torch.log(1.0 - u)
            s = torch.sigmoid((logistic + a) / self.temperature)  # (0,1)
            # stretch to (limit_low, limit_high) then clamp to [0,1]
            s_bar = s * (self.limit_high - self.limit_low) + self.limit_low
            z = torch.clamp(s_bar, 0.0, 1.0)
            gate_hard = (z > 0.5).float()
            return gate_hard, z, prob_active
        else:
            # inference: deterministic gating by thresholding expected prob
            gate_hard = (prob_active > self.droprate).float()
            # soft = prob_active for interpretability
            return gate_hard, prob_active, prob_active

    def expected_l0(self, user_ids):
        """
        배치의 expected active 비율 (평균)
        사용해 L0 penalty 계산
        """
        a = self.alpha[user_ids]  # [B, K]
        prob_active = torch.sigmoid(a)
        return prob_active.mean()  # 평균 활성 비율 (스칼라)


class CSAR_R_L0(BaseModel):
    """
    CSAR + User-adaptive L0 gating (Hard-Concrete style)
    - config['model'] expected keys:
        - embedding_dim
        - num_interests
        - l1_lambda (optional, not used here)
        - lam_l0 (float) : weight for L0 penalty
        - lam_orth (float) : orth loss weight
        - droprate (float) : inference threshold / target sparsity (0~1)
        - l0_temp (float) : concrete temperature
    """
    def __init__(self, config, data_loader):
        super(CSAR_R_L0, self).__init__(config, data_loader)

        self.embedding_dim = int(self.config['model']['embedding_dim'])
        self.lamda_orth = float(self.config['model'].get('orth_loss_weight', 0.1))
        self.lamda_l0 = float(self.config['model'].get('lamda_l0', 1e-3))
        self.droprate = float(self.config['model'].get('droprate', 0.5))

        # Embeddings & Layers
        self.user_embedding = nn.Embedding(self.data_loader.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.data_loader.n_items, self.embedding_dim)

        # CoSupportAttentionLayer: (num_interests, embedding_dim) -> when called with user_emb returns [B, K]
        # signature assumed: CoSupportAttentionLayer(num_interests, embedding_dim, scale=...)
        self.attention_layer = CoSupportAttentionLayer(self.embedding_dim, self.embedding_dim, scale=self.config['model'].get('scale', True))

        # L0 gate (user-specific)
        self.l0_gate = L0Gate(self.data_loader.n_users, self.embedding_dim, droprate=self.droprate)

        self.loss_fn = BPRLoss()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _get_user_interests(self, user_ids, training=True):
        """
        user_ids: LongTensor [B]
        returns: gated interest vectors [B, K], plus optionally gate info if needed
        """
        # user embedding
        u_emb = self.user_embedding(user_ids)         # [B, emb_dim]
        interests = self.attention_layer(u_emb)       # [B, K]

        gate_hard, gate_soft, prob_active = self.l0_gate.sample_gate(user_ids, training=training)
        # apply hard mask on forward but allow gradient via soft (we already used concrete soft in sampling)
        # here we use hard gating for actual forward (consistent with STE-like behavior)
        gated = interests * gate_hard

        # To keep gradient flow through soft relaxation, use "straight-through" trick:
        # gated = (gated - interests * gate_soft).detach() + interests * gate_soft
        # but because gate_soft used above is in (0,1) and we already used gate_hard, better to use:
        gated = (interests * gate_hard - interests * gate_soft).detach() + interests * gate_soft

        return gated, gate_hard, gate_soft, prob_active

    def get_final_item_embeddings(self):
        """
        Topic-space analysis용 아이템 interest vector 반환 (detached)
        """
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()

    def forward(self, users):
        """
        users: LongTensor [B] -> returns [B, n_items] score matrix
        """
        u_emb = self.user_embedding(users)                 # [B, emb]
        all_i_emb = self.item_embedding.weight             # [n_items, emb]

        # MF (base)
        mf_score = torch.matmul(u_emb, all_i_emb.t())      # [B, n_items]

        # CSAR detail (gated)
        u_int, _, _, _ = self._get_user_interests(users, training=self.training)
        i_int = self.attention_layer(all_i_emb)            # [n_items, K]
        csar_score = torch.einsum('bk,nk->bn', u_int, i_int)

        return mf_score + csar_score

    def predict_for_pairs(self, user_ids, item_ids, training=False):
        """
        pairwise prediction - use deterministic gating in inference (training flag switches sampling mode)
        """
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        mf_score = (u_emb * i_emb).sum(dim=-1)

        # gating
        if training:
            u_int, _, _, _ = self._get_user_interests(user_ids, training=True)
        else:
            # inference deterministic gating: use expected prob thresholding
            interests = self.attention_layer(u_emb)   # [B, K]
            # compute deterministic gate from alpha
            prob_active = torch.sigmoid(self.l0_gate.alpha[user_ids])  # [B, K]
            gate_hard = (prob_active > self.droprate).float()
            # Use straight-through with soft=prob_active
            u_int = (interests * gate_hard - interests * prob_active).detach() + interests * prob_active

        i_int = self.attention_layer(i_emb)
        csar_score = (u_int * i_int).sum(dim=-1)

        return mf_score + csar_score

    def calc_loss(self, batch_data):
        """
        batch_data expected keys:
            - 'user_id', 'pos_item_id', 'neg_item_id' (each [B,1] typically)
        returns:
            tuple of losses (bpr_loss, orth_loss * lamda_orth, l0_loss * lam_l0)
            params_to_log dict
        """
        users = batch_data['user_id'].squeeze(-1).long()
        pos_items = batch_data['pos_item_id'].squeeze(-1).long()
        neg_items = batch_data['neg_item_id'].squeeze(-1).long()

        # scores
        pos_scores = self.predict_for_pairs(users, pos_items, training=True)
        neg_scores = self.predict_for_pairs(users, neg_items, training=True)

        # bpr
        bpr_loss = self.loss_fn(pos_scores, neg_scores)

        # orth loss (from attention layer)
        orth_loss = self.attention_layer.get_orth_loss(loss_type="l2")

        # expected L0 penalty (batch mean)
        l0_penalty = self.l0_gate.expected_l0(users)  # scalar batch mean of prob_active

        params_to_log = {
            'l0_prob': l0_penalty.item(),
            'scale': float(getattr(self.attention_layer, 'scale', 1.0))
        }

        total_losses = (bpr_loss, self.lamda_orth * orth_loss, self.lamda_l0 * l0_penalty)
        return total_losses, params_to_log

    def _get_all_item_interests(self):
        """
        *모든* 아이템의 K-dim 관심사 벡터를 계산합니다.
        """
        all_item_embs = self.item_embedding.weight
        return self.attention_layer(all_item_embs).detach()