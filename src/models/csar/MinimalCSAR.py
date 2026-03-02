import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ...loss import SampledSoftmaxLoss
import numpy as np

def _safe_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    MPS에서 linalg.solve가 불안정하므로 CPU로 내려서 계산 후 복귀.
    """
    device = A.device
    if device.type in ('mps',):
        return torch.linalg.solve(A.cpu(), B.cpu()).to(device)
    return torch.linalg.solve(A, B)

class MinimalCSAR(BaseModel):
    """
    Minimal CSAR (v3.5) - Sampled Softplus Edition
    
    score(u, i) = scale * (E_u @ G @ E_i^T)
    
    - Loss: SampledSoftmaxLoss (InfoNCE)
    - Scale: Learnable logit scaling.
    - G: Centered OLS with EMA.
    """
    def __init__(self, config, data_loader):
        super(MinimalCSAR, self).__init__(config, data_loader)
        
        self.embedding_dim = self.config['model']['embedding_dim']
        self.ols_eps = self.config['model'].get('ols_eps', 1e-4)
        if isinstance(self.ols_eps, (list, np.ndarray)):
            self.ols_eps = self.ols_eps[0]
        self.reg_lambda = self.config['model'].get('reg_lambda', 500.0)
        if isinstance(self.reg_lambda, (list, np.ndarray)):
            self.reg_lambda = self.reg_lambda[0]
        self.ema_momentum = self.config['model'].get('ema_momentum', 0.05)

        self.n_users = self.data_loader.n_users
        self.n_items = self.data_loader.n_items

        self.E_u = nn.Embedding(self.n_users, self.embedding_dim)
        self.E_i = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Softplus와 일관된 스케일 파라미터
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        
        # Loss 함수 정의
        self.loss_fn = SampledSoftmaxLoss(temperature=0.1)
        
        # G 버퍼: 학습 중 EMA로 업데이트
        self.register_buffer('G', torch.eye(self.embedding_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.E_u.weight)
        nn.init.xavier_uniform_(self.E_i.weight)

    def _center(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=0, keepdim=True)

    def _compute_G(self, e_u_c: torch.Tensor, e_i_c: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Simple Covariance (Correlation) instead of OLS
            n = e_u_c.size(0)
            G = (e_u_c.t() @ e_i_c) / (n + 1e-6)
            return G.clamp(-5.0, 5.0)

    # ──────────────────────────────────────────────
    # Training / Prediction
    # ──────────────────────────────────────────────
    def forward(self, users):
        """Evaluation용 Score"""
        e_u = self.E_u(users)
        e_i = self.E_i.weight
        return self.score_scale * (e_u @ self.G @ e_i.t())

    def predict_for_pairs(self, user_ids, item_ids):
        e_u = self.E_u(user_ids)
        e_i = self.E_i(item_ids)
        return self.score_scale * (e_u @ self.G * e_i).sum(-1)

    def calc_loss(self, batch_data):
        users     = batch_data['user_id']
        pos_items = batch_data['pos_item_id']
        neg_items = batch_data['neg_item_id'] # [batch_size, num_negatives]
        
        e_u   = self.E_u(users).view(-1, self.embedding_dim)
        e_pos = self.E_i(pos_items).view(-1, self.embedding_dim)
        
        # 1. Update G (Centered OLS)
        with torch.no_grad():
            e_u_c   = self._center(e_u)
            e_pos_c = self._center(e_pos)
            G_batch = self._compute_G(e_u_c, e_pos_c)
            m = self.ema_momentum
            self.G.data.copy_((1.0 - m) * self.G + m * G_batch)

        # 2. Scores calculation
        pos_score = (e_u @ self.G * e_pos).sum(-1).unsqueeze(1) * self.score_scale
        
        if neg_items.dim() == 1:
            neg_items = neg_items.unsqueeze(1)
        
        e_neg = self.E_i(neg_items) # [batch_size, num_negs, dim]
        u_g = (e_u @ self.G).unsqueeze(1)
        neg_score = (u_g * e_neg).sum(-1) * self.score_scale

        # 3. SampledSoftmax Loss
        ssm_loss = self.loss_fn(pos_score, neg_score)
        
        return (ssm_loss,), {
            "loss": ssm_loss.item(),
            "scale": self.score_scale.item(),
            "g_sum": self.G.abs().sum().item()
        }

    def get_final_item_embeddings(self):
        return self.E_i.weight.detach()
