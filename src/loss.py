import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) 손실 함수
    """
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return loss


class DynamicMarginBPRLoss(nn.Module):
    """
    동적 마진 BPR 손실 함수
    모델이 쉽게 구분하는 쌍에는 더 큰 마진을 요구
    """
    def __init__(self, alpha=0.5):
        super(DynamicMarginBPRLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pos_scores, neg_scores):
        with torch.no_grad():
            diff = pos_scores - neg_scores
            dynamic_margin = self.alpha * torch.sigmoid(diff) 
        loss = F.softplus(-(pos_scores - neg_scores - dynamic_margin)).mean()
        return loss


class MSELoss(nn.Module):
    """
    Mean Squared Error (MSE) 손실 함수
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predictions, ratings):
        loss = self.loss_fn(predictions, ratings)
        return loss


class SampledSoftmaxLoss(nn.Module):
    """
    Sampled Softmax 손실 함수 (InfoNCE)
    
    특징:
    1. Temperature Scaling: 점수 분포 조절
    2. Scale Factor: Negative에 가중치 적용하여 Hard Negative 효과 강화 (1.0이면 표준 Softmax)
    3. logQ Correction (선택): Popularity 기반 Negative Sampling 시 인기편향 보정
       - 보정 공식: score - log(Q(i)), Q(i) = 아이템 i의 샘플링 확률
    """
    def __init__(self, temperature=0.1, scale_factor=1.0, log_q_correction=False):
        """
        Args:
            temperature (float): 온도 파라미터 (낮을수록 sharp한 분포)
            scale_factor (float): Negative 가중치 (1.0이면 표준 Softmax, >1이면 Hard Negative 강화)
            log_q_correction (bool): 인기편향 보정 사용 여부
        """
        super(SampledSoftmaxLoss, self).__init__()
        self.temperature = temperature
        self.scale_factor = scale_factor
        self.log_q_correction = log_q_correction
        self.log_q = None

    def set_log_q(self, sampling_weights):
        """
        Popularity 기반 샘플링 가중치로부터 log(Q(i)) 계산
        
        Args:
            sampling_weights (Tensor): 아이템별 샘플링 확률 [n_items]
        """
        if sampling_weights is not None:
            self.log_q = torch.log(sampling_weights + 1e-10)
        else:
            self.log_q = None

    def forward(self, pos_scores, neg_scores, neg_item_ids=None):
        """
        Args:
            pos_scores: [Batch, 1] Positive 점수
            neg_scores: [Batch, K] Negative 점수
            neg_item_ids: [Batch, K] Negative 아이템 ID (logQ correction 시 필요)
        """
        pos_logits = pos_scores / self.temperature
        neg_logits = neg_scores / self.temperature
        
        # logQ Correction (인기편향 보정)
        if self.log_q_correction and self.log_q is not None and neg_item_ids is not None:
            log_q_neg = self.log_q[neg_item_ids]
            neg_logits = neg_logits - log_q_neg
        
        # Scale Factor 적용 (Hard Negative 강화)
        if self.scale_factor != 1.0:
            log_scale = torch.log(torch.tensor(self.scale_factor, device=pos_logits.device, dtype=pos_logits.dtype))
            neg_logits = log_scale + neg_logits
        
        # [Batch, 1 + K]
        all_logits = torch.cat([pos_logits, neg_logits], dim=1)
        
        # Positive는 항상 Column 0
        labels = torch.zeros(all_logits.shape[0], dtype=torch.long, device=all_logits.device)
        
        return F.cross_entropy(all_logits, labels)