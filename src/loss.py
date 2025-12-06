import torch
import torch.nn as nn
import numpy as np  
import torch.nn.functional as F
import math

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
    def __init__(self, alpha=0.5):
        super(DynamicMarginBPRLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pos_scores, neg_scores):
        # 1. 현재 모델이 느끼는 난이도 (점수 차이)
        with torch.no_grad():
            diff = pos_scores - neg_scores
            # 차이가 클수록(잘할수록) 마진을 세게 검
            dynamic_margin = self.alpha * torch.sigmoid(diff) 
        # 2. 동적 마진 적용
        loss = F.softplus(-(pos_scores - neg_scores - dynamic_margin)).mean()
        return loss

class PolyLoss(nn.Module):
    def __init__(self, epsilon=1.0, reduction='mean'):
        """
        epsilon: 
          > 0 : 쉬운 문제도 더 파고들어라! (희소 데이터 추천, Gradient 강화)
          < 0 : 어려운 문제에 집중해라! (Focal Loss 느낌)
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        # 1. 기본 CE Loss 계산 (Base)
        ce_loss = self.ce(logits, targets)
        
        # 2. 정답 클래스의 확률(Pt) 계산
        # Softmax를 통과시킨 후 정답 인덱스의 확률만 가져옴
        pt = torch.gather(F.softmax(logits, dim=-1), 1, targets.unsqueeze(1)).squeeze(1)
        
        # 3. PolyLoss 수식 적용
        # Loss = CE + eps * (1 - Pt)
        poly_loss = ce_loss + self.epsilon * (1 - pt)
        
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        else:
            return poly_loss

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
    

class scaled_sampled_softmax(nn.Module):

    def __init__(self, temperature=0.4, scale_factor=20.0):
        super(scaled_sampled_softmax, self).__init__()
        self.temperature = temperature
        self.scale_factor = scale_factor

    def forward(self, pos_scores, neg_scores):
        pos_scores = pos_scores / self.temperature  # (B,)
        neg_scores = neg_scores / self.temperature  # (B, N_neg)

        # scale_factor를 tensor로 만들어 autograd 연결 유지
        scale_factor = torch.tensor(self.scale_factor, device=pos_scores.device, dtype=pos_scores.dtype)
        log_scale = torch.log(scale_factor)  # scalar tensor, requires_grad=False
        # log(C * exp(neg/T)) = log(C) + neg/T
        neg_terms = log_scale + neg_scores  # (B, N_neg)
        # (B, 1 + N_neg)
        all_terms = torch.cat([pos_scores.unsqueeze(1), neg_terms], dim=1)
        # log(exp(pos/T) + C * sum(exp(neg/T)))
        log_denominator = torch.logsumexp(all_terms, dim=1)  # (B,)
        # -log(P(pos)) = log_denominator - pos/T
        loss = log_denominator - pos_scores
        return loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE 손실 함수
    """
    def __init__(self, temperature = 0.4):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores (torch.Tensor): (batch_size, 1)
            neg_scores (torch.Tensor): (batch_size, num_negatives)
        """
        # (batch_size, 1 + num_negatives)
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)
        all_scores = all_scores / self.temperature
        
        # Positive score에 해당하는 인덱스는 항상 0
        labels = torch.zeros(all_scores.shape[0], dtype=torch.long, device=all_scores.device)
        
        loss = F.cross_entropy(all_scores, labels)
        return loss

class CSARLoss(nn.Module):
    """
    CSAR 모델 전용 Log-Energy InfoNCE Loss
    
    Theoretical Background:
    1. Energy-based Model: Softplus 활성화를 통한 Intensity 내적(Energy)을 점수로 사용.
    2. Log-Space Transform: 곱셈 기반 에너지(Energy)를 덧셈 기반 로짓(Log-Energy)으로 변환하여
       Softmax의 지수 폭발(Explosion)을 방지하고 Magnitude 정보를 비율(Ratio)로 학습.
    3. Fixed Temperature: 학습 가능한 스케일 대신 고정된 온도를 사용하여 
       Gradient가 온전히 임베딩 학습에 집중되도록 강제 (Gradient Stealing 방지).
    
    * Note: CSAR 모델의 Co-Support 구조가 이미 내부적으로 관심사 필터링을 수행하므로, 
      LogQ Correction을 통한 강제적인 인기도 편향 제거는 오히려 성능을 저해할 수 있어 제거함.
    """
    def __init__(self, temperature=0.1):
        """
        Args:
            temperature (float): 고정된 온도 값. 
                                 값이 작을수록(0.1) 분포가 뾰족해져 Hard Negative에 집중하고,
                                 값이 클수록(1.0) 분포가 부드러워짐.
        """
        super(CSARLoss, self).__init__()
        
        # Fixed Temperature
        # 학습 가능한 파라미터를 제거하여 Gradient가 임베딩 업데이트에만 쓰이도록 함.
        self.temperature = temperature

    def forward(self, user_intensities, batch_item_intensities):
        """
        Args:
            user_intensities (Tensor): [Batch, K] - Softplus 통과된 유저 벡터 (Intensity)
            batch_item_intensities (Tensor): [Batch, K] - 배치 내 아이템들의 벡터 (Candidates)
            
        Returns:
            loss (Tensor): Scalar Loss
        """
        # 1. Energy Score Calculation (Co-Support)
        # In-Batch Negative Sampling: 배치 내 모든 유저 vs 배치 내 모든 아이템
        # scores: [B, B] (i행 j열 = 유저 i와 아이템 j의 에너지)
        # Softplus * Softplus 구조라 값이 매우 클 수 있음 (0 ~ inf)
        energy_scores = torch.matmul(user_intensities, batch_item_intensities.t())
        
        # 3. Fixed Temperature Scaling
        # Log로 변환된 에너지(비율)를 온도로 나누어 분포의 선명도(Sharpness) 조절
        # Scale 학습을 제거하고 고정된 상수로 나눔
        final_logits = energy_scores / self.temperature
        
        # 4. Cross Entropy (Standard InfoNCE)
        # In-Batch 상황이므로 유저 i의 정답은 i번째 아이템임 (Diagonal is positive)
        labels = torch.arange(len(user_intensities), device=user_intensities.device)
        loss = F.cross_entropy(final_logits, labels)
        
        return loss

class NormalizedSampledSoftmaxLoss(nn.Module):
    """
    Normalized Sampled Softmax Loss
    
    특징:
    1. Z-Score Normalization: 입력된 에너지 점수의 분포를 정규화하여(Mean=0, Std=1) 학습 안정성을 극대화함.
       - Temperature 의존성 문제를 해결하고, Gradient Explosion을 방지.
    2. Sampled Softmax Correction: In-Batch Negative Sampling으로 인한 편향을 보정.
       - Correction Term: log((N-1)/(B-1))을 Negative Logits에 더해줌으로써, 
         마치 전체 아이템(N)에 대해 Softmax를 수행하는 것과 유사한 효과를 냄.
    3. Fixed Temperature: 정규화된 분포에 맞춰 고정된 온도를 사용하여 학습의 일관성 유지.
    """
    def __init__(self, n_items, temperature=0.1):
        """
        Args:
            n_items (int): 전체 아이템 수 (Sampled Softmax 보정용)
            temperature (float): 고정된 온도 값.
        """
        super(NormalizedSampledSoftmaxLoss, self).__init__()
        
        self.n_items = n_items
        self.temperature = temperature

    def forward(self, user_intensities, batch_item_intensities):
        """
        Args:
            user_intensities (Tensor): [Batch, K] - Softplus 통과된 유저 벡터
            batch_item_intensities (Tensor): [Batch, K] - 배치 내 아이템 벡터
        """
        # 1. Energy Score Calculation
        # scores: [B, B] (0 ~ inf)
        energy_scores = torch.matmul(user_intensities, batch_item_intensities.t())
        
        # 2. Global Z-Score Normalization (Standardization)
        # Final Score Matrix 자체를 정규화하여 스케일 문제를 원천 차단
        mean = energy_scores.mean()
        std = energy_scores.std()
        energy_scores = (energy_scores - mean) / (std + 1e-9)
        
        # 3. Fixed Temperature Scaling
        logits = energy_scores / self.temperature
        
        # 4. Sampled Softmax Correction (Consistent Estimation)
        # In-Batch Negative들이 전체 Negative의 일부임을 감안하여 점수 보정
        # 보정: Neg Logits += log((N-1)/(B-1))
        if self.training:
            batch_size = logits.size(0)
            # 정확한 추정을 위해 (N-1)/(B-1) 사용 (User 본인의 Pos 제외)
            correction = math.log((self.n_items - 1) / (batch_size - 1 + 1e-9))
            
            # Diagonal(Pos)은 건드리지 않고, Off-Diagonal(Neg)에만 보정항 추가
            # Mask 생성: Identity Matrix (1 on diagonal)
            mask = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
            
            # Negatives(False in mask) apply correction
            # In-place add to save memory
            logits = logits + (~mask) * correction
        
        # 5. Cross Entropy
        labels = torch.arange(len(user_intensities), device=user_intensities.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss



class orthogonal_loss(nn.Module):
    """
    직교성(Orthogonality) 손실 함수
    """
    def __init__(self, losstype):
        super(orthogonal_loss, self).__init__()
        self.loss_type = losstype

    def forward(self, keys):
        K = keys.size(0)
        keys_normalized = F.normalize(keys, p=2, dim=1)
        cosine_similarity = torch.matmul(keys_normalized, keys_normalized.t())
        identity_matrix = torch.eye(K, device=keys.device)
        num_off_diagonal_elements = K * K - K
        if self.loss_type == 'l1':
            identity_matrix = torch.eye(keys.size(0), device=keys.device)
        # 제곱 대신 절댓값을 사용하여 수치적 안정성 확보    
            loss = torch.abs(cosine_similarity - identity_matrix).sum()
            return loss / num_off_diagonal_elements 
        elif self.loss_type == 'l2':
            l2_sum_loss = ((cosine_similarity - identity_matrix) ** 2).sum()
            return l2_sum_loss / num_off_diagonal_elements 
        
    def l1_orthogonal_loss(keys):
        """
        Enforces orthogonality on the interest keys.

        Args:
            keys (torch.Tensor): The global interest keys tensor. [num_interests, D]

        Returns:
            torch.Tensor: The orthogonality loss.
        """
        K = keys.size(0)
        keys_normalized = F.normalize(keys, p=2, dim=1)
        cosine_similarity = torch.matmul(keys_normalized, keys_normalized.t())
        identity_matrix = torch.eye(K, device=keys.device)
        num_off_diagonal_elements = K * K - K
        identity_matrix = torch.eye(keys.size(0), device=keys.device)
        # 제곱 대신 절댓값을 사용하여 수치적 안정성 확보
        loss = torch.abs(cosine_similarity - identity_matrix).sum()
        return loss / num_off_diagonal_elements 

    def l2_orthogonal_loss(keys):
        """
        Enforces orthogonality on the interest keys.

        Args:
            keys (torch.Tensor): The global interest keys tensor. [num_interests, D]

        Returns:
            torch.Tensor: The orthogonality loss.
        """
        K = keys.size(0)
        keys_normalized = F.normalize(keys, p=2, dim=1)
        cosine_similarity = torch.matmul(keys_normalized, keys_normalized.t())
        identity_matrix = torch.eye(K, device=keys.device)
        off_diag = cosine_similarity - identity_matrix
            # 1. 제곱의 합 (L2 Loss)
        l2_sum_loss = (off_diag ** 2).sum()
            # 2. 비대각선 요소의 개수로 나누어 정규화 (Normalization)
        num_off_diagonal_elements = K * K - K
        normalized_loss = l2_sum_loss / num_off_diagonal_elements 
        return normalized_loss 

class EntropyAdaptiveInfoNCE(nn.Module):
    """
    SCI 논문용: Decoupled Adaptive InfoNCE
    - get_adaptive_tau: 학습 상태(Entropy, Orthogonality)를 진단하여 온도를 결정 (Gradient 차단됨)
    - forward: 결정된 온도로 실제 Loss 계산 (Unidirectional for efficiency)
    """
    def __init__(self, base_tau=0.1, alpha=0.5, eps=1e-8):
        super().__init__()
        self.base_tau = base_tau
        self.alpha = alpha
        self.eps = eps

    def _calc_uncertainty(self, z):
        """Interest Vector의 불확실성(Entropy) 측정"""
        B, K = z.size()
        p = F.softmax(z, dim=1)
        log_p = F.log_softmax(z, dim=1)
        entropy = -(p * log_p).sum(dim=1)
        max_entropy = math.log(K + self.eps)
        return entropy / max_entropy # [B]

    def _calc_key_quality(self, W):
        """Key의 직교성 품질 측정 (1.0 = 완벽)"""
        K = W.size(0)
        W_norm = F.normalize(W, dim=1)
        gram = W_norm @ W_norm.T
        eye = torch.eye(K, device=W.device)
        # 비대각 원소의 평균 크기
        off_diag_mean = (gram - eye).abs().sum() / (K * (K - 1) + self.eps)
        return 1.0 - off_diag_mean

    def get_adaptive_tau(self, interests, W):
        """
        외부에서 호출: 현재 배치의 Interest와 Key 상태를 보고 Tau를 반환
        """
        # 1. Key Quality (Global) - 안전장치: no_grad
        with torch.no_grad():
            key_quality = self._calc_key_quality(W)
        
        # 2. Instance Uncertainty (Local)
        # interest 자체는 gradient가 필요할 수 있으나, tau 계산 목적으론 끊는게 안전함
        with torch.no_grad(): 
            uncertainty = self._calc_uncertainty(interests)
            
            # 3. Decoupled Adaptation Logic
            # Uncertainty 높음(Sparse) -> Tau 증가 (Soft)
            # Quality 높음(Orthogonal) -> Tau 감소 (Sharp)
            net_difficulty = uncertainty - key_quality
            
            tau = self.base_tau * torch.exp(self.alpha * net_difficulty)
            tau = torch.clamp(tau, min=0.05, max=0.5).unsqueeze(1) # [B, 1]
        
        return tau

    def forward(self, view1, view2, tau):
        """
        Unidirectional InfoNCE
        - view1, view2: [B, K] (Augmented Views)
        - tau: [B, 1] (Adaptive Temperature)
        """
        B = view1.size(0)
        z1 = F.normalize(view1, dim=1)
        z2 = F.normalize(view2, dim=1)
        
        # Cosine Similarity / Adaptive Tau
        logits = (z1 @ z2.T) / tau
        
        # Labels: Diagonal is positive
        labels = torch.arange(B, device=z1.device)
        
        # One-way Cross Entropy (속도 2배, 메모리 절약)
        return F.cross_entropy(logits, labels)