import torch
import torch.nn as nn
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